from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.filings import FilingRecord
from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlInstanceRecord
from src.data_pipeline.quality.sec_filings.xbrl_core import instance_problem
from src.data_pipeline.storage.sec_filings.xbrl_core import (
    XBRL_CONTEXTS_TABLE,
    XBRL_UNITS_TABLE,
)

StoredRows = Mapping[str, Sequence[dict[str, Any]]]


@dataclass
class XbrlPlan:
    """How each filing relates to the stored XBRL groups, decided before parsing.

    `new` filings have no stored instance. `recover` filings have exact stored
    instances of which some context or unit groups are entirely missing:
    `missing` names those (instance_id, table) pairs. Complete filings are
    `skipped`; `conflicts` hold (context, reason) pairs.
    """

    new: list[FilingRecord] = field(default_factory=list)
    recover: dict[str, list[XbrlInstanceRecord]] = field(default_factory=dict)
    missing: set[tuple[str, str]] = field(default_factory=set)
    skipped: int = 0
    conflicts: list[tuple[dict[str, str], str]] = field(default_factory=list)


def _children_problem(
    instance: XbrlInstanceRecord, table: str, stored: int, declared: int
) -> str | None:
    if stored in (0, declared):
        return None
    return f"{instance.instance_id} in {table}: {stored} rows, the instance declares {declared}"


def _classify_stored(
    plan: XbrlPlan,
    filing: FilingRecord,
    rows: Sequence[dict[str, Any]],
    documents: Mapping[str, DocumentRecord],
    children: Mapping[str, Mapping[str, int]],
) -> str | None:
    """Place a filing with stored instances in the plan; return a conflict reason.

    Context and unit groups are judged by their row counts: a group is
    missing (recovered now) or holds exactly the declared count; the
    integrity audit checks the rows themselves.
    """

    try:
        instances = [XbrlInstanceRecord.model_validate(row) for row in rows]
    except ValidationError as error:
        return f"stored instance is invalid: {error.errors()[0]['msg']}"
    missing = set()
    for instance in instances:
        problem = instance_problem(instance, documents)
        if problem is not None:
            return f"{instance.instance_id}: {problem}"
        for table, declared in (
            (XBRL_CONTEXTS_TABLE, instance.context_count),
            (XBRL_UNITS_TABLE, instance.unit_count),
        ):
            stored = children[table].get(instance.instance_id, 0)
            problem = _children_problem(instance, table, stored, declared)
            if problem is not None:
                return problem
            if declared and not stored:
                missing.add((instance.instance_id, table))
    if missing:
        plan.recover[filing.filing_id] = instances
        plan.missing |= missing
    else:
        plan.skipped += 1
    return None


def classify_filings(
    filings: Sequence[FilingRecord],
    documents: Mapping[str, DocumentRecord],
    stored_instances: StoredRows,
    children: Mapping[str, Mapping[str, int]],
) -> XbrlPlan:
    """Classify every filing against the three tables before reading Bronze.

    `stored_instances` groups instance rows by filing_id; `children` maps the
    contexts and units tables to their row counts by instance_id;
    `documents` holds at least the stored instances' source documents.
    Child rows without a stored instance, instances of unknown filings,
    invalid instances, partial groups and changed source sets are conflicts.
    """

    plan = XbrlPlan()
    known = {row["instance_id"] for rows in stored_instances.values() for row in rows}
    for table, groups in children.items():
        for instance_id in sorted(groups.keys() - known):
            context = {"instance_id": instance_id, "table": table}
            plan.conflicts.append((context, "rows exist without their instance"))
    filing_ids = {filing.filing_id for filing in filings}
    for filing_id in sorted(stored_instances.keys() - filing_ids):
        plan.conflicts.append(({"filing_id": filing_id}, "instances have no filing"))
    for filing in filings:
        rows = stored_instances.get(filing.filing_id)
        if rows is None:
            plan.new.append(filing)
            continue
        reason = _classify_stored(plan, filing, rows, documents, children)
        if reason is not None:
            plan.conflicts.append(({"filing_id": filing.filing_id}, reason))
    return plan


def instances_difference(
    stored: Sequence[XbrlInstanceRecord], expected: Sequence[XbrlInstanceRecord]
) -> str | None:
    """Compare a filing's stored and re-parsed instances, ignoring `processed_at`."""

    old = {i.instance_id: i.model_dump(exclude={"processed_at"}) for i in stored}
    new = {i.instance_id: i.model_dump(exclude={"processed_at"}) for i in expected}
    if old.keys() != new.keys():
        return f"stored instances {sorted(old)} differ from {sorted(new)}"
    changed = sorted(
        {
            name
            for key, row in new.items()
            for name in row
            if old[key][name] != row[name]
        }
    )
    return f"stored instances differ in {changed}" if changed else None

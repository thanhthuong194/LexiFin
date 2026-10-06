from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel
from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.pipelines.sec_filings.xbrl_core_extraction import (
    BuiltFiling,
    build_rows,
    extract_filings,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_core_state import (
    XbrlPlan,
    classify_filings,
    instances_difference,
)
from src.data_pipeline.quality.sec_filings.xbrl_core import (
    contexts_problem,
    instance_problem,
    units_problem,
)
from src.data_pipeline.storage.iceberg import (
    append_rows,
    current_snapshot_id,
    group_rows_by,
    open_silver_catalog,
    verify_append,
)
from src.data_pipeline.storage.sec_filings.documents import (
    open_documents_table,
    read_documents,
)
from src.data_pipeline.storage.sec_filings.filings import (
    open_filings_table,
    read_filings,
)
from src.data_pipeline.storage.sec_filings.xbrl_core import (
    XBRL_CONTEXTS_TABLE,
    XBRL_CORE_SCHEMAS,
    XBRL_INSTANCES_TABLE,
    XBRL_UNITS_TABLE,
    open_xbrl_core_table,
    read_instance_groups,
)
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)

_KEYS = {
    XBRL_INSTANCES_TABLE: "instance_id",
    XBRL_CONTEXTS_TABLE: "context_key",
    XBRL_UNITS_TABLE: "unit_key",
}


@dataclass
class XbrlCoreStageSummary:
    """Outcome of one XBRL core run (instances, contexts and units).

    Each filing is counted once: without XBRL (parsed, nothing to store, so
    parsed again next run), inserted, skipped (complete, not re-read),
    recovered (missing context or unit groups appended), conflicted or failed.
    A rejected run writes nothing. Instance, context and unit breakdowns
    describe the rows inserted in this run.
    """

    status: Literal["completed", "rejected"] = "rejected"
    filings_considered: int = 0
    filings_without_xbrl: int = 0
    filings_inserted: int = 0
    filings_skipped: int = 0
    filings_recovered: int = 0
    filings_conflicted: int = 0
    filings_failed: int = 0
    instances_inserted: int = 0
    contexts_inserted: int = 0
    units_inserted: int = 0
    inline_instances: int = 0
    xml_only_instances: int = 0
    default_targets: int = 0
    named_targets: int = 0
    instant_contexts: int = 0
    duration_contexts: int = 0
    forever_contexts: int = 0
    simple_units: int = 0
    divide_units: int = 0
    duration_seconds: float = 0.0
    instances_snapshot_before: int | None = None
    instances_snapshot_after: int | None = None
    contexts_snapshot_before: int | None = None
    contexts_snapshot_after: int | None = None
    units_snapshot_before: int | None = None
    units_snapshot_after: int | None = None


def _validate(
    built: Mapping[str, BuiltFiling],
    plan: XbrlPlan,
    documents: Mapping[str, DocumentRecord],
    summary: XbrlCoreStageSummary,
) -> None:
    """Check every built group; recovered filings must re-parse to their stored rows."""

    for filing_id, rows in built.items():
        problem = None
        for instance in rows.instances:
            problem = (
                instance_problem(instance, documents)
                or contexts_problem(instance, rows.contexts[instance.instance_id])
                or units_problem(instance, rows.units[instance.instance_id])
            )
            if problem is not None:
                break
        if problem is None and filing_id in plan.recover:
            problem = instances_difference(plan.recover[filing_id], rows.instances)
        if problem is not None:
            summary.filings_conflicted += 1
            logger.error("xbrl_instance_conflict", filing_id=filing_id, reason=problem)


def _new_rows(
    built: Mapping[str, BuiltFiling], plan: XbrlPlan
) -> dict[str, list[BaseModel]]:
    """Rows to append per table: whole new filings, and missing recovered groups."""

    new_ids = {filing.filing_id for filing in plan.new}
    instances: list[BaseModel] = []
    contexts: list[BaseModel] = []
    units: list[BaseModel] = []
    for filing_id, rows in built.items():
        is_new = filing_id in new_ids
        if is_new:
            instances.extend(rows.instances)
        for instance_id, group in rows.contexts.items():
            if is_new or (instance_id, XBRL_CONTEXTS_TABLE) in plan.missing:
                contexts.extend(group)
        for instance_id, unit_group in rows.units.items():
            if is_new or (instance_id, XBRL_UNITS_TABLE) in plan.missing:
                units.extend(unit_group)
    return {
        XBRL_INSTANCES_TABLE: sorted(instances, key=_order("filing_id", "instance_id")),
        XBRL_CONTEXTS_TABLE: sorted(contexts, key=_order("instance_id", "context_ref")),
        XBRL_UNITS_TABLE: sorted(units, key=_order("instance_id", "unit_ref")),
    }


def _order(*columns: str) -> Callable[[BaseModel], tuple[str, ...]]:
    return lambda row: tuple(getattr(row, column) for column in columns)


def _verify_groups(table: Table, identifier: str, written: Sequence[BaseModel]) -> None:
    """Re-read the table: every group written now is stored exactly."""

    is_instances = identifier == XBRL_INSTANCES_TABLE
    group_key = "filing_id" if is_instances else "instance_id"
    stored = (
        read_instance_groups(table) if is_instances else group_rows_by(table, group_key)
    )
    key = _KEYS[identifier]
    expected: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in written:
        expected[getattr(row, group_key)][getattr(row, key)] = row.model_dump(
            exclude={"processed_at"}
        )
    for group, rows in expected.items():
        found = {
            row[key]: {
                name: value for name, value in row.items() if name != "processed_at"
            }
            for row in stored.get(group, [])
        }
        if found != rows:
            raise ValueError(f"{identifier} rows of {group} differ after commit")


def _write(
    tables: Mapping[str, Table],
    rows: Mapping[str, list[BaseModel]],
    stored_counts: Mapping[str, int],
    snapshots: dict[str, int | None],
) -> None:
    """Append and verify instances, then contexts, then units (parent first)."""

    for identifier, timer in (
        (XBRL_INSTANCES_TABLE, "xbrl_core.write_instances"),
        (XBRL_CONTEXTS_TABLE, "xbrl_core.write_contexts"),
        (XBRL_UNITS_TABLE, "xbrl_core.write_units"),
    ):
        new = rows[identifier]
        if new:
            with timed_operation(timer, records=len(new)):
                append_rows(tables[identifier], new, XBRL_CORE_SCHEMAS[identifier])
        with timed_operation("xbrl_core.verify", table=identifier):
            snapshots[identifier] = verify_append(
                tables[identifier],
                key_column=_KEYS[identifier],
                expected_rows=stored_counts[identifier] + len(new),
                inserted_keys=[getattr(row, _KEYS[identifier]) for row in new],
                snapshot_before=snapshots[identifier],
            )
            _verify_groups(tables[identifier], identifier, new)


def _count_inserted(
    rows: Mapping[str, list[Any]], summary: XbrlCoreStageSummary
) -> None:
    instances = rows[XBRL_INSTANCES_TABLE]
    periods = Counter(context.period_type for context in rows[XBRL_CONTEXTS_TABLE])
    kinds = Counter(unit.unit_kind for unit in rows[XBRL_UNITS_TABLE])
    summary.instances_inserted = len(instances)
    summary.contexts_inserted = len(rows[XBRL_CONTEXTS_TABLE])
    summary.units_inserted = len(rows[XBRL_UNITS_TABLE])
    summary.inline_instances = sum(i.is_inline for i in instances)
    summary.xml_only_instances = len(instances) - summary.inline_instances
    summary.default_targets = sum(
        i.is_inline and i.target_name is None for i in instances
    )
    summary.named_targets = summary.inline_instances - summary.default_targets
    summary.instant_contexts = periods["instant"]
    summary.duration_contexts = periods["duration"]
    summary.forever_contexts = periods["forever"]
    summary.simple_units, summary.divide_units = kinds["simple"], kinds["divide"]


def _run_xbrl(
    catalog: Catalog,
    data_dir: Path,
    processed_at: datetime,
    summary: XbrlCoreStageSummary,
) -> None:
    tables = {}
    for identifier in XBRL_CORE_SCHEMAS:
        tables[identifier], created = open_xbrl_core_table(
            catalog, data_dir, identifier
        )
        name = identifier.split(".", 1)[1]
        logger.info(
            f"{name}_table_{'created' if created else 'loaded'}", table=identifier
        )
    snapshots = {identifier: current_snapshot_id(t) for identifier, t in tables.items()}
    summary.instances_snapshot_before = snapshots[XBRL_INSTANCES_TABLE]
    summary.contexts_snapshot_before = snapshots[XBRL_CONTEXTS_TABLE]
    summary.units_snapshot_before = snapshots[XBRL_UNITS_TABLE]
    summary.instances_snapshot_after = summary.instances_snapshot_before
    summary.contexts_snapshot_after = summary.contexts_snapshot_before
    summary.units_snapshot_after = summary.units_snapshot_before
    with timed_operation("xbrl_core.read_parents"):
        filings = sorted(
            read_filings(open_filings_table(catalog, data_dir)[0]),
            key=lambda f: (f.filing_date, f.filing_id),
        )
        documents = {
            d.document_id: d
            for d in read_documents(open_documents_table(catalog, data_dir)[0])
        }
        stored = read_instance_groups(tables[XBRL_INSTANCES_TABLE])
        children = {
            table: group_rows_by(tables[table], "instance_id")
            for table in (XBRL_CONTEXTS_TABLE, XBRL_UNITS_TABLE)
        }
    summary.filings_considered = len(filings)
    with timed_operation("xbrl_core.validate", phase="existing_state"):
        plan = classify_filings(filings, documents, stored, children)
    for context, reason in plan.conflicts:
        logger.error("xbrl_instance_conflict", reason=reason, **context)
    summary.filings_conflicted, summary.filings_skipped = (
        len(plan.conflicts),
        plan.skipped,
    )
    logger.info("xbrl_instance_skipped", filings=plan.skipped)
    if plan.conflicts:
        return

    by_filing: dict[str, list[DocumentRecord]] = defaultdict(list)
    for document in documents.values():
        by_filing[document.filing_id].append(document)
    candidates = plan.new + [f for f in filings if f.filing_id in plan.recover]
    with timed_operation("xbrl_core.parse", filings=len(candidates)):
        extraction = extract_filings(candidates, by_filing, data_dir)
    summary.filings_conflicted += extraction.conflicts
    summary.filings_failed = extraction.failures
    summary.filings_without_xbrl = extraction.without_xbrl
    built = build_rows(extraction.discovered, processed_at)
    with timed_operation("xbrl_core.validate", phase="built_records"):
        _validate(built, plan, documents, summary)
    if summary.filings_conflicted or summary.filings_failed:
        return

    rows = _new_rows(built, plan)
    stored_counts = {
        XBRL_INSTANCES_TABLE: sum(map(len, stored.values())),
        **{table: sum(map(len, groups.values())) for table, groups in children.items()},
    }
    _write(tables, rows, stored_counts, snapshots)
    summary.instances_snapshot_after = snapshots[XBRL_INSTANCES_TABLE]
    summary.contexts_snapshot_after = snapshots[XBRL_CONTEXTS_TABLE]
    summary.units_snapshot_after = snapshots[XBRL_UNITS_TABLE]
    summary.filings_recovered = len(plan.recover)
    summary.filings_inserted = sum(
        1 for f in plan.new if built.get(f.filing_id) and built[f.filing_id].instances
    )
    _count_inserted(rows, summary)
    logger.info(
        "xbrl_instance_inserted",
        filings=summary.filings_inserted,
        instances=summary.instances_inserted,
    )
    logger.info("xbrl_instance_recovered", filings=summary.filings_recovered)
    summary.status = "completed"


def run_xbrl_core_stage(data_dir: Path, processed_at: datetime) -> XbrlCoreStageSummary:
    """Build `xbrl_instances`, `xbrl_contexts` and `xbrl_units` from XBRL sources.

    Filings are classified against the three tables before reading Bronze:
    complete ones are skipped, recoverable ones re-parsed to append missing
    context or unit groups. Each candidate filing's Inline and XML instance
    documents are verified by offsets and checksum and parsed once. The
    whole batch is validated before writing instances, then contexts, then
    units, each in one append with read-back verification. Any conflict or
    invalid filing rejects the batch; unexpected errors are logged and raised.
    """

    started = perf_counter()
    summary = XbrlCoreStageSummary()
    with log_context(source="sec_filings"):
        logger.info("xbrl_core_stage_started")
        try:
            with (
                timed_operation("xbrl_core.total"),
                open_silver_catalog(data_dir) as catalog,
            ):
                _run_xbrl(catalog, data_dir, processed_at, summary)
        except Exception as error:
            logger.error(
                "xbrl_core_stage_failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise

        summary.duration_seconds = round(perf_counter() - started, 3)
        if summary.status == "completed":
            logger.info("xbrl_core_stage_completed", **asdict(summary))
        else:
            logger.error("xbrl_core_stage_failed", **asdict(summary))
    return summary

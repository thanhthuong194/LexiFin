from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import batched
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel
from pyiceberg.catalog import Catalog
from pyiceberg.expressions import BooleanExpression
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.filings import FilingRecord
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
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.storage.iceberg import (
    append_rows,
    count_groups,
    current_snapshot_id,
    filing_dates_filter,
    open_silver_catalog,
    read_groups_in,
    snapshot_total,
    verify_append,
    verify_written_values,
)
from src.data_pipeline.storage.sec_filings.documents import open_documents_table
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


# Filings whose XBRL sources are parsed, validated and committed together.
FILINGS_PER_COMMIT = 200
# Columns that pin down a written context or unit for read-back verification.
_PINNED = {
    XBRL_CONTEXTS_TABLE: ("instance_id", "context_ref", "context_hash"),
    XBRL_UNITS_TABLE: ("instance_id", "unit_ref", "normalized_expression"),
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


def _verify_written(
    table: Table, identifier: str, written: Sequence[BaseModel]
) -> None:
    """Re-read the rows written now: each is stored with the same values.

    Instances are compared whole; contexts and units by the columns that pin
    them down (their hashes and refs), read without the other columns.
    """

    if identifier == XBRL_INSTANCES_TABLE:
        stored = read_instance_groups(table)
        for row in written:
            values = row.model_dump(exclude={"processed_at"})
            found = [
                {k: v for k, v in r.items() if k != "processed_at"}
                for r in stored.get(values["filing_id"], [])
                if r["instance_id"] == values["instance_id"]
            ]
            if found != [values]:
                raise ValueError(
                    f"{identifier} row {values['instance_id']} differs after commit"
                )
        return
    verify_written_values(
        table, written, key_column=_KEYS[identifier], columns=_PINNED[identifier]
    )


def _write(
    tables: Mapping[str, Table],
    rows: Mapping[str, list[BaseModel]],
    snapshots: dict[str, int | None],
) -> None:
    """Append and verify instances, then contexts, then units (parent first)."""

    for identifier, timer in (
        (XBRL_INSTANCES_TABLE, "xbrl_core.write_instances"),
        (XBRL_CONTEXTS_TABLE, "xbrl_core.write_contexts"),
        (XBRL_UNITS_TABLE, "xbrl_core.write_units"),
    ):
        new = rows[identifier]
        stored = snapshot_total(tables[identifier], "total-records")
        if new:
            with timed_operation(timer, records=len(new)):
                append_rows(tables[identifier], new, XBRL_CORE_SCHEMAS[identifier])
        with timed_operation("xbrl_core.verify", table=identifier):
            snapshots[identifier] = verify_append(
                tables[identifier],
                key_column=_KEYS[identifier],
                expected_rows=stored + len(new),
                inserted_keys=[getattr(row, _KEYS[identifier]) for row in new],
                snapshot_before=snapshots[identifier],
            )
            _verify_written(tables[identifier], identifier, new)


def _count_inserted(
    rows: Mapping[str, list[Any]], summary: XbrlCoreStageSummary
) -> None:
    instances = rows[XBRL_INSTANCES_TABLE]
    periods = Counter(context.period_type for context in rows[XBRL_CONTEXTS_TABLE])
    kinds = Counter(unit.unit_kind for unit in rows[XBRL_UNITS_TABLE])
    inline = sum(i.is_inline for i in instances)
    default = sum(i.is_inline and i.target_name is None for i in instances)
    summary.instances_inserted += len(instances)
    summary.contexts_inserted += len(rows[XBRL_CONTEXTS_TABLE])
    summary.units_inserted += len(rows[XBRL_UNITS_TABLE])
    summary.inline_instances += inline
    summary.xml_only_instances += len(instances) - inline
    summary.default_targets += default
    summary.named_targets += inline - default
    summary.instant_contexts += periods["instant"]
    summary.duration_contexts += periods["duration"]
    summary.forever_contexts += periods["forever"]
    summary.simple_units += kinds["simple"]
    summary.divide_units += kinds["divide"]


def _read_documents(
    table: Table,
    column: str,
    values: Collection[str],
    files: BooleanExpression | None = None,
) -> list[DocumentRecord]:
    groups = (
        read_groups_in(table, column, values)
        if files is None
        else read_groups_in(table, column, values, files=files)
    )
    return [
        DocumentRecord.model_validate(row) for rows in groups.values() for row in rows
    ]


def _run_batch(
    tables: Mapping[str, Table],
    batch: Sequence[FilingRecord],
    plan: XbrlPlan,
    run: tuple[SecBronze, datetime, Table],
    summary: XbrlCoreStageSummary,
) -> bool:
    """Parse, validate and commit one batch of filings; return whether it was written."""

    bronze, processed_at, documents_table = run
    by_filing: dict[str, list[DocumentRecord]] = defaultdict(list)
    for document in _read_documents(
        documents_table,
        "filing_id",
        {filing.filing_id for filing in batch},
        filing_dates_filter({filing.filing_date for filing in batch}),
    ):
        by_filing[document.filing_id].append(document)
    with timed_operation("xbrl_core.parse", filings=len(batch)):
        extraction = extract_filings(batch, by_filing, bronze)
    summary.filings_conflicted += extraction.conflicts
    summary.filings_failed += extraction.failures
    summary.filings_without_xbrl += extraction.without_xbrl
    built = build_rows(extraction.discovered, processed_at)
    documents = {d.document_id: d for group in by_filing.values() for d in group}
    with timed_operation("xbrl_core.validate", phase="built_records"):
        _validate(built, plan, documents, summary)
    if summary.filings_conflicted or summary.filings_failed:
        return False
    rows = _new_rows(built, plan)
    snapshots = {
        XBRL_INSTANCES_TABLE: summary.instances_snapshot_after,
        XBRL_CONTEXTS_TABLE: summary.contexts_snapshot_after,
        XBRL_UNITS_TABLE: summary.units_snapshot_after,
    }
    _write(tables, rows, snapshots)
    summary.instances_snapshot_after = snapshots[XBRL_INSTANCES_TABLE]
    summary.contexts_snapshot_after = snapshots[XBRL_CONTEXTS_TABLE]
    summary.units_snapshot_after = snapshots[XBRL_UNITS_TABLE]
    new_ids = {filing.filing_id for filing in plan.new}
    summary.filings_inserted += sum(
        1
        for filing in batch
        if filing.filing_id in new_ids
        and built.get(filing.filing_id)
        and built[filing.filing_id].instances
    )
    _count_inserted(rows, summary)
    return True


def _run_xbrl(
    catalog: Catalog,
    data_dir: Path,
    run_inputs: tuple[SecBronze, datetime],
    summary: XbrlCoreStageSummary,
) -> None:
    bronze, processed_at = run_inputs
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
        documents_table, _ = open_documents_table(catalog, data_dir)
        stored = read_instance_groups(tables[XBRL_INSTANCES_TABLE])
        sources = {
            d.document_id: d
            for d in _read_documents(
                documents_table,
                "document_id",
                {
                    i
                    for rows in stored.values()
                    for r in rows
                    for i in r["source_document_ids"]
                },
            )
        }
        children = {
            table: count_groups(tables[table], "instance_id")
            for table in (XBRL_CONTEXTS_TABLE, XBRL_UNITS_TABLE)
        }
    summary.filings_considered = len(filings)
    with timed_operation("xbrl_core.validate", phase="existing_state"):
        plan = classify_filings(filings, sources, stored, children)
    for context, reason in plan.conflicts:
        logger.error("xbrl_instance_conflict", reason=reason, **context)
    summary.filings_conflicted, summary.filings_skipped = (
        len(plan.conflicts),
        plan.skipped,
    )
    logger.info("xbrl_instance_skipped", filings=plan.skipped)
    if plan.conflicts:
        return

    candidates = plan.new + [f for f in filings if f.filing_id in plan.recover]
    for batch in batched(candidates, FILINGS_PER_COMMIT, strict=False):
        if not _run_batch(
            tables, batch, plan, (bronze, processed_at, documents_table), summary
        ):
            return
    summary.filings_recovered = len(plan.recover)
    logger.info(
        "xbrl_instance_inserted",
        filings=summary.filings_inserted,
        instances=summary.instances_inserted,
    )
    logger.info("xbrl_instance_recovered", filings=summary.filings_recovered)
    summary.status = "completed"


def run_xbrl_core_stage(
    data_dir: Path, processed_at: datetime, bronze: SecBronze | None = None
) -> XbrlCoreStageSummary:
    """Build `xbrl_instances`, `xbrl_contexts` and `xbrl_units` from XBRL sources.

    Filings are classified against the three tables before reading Bronze:
    complete ones are skipped, recoverable ones re-parsed to append missing
    context or unit groups. Each candidate filing's Inline and XML instance
    documents are verified by offsets and checksum and parsed once. Stored
    contexts and units are judged by their row counts; any conflict rejects
    the run before the first write. Candidates are validated and committed
    in batches of FILINGS_PER_COMMIT filings: instances, then contexts, then
    units, each verified after its append. An invalid filing rejects its
    batch; earlier batches stay committed. Unexpected errors are logged and
    raised.
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
                _run_xbrl(
                    catalog,
                    data_dir,
                    (bronze or SecBronze.under(data_dir), processed_at),
                    summary,
                )
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

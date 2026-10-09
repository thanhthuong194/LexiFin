from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from itertools import batched
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.content_blocks import ContentBlockRecord
from src.data_pipeline.contracts.sec_filings.document_table_cells import (
    DOCUMENT_TABLE_CELLS_ARROW_SCHEMA,
)
from src.data_pipeline.contracts.sec_filings.document_tables import (
    DOCUMENT_TABLES_ARROW_SCHEMA,
)
from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.pipelines.sec_filings.document_table_extraction import (
    BuiltTables,
    extract_document_tables,
)
from src.data_pipeline.pipelines.sec_filings.document_table_state import (
    BlockHead,
    TableHead,
    TablePlan,
    classify_documents,
    document_context,
    stored_state,
    table_difference,
    verify_written,
)
from src.data_pipeline.quality.sec_filings.document_tables import (
    cells_problem,
    tables_problem,
)
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.storage.iceberg import (
    GroupSummary,
    append_rows,
    current_snapshot_id,
    filing_dates_filter,
    iter_rows,
    open_silver_catalog,
    read_groups_in,
    snapshot_total,
    summarize_groups,
    verify_append,
)
from src.data_pipeline.storage.sec_filings.content_blocks import (
    open_content_blocks_table,
)
from src.data_pipeline.storage.sec_filings.document_table_cells import (
    DOCUMENT_TABLE_CELLS_TABLE,
    open_document_table_cells_table,
)
from src.data_pipeline.storage.sec_filings.document_tables import (
    DOCUMENT_TABLES_TABLE,
    open_document_tables_table,
)
from src.data_pipeline.storage.sec_filings.documents import (
    open_documents_table,
    read_primary_html_documents,
)
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)

# Documents parsed, validated and committed together: memory holds one
# batch's tables and cells, and each batch is one commit per table.
DOCUMENTS_PER_COMMIT = 100
# Columns read to classify documents, without block text or cell content.
_LINEAGE = ("filing_id", "filing_date", "source_document_sha256")
_BLOCK_HEAD = ("document_id", "block_id", "block_ordinal", "block_type", "source_xpath")
_TABLE_HEAD = (
    "document_id",
    "table_id",
    "table_ordinal",
    "document_table_count",
    "block_id",
    "source_xpath",
    "cell_count",
)


@dataclass
class DocumentTablesStageSummary:
    """Outcome of one `document_tables` + `document_table_cells` run.

    Each primary HTML document is counted once: without tables (no table
    blocks, nothing stored), inserted, skipped (both groups complete), resumed
    (exact stored tables, cells appended now), conflicted or failed. A rejected
    run counts would-be inserts and resumes in neither. `empty_cells` and
    `invalid_span_warnings` describe the documents built in this run.
    """

    status: Literal["completed", "rejected"] = "rejected"
    documents_considered: int = 0
    documents_without_tables: int = 0
    documents_inserted: int = 0
    documents_skipped: int = 0
    documents_resumed: int = 0
    documents_conflicted: int = 0
    documents_failed: int = 0
    table_blocks: int = 0
    tables_written: int = 0
    cells_written: int = 0
    empty_cells: int = 0
    invalid_span_warnings: int = 0
    duration_seconds: float = 0.0
    tables_snapshot_before: int | None = None
    tables_snapshot_after: int | None = None
    cells_snapshot_before: int | None = None
    cells_snapshot_after: int | None = None


def _conflict(summary: DocumentTablesStageSummary, reason: str, **context: Any) -> None:
    summary.documents_conflicted += 1
    logger.error("table_document_conflict", reason=reason, **context)


def _validate_built(
    plan: TablePlan,
    built: Mapping[str, BuiltTables],
    summary: DocumentTablesStageSummary,
) -> None:
    """Check every built group; resumed tables must equal the stored ones."""

    for parent in plan.candidates:
        result = built.get(parent.document_id)
        if result is None:
            continue
        blocks = plan.blocks[parent.document_id]
        problem = tables_problem(parent, blocks, result.tables) or cells_problem(
            blocks, result.tables, result.cells
        )
        if problem is not None:
            summary.documents_failed += 1
            logger.error(
                "table_document_failed", error=problem, **document_context(parent)
            )
        elif parent.document_id in plan.resumed:
            difference = table_difference(
                plan.resumed[parent.document_id], result.tables
            )
            if difference is not None:
                _conflict(summary, difference, **document_context(parent))


def _write(
    tables: tuple[Table, Table],
    plan: TablePlan,
    built: Mapping[str, BuiltTables],
    summary: DocumentTablesStageSummary,
) -> None:
    """Append the batch's new tables, verify them, then append and verify its cells."""

    tables_table, cells_table = tables
    new_tables = [
        table
        for parent in plan.parents
        if parent.document_id not in plan.resumed
        for table in built[parent.document_id].tables
    ]
    new_cells = [cell for p in plan.parents for cell in built[p.document_id].cells]
    built_tables = {doc: result.tables for doc, result in built.items()}
    rows = (
        snapshot_total(tables_table, "total-records"),
        snapshot_total(cells_table, "total-records"),
    )
    if new_tables:
        with timed_operation("tables.write_tables", records=len(new_tables)):
            append_rows(tables_table, new_tables, DOCUMENT_TABLES_ARROW_SCHEMA)
    with timed_operation("tables.verify", table=DOCUMENT_TABLES_TABLE):
        summary.tables_snapshot_after = verify_append(
            tables_table,
            key_column="table_id",
            expected_rows=rows[0] + len(new_tables),
            inserted_keys=[table.table_id for table in new_tables],
            snapshot_before=summary.tables_snapshot_after,
        )
        verify_written((tables_table, None), plan, built_tables)
    if new_cells:
        with timed_operation("tables.write_cells", records=len(new_cells)):
            append_rows(cells_table, new_cells, DOCUMENT_TABLE_CELLS_ARROW_SCHEMA)
    with timed_operation("tables.verify", table=DOCUMENT_TABLE_CELLS_TABLE):
        summary.cells_snapshot_after = verify_append(
            cells_table,
            key_column="cell_id",
            expected_rows=rows[1] + len(new_cells),
            inserted_keys=[cell.cell_id for cell in new_cells],
            snapshot_before=summary.cells_snapshot_after,
        )
        verify_written((tables_table, cells_table), plan, built_tables)
    summary.documents_inserted += len(plan.parents) - len(plan.resumed)
    summary.documents_resumed += len(plan.resumed)
    summary.tables_written += len(new_tables)
    summary.cells_written += len(new_cells)
    logger.info(
        "table_document_committed",
        documents=len(plan.parents),
        resumed=len(plan.resumed),
        tables=len(new_tables),
        cells=len(new_cells),
    )


def _open_tables(catalog: Catalog, data_dir: Path) -> tuple[Table, Table]:
    tables_table, created = open_document_tables_table(catalog, data_dir)
    logger.info(
        "document_tables_created" if created else "document_tables_loaded",
        table=DOCUMENT_TABLES_TABLE,
        location=tables_table.location(),
    )
    cells_table, created = open_document_table_cells_table(catalog, data_dir)
    logger.info(
        "document_table_cells_created" if created else "document_table_cells_loaded",
        table=DOCUMENT_TABLE_CELLS_TABLE,
        location=cells_table.location(),
    )
    return tables_table, cells_table


def _lineage(row: Mapping[str, Any]) -> tuple[str, date, str]:
    return (row["filing_id"], row["filing_date"], row["source_document_sha256"])


def _read_state(
    catalog: Catalog, data_dir: Path, tables: tuple[Table, Table]
) -> tuple[
    list[DocumentRecord],
    dict[str, list[BlockHead]],
    dict[str, list[TableHead]],
    dict[str, GroupSummary],
]:
    """Primary HTML documents, their table blocks' heads in block order, the
    stored tables' heads, and a summary of each stored table's cells.

    Only key and lineage columns are read: memory follows the number of
    tables, not of cells.
    """

    documents_table, _ = open_documents_table(catalog, data_dir)
    blocks_table, _ = open_content_blocks_table(catalog, data_dir)
    parents = sorted(
        read_primary_html_documents(documents_table),
        key=lambda parent: (parent.filing_date, parent.document_id),
    )
    blocks: dict[str, list[BlockHead]] = defaultdict(list)
    for row in iter_rows(blocks_table, (*_BLOCK_HEAD, *_LINEAGE)):
        if row["block_type"] == "table":
            blocks[row["document_id"]].append(
                BlockHead(
                    row["block_id"],
                    row["block_ordinal"],
                    row["source_xpath"],
                    _lineage(row),
                )
            )
    for heads in blocks.values():
        heads.sort(key=lambda head: head.block_ordinal)
    stored: dict[str, list[TableHead]] = defaultdict(list)
    for row in iter_rows(tables[0], (*_TABLE_HEAD, *_LINEAGE)):
        stored[row.pop("document_id")].append(
            TableHead(
                **{name: row[name] for name in _TABLE_HEAD[1:]}, lineage=_lineage(row)
            )
        )
    cells = summarize_groups(
        tables[1],
        "table_id",
        "cell_ordinal",
        declared="table_cell_count",
        uniform=("document_id", *_LINEAGE),
    )
    return parents, dict(blocks), dict(stored), cells


def _batch_plan(
    blocks_table: Table,
    tables_table: Table,
    batch: Sequence[DocumentRecord],
    resumed: set[str],
) -> tuple[TablePlan, list[tuple[dict[str, str], str]]]:
    """The batch's full table blocks and its resumed documents' stored tables.

    Returns the plan and conflicts found in the stored tables.
    """

    ids = {parent.document_id for parent in batch}
    files = filing_dates_filter({parent.filing_date for parent in batch})
    block_rows = read_groups_in(blocks_table, "document_id", ids, files=files)
    blocks = {
        document_id: sorted(
            (
                ContentBlockRecord.model_validate(r)
                for r in rows
                if r["block_type"] == "table"
            ),
            key=lambda block: block.block_ordinal,
        )
        for document_id, rows in block_rows.items()
    }
    table_rows = read_groups_in(tables_table, "document_id", ids & resumed, files=files)
    plan = TablePlan(batch, blocks)
    conflicts = []
    for parent in batch:
        if parent.document_id not in resumed:
            continue
        tables, problem = stored_state(
            parent,
            blocks.get(parent.document_id, []),
            table_rows.get(parent.document_id, ()),
            None,
        )
        if problem is not None:
            conflicts.append((document_context(parent), problem))
        plan.resumed[parent.document_id] = tables
    return plan, conflicts


def _run_batch(
    tables: tuple[Table, Table, Table],
    batch: Sequence[DocumentRecord],
    resumed: set[str],
    run: tuple[SecBronze, datetime],
    summary: DocumentTablesStageSummary,
) -> bool:
    """Parse, validate and commit one batch; return whether it was written."""

    blocks_table, tables_table, cells_table = tables
    plan, conflicts = _batch_plan(blocks_table, tables_table, batch, resumed)
    for context, reason in conflicts:
        _conflict(summary, reason, **context)
    if conflicts:
        return False
    extraction = extract_document_tables(plan, *run)
    summary.documents_failed += extraction.failures
    summary.documents_conflicted += extraction.conflicts
    summary.empty_cells += extraction.empty_cells
    summary.invalid_span_warnings += extraction.invalid_spans
    with timed_operation("tables.validate", phase="built_records"):
        _validate_built(plan, extraction.built, summary)
    if summary.documents_failed or summary.documents_conflicted:
        return False
    _write((tables_table, cells_table), plan, extraction.built, summary)
    return True


def _run_tables(
    catalog: Catalog,
    data_dir: Path,
    run: tuple[SecBronze, datetime],
    summary: DocumentTablesStageSummary,
) -> None:
    tables_table, cells_table = _open_tables(catalog, data_dir)
    summary.tables_snapshot_before = summary.tables_snapshot_after = (
        current_snapshot_id(tables_table)
    )
    summary.cells_snapshot_before = summary.cells_snapshot_after = current_snapshot_id(
        cells_table
    )
    with timed_operation("tables.read_state"):
        parents, blocks, stored, cells = _read_state(
            catalog, data_dir, (tables_table, cells_table)
        )
    summary.documents_considered = len(parents)
    summary.table_blocks = sum(len(group) for group in blocks.values())
    logger.info(
        "table_blocks_loaded", documents=len(parents), table_blocks=summary.table_blocks
    )

    with timed_operation("tables.validate", phase="existing_state"):
        result = classify_documents(parents, blocks, stored, cells)
    summary.documents_without_tables = result.without_tables
    summary.documents_skipped = result.skipped
    for context, reason in result.conflicts:
        _conflict(summary, reason, **context)
    logger.info(
        "table_document_skipped",
        documents=result.skipped,
        documents_without_tables=result.without_tables,
    )
    if summary.documents_conflicted:
        return
    blocks_table, _ = open_content_blocks_table(catalog, data_dir)
    resumed = {parent.document_id for parent in result.resumed}
    candidates = sorted(
        [*result.new, *result.resumed],
        key=lambda parent: (parent.filing_date, parent.document_id),
    )
    for batch in batched(candidates, DOCUMENTS_PER_COMMIT, strict=False):
        if not _run_batch(
            (blocks_table, tables_table, cells_table),
            batch,
            resumed,
            run,
            summary,
        ):
            return
    summary.status = "completed"


def run_document_tables_stage(
    data_dir: Path, processed_at: datetime, bronze: SecBronze | None = None
) -> DocumentTablesStageSummary:
    """Build `sec_filings.document_tables` and `sec_filings.document_table_cells`.

    Every `table` content block gets one table row and its cells. Documents
    are classified against table heads and per-table cell summaries before
    any parsing; any conflict rejects the run before the first write.
    Documents to build are parsed, validated and committed in batches of
    DOCUMENTS_PER_COMMIT: tables first, verified, then cells, so a failed
    cells commit is resumed by the next run. An invalid document rejects its
    batch before that batch's first write; earlier batches stay committed.
    Unexpected errors are logged and re-raised.
    """

    started = perf_counter()
    summary = DocumentTablesStageSummary()
    with log_context(source="sec_filings"):
        logger.info("tables_stage_started")
        try:
            with (
                timed_operation("tables.total"),
                open_silver_catalog(data_dir) as catalog,
            ):
                _run_tables(
                    catalog,
                    data_dir,
                    (bronze or SecBronze.under(data_dir), processed_at),
                    summary,
                )
        except Exception as error:
            logger.error(
                "tables_stage_failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise

        summary.duration_seconds = round(perf_counter() - started, 3)
        if summary.status == "completed":
            logger.info("tables_stage_completed", **asdict(summary))
        else:
            logger.error("tables_stage_failed", **asdict(summary))
    return summary

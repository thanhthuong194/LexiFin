from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
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
from src.data_pipeline.pipelines.sec_filings.document_table_extraction import (
    BuiltTables,
    extract_document_tables,
)
from src.data_pipeline.pipelines.sec_filings.document_table_state import (
    TablePlan,
    classify_documents,
    document_context,
    table_difference,
    verify_written,
)
from src.data_pipeline.quality.sec_filings.document_tables import (
    cells_problem,
    tables_problem,
)
from src.data_pipeline.storage.iceberg import (
    append_rows,
    current_snapshot_id,
    group_rows_by,
    open_silver_catalog,
    verify_append,
)
from src.data_pipeline.storage.sec_filings.content_blocks import (
    open_content_blocks_table,
    read_table_blocks,
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
    stored_rows: tuple[int, int],
    summary: DocumentTablesStageSummary,
) -> None:
    """Append new tables, verify them, then append and verify the cells."""

    tables_table, cells_table = tables
    new_ids = {parent.document_id for parent in plan.new}
    new_tables = sorted(
        (table for doc in new_ids for table in built[doc].tables),
        key=lambda t: (t.filing_date, t.document_id, t.table_ordinal),
    )
    new_cells = sorted(
        (
            cell
            for parent in plan.candidates
            for cell in built[parent.document_id].cells
        ),
        key=lambda c: (c.filing_date, c.document_id, c.table_id, c.cell_ordinal),
    )
    built_tables = {doc: result.tables for doc, result in built.items()}
    if new_tables:
        with timed_operation("tables.write_tables", records=len(new_tables)):
            append_rows(tables_table, new_tables, DOCUMENT_TABLES_ARROW_SCHEMA)
    with timed_operation("tables.verify", table=DOCUMENT_TABLES_TABLE):
        summary.tables_snapshot_after = verify_append(
            tables_table,
            key_column="table_id",
            expected_rows=stored_rows[0] + len(new_tables),
            inserted_keys=[table.table_id for table in new_tables],
            snapshot_before=summary.tables_snapshot_before,
        )
        verify_written(tables_table, None, plan, built_tables)
    if new_cells:
        with timed_operation("tables.write_cells", records=len(new_cells)):
            append_rows(cells_table, new_cells, DOCUMENT_TABLE_CELLS_ARROW_SCHEMA)
    with timed_operation("tables.verify", table=DOCUMENT_TABLE_CELLS_TABLE):
        summary.cells_snapshot_after = verify_append(
            cells_table,
            key_column="cell_id",
            expected_rows=stored_rows[1] + len(new_cells),
            inserted_keys=[cell.cell_id for cell in new_cells],
            snapshot_before=summary.cells_snapshot_before,
        )
        verify_written(tables_table, cells_table, plan, built_tables)
    summary.documents_inserted, summary.documents_resumed = (
        len(new_ids),
        len(plan.resumed),
    )
    summary.tables_written, summary.cells_written = len(new_tables), len(new_cells)
    logger.info(
        "table_document_inserted", documents=len(new_ids), tables=len(new_tables)
    )
    logger.info("table_document_resumed", documents=len(plan.resumed))


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


def _read_parents(
    catalog: Catalog, data_dir: Path
) -> tuple[list[Any], dict[str, list[ContentBlockRecord]]]:
    """Primary HTML documents and their `table` blocks in block order."""

    documents_table, _ = open_documents_table(catalog, data_dir)
    blocks_table, _ = open_content_blocks_table(catalog, data_dir)
    parents = sorted(
        read_primary_html_documents(documents_table),
        key=lambda parent: (parent.filing_date, parent.document_id),
    )
    blocks: dict[str, list[ContentBlockRecord]] = defaultdict(list)
    for block in sorted(read_table_blocks(blocks_table), key=lambda b: b.block_ordinal):
        blocks[block.document_id].append(block)
    return parents, dict(blocks)


def _run_tables(
    catalog: Catalog,
    data_dir: Path,
    processed_at: datetime,
    summary: DocumentTablesStageSummary,
) -> None:
    tables_table, cells_table = _open_tables(catalog, data_dir)
    summary.tables_snapshot_before = summary.tables_snapshot_after = (
        current_snapshot_id(tables_table)
    )
    summary.cells_snapshot_before = summary.cells_snapshot_after = current_snapshot_id(
        cells_table
    )
    with timed_operation("tables.read_parents"):
        parents, blocks = _read_parents(catalog, data_dir)
        stored_tables = group_rows_by(tables_table, "document_id")
        stored_cells = group_rows_by(cells_table, "document_id")
    summary.documents_considered = len(parents)
    summary.table_blocks = sum(len(group) for group in blocks.values())
    logger.info(
        "table_blocks_loaded", documents=len(parents), table_blocks=summary.table_blocks
    )

    with timed_operation("tables.validate", phase="existing_state"):
        plan = classify_documents(parents, blocks, stored_tables, stored_cells)
    summary.documents_without_tables = plan.without_tables
    summary.documents_skipped = plan.skipped
    for context, reason in plan.conflicts:
        _conflict(summary, reason, **context)
    logger.info(
        "table_document_skipped",
        documents=plan.skipped,
        documents_without_tables=plan.without_tables,
    )
    if summary.documents_conflicted:
        return
    extraction = extract_document_tables(plan, data_dir, processed_at)
    summary.documents_failed += extraction.failures
    summary.documents_conflicted += extraction.conflicts
    summary.empty_cells, summary.invalid_span_warnings = (
        extraction.empty_cells,
        extraction.invalid_spans,
    )
    built = extraction.built
    with timed_operation("tables.validate", phase="built_records"):
        _validate_built(plan, built, summary)
    if summary.documents_failed or summary.documents_conflicted:
        return

    stored_rows = (
        sum(len(rows) for rows in stored_tables.values()),
        sum(len(rows) for rows in stored_cells.values()),
    )
    _write((tables_table, cells_table), plan, built, stored_rows, summary)
    summary.status = "completed"


def run_document_tables_stage(
    data_dir: Path, processed_at: datetime
) -> DocumentTablesStageSummary:
    """Build `sec_filings.document_tables` and `sec_filings.document_table_cells`.

    Every `table` content block gets one table row and its cells. Documents
    are classified against both tables before any parsing; documents to build
    are parsed once and validated as a batch. Tables are appended first and
    verified, then cells, so a failed cells commit is resumed by the next run.
    Any conflict or invalid document rejects the batch before the first
    write; unexpected errors are logged and re-raised.
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
                _run_tables(catalog, data_dir, processed_at, summary)
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

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.data_pipeline.contracts.sec_filings.document_table_cells import (
    DocumentTableCellRecord,
)
from src.data_pipeline.contracts.sec_filings.document_tables import DocumentTableRecord
from src.data_pipeline.pipelines.sec_filings.document_table_state import (
    TablePlan,
    document_context,
)
from src.data_pipeline.quality.sec_filings.document_tables import (
    table_blocks_sync_problem,
)
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.storage.sec_filings.documents import read_document_content
from src.data_pipeline.transforms.sec_filings.document_tables import (
    build_document_tables,
)
from src.data_pipeline.transforms.sec_filings.html_blocks import extract_blocks
from src.data_pipeline.transforms.sec_filings.html_dom import parse_html
from src.data_pipeline.transforms.sec_filings.html_tables import (
    RawTable,
    build_grid,
    extract_table,
)
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)


@dataclass(frozen=True)
class BuiltTables:
    """The table and cell records built for one document."""

    tables: list[DocumentTableRecord]
    cells: list[DocumentTableCellRecord]


@dataclass
class TableExtraction:
    """Records built from Bronze, with the documents that could not be built.

    `failures` are unreadable or unparsable sources; `conflicts` are documents
    whose HTML no longer yields their stored table blocks.
    """

    built: dict[str, BuiltTables] = field(default_factory=dict)
    failures: int = 0
    conflicts: int = 0
    empty_cells: int = 0
    invalid_spans: int = 0


def _extract(
    plan: TablePlan, bronze: SecBronze, result: TableExtraction
) -> dict[str, tuple[RawTable, ...]]:
    """Parse each candidate's verified HTML once and read its tables.

    Tables are found through the stored table blocks' XPaths, after checking
    that the HTML yields exactly those blocks.
    """

    extracted: dict[str, tuple[RawTable, ...]] = {}
    verified_documents = verified_bytes = 0
    for parent in plan.candidates:
        stored = plan.blocks[parent.document_id]
        context = document_context(parent)
        with log_context(document_id=parent.document_id, form=parent.document_type):
            try:
                content = read_document_content(parent, bronze)
                verified_documents += 1
                verified_bytes += len(content)
                root = parse_html(content)
                html = extract_blocks(root)
                problem = table_blocks_sync_problem(html.blocks, stored)
                if problem is not None:
                    result.conflicts += 1
                    logger.error("table_document_conflict", reason=problem, **context)
                    continue
                if html.empty_tables:
                    logger.warning(
                        "empty_table_markup_skipped", tables=html.empty_tables
                    )
                tree = root.getroottree()
                elements = {tree.getpath(table): table for table in root.iter("table")}
                extracted[parent.document_id] = tuple(
                    extract_table(elements[block.source_xpath]) for block in stored
                )
            except (OSError, ValueError) as error:
                result.failures += 1
                logger.error("table_document_failed", error=str(error), **context)
    logger.info(
        "table_document_content_verified",
        documents=verified_documents,
        bytes=verified_bytes,
    )
    return extracted


def _build(
    plan: TablePlan,
    extracted: dict[str, tuple[RawTable, ...]],
    processed_at: datetime,
    result: TableExtraction,
) -> None:
    """Lay out every extracted table and build each document's records."""

    for document_id, raw_tables in extracted.items():
        blocks = plan.blocks[document_id]
        layouts = [build_grid(raw) for raw in raw_tables]
        invalid = sum(layout.invalid_spans for layout in layouts)
        if invalid:
            result.invalid_spans += invalid
            logger.warning("invalid_table_span", document_id=document_id, spans=invalid)
        for block, raw in zip(blocks, raw_tables, strict=True):
            if len(raw.captions) > 1:
                logger.warning(
                    "multiple_table_captions",
                    block_id=block.block_id,
                    captions=len(raw.captions),
                )
        tables, cells = build_document_tables(blocks, layouts, processed_at)
        result.empty_cells += sum(not cell.text for cell in cells)
        logger.info(
            "document_tables_extracted",
            document_id=document_id,
            tables=len(tables),
            cells=len(cells),
        )
        result.built[document_id] = BuiltTables(tables, cells)


def extract_document_tables(
    plan: TablePlan, bronze: SecBronze, processed_at: datetime
) -> TableExtraction:
    """Build table and cell records for the plan's candidate documents.

    Each document's HTML is parsed once (`tables.extract_tables`); the grids
    are then laid out from plain extracted values (`tables.build_grid`), so
    no DOM is kept across documents.
    """

    result = TableExtraction()
    with timed_operation("tables.extract_tables", documents=len(plan.candidates)):
        extracted = _extract(plan, bronze, result)
    with timed_operation("tables.build_grid", documents=len(extracted)):
        _build(plan, extracted, processed_at, result)
    return result

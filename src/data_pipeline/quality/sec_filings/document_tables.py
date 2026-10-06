from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from src.data_pipeline.contracts.sec_filings.content_blocks import ContentBlockRecord
from src.data_pipeline.contracts.sec_filings.document_table_cells import (
    DocumentTableCellRecord,
)
from src.data_pipeline.contracts.sec_filings.document_tables import DocumentTableRecord
from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.quality.sec_filings.narrative import declared_count_problem
from src.data_pipeline.transforms.sec_filings.html_blocks import HtmlBlock
from src.data_pipeline.transforms.sec_filings.html_tables import table_content_sha256


def table_blocks_sync_problem(
    html_blocks: Sequence[HtmlBlock], stored: Sequence[ContentBlockRecord]
) -> str | None:
    """Compare a document's table blocks parsed now with the stored ones.

    Both lists are in block order; XPaths and texts must match one to one.
    A mismatch means the narrative tables predate the current parser.
    """

    actual = [(b.source_xpath, b.text) for b in html_blocks if b.block_type == "table"]
    expected = [(block.source_xpath, block.text) for block in stored]
    if actual == expected:
        return None
    missing = sorted({xpath for xpath, _ in actual} - {xpath for xpath, _ in expected})
    extra = sorted({xpath for xpath, _ in expected} - {xpath for xpath, _ in actual})
    return (
        f"HTML has {len(actual)} tables, content_blocks has {len(expected)} table "
        f"blocks (missing {missing[:3]}, unexpected {extra[:3]}, or text differs); "
        "rebuild the narrative stage"
    )


def tables_problem(
    parent: DocumentRecord,
    blocks: Sequence[ContentBlockRecord],
    tables: Sequence[DocumentTableRecord],
) -> str | None:
    """Return why a document's tables are not exactly one per `table` block.

    `blocks` are the document's table blocks. Checks lineage and source hash,
    declared count against ordinals 1..N, and that ordinal N is the N-th table
    block by block order with the same source XPath.
    """

    if any(
        (t.document_id, t.filing_id, t.filing_date)
        != (parent.document_id, parent.filing_id, parent.filing_date)
        or t.source_document_sha256 != parent.content_sha256
        for t in tables
    ):
        return "tables do not carry their parent's lineage or source hash"
    problem = declared_count_problem(
        [t.table_ordinal for t in tables],
        {t.document_table_count for t in tables},
        "tables",
    )
    if problem is not None:
        return problem
    ordered_blocks = sorted(blocks, key=lambda block: block.block_ordinal)
    ordered_tables = sorted(tables, key=lambda table: table.table_ordinal)
    expected = [(b.block_id, b.source_xpath) for b in ordered_blocks]
    if [(t.block_id, t.source_xpath) for t in ordered_tables] != expected:
        return f"{len(tables)} tables do not match the {len(blocks)} table blocks"
    return None


def _cell_group_problem(
    table: DocumentTableRecord, cells: Sequence[DocumentTableCellRecord]
) -> str | None:
    """Check that a table's cells are complete and carry the table's lineage."""

    problem = declared_count_problem(
        [c.cell_ordinal for c in cells], {c.table_cell_count for c in cells}, "cells"
    )
    if problem is not None or len(cells) != table.cell_count:
        return (
            problem or f"{len(cells)} cells, but the table declares {table.cell_count}"
        )
    if any(
        (c.document_id, c.filing_id, c.filing_date, c.source_document_sha256)
        != (
            table.document_id,
            table.filing_id,
            table.filing_date,
            table.source_document_sha256,
        )
        for c in cells
    ):
        return "cells do not carry their table's lineage or source hash"
    if sum(c.is_header for c in cells) != table.header_cell_count:
        return "header_cell_count does not match the <th> cells"
    return None


def _cell_content_problem(
    block: ContentBlockRecord,
    table: DocumentTableRecord,
    cells: Sequence[DocumentTableCellRecord],
) -> str | None:
    """Check grid extent, content hash and block text against the cells."""

    if max(c.row_ordinal for c in cells) > table.row_count or table.column_count != max(
        c.column_ordinal + c.colspan - 1 for c in cells
    ):
        return "cells fall outside the table's rows or do not span its columns"
    if table.table_content_sha256 != table_content_sha256(
        table.caption, table.row_count, table.column_count, cells
    ):
        return "table_content_sha256 does not match the cells"
    ordered = sorted(cells, key=lambda cell: cell.cell_ordinal)
    if block.text != "\n".join(cell.text for cell in ordered if cell.text):
        return "the table block's text differs from its non-empty cell texts"
    return None


def cells_problem(
    blocks: Sequence[ContentBlockRecord],
    tables: Sequence[DocumentTableRecord],
    cells: Sequence[DocumentTableCellRecord],
) -> str | None:
    """Return why a document's cells do not complete its tables, or None.

    `tables` must already pass `tables_problem`. Every table needs exactly its
    declared cells (ordinals 1..N), matching header count, grid extent and
    content hash, and its block text must join the non-empty cell texts.
    """

    by_table: dict[str, list[DocumentTableCellRecord]] = defaultdict(list)
    for cell in cells:
        by_table[cell.table_id].append(cell)
    if set(by_table) - {table.table_id for table in tables}:
        return "cells belong to tables that are not stored"
    blocks_by_id = {block.block_id: block for block in blocks}
    for table in tables:
        group = by_table.get(table.table_id)
        if not group:
            return f"{table.table_id} has no cells"
        problem = _cell_group_problem(table, group) or _cell_content_problem(
            blocks_by_id[table.block_id], table, group
        )
        if problem is not None:
            return f"{table.table_id}: {problem}"
    return None

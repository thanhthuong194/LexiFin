from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from src.data_pipeline.contracts.sec_filings.content_blocks import (
    ContentBlockRecord,
    sha256_text,
)
from src.data_pipeline.contracts.sec_filings.document_table_cells import (
    DocumentTableCellRecord,
    cell_id_for,
)
from src.data_pipeline.contracts.sec_filings.document_tables import (
    DocumentTableRecord,
    table_id_for,
)
from src.data_pipeline.transforms.sec_filings.html_tables import (
    TableLayout,
    table_content_sha256,
)


def build_document_tables(
    blocks: Sequence[ContentBlockRecord],
    layouts: Sequence[TableLayout],
    processed_at: datetime,
) -> tuple[list[DocumentTableRecord], list[DocumentTableCellRecord]]:
    """Build the table and cell records of one document.

    `blocks` are the document's `table` content blocks in block order and
    `layouts` their laid-out tables, one each; table ordinals follow that order.
    """

    tables: list[DocumentTableRecord] = []
    cells: list[DocumentTableCellRecord] = []
    for ordinal, (block, layout) in enumerate(zip(blocks, layouts, strict=True), 1):
        table_id = table_id_for(block.block_id)
        tables.append(
            DocumentTableRecord(
                table_id=table_id,
                block_id=block.block_id,
                filing_id=block.filing_id,
                document_id=block.document_id,
                filing_date=block.filing_date,
                table_ordinal=ordinal,
                document_table_count=len(blocks),
                caption=layout.caption,
                row_count=layout.row_count,
                column_count=layout.column_count,
                cell_count=len(layout.cells),
                header_cell_count=sum(cell.is_header for cell in layout.cells),
                source_xpath=block.source_xpath,
                source_document_sha256=block.source_document_sha256,
                table_content_sha256=table_content_sha256(
                    layout.caption, layout.row_count, layout.column_count, layout.cells
                ),
                processed_at=processed_at,
            )
        )
        cells.extend(
            DocumentTableCellRecord.model_validate(
                {
                    "cell_id": cell_id_for(table_id, cell.cell_ordinal),
                    "filing_id": block.filing_id,
                    "document_id": block.document_id,
                    "table_id": table_id,
                    "filing_date": block.filing_date,
                    "cell_ordinal": cell.cell_ordinal,
                    "table_cell_count": len(layout.cells),
                    "row_ordinal": cell.row_ordinal,
                    "row_group": cell.row_group,
                    "row_cell_ordinal": cell.row_cell_ordinal,
                    "column_ordinal": cell.column_ordinal,
                    "rowspan": cell.rowspan,
                    "colspan": cell.colspan,
                    "is_header": cell.is_header,
                    "scope": cell.scope,
                    "text": cell.text,
                    "text_length": len(cell.text),
                    "text_sha256": sha256_text(cell.text),
                    "source_tag": cell.source_tag,
                    "source_xpath": cell.source_xpath,
                    "source_document_sha256": block.source_document_sha256,
                    "processed_at": processed_at,
                }
            )
            for cell in layout.cells
        )
    return tables, cells

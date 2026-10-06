from __future__ import annotations

from datetime import date
from typing import Literal

import pyarrow as pa
from pydantic import AwareDatetime, Field, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel
from src.data_pipeline.contracts.sec_filings.content_blocks import sha256_text
from src.data_pipeline.contracts.sec_filings.filings import INT32_MAX, SHA256_PATTERN

RowGroup = Literal["header", "body", "footer", "unknown"]

DOCUMENT_TABLE_CELLS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("cell_id", pa.string(), nullable=False),
        pa.field("filing_id", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("table_id", pa.string(), nullable=False),
        pa.field("filing_date", pa.date32(), nullable=False),
        pa.field("cell_ordinal", pa.int32(), nullable=False),
        pa.field("table_cell_count", pa.int32(), nullable=False),
        pa.field("row_ordinal", pa.int32(), nullable=False),
        pa.field("row_group", pa.string(), nullable=False),
        pa.field("row_cell_ordinal", pa.int32(), nullable=False),
        pa.field("column_ordinal", pa.int32(), nullable=False),
        pa.field("rowspan", pa.int32(), nullable=False),
        pa.field("colspan", pa.int32(), nullable=False),
        pa.field("is_header", pa.bool_(), nullable=False),
        pa.field("scope", pa.string(), nullable=True),
        pa.field("text", pa.string(), nullable=False),
        pa.field("text_length", pa.int64(), nullable=False),
        pa.field("text_sha256", pa.string(), nullable=False),
        pa.field("source_tag", pa.string(), nullable=False),
        pa.field("source_xpath", pa.string(), nullable=False),
        pa.field("source_document_sha256", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def cell_id_for(table_id: str, cell_ordinal: int) -> str:
    """Return the internal key of the N-th physical cell of a table."""

    return f"{table_id}:cell:{cell_ordinal:06d}"


class DocumentTableCellRecord(ContractModel):
    """One physical `<td>`/`<th>` of a document table, placed on its logical grid.

    Field order matches DOCUMENT_TABLE_CELLS_ARROW_SCHEMA. Ordinals start at 1:
    `cell_ordinal` in DOM order within the table, `row_ordinal` among the
    table's physical rows, `column_ordinal` the first logical column the cell
    covers. Empty cells keep `text=""`. `is_header` means a `<th>`.
    """

    cell_id: str
    filing_id: str
    document_id: str
    table_id: str
    filing_date: date
    cell_ordinal: int = Field(ge=1, le=INT32_MAX)
    table_cell_count: int = Field(ge=1, le=INT32_MAX)
    row_ordinal: int = Field(ge=1, le=INT32_MAX)
    row_group: RowGroup
    row_cell_ordinal: int = Field(ge=1, le=INT32_MAX)
    column_ordinal: int = Field(ge=1, le=INT32_MAX)
    rowspan: int = Field(ge=1, le=INT32_MAX)
    colspan: int = Field(ge=1, le=INT32_MAX)
    is_header: bool
    scope: str | None = Field(min_length=1)
    text: str
    text_length: int = Field(ge=0)
    text_sha256: str = Field(pattern=SHA256_PATTERN)
    source_tag: Literal["td", "th"]
    source_xpath: str = Field(min_length=1)
    source_document_sha256: str = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_derived_fields(self) -> DocumentTableCellRecord:
        if self.cell_id != cell_id_for(self.table_id, self.cell_ordinal):
            raise ValueError("cell_id must be derived from table_id and ordinal")
        if self.cell_ordinal > self.table_cell_count:
            raise ValueError("cell_ordinal exceeds table_cell_count")
        if self.text_length != len(self.text):
            raise ValueError("text_length must equal the character count of text")
        if self.text_sha256 != sha256_text(self.text):
            raise ValueError("text_sha256 does not match text")
        if self.is_header != (self.source_tag == "th"):
            raise ValueError("is_header must be true exactly for <th> cells")
        return self

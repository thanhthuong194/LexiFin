from __future__ import annotations

from datetime import date

import pyarrow as pa
from pydantic import AwareDatetime, Field, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel
from src.data_pipeline.contracts.sec_filings.filings import INT32_MAX, SHA256_PATTERN

DOCUMENT_TABLES_ARROW_SCHEMA = pa.schema(
    [
        pa.field("table_id", pa.string(), nullable=False),
        pa.field("block_id", pa.string(), nullable=False),
        pa.field("filing_id", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("filing_date", pa.date32(), nullable=False),
        pa.field("table_ordinal", pa.int32(), nullable=False),
        pa.field("document_table_count", pa.int32(), nullable=False),
        pa.field("caption", pa.string(), nullable=True),
        pa.field("row_count", pa.int32(), nullable=False),
        pa.field("column_count", pa.int32(), nullable=False),
        pa.field("cell_count", pa.int32(), nullable=False),
        pa.field("header_cell_count", pa.int32(), nullable=False),
        pa.field("source_xpath", pa.string(), nullable=False),
        pa.field("source_document_sha256", pa.string(), nullable=False),
        pa.field("table_content_sha256", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def table_id_for(block_id: str) -> str:
    """Return the key of the table represented by a `table` content block."""

    return f"{block_id}:table"


class DocumentTableRecord(ContractModel):
    """One HTML `<table>` of a primary document; its parent is a `table` block.

    Field order matches DOCUMENT_TABLES_ARROW_SCHEMA. `table_ordinal` orders the
    document's table blocks by block ordinal. `row_count` and `cell_count`
    count physical rows and cells; `column_count` is the logical width after
    row/column spans. `table_content_sha256` hashes the caption, size and
    ordered cells as canonical JSON.
    """

    table_id: str
    block_id: str
    filing_id: str
    document_id: str
    filing_date: date
    table_ordinal: int = Field(ge=1, le=INT32_MAX)
    document_table_count: int = Field(ge=1, le=INT32_MAX)
    caption: str | None = Field(min_length=1)
    row_count: int = Field(ge=1, le=INT32_MAX)
    column_count: int = Field(ge=1, le=INT32_MAX)
    cell_count: int = Field(ge=1, le=INT32_MAX)
    header_cell_count: int = Field(ge=0, le=INT32_MAX)
    source_xpath: str = Field(min_length=1)
    source_document_sha256: str = Field(pattern=SHA256_PATTERN)
    table_content_sha256: str = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_derived_fields(self) -> DocumentTableRecord:
        if self.table_id != table_id_for(self.block_id):
            raise ValueError("table_id must be derived from block_id")
        if self.table_ordinal > self.document_table_count:
            raise ValueError("table_ordinal exceeds document_table_count")
        if self.header_cell_count > self.cell_count:
            raise ValueError("header_cell_count exceeds cell_count")
        return self

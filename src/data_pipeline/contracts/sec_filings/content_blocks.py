from __future__ import annotations

import hashlib
from datetime import date
from typing import Literal

import pyarrow as pa
from pydantic import AwareDatetime, Field, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel
from src.data_pipeline.contracts.sec_filings.filings import INT32_MAX, SHA256_PATTERN

BlockType = Literal[
    "heading", "paragraph", "list_item", "table", "preformatted", "other"
]

CONTENT_BLOCKS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("block_id", pa.string(), nullable=False),
        pa.field("filing_id", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("filing_date", pa.date32(), nullable=False),
        pa.field("section_id", pa.string(), nullable=False),
        pa.field("block_ordinal", pa.int32(), nullable=False),
        pa.field("document_block_count", pa.int32(), nullable=False),
        pa.field("block_type", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("text_length", pa.int64(), nullable=False),
        pa.field("text_sha256", pa.string(), nullable=False),
        pa.field("source_tag", pa.string(), nullable=False),
        pa.field("source_xpath", pa.string(), nullable=False),
        pa.field("heading_level", pa.int32(), nullable=True),
        pa.field("inside_table", pa.bool_(), nullable=False),
        pa.field("source_table_xpath", pa.string(), nullable=True),
        pa.field("source_document_sha256", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def block_id_for(document_id: str, block_ordinal: int) -> str:
    """Return the internal key of the N-th visible-text block of a document."""

    return f"{document_id}:block:{block_ordinal:06d}"


def sha256_text(text: str) -> str:
    """Return the SHA-256 of normalized block or section text (UTF-8)."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ContentBlockRecord(ContractModel):
    """One ordered visible-text block of a primary HTML document.

    Field order matches CONTENT_BLOCKS_ARROW_SCHEMA. `text` is normalized
    visible text; `text_length` counts Unicode characters. `source_xpath` and
    `source_table_xpath` are lxml `getpath()` paths in the parsed (repaired)
    DOM of the document's exact `<TEXT>` content, not raw byte offsets; Inline
    XBRL steps keep their prefixed tag name (`ix:nonnumeric[1]`), which
    standard XPath evaluates as `*[name()='ix:nonnumeric'][1]`. `section_id` is
    the most specific section whose range contains the block.

    Each HTML `<table>` with at least one visible cell is one atomic `table`
    block: `source_xpath` is the table's, `text` joins its non-empty cell
    texts with newlines (empty when every cell is empty), and its rows and
    cells live in `document_tables` and `document_table_cells`. A nested table
    is its own block with `source_table_xpath` naming the enclosing table.
    """

    block_id: str
    filing_id: str
    document_id: str
    filing_date: date
    section_id: str
    block_ordinal: int = Field(ge=1, le=INT32_MAX)
    document_block_count: int = Field(ge=1, le=INT32_MAX)
    block_type: BlockType
    text: str
    text_length: int = Field(ge=0)
    text_sha256: str = Field(pattern=SHA256_PATTERN)
    source_tag: str = Field(min_length=1)
    source_xpath: str = Field(min_length=1)
    heading_level: int | None = Field(ge=1, le=6)
    inside_table: bool
    source_table_xpath: str | None
    source_document_sha256: str = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_derived_fields(self) -> ContentBlockRecord:
        if self.block_id != block_id_for(self.document_id, self.block_ordinal):
            raise ValueError("block_id must be derived from document_id and ordinal")
        if self.block_ordinal > self.document_block_count:
            raise ValueError("block_ordinal exceeds document_block_count")
        if self.text_length != len(self.text):
            raise ValueError("text_length must equal the character count of text")
        if self.text_sha256 != sha256_text(self.text):
            raise ValueError("text_sha256 does not match text")
        is_table = self.block_type == "table"
        if not self.text and not is_table:
            raise ValueError("only a table block may have empty text")
        if is_table != (self.source_tag == "table") or (
            is_table and self.heading_level is not None
        ):
            raise ValueError(
                "a table block has source_tag 'table' and no heading level"
            )
        if self.inside_table != bool(self.source_table_xpath):
            raise ValueError("source_table_xpath must be set exactly when inside_table")
        return self

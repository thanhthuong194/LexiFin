from __future__ import annotations

from datetime import date
from typing import Literal

import pyarrow as pa
from pydantic import AwareDatetime, Field, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel
from src.data_pipeline.contracts.sec_filings.filings import INT32_MAX, SHA256_PATTERN

SectionKind = Literal[
    "document", "table_of_contents", "part", "item", "signature", "exhibit_index"
]

SECTIONS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("section_id", pa.string(), nullable=False),
        pa.field("filing_id", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("filing_date", pa.date32(), nullable=False),
        pa.field("section_ordinal", pa.int32(), nullable=False),
        pa.field("document_section_count", pa.int32(), nullable=False),
        pa.field("parent_section_id", pa.string(), nullable=True),
        pa.field("section_kind", pa.string(), nullable=False),
        pa.field("section_code", pa.string(), nullable=True),
        pa.field("title", pa.string(), nullable=False),
        pa.field("level", pa.int32(), nullable=False),
        pa.field("start_block_ordinal", pa.int32(), nullable=False),
        pa.field("end_block_ordinal", pa.int32(), nullable=False),
        pa.field("source_xpath", pa.string(), nullable=True),
        pa.field("source_document_sha256", pa.string(), nullable=False),
        pa.field("section_text_sha256", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def section_id_for(document_id: str, section_ordinal: int) -> str:
    """Return the internal key of the N-th section of a document (root is 1)."""

    return f"{document_id}:section:{section_ordinal:04d}"


class SectionRecord(ContractModel):
    """One structural section (root, TOC, PART, ITEM, ...) of a primary HTML document.

    Field order matches SECTIONS_ARROW_SCHEMA. The block range is
    [start_block_ordinal, end_block_ordinal): inclusive start, exclusive end,
    covering descendant sections too. `section_text_sha256` hashes the
    newline-joined text of every block in that range. The root section
    (ordinal 1, kind `document`) has no parent, code or source XPath.
    """

    section_id: str
    filing_id: str
    document_id: str
    filing_date: date
    section_ordinal: int = Field(ge=1, le=INT32_MAX)
    document_section_count: int = Field(ge=1, le=INT32_MAX)
    parent_section_id: str | None
    section_kind: SectionKind
    section_code: str | None
    title: str = Field(min_length=1)
    level: int = Field(ge=0, le=INT32_MAX)
    start_block_ordinal: int = Field(ge=1, le=INT32_MAX)
    end_block_ordinal: int = Field(ge=2, le=INT32_MAX)
    source_xpath: str | None
    source_document_sha256: str = Field(pattern=SHA256_PATTERN)
    section_text_sha256: str = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_derived_fields(self) -> SectionRecord:
        if self.section_id != section_id_for(self.document_id, self.section_ordinal):
            raise ValueError("section_id must be derived from document_id and ordinal")
        if self.section_ordinal > self.document_section_count:
            raise ValueError("section_ordinal exceeds document_section_count")
        if self.start_block_ordinal >= self.end_block_ordinal:
            raise ValueError("start_block_ordinal must be before end_block_ordinal")
        is_root = self.section_kind == "document"
        if is_root != (self.section_ordinal == 1):
            raise ValueError("exactly the first section must be the document root")
        if is_root:
            if (self.parent_section_id, self.section_code, self.source_xpath) != (
                None,
                None,
                None,
            ) or (self.level, self.start_block_ordinal) != (0, 1):
                raise ValueError(
                    "root section has no parent/code/xpath, level 0, start 1"
                )
        elif not (self.parent_section_id and self.section_code and self.source_xpath):
            raise ValueError("detected sections need a parent, a code and an xpath")
        return self

from __future__ import annotations

from datetime import date
from typing import Literal

import pyarrow as pa
from pydantic import AwareDatetime, Field, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel
from src.data_pipeline.contracts.sec_filings.filings import INT32_MAX, SHA256_PATTERN

DocumentRole = Literal[
    "primary_document",
    "exhibit",
    "xbrl_schema",
    "xbrl_calculation",
    "xbrl_definition",
    "xbrl_label",
    "xbrl_presentation",
    "xbrl_instance",
    "graphic",
    "generated",
    "other",
]

ContentKind = Literal["html", "xml", "json", "text", "graphic", "binary", "unknown"]

DOCUMENTS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("filing_id", pa.string(), nullable=False),
        pa.field("filing_date", pa.date32(), nullable=False),
        pa.field("document_ordinal", pa.int32(), nullable=False),
        pa.field("sequence", pa.string(), nullable=True),
        pa.field("document_type", pa.string(), nullable=False),
        pa.field("filename", pa.string(), nullable=True),
        pa.field("description", pa.string(), nullable=True),
        pa.field("role", pa.string(), nullable=False),
        pa.field("content_kind", pa.string(), nullable=False),
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("source_submission_sha256", pa.string(), nullable=False),
        pa.field("document_start_byte", pa.int64(), nullable=False),
        pa.field("document_end_byte", pa.int64(), nullable=False),
        pa.field("content_start_byte", pa.int64(), nullable=True),
        pa.field("content_end_byte", pa.int64(), nullable=True),
        pa.field("content_size_bytes", pa.int64(), nullable=False),
        pa.field("content_sha256", pa.string(), nullable=True),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def document_id_for(filing_id: str, document_ordinal: int) -> str:
    """Return the internal key of the N-th `<DOCUMENT>` block of a filing."""

    return f"{filing_id}:document:{document_ordinal:04d}"


class DocumentRecord(ContractModel):
    """One `<DOCUMENT>` block of a full submission in `sec_filings.documents`.

    Field order matches DOCUMENTS_ARROW_SCHEMA. Offsets are zero-based byte
    positions in the raw `source_path` file, start inclusive and end exclusive:
    the document span covers `<DOCUMENT>` through `</DOCUMENT>`, the content
    span covers the exact bytes between `<TEXT>` and `</TEXT>`. A block without
    `<TEXT>` has null content offsets and checksum and a content size of 0.
    """

    document_id: str
    filing_id: str
    filing_date: date
    document_ordinal: int = Field(ge=1, le=INT32_MAX)
    sequence: str | None
    document_type: str = Field(min_length=1)
    filename: str | None
    description: str | None
    role: DocumentRole
    content_kind: ContentKind
    source_path: str
    source_submission_sha256: str = Field(pattern=SHA256_PATTERN)
    document_start_byte: int = Field(ge=0)
    document_end_byte: int = Field(ge=0)
    content_start_byte: int | None = Field(ge=0)
    content_end_byte: int | None = Field(ge=0)
    content_size_bytes: int = Field(ge=0)
    content_sha256: str | None = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_identity_and_offsets(self) -> DocumentRecord:
        if self.document_id != document_id_for(self.filing_id, self.document_ordinal):
            raise ValueError("document_id must be derived from filing_id and ordinal")
        if self.document_start_byte >= self.document_end_byte:
            raise ValueError("document_start_byte must be before document_end_byte")
        content = (self.content_start_byte, self.content_end_byte, self.content_sha256)
        if content == (None, None, None):
            if self.content_size_bytes != 0:
                raise ValueError("A document without <TEXT> must have content size 0")
            return self
        start, end, _ = content
        if start is None or end is None or self.content_sha256 is None:
            raise ValueError("content offsets and checksum must all be set or all null")
        if not self.document_start_byte < start <= end < self.document_end_byte:
            raise ValueError("content offsets must lie inside the document span")
        if self.content_size_bytes != end - start:
            raise ValueError(
                "content_size_bytes must equal content_end - content_start"
            )
        return self

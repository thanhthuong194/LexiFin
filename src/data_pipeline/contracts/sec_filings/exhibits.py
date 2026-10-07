from __future__ import annotations

from datetime import date
from typing import Any

import pyarrow as pa
from pydantic import AwareDatetime, Field, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel
from src.data_pipeline.contracts.sec_filings.filings import INT32_MAX, SHA256_PATTERN
from src.data_pipeline.contracts.sec_filings.xbrl_core import canonical_sha256

EXHIBITS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("exhibit_id", pa.string(), nullable=False),
        pa.field("filing_id", pa.string(), nullable=False),
        pa.field("source_document_id", pa.string(), nullable=False),
        pa.field("filing_date", pa.date32(), nullable=False),
        pa.field("exhibit_ordinal", pa.int32(), nullable=False),
        pa.field("filing_exhibit_count", pa.int32(), nullable=False),
        pa.field("exhibit_number", pa.string(), nullable=False),
        pa.field("description", pa.string(), nullable=True),
        pa.field("href", pa.string(), nullable=True),
        pa.field("resolved_document_id", pa.string(), nullable=True),
        pa.field("source_xpath", pa.string(), nullable=False),
        pa.field("source_text_sha256", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)
# What an Exhibit Index entry discloses.
EXHIBIT_CONTENT_FIELDS = ("exhibit_number", "description", "href")


def exhibit_id_for(filing_id: str, exhibit_ordinal: int) -> str:
    """Key of the N-th Exhibit Index entry of a filing, in document order."""

    return f"{filing_id}:exhibit:{exhibit_ordinal:04d}"


def exhibit_text_sha256(values: dict[str, Any]) -> str:
    """Hash the EXHIBIT_CONTENT_FIELDS of an entry as canonical JSON."""

    return canonical_sha256({name: values[name] for name in EXHIBIT_CONTENT_FIELDS})


class ExhibitRecord(ContractModel):
    """One entry disclosed in the Exhibit Index of a filing's primary document.

    Field order matches EXHIBITS_ARROW_SCHEMA. An entry is not an attachment:
    it may name an exhibit filed with this filing (`resolved_document_id`),
    one incorporated by reference from an earlier filing, or none at all.
    `exhibit_number` is the observed number without footnote markers.
    """

    exhibit_id: str
    filing_id: str
    source_document_id: str
    filing_date: date
    exhibit_ordinal: int = Field(ge=1, le=INT32_MAX)
    filing_exhibit_count: int = Field(ge=1, le=INT32_MAX)
    exhibit_number: str = Field(min_length=1)
    description: str | None = Field(min_length=1)
    href: str | None = Field(min_length=1)
    resolved_document_id: str | None
    source_xpath: str = Field(min_length=1)
    source_text_sha256: str = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_derived_fields(self) -> ExhibitRecord:
        if self.exhibit_id != exhibit_id_for(self.filing_id, self.exhibit_ordinal):
            raise ValueError("exhibit_id must be derived from filing and ordinal")
        if self.exhibit_ordinal > self.filing_exhibit_count:
            raise ValueError("exhibit_ordinal exceeds filing_exhibit_count")
        documents = (self.source_document_id, self.resolved_document_id)
        if any(
            document is not None and not document.startswith(f"{self.filing_id}:")
            for document in documents
        ):
            raise ValueError("source and resolved documents must be of the filing")
        if self.resolved_document_id is not None and self.href is None:
            raise ValueError("only an entry with a link resolves to a document")
        if self.source_text_sha256 != exhibit_text_sha256(self.model_dump()):
            raise ValueError("source_text_sha256 does not match the entry")
        return self

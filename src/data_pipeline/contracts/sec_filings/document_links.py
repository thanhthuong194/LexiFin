from __future__ import annotations

from datetime import date
from typing import Literal

import pyarrow as pa
from pydantic import AwareDatetime, Field, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel
from src.data_pipeline.contracts.sec_filings.filings import INT32_MAX, SHA256_PATTERN

LinkKind = Literal[
    "fragment",
    "relative_document",
    "sec_url",
    "external_url",
    "email",
    "telephone",
    "javascript",
    "data",
    "other",
]
ResolutionStatus = Literal["resolved", "unresolved", "ambiguous", "not_applicable"]

DOCUMENT_LINKS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("link_id", pa.string(), nullable=False),
        pa.field("filing_id", pa.string(), nullable=False),
        pa.field("source_document_id", pa.string(), nullable=False),
        pa.field("target_document_id", pa.string(), nullable=True),
        pa.field("filing_date", pa.date32(), nullable=False),
        pa.field("link_ordinal", pa.int32(), nullable=False),
        pa.field("document_link_count", pa.int32(), nullable=False),
        pa.field("href", pa.string(), nullable=False),
        pa.field("normalized_href", pa.string(), nullable=False),
        pa.field("link_kind", pa.string(), nullable=False),
        pa.field("resolution_status", pa.string(), nullable=False),
        pa.field("target_fragment", pa.string(), nullable=True),
        pa.field("link_text", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=True),
        pa.field("source_xpath", pa.string(), nullable=False),
        pa.field("source_document_sha256", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

# The resolution outcomes each kind of link may have: only links that name a
# document of the filing are resolved against the document inventory.
_STATUSES_BY_KIND: dict[str, frozenset[str]] = {
    "fragment": frozenset({"resolved"}),
    "relative_document": frozenset({"resolved", "unresolved", "ambiguous"}),
    "sec_url": frozenset({"resolved", "unresolved", "ambiguous"}),
    "other": frozenset({"unresolved"}),
    **dict.fromkeys(
        ("external_url", "email", "telephone", "javascript", "data"),
        frozenset({"not_applicable"}),
    ),
}


def link_id_for(source_document_id: str, link_ordinal: int) -> str:
    """Return the internal key of the N-th `<a href>` of a source document."""

    return f"{source_document_id}:link:{link_ordinal:06d}"


class DocumentLinkRecord(ContractModel):
    """One `<a href>` element of an HTML document and the document it targets.

    Field order matches DOCUMENT_LINKS_ARROW_SCHEMA. `source_document_id` and
    `target_document_id` both reference `documents.document_id`; a target is
    set exactly when the link resolves to one document of the same filing (a
    fragment targets its own source). `href` is the parsed attribute value
    (entities decoded), `normalized_href` its NFC, trimmed form. `link_text`
    is normalized visible text, empty for image-only or hidden anchors.
    """

    link_id: str
    filing_id: str
    source_document_id: str
    target_document_id: str | None
    filing_date: date
    link_ordinal: int = Field(ge=1, le=INT32_MAX)
    document_link_count: int = Field(ge=1, le=INT32_MAX)
    href: str
    normalized_href: str
    link_kind: LinkKind
    resolution_status: ResolutionStatus
    target_fragment: str | None = Field(min_length=1)
    link_text: str
    title: str | None = Field(min_length=1)
    source_xpath: str = Field(min_length=1)
    source_document_sha256: str = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_derived_fields(self) -> DocumentLinkRecord:
        if self.link_id != link_id_for(self.source_document_id, self.link_ordinal):
            raise ValueError("link_id must be derived from source document and ordinal")
        if self.link_ordinal > self.document_link_count:
            raise ValueError("link_ordinal exceeds document_link_count")
        if self.resolution_status not in _STATUSES_BY_KIND[self.link_kind]:
            raise ValueError(
                f"a {self.link_kind} link cannot be {self.resolution_status}"
            )
        if (self.resolution_status == "resolved") != (
            self.target_document_id is not None
        ):
            raise ValueError("target_document_id must be set exactly when resolved")
        if (
            self.link_kind == "fragment"
            and self.target_document_id != self.source_document_id
        ):
            raise ValueError("a fragment link targets its own source document")
        if self.target_fragment is not None and self.target_fragment.startswith("#"):
            raise ValueError("target_fragment must not start with '#'")
        return self

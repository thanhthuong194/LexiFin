from __future__ import annotations

import hashlib
from typing import Any

import pyarrow as pa
from pydantic import AwareDatetime, Field, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel
from src.data_pipeline.contracts.sec_filings.filings import INT32_MAX, SHA256_PATTERN
from src.data_pipeline.contracts.sec_filings.xbrl_core import canonical_sha256

XBRL_FACT_FOOTNOTES_ARROW_SCHEMA = pa.schema(
    [
        pa.field("footnote_relation_id", pa.string(), nullable=False),
        pa.field("fact_id", pa.string(), nullable=False),
        pa.field("footnote_ordinal", pa.int32(), nullable=False),
        pa.field("fact_footnote_count", pa.int32(), nullable=False),
        pa.field("footnote_resource_id", pa.string(), nullable=False),
        pa.field("footnote_text", pa.string(), nullable=False),
        pa.field("footnote_html", pa.string(), nullable=True),
        pa.field("role_uri", pa.string(), nullable=False),
        pa.field("arcrole_uri", pa.string(), nullable=False),
        pa.field("linkrole_uri", pa.string(), nullable=False),
        pa.field("language", pa.string(), nullable=True),
        pa.field("resource_label", pa.string(), nullable=True),
        pa.field("source_xpath", pa.string(), nullable=False),
        pa.field("arc_xpath", pa.string(), nullable=True),
        pa.field("footnote_content_sha256", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)
# What the footnote resource says, independent of the facts it is linked to.
FOOTNOTE_CONTENT_FIELDS = (
    "footnote_text",
    "footnote_html",
    "role_uri",
    "language",
    "resource_label",
)


def footnote_relation_id_for(fact_id: str, footnote_ordinal: int) -> str:
    """Key of the N-th footnote relationship of a fact, in canonical order."""

    return f"{fact_id}:footnote:{footnote_ordinal:04d}"


def footnote_resource_id_for(
    instance_id: str, document_id: str, source_xpath: str
) -> str:
    """Key of a footnote resource: its location, shared by every fact it notes."""

    locator = f"{document_id}\x00{source_xpath}".encode()
    return f"{instance_id}:footnote-resource:{hashlib.sha256(locator).hexdigest()}"


def footnote_content_sha256(values: dict[str, Any]) -> str:
    """Hash the FOOTNOTE_CONTENT_FIELDS of a footnote as canonical JSON."""

    return canonical_sha256({name: values[name] for name in FOOTNOTE_CONTENT_FIELDS})


class XbrlFactFootnoteRecord(ContractModel):
    """One effective fact-footnote relationship of an XBRL fact.

    Field order matches XBRL_FACT_FOOTNOTES_ARROW_SCHEMA. `footnote_text` is
    the normalized text (may be empty) and `footnote_html` the inner markup,
    null for plain text; `language` is the lowercase xml:lang in scope.
    Ordinals 1..N order a fact's relationships; `fact_footnote_count` is N.
    """

    footnote_relation_id: str
    fact_id: str
    footnote_ordinal: int = Field(ge=1, le=INT32_MAX)
    fact_footnote_count: int = Field(ge=1, le=INT32_MAX)
    footnote_resource_id: str
    footnote_text: str
    footnote_html: str | None = Field(min_length=1)
    role_uri: str = Field(min_length=1)
    arcrole_uri: str = Field(min_length=1)
    linkrole_uri: str = Field(min_length=1)
    language: str | None = Field(min_length=1)
    resource_label: str | None = Field(min_length=1)
    source_xpath: str = Field(min_length=1)
    arc_xpath: str | None = Field(min_length=1)
    footnote_content_sha256: str = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_derived_fields(self) -> XbrlFactFootnoteRecord:
        if self.footnote_relation_id != footnote_relation_id_for(
            self.fact_id, self.footnote_ordinal
        ):
            raise ValueError(
                "footnote_relation_id must be derived from fact and ordinal"
            )
        if self.footnote_ordinal > self.fact_footnote_count:
            raise ValueError("footnote_ordinal exceeds fact_footnote_count")
        instance_id = self.fact_id.rpartition(":fact:")[0]
        if not self.footnote_resource_id.startswith(
            f"{instance_id}:footnote-resource:"
        ):
            raise ValueError("footnote resource belongs to another instance")
        if self.language is not None and self.language != self.language.strip().lower():
            raise ValueError("language must be a trimmed lowercase tag")
        if self.footnote_content_sha256 != footnote_content_sha256(self.model_dump()):
            raise ValueError("footnote_content_sha256 does not match the footnote")
        return self

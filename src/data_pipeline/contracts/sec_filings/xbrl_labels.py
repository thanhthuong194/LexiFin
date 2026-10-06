from __future__ import annotations

import hashlib
import unicodedata
from typing import Any

import pyarrow as pa
from pydantic import AwareDatetime, Field, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel
from src.data_pipeline.contracts.sec_filings.filings import SHA256_PATTERN
from src.data_pipeline.contracts.sec_filings.xbrl_core import canonical_sha256

XBRL_LABELS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("label_id", pa.string(), nullable=False),
        pa.field("concept_id", pa.string(), nullable=False),
        pa.field("label_resource_id", pa.string(), nullable=False),
        pa.field("role_uri", pa.string(), nullable=False),
        pa.field("language", pa.string(), nullable=True),
        pa.field("label_text", pa.string(), nullable=False),
        pa.field("resource_label", pa.string(), nullable=True),
        pa.field("linkrole_uri", pa.string(), nullable=False),
        pa.field("source_document_id", pa.string(), nullable=False),
        pa.field("source_xpath", pa.string(), nullable=False),
        pa.field("arc_xpath", pa.string(), nullable=True),
        pa.field("label_text_sha256", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)
# What makes one concept-label relationship: the concept, the resource and
# its content, and where the relationship is declared.
LABEL_IDENTITY_FIELDS = (
    "concept_id",
    "label_resource_id",
    "linkrole_uri",
    "role_uri",
    "language",
    "arc_xpath",
    "label_text_sha256",
)


def normalize_label(text: str) -> str:
    """NFC, LF line endings, trimmed; case, punctuation and inner spaces kept."""

    unix = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", unix).strip()


def label_resource_id_for(source_document_id: str, source_xpath: str) -> str:
    """Key of a label resource: its DTS document and its place in it."""

    payload = {"source_document_id": source_document_id, "source_xpath": source_xpath}
    return f"xbrl:label-resource:{canonical_sha256(payload)}"


def label_id_for(values: dict[str, Any]) -> str:
    """Key of a concept-label relationship, from its LABEL_IDENTITY_FIELDS."""

    payload = {name: values[name] for name in LABEL_IDENTITY_FIELDS}
    return f"xbrl:label:{canonical_sha256(payload)}"


def is_dts_document_id(value: str) -> bool:
    """Whether a DTS document identity is a filing document_id or a URL.

    Neither is a local path, which would differ between runs.
    """

    return value.startswith("sec:") or "://" in value


class XbrlLabelRecord(ContractModel):
    """One effective relationship from a concept to a label resource.

    Field order matches XBRL_LABELS_ARROW_SCHEMA. Labels are shared by every
    DTS that holds the same resource: a filing's own files are identified by
    their document_id, published taxonomy files by URL. `label_text` is
    normalized (may be empty) and `language` the lowercase xml:lang in scope.
    """

    label_id: str
    concept_id: str
    label_resource_id: str
    role_uri: str = Field(min_length=1)
    language: str | None = Field(min_length=1)
    label_text: str
    resource_label: str | None = Field(min_length=1)
    linkrole_uri: str = Field(min_length=1)
    source_document_id: str = Field(min_length=1)
    source_xpath: str = Field(min_length=1)
    arc_xpath: str | None = Field(min_length=1)
    label_text_sha256: str = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_derived_fields(self) -> XbrlLabelRecord:
        if not is_dts_document_id(self.source_document_id):
            raise ValueError("source_document_id must be a document_id or a URL")
        if self.label_resource_id != label_resource_id_for(
            self.source_document_id, self.source_xpath
        ):
            raise ValueError("label_resource_id must be derived from its location")
        if self.label_text != normalize_label(self.label_text):
            raise ValueError("label_text must be normalized")
        if self.language is not None and self.language != self.language.strip().lower():
            raise ValueError("language must be a trimmed lowercase tag")
        digest = hashlib.sha256(self.label_text.encode()).hexdigest()
        if self.label_text_sha256 != digest:
            raise ValueError("label_text_sha256 does not match label_text")
        if self.label_id != label_id_for(self.model_dump()):
            raise ValueError("label_id must be derived from the relationship")
        return self

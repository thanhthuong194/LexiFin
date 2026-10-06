from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Literal

import pyarrow as pa
from pydantic import AwareDatetime, Field, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel
from src.data_pipeline.contracts.sec_filings.filings import INT32_MAX, SHA256_PATTERN
from src.data_pipeline.contracts.sec_filings.xbrl_core import (
    CLARK_QNAME,
    canonical_sha256,
)

FactKind = Literal["numeric", "non_numeric", "fraction"]

XBRL_FACTS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("fact_id", pa.string(), nullable=False),
        pa.field("instance_id", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("concept_id", pa.string(), nullable=False),
        pa.field("context_key", pa.string(), nullable=False),
        pa.field("unit_key", pa.string(), nullable=True),
        pa.field("block_id", pa.string(), nullable=True),
        pa.field("cell_id", pa.string(), nullable=True),
        pa.field("source_fact_id", pa.string(), nullable=True),
        pa.field("fact_kind", pa.string(), nullable=False),
        pa.field("raw_value", pa.string(), nullable=True),
        pa.field("normalized_value", pa.string(), nullable=True),
        pa.field("numeric_value", pa.string(), nullable=True),
        pa.field("is_nil", pa.bool_(), nullable=False),
        pa.field("is_hidden", pa.bool_(), nullable=False),
        pa.field("decimals", pa.string(), nullable=True),
        pa.field("precision", pa.string(), nullable=True),
        pa.field("scale", pa.int32(), nullable=True),
        pa.field("sign", pa.string(), nullable=True),
        pa.field("format", pa.string(), nullable=True),
        pa.field("escape", pa.bool_(), nullable=True),
        pa.field("xml_language", pa.string(), nullable=True),
        pa.field("continuation_refs", pa.list_(pa.string()), nullable=False),
        pa.field("source_xpath", pa.string(), nullable=False),
        pa.field("fact_order", pa.int64(), nullable=False),
        pa.field("fact_sha256", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)
# Source and semantic fields of a fact occurrence. Its key, batch time and
# block/cell alignment (an enrichment that a narrative rebuild may change)
# are left out.
FACT_HASHED_FIELDS = (
    "instance_id",
    "document_id",
    "concept_id",
    "context_key",
    "unit_key",
    "source_fact_id",
    "fact_kind",
    "raw_value",
    "normalized_value",
    "numeric_value",
    "is_nil",
    "is_hidden",
    "decimals",
    "precision",
    "scale",
    "sign",
    "format",
    "escape",
    "xml_language",
    "continuation_refs",
    "source_xpath",
    "fact_order",
)


def fact_id_for(instance_id: str, fact_order: int) -> str:
    """Key of the N-th fact occurrence of an instance (document, then DOM order)."""

    return f"{instance_id}:fact:{fact_order:08d}"


def fact_sha256(values: dict[str, Any]) -> str:
    """Hash the FACT_HASHED_FIELDS of a fact as canonical JSON."""

    return canonical_sha256({name: values[name] for name in FACT_HASHED_FIELDS})


def canonical_decimal(value: str) -> str | None:
    """The exact plain-notation form of a finite decimal, or None if it is not one."""

    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    return format(number, "f") if number.is_finite() else None


class XbrlFactRecord(ContractModel):
    """One fact occurrence of a logical instance; equal facts stay separate rows.

    Field order matches XBRL_FACTS_ARROW_SCHEMA. `raw_value` is the source
    text (continuations joined; "numerator/denominator" for a fraction) before
    format, scale and sign; `normalized_value` is the processor's value after
    them (null when it cannot apply the format), and
    `numeric_value` its exact plain decimal for finite numeric facts.
    `escape` and `xml_language` (the language in scope) only qualify
    nonnumeric facts. At most one of `block_id` and `cell_id` locates the
    fact in narrative content. `fact_sha256` hashes FACT_HASHED_FIELDS.
    """

    fact_id: str
    instance_id: str
    document_id: str
    concept_id: str
    context_key: str
    unit_key: str | None
    block_id: str | None
    cell_id: str | None
    source_fact_id: str | None = Field(min_length=1)
    fact_kind: FactKind
    raw_value: str | None
    normalized_value: str | None
    numeric_value: str | None
    is_nil: bool
    is_hidden: bool
    decimals: str | None = Field(min_length=1)
    precision: str | None = Field(min_length=1)
    scale: int | None = Field(ge=-INT32_MAX, le=INT32_MAX)
    sign: str | None = Field(min_length=1)
    format: str | None
    escape: bool | None
    xml_language: str | None = Field(min_length=1)
    continuation_refs: list[str]
    source_xpath: str = Field(min_length=1)
    fact_order: int = Field(ge=1)
    fact_sha256: str = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_derived_fields(self) -> XbrlFactRecord:
        if self.fact_id != fact_id_for(self.instance_id, self.fact_order):
            raise ValueError("fact_id must be derived from instance and fact_order")
        if (self.fact_kind == "non_numeric") != (self.unit_key is None):
            raise ValueError("exactly numeric and fraction facts have a unit")
        values = (self.raw_value, self.normalized_value, self.numeric_value)
        if self.is_nil and values != (None, None, None):
            raise ValueError("a nil fact has no value")
        if self.numeric_value is not None and (
            self.fact_kind != "numeric"
            or canonical_decimal(self.numeric_value) != self.numeric_value
        ):
            raise ValueError("numeric_value must be a plain finite decimal")
        text_qualifiers = (self.escape, self.xml_language)
        if self.fact_kind != "non_numeric" and text_qualifiers != (None, None):
            raise ValueError("only a nonnumeric fact has an escape flag or language")
        if self.block_id is not None and self.cell_id is not None:
            raise ValueError("a fact is in a block or a cell, not both")
        if self.format is not None and not CLARK_QNAME.fullmatch(self.format):
            raise ValueError("format must be a {namespace}local QName")
        if self.fact_sha256 != fact_sha256(self.model_dump()):
            raise ValueError("fact_sha256 does not match the fact")
        return self

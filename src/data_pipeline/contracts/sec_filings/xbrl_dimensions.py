from __future__ import annotations

from typing import Any, Literal

import pyarrow as pa
from pydantic import AwareDatetime, Field, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel
from src.data_pipeline.contracts.sec_filings.filings import INT32_MAX, SHA256_PATTERN
from src.data_pipeline.contracts.sec_filings.xbrl_core import (
    CLARK_QNAME,
    canonical_sha256,
)

ContextElement = Literal["segment", "scenario"]
MemberType = Literal["explicit", "typed"]

XBRL_DIMENSIONS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("dimension_id", pa.string(), nullable=False),
        pa.field("context_key", pa.string(), nullable=False),
        pa.field("dimension_ordinal", pa.int32(), nullable=False),
        pa.field("context_dimension_count", pa.int32(), nullable=False),
        pa.field("context_element", pa.string(), nullable=False),
        pa.field("dimension_concept_id", pa.string(), nullable=False),
        pa.field("member_type", pa.string(), nullable=False),
        pa.field("member_concept_id", pa.string(), nullable=True),
        pa.field("typed_member_qname", pa.string(), nullable=True),
        pa.field("typed_member_value", pa.string(), nullable=True),
        pa.field("typed_member_xml", pa.string(), nullable=True),
        pa.field("source_xpath", pa.string(), nullable=False),
        pa.field("dimension_hash", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)
# What a dimension states. Its position in the context, batch time and source
# location are left out; the location is compared separately.
DIMENSION_HASHED_FIELDS = (
    "context_element",
    "dimension_concept_id",
    "member_type",
    "member_concept_id",
    "typed_member_qname",
    "typed_member_value",
    "typed_member_xml",
)


def dimension_id_for(context_key: str, dimension_ordinal: int) -> str:
    """Key of the N-th dimension of a context, in canonical dimension order."""

    return f"{context_key}:dimension:{dimension_ordinal:04d}"


def dimension_hash(values: dict[str, Any]) -> str:
    """Hash the DIMENSION_HASHED_FIELDS of a dimension as canonical JSON."""

    return canonical_sha256({name: values[name] for name in DIMENSION_HASHED_FIELDS})


class XbrlDimensionRecord(ContractModel):
    """One explicit or typed dimension reported in a context's segment or scenario.

    Field order matches XBRL_DIMENSIONS_ARROW_SCHEMA. An explicit dimension
    has a member concept; a typed one the Clark QName, normalized text (may
    be empty) and canonical XML of its member element. Ordinals 1..N follow
    (context element, dimension QName); `context_dimension_count` is N.
    """

    dimension_id: str
    context_key: str
    dimension_ordinal: int = Field(ge=1, le=INT32_MAX)
    context_dimension_count: int = Field(ge=1, le=INT32_MAX)
    context_element: ContextElement
    dimension_concept_id: str
    member_type: MemberType
    member_concept_id: str | None
    typed_member_qname: str | None
    typed_member_value: str | None
    typed_member_xml: str | None = Field(min_length=1)
    source_xpath: str = Field(min_length=1)
    dimension_hash: str = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_derived_fields(self) -> XbrlDimensionRecord:
        if self.dimension_id != dimension_id_for(
            self.context_key, self.dimension_ordinal
        ):
            raise ValueError("dimension_id must be derived from context and ordinal")
        if self.dimension_ordinal > self.context_dimension_count:
            raise ValueError("dimension_ordinal exceeds context_dimension_count")
        typed = (
            self.typed_member_qname,
            self.typed_member_value,
            self.typed_member_xml,
        )
        if self.member_type == "explicit" and (
            self.member_concept_id is None or typed != (None, None, None)
        ):
            raise ValueError("an explicit dimension has a member concept only")
        if self.member_type == "typed" and (
            self.member_concept_id is not None or None in typed
        ):
            raise ValueError("a typed dimension has a typed member only")
        if self.typed_member_qname is not None and not CLARK_QNAME.fullmatch(
            self.typed_member_qname
        ):
            raise ValueError("typed_member_qname must be a {namespace}local QName")
        if self.dimension_hash != dimension_hash(self.model_dump()):
            raise ValueError("dimension_hash does not match the dimension")
        return self

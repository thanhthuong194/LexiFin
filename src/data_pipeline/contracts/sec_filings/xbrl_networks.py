from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

import pyarrow as pa
from pydantic import AwareDatetime, Field, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel
from src.data_pipeline.contracts.sec_filings.filings import INT32_MAX, SHA256_PATTERN
from src.data_pipeline.contracts.sec_filings.xbrl_core import canonical_sha256
from src.data_pipeline.contracts.sec_filings.xbrl_labels import is_dts_document_id

LinkbaseType = Literal["presentation", "calculation", "definition"]
LINKBASE_ORDER: dict[str, int] = {"presentation": 1, "calculation": 2, "definition": 3}

XBRL_NETWORKS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("network_id", pa.string(), nullable=False),
        pa.field("instance_id", pa.string(), nullable=False),
        pa.field("network_ordinal", pa.int32(), nullable=False),
        pa.field("instance_network_count", pa.int32(), nullable=False),
        pa.field("linkbase_type", pa.string(), nullable=False),
        pa.field("role_uri", pa.string(), nullable=False),
        pa.field("role_definition", pa.string(), nullable=True),
        pa.field("relationship_count", pa.int32(), nullable=False),
        pa.field("network_hash", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)
XBRL_RELATIONSHIPS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("relationship_id", pa.string(), nullable=False),
        pa.field("network_id", pa.string(), nullable=False),
        pa.field("relationship_ordinal", pa.int32(), nullable=False),
        pa.field("network_relationship_count", pa.int32(), nullable=False),
        pa.field("source_concept_id", pa.string(), nullable=False),
        pa.field("target_concept_id", pa.string(), nullable=False),
        pa.field("arcrole_uri", pa.string(), nullable=False),
        pa.field("order_value", pa.string(), nullable=False),
        pa.field("weight", pa.string(), nullable=True),
        pa.field("preferred_label_role", pa.string(), nullable=True),
        pa.field("target_role_uri", pa.string(), nullable=True),
        pa.field("context_element", pa.string(), nullable=True),
        pa.field("is_closed", pa.bool_(), nullable=True),
        pa.field("is_usable", pa.bool_(), nullable=True),
        pa.field("source_document_id", pa.string(), nullable=False),
        pa.field("source_xpath", pa.string(), nullable=False),
        pa.field("relationship_hash", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)
# What a relationship states and where its arc is; its ID is derived from it.
RELATIONSHIP_HASHED_FIELDS = (
    "network_id",
    "source_concept_id",
    "target_concept_id",
    "arcrole_uri",
    "source_document_id",
    "source_xpath",
    "order_value",
    "weight",
    "preferred_label_role",
    "target_role_uri",
    "context_element",
    "is_closed",
    "is_usable",
)


def plain_decimal(value: Decimal) -> str:
    """Exact plain notation without insignificant zeros; -0 becomes 0."""

    if not value.is_finite():
        raise ValueError(f"{value} is not a finite decimal")
    normalized = value.normalize()
    return format(normalized if normalized else Decimal(0), "f")


def _is_plain_decimal(text: str) -> bool:
    try:
        return plain_decimal(Decimal(text)) == text
    except (InvalidOperation, ValueError):
        return False


def network_id_for(instance_id: str, linkbase_type: str, role_uri: str) -> str:
    """Key of a network: its instance, linkbase type and link role."""

    payload = {"linkbase_type": linkbase_type, "role_uri": role_uri}
    return f"{instance_id}:network:{canonical_sha256(payload)}"


def relationship_hash(values: dict[str, Any]) -> str:
    """Hash the RELATIONSHIP_HASHED_FIELDS of a relationship as canonical JSON."""

    return canonical_sha256({name: values[name] for name in RELATIONSHIP_HASHED_FIELDS})


def relationship_id_for(network_id: str, relationship_digest: str) -> str:
    """Key of a relationship: its network and its relationship_hash."""

    return f"{network_id}:relationship:{relationship_digest}"


def network_hash(values: dict[str, Any], relationship_hashes: Sequence[str]) -> str:
    """Hash a network's identity, definition and relationships in ordinal order."""

    payload = {
        name: values[name]
        for name in ("instance_id", "linkbase_type", "role_uri", "role_definition")
    }
    return canonical_sha256(
        {**payload, "relationship_hashes": list(relationship_hashes)}
    )


class XbrlNetworkRecord(ContractModel):
    """One network of effective relationships of an instance: linkbase type and role.

    Field order matches XBRL_NETWORKS_ARROW_SCHEMA. Ordinals follow linkbase
    type (presentation, calculation, definition), then role URI;
    `network_hash` covers the ordered relationship hashes.
    """

    network_id: str
    instance_id: str
    network_ordinal: int = Field(ge=1, le=INT32_MAX)
    instance_network_count: int = Field(ge=1, le=INT32_MAX)
    linkbase_type: LinkbaseType
    role_uri: str = Field(min_length=1)
    role_definition: str | None
    relationship_count: int = Field(ge=1, le=INT32_MAX)
    network_hash: str = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_derived_fields(self) -> XbrlNetworkRecord:
        if self.network_id != network_id_for(
            self.instance_id, self.linkbase_type, self.role_uri
        ):
            raise ValueError("network_id must be derived from instance, type and role")
        if self.network_ordinal > self.instance_network_count:
            raise ValueError("network_ordinal exceeds instance_network_count")
        return self


class XbrlRelationshipRecord(ContractModel):
    """One effective arc from a source to a target concept in a network.

    Field order matches XBRL_RELATIONSHIPS_ARROW_SCHEMA. `order_value` and
    `weight` are plain decimals; attributes that do not apply to the arc are
    null. Ordinals follow source concept, order, target concept, arcrole and
    arc location.
    """

    relationship_id: str
    network_id: str
    relationship_ordinal: int = Field(ge=1, le=INT32_MAX)
    network_relationship_count: int = Field(ge=1, le=INT32_MAX)
    source_concept_id: str
    target_concept_id: str
    arcrole_uri: str = Field(min_length=1)
    order_value: str
    weight: str | None
    preferred_label_role: str | None = Field(min_length=1)
    target_role_uri: str | None = Field(min_length=1)
    context_element: Literal["segment", "scenario"] | None
    is_closed: bool | None
    is_usable: bool | None
    source_document_id: str = Field(min_length=1)
    source_xpath: str = Field(min_length=1)
    relationship_hash: str = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_derived_fields(self) -> XbrlRelationshipRecord:
        if self.relationship_ordinal > self.network_relationship_count:
            raise ValueError("relationship_ordinal exceeds network_relationship_count")
        for number in (self.order_value, self.weight):
            if number is not None and not _is_plain_decimal(number):
                raise ValueError(f"{number!r} is not a plain decimal")
        if not is_dts_document_id(self.source_document_id):
            raise ValueError("source_document_id must be a document_id or a URL")
        if self.relationship_hash != relationship_hash(self.model_dump()):
            raise ValueError("relationship_hash does not match the relationship")
        if self.relationship_id != relationship_id_for(
            self.network_id, self.relationship_hash
        ):
            raise ValueError("relationship_id must be derived from its hash")
        return self

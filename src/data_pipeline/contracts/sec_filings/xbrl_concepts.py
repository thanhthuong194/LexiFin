from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Literal

import pyarrow as pa
from pydantic import AwareDatetime, Field, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel
from src.data_pipeline.contracts.sec_filings.filings import SHA256_PATTERN
from src.data_pipeline.contracts.sec_filings.xbrl_core import (
    CLARK_QNAME,
    canonical_sha256,
)

ConceptPeriodType = Literal["instant", "duration"]
Balance = Literal["debit", "credit"]

XBRL_CONCEPTS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("concept_id", pa.string(), nullable=False),
        pa.field("namespace_uri", pa.string(), nullable=False),
        pa.field("local_name", pa.string(), nullable=False),
        pa.field("data_type", pa.string(), nullable=True),
        pa.field("substitution_group", pa.string(), nullable=True),
        pa.field("period_type", pa.string(), nullable=True),
        pa.field("balance", pa.string(), nullable=True),
        pa.field("is_abstract", pa.bool_(), nullable=False),
        pa.field("is_nillable", pa.bool_(), nullable=False),
        pa.field("is_extension", pa.bool_(), nullable=False),
        pa.field("source_schema_uri", pa.string(), nullable=False),
        pa.field("concept_definition_sha256", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def concept_id_for(namespace_uri: str, local_name: str) -> str:
    """Key of a concept: its expanded QName, whatever prefix a document uses."""

    payload = f"{namespace_uri}\x00{local_name}".encode()
    return f"xbrl:concept:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class ConceptDefinition:
    """What a concept is, independent of where it was read and of its key.

    `data_type` and `substitution_group` are Clark-notation QNames.
    """

    namespace_uri: str
    local_name: str
    data_type: str | None
    substitution_group: str | None
    period_type: ConceptPeriodType | None
    balance: Balance | None
    is_abstract: bool
    is_nillable: bool

    def sha256(self) -> str:
        """SHA-256 of the definition as canonical JSON."""

        return canonical_sha256(asdict(self))


class XbrlConceptRecord(ContractModel):
    """One concept of the global dictionary, keyed by expanded QName.

    Field order matches XBRL_CONCEPTS_ARROW_SCHEMA. `source_schema_uri` is the
    schema it was first read from (a URL, or the filing's schema file name);
    `is_extension` marks concepts of a filing's own extension taxonomy. Both
    are lineage and left out of `concept_definition_sha256`.
    """

    concept_id: str
    namespace_uri: str = Field(min_length=1)
    local_name: str = Field(min_length=1)
    data_type: str | None
    substitution_group: str | None
    period_type: ConceptPeriodType | None
    balance: Balance | None
    is_abstract: bool
    is_nillable: bool
    is_extension: bool
    source_schema_uri: str = Field(min_length=1)
    concept_definition_sha256: str = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    def definition(self) -> ConceptDefinition:
        """The concept's definition, without key, lineage and batch time."""

        return ConceptDefinition(
            namespace_uri=self.namespace_uri,
            local_name=self.local_name,
            data_type=self.data_type,
            substitution_group=self.substitution_group,
            period_type=self.period_type,
            balance=self.balance,
            is_abstract=self.is_abstract,
            is_nillable=self.is_nillable,
        )

    @model_validator(mode="after")
    def _check_derived_fields(self) -> XbrlConceptRecord:
        if self.concept_id != concept_id_for(self.namespace_uri, self.local_name):
            raise ValueError("concept_id must be derived from the expanded QName")
        for qname in (self.data_type, self.substitution_group):
            if qname is not None and not CLARK_QNAME.fullmatch(qname):
                raise ValueError(f"{qname!r} is not a {{namespace}}local QName")
        if self.concept_definition_sha256 != self.definition().sha256():
            raise ValueError("concept_definition_sha256 does not match the definition")
        return self

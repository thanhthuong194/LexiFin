from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from typing import Literal

import pyarrow as pa
from pydantic import AwareDatetime, Field, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel
from src.data_pipeline.contracts.sec_filings.filings import INT32_MAX, SHA256_PATTERN

PeriodType = Literal["instant", "duration", "forever"]
UnitKind = Literal["simple", "divide"]

XBRL_INSTANCES_ARROW_SCHEMA = pa.schema(
    [
        pa.field("instance_id", pa.string(), nullable=False),
        pa.field("filing_id", pa.string(), nullable=False),
        pa.field("target_name", pa.string(), nullable=True),
        pa.field("is_inline", pa.bool_(), nullable=False),
        pa.field("source_document_ids", pa.list_(pa.string()), nullable=False),
        pa.field("schema_refs", pa.list_(pa.string()), nullable=False),
        pa.field("namespace_map", pa.map_(pa.string(), pa.string()), nullable=False),
        pa.field("fact_count", pa.int64(), nullable=False),
        pa.field("context_count", pa.int32(), nullable=False),
        pa.field("unit_count", pa.int32(), nullable=False),
        pa.field("source_set_sha256", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)
XBRL_CONTEXTS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("context_key", pa.string(), nullable=False),
        pa.field("instance_id", pa.string(), nullable=False),
        pa.field("context_ref", pa.string(), nullable=False),
        pa.field("entity_identifier_scheme", pa.string(), nullable=False),
        pa.field("entity_identifier", pa.string(), nullable=False),
        pa.field("period_type", pa.string(), nullable=False),
        pa.field("start_date", pa.date32(), nullable=True),
        pa.field("end_date", pa.date32(), nullable=True),
        pa.field("instant_date", pa.date32(), nullable=True),
        pa.field("segment_xml", pa.string(), nullable=True),
        pa.field("scenario_xml", pa.string(), nullable=True),
        pa.field("context_hash", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)
XBRL_UNITS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("unit_key", pa.string(), nullable=False),
        pa.field("instance_id", pa.string(), nullable=False),
        pa.field("unit_ref", pa.string(), nullable=False),
        pa.field("unit_kind", pa.string(), nullable=False),
        pa.field("numerator_measures", pa.list_(pa.string()), nullable=False),
        pa.field("denominator_measures", pa.list_(pa.string()), nullable=False),
        pa.field("normalized_expression", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)
# An expanded QName (a measure, type or format) in Clark notation:
# {namespace-uri}local-name.
CLARK_QNAME = re.compile(r"\{[^{}\s]+\}[^{}\s:]+")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(payload: object) -> str:
    """SHA-256 of a JSON payload serialized canonically (sorted keys, no spaces)."""

    return _sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def inline_instance_id(filing_id: str, target_name: str | None) -> str:
    """Key of an Inline XBRL target: `default`, or the SHA-256 of its name."""

    target = "default" if target_name is None else _sha256(target_name)
    return f"{filing_id}:xbrl:inline:{target}"


def xml_instance_id(filing_id: str, document_id: str) -> str:
    """Key of a standalone XML instance not merged into an Inline target."""

    return f"{filing_id}:xbrl:xml:{_sha256(document_id)}"


def context_key_for(instance_id: str, context_ref: str) -> str:
    """Key of a context; its source ID is only unique within the instance."""

    return f"{instance_id}:context:{_sha256(context_ref)}"


def unit_key_for(instance_id: str, unit_ref: str) -> str:
    """Key of a unit; its source ID is only unique within the instance."""

    return f"{instance_id}:unit:{_sha256(unit_ref)}"


def source_set_sha256(sources: Sequence[tuple[str, str]]) -> str:
    """Hash an instance's (document_id, content_sha256) pairs, in source order."""

    return canonical_sha256(
        [{"document_id": doc, "content_sha256": digest} for doc, digest in sources]
    )


@dataclass(frozen=True)
class ContextDefinition:
    """What a context means: entity, period, segment and scenario, without its ID.

    Two contexts with equal definitions hash alike, whatever their IDs.
    """

    entity_identifier_scheme: str
    entity_identifier: str
    period_type: PeriodType
    start_date: date | None
    end_date: date | None
    instant_date: date | None
    segment_xml: str | None
    scenario_xml: str | None

    def sha256(self) -> str:
        """SHA-256 of the definition as canonical JSON (dates in ISO format)."""

        payload = {
            name: value.isoformat() if isinstance(value, date) else value
            for name, value in asdict(self).items()
        }
        return canonical_sha256(payload)


def unit_expression(numerators: Sequence[str], denominators: Sequence[str]) -> str:
    """`a*b` for a simple unit, `(a*b)/(c)` for a divide unit; measures sorted."""

    numerator = "*".join(sorted(numerators))
    if not denominators:
        return numerator
    return f"({numerator})/({'*'.join(sorted(denominators))})"


class XbrlInstanceRecord(ContractModel):
    """One logical XBRL instance of a filing: an Inline target or an XML instance.

    Field order matches XBRL_INSTANCES_ARROW_SCHEMA. `source_document_ids`
    lists every representation (Inline documents and merged XML) by document
    ordinal; `source_set_sha256` hashes them with their content hashes. The
    counts are the expected context, unit and fact rows.
    """

    instance_id: str
    filing_id: str
    target_name: str | None = Field(min_length=1)
    is_inline: bool
    source_document_ids: list[str] = Field(min_length=1)
    schema_refs: list[str] = Field(min_length=1)
    namespace_map: dict[str, str]
    fact_count: int = Field(ge=0)
    context_count: int = Field(ge=0, le=INT32_MAX)
    unit_count: int = Field(ge=0, le=INT32_MAX)
    source_set_sha256: str = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_identity(self) -> XbrlInstanceRecord:
        expected = (
            inline_instance_id(self.filing_id, self.target_name)
            if self.is_inline
            else xml_instance_id(self.filing_id, self.source_document_ids[0])
        )
        if self.instance_id != expected:
            raise ValueError(
                "instance_id must follow the Inline target or XML document"
            )
        if not self.is_inline and (
            self.target_name is not None or len(self.source_document_ids) != 1
        ):
            raise ValueError("an XML-only instance has one document and no target")
        if len(set(self.source_document_ids)) != len(self.source_document_ids):
            raise ValueError("source_document_ids must be unique")
        if self.schema_refs != sorted(set(self.schema_refs)):
            raise ValueError("schema_refs must be sorted and unique")
        if list(self.namespace_map) != sorted(self.namespace_map):
            raise ValueError("namespace_map must be ordered by prefix")
        return self


class XbrlContextRecord(ContractModel):
    """One context of a logical instance, with its period and dimensional XML.

    Field order matches XBRL_CONTEXTS_ARROW_SCHEMA. Dates are the source's
    lexical dates; `segment_xml` and `scenario_xml` are canonical XML (see
    `canonical_xml`), and `context_hash` hashes the definition without its ID.
    """

    context_key: str
    instance_id: str
    context_ref: str = Field(min_length=1)
    entity_identifier_scheme: str = Field(min_length=1)
    entity_identifier: str = Field(min_length=1)
    period_type: PeriodType
    start_date: date | None
    end_date: date | None
    instant_date: date | None
    segment_xml: str | None = Field(min_length=1)
    scenario_xml: str | None = Field(min_length=1)
    context_hash: str = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    def definition(self) -> ContextDefinition:
        """The context's meaning, without its key, ref and instance."""

        return ContextDefinition(
            entity_identifier_scheme=self.entity_identifier_scheme,
            entity_identifier=self.entity_identifier,
            period_type=self.period_type,
            start_date=self.start_date,
            end_date=self.end_date,
            instant_date=self.instant_date,
            segment_xml=self.segment_xml,
            scenario_xml=self.scenario_xml,
        )

    @model_validator(mode="after")
    def _check_derived_fields(self) -> XbrlContextRecord:
        if self.context_key != context_key_for(self.instance_id, self.context_ref):
            raise ValueError("context_key must be derived from instance and ref")
        dates = (self.start_date, self.end_date, self.instant_date)
        present = tuple(day is not None for day in dates)
        expected = {
            "instant": (False, False, True),
            "duration": (True, True, False),
            "forever": (False, False, False),
        }[self.period_type]
        if present != expected:
            raise ValueError(f"period dates do not fit a {self.period_type} period")
        if self.context_hash != self.definition().sha256():
            raise ValueError("context_hash does not match the definition")
        return self


class XbrlUnitRecord(ContractModel):
    """One unit of a logical instance; measures are sorted Clark-notation QNames.

    Field order matches XBRL_UNITS_ARROW_SCHEMA. A simple unit has only
    numerator measures (several multiply); a divide unit has both.
    """

    unit_key: str
    instance_id: str
    unit_ref: str = Field(min_length=1)
    unit_kind: UnitKind
    numerator_measures: list[str] = Field(min_length=1)
    denominator_measures: list[str]
    normalized_expression: str
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_derived_fields(self) -> XbrlUnitRecord:
        if self.unit_key != unit_key_for(self.instance_id, self.unit_ref):
            raise ValueError("unit_key must be derived from instance and ref")
        if (self.unit_kind == "divide") != bool(self.denominator_measures):
            raise ValueError("exactly divide units have denominator measures")
        measures = (self.numerator_measures, self.denominator_measures)
        if not all(CLARK_QNAME.fullmatch(m) for group in measures for m in group):
            raise ValueError("measures must be {namespace}local QNames")
        if any(group != sorted(group) for group in measures):
            raise ValueError("measures must be sorted")
        if self.normalized_expression != unit_expression(*measures):
            raise ValueError("normalized_expression does not match the measures")
        return self

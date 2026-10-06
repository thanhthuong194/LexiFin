from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from datetime import datetime
from typing import Any

from lxml import etree

from src.data_pipeline.contracts.sec_filings.xbrl_concepts import (
    ConceptDefinition,
    XbrlConceptRecord,
    concept_id_for,
)
from src.data_pipeline.contracts.sec_filings.xbrl_core import (
    context_key_for,
    unit_key_for,
)
from src.data_pipeline.contracts.sec_filings.xbrl_facts import (
    XbrlFactRecord,
    canonical_decimal,
    fact_id_for,
    fact_sha256,
)
from src.data_pipeline.transforms.sec_filings.html_blocks import is_block_container
from src.data_pipeline.transforms.sec_filings.html_dom import element_tag, is_hidden
from src.data_pipeline.transforms.sec_filings.xbrl_model_reader import (
    LoadedConcept,
    LoadedFact,
)

_HTML_FACT_TAGS = frozenset({"ix:nonnumeric", "ix:nonfraction", "ix:fraction"})
_CELL_TAGS = frozenset({"td", "th"})
Location = tuple[str | None, str | None]


class XbrlCoreMismatch(ValueError):
    """The processor's reading of an instance disagrees with the XBRL core rows."""


def concept_definition(concept: LoadedConcept) -> ConceptDefinition:
    """The definition of a loaded concept, without its lineage."""

    fields: dict[str, Any] = {
        "namespace_uri": concept.namespace_uri,
        "local_name": concept.local_name,
        "data_type": concept.data_type,
        "substitution_group": concept.substitution_group,
        "period_type": concept.period_type,
        "balance": concept.balance,
        "is_abstract": concept.is_abstract,
        "is_nillable": concept.is_nillable,
    }
    return ConceptDefinition(**fields)


def concept_record(concept: LoadedConcept, processed_at: datetime) -> XbrlConceptRecord:
    """The `xbrl_concepts` row of a loaded concept."""

    definition = concept_definition(concept)
    return XbrlConceptRecord.model_validate(
        {
            "concept_id": concept_id_for(concept.namespace_uri, concept.local_name),
            **vars(definition),
            "is_extension": concept.is_extension,
            "source_schema_uri": concept.source_schema_uri,
            "concept_definition_sha256": definition.sha256(),
            "processed_at": processed_at,
        }
    )


def _unit_key(
    instance_id: str, fact: LoadedFact, unit_refs: Collection[str], order: int
) -> str | None:
    if fact.fact_kind == "non_numeric":
        if fact.unit_ref is not None:
            raise XbrlCoreMismatch(
                f"nonnumeric fact {order} has unitRef {fact.unit_ref!r}"
            )
        return None
    if fact.unit_ref is None or fact.unit_ref not in unit_refs:
        raise XbrlCoreMismatch(
            f"fact {order}: unit {fact.unit_ref!r} is not in xbrl_units"
        )
    return unit_key_for(instance_id, fact.unit_ref)


def fact_records(
    instance_id: str,
    facts: Sequence[LoadedFact],
    references: tuple[Collection[str], Collection[str]],
    locations: Sequence[Location],
    processed_at: datetime,
) -> tuple[list[XbrlFactRecord], int]:
    """Build an instance's fact rows in the given order, and count unusable values.

    `references` holds the instance's stored context and unit refs; a fact
    whose refs are not among them raises XbrlCoreMismatch. `locations` are
    the (block_id, cell_id) of each fact. The count is of facts whose value
    the processor could not normalize, or whose numeric value is not an exact
    plain decimal: those keep their raw value and get no normalized or
    numeric value.
    """

    context_refs, unit_refs = references
    records, unusable = [], 0
    for order, (fact, (block_id, cell_id)) in enumerate(
        zip(facts, locations, strict=True), start=1
    ):
        if fact.context_ref not in context_refs:
            raise XbrlCoreMismatch(
                f"fact {order}: context {fact.context_ref!r} is not in xbrl_contexts"
            )
        numeric = None
        if fact.fact_kind == "numeric" and fact.normalized_value is not None:
            numeric = canonical_decimal(fact.normalized_value)
        unusable += fact.normalization_error is not None or (
            fact.fact_kind == "numeric" and not fact.is_nil and numeric is None
        )
        values: dict[str, Any] = {
            "fact_id": fact_id_for(instance_id, order),
            "instance_id": instance_id,
            "document_id": fact.document_id,
            "concept_id": concept_id_for(fact.namespace_uri, fact.local_name),
            "context_key": context_key_for(instance_id, fact.context_ref),
            "unit_key": _unit_key(instance_id, fact, unit_refs, order),
            "block_id": block_id,
            "cell_id": cell_id,
            "source_fact_id": fact.source_fact_id,
            "fact_kind": fact.fact_kind,
            "raw_value": fact.raw_value,
            "normalized_value": fact.normalized_value,
            "numeric_value": numeric,
            "is_nil": fact.is_nil,
            "is_hidden": fact.is_hidden,
            "decimals": fact.decimals,
            "precision": fact.precision,
            "scale": fact.scale,
            "sign": fact.sign,
            "format": fact.format,
            "escape": fact.escape,
            "xml_language": fact.xml_language,
            "continuation_refs": list(fact.continuation_refs),
            "source_xpath": fact.source_xpath,
            "fact_order": order,
            "processed_at": processed_at,
        }
        values["fact_sha256"] = fact_sha256(values)
        records.append(XbrlFactRecord.model_validate(values))
    return records, unusable


def _locate(
    element: etree._Element,
    cells: Mapping[str, str],
    blocks: Mapping[str, Sequence[str]],
) -> Location:
    """The stored cell, else block, holding a visible fact; (None, None) if unsure."""

    ancestry = [element, *element.iterancestors()]
    if any(is_hidden(node, element_tag(node)) for node in ancestry):
        return None, None
    spans_blocks = any(
        is_block_container(element_tag(node)) or element_tag(node) == "table"
        for node in element.iter(etree.Element)
        if node is not element
    )
    if spans_blocks:
        return None, None
    tree = element.getroottree()
    cell = next((n for n in ancestry if element_tag(n) in _CELL_TAGS), None)
    if cell is not None:
        return None, cells.get(tree.getpath(cell))
    container = next((n for n in ancestry if is_block_container(element_tag(n))), None)
    ids = () if container is None else blocks.get(tree.getpath(container), ())
    return (ids[0], None) if len(ids) == 1 else (None, None)


def align_document(
    root: etree._Element,
    positions: Sequence[int],
    total: int,
    content: tuple[Mapping[str, str], Mapping[str, Sequence[str]]],
) -> list[Location]:
    """Locate a document's facts in its stored table cells or narrative blocks.

    `root` is the document as the narrative stage parses it; `positions`
    rank each fact among the document's `total` Inline fact elements, which
    must all be found again. `content` maps cell XPaths to cell IDs and block
    XPaths to block IDs. A cell wins over a block; a hidden fact, a fact
    spanning several blocks or tables, or an XPath shared by several blocks
    is left unaligned (None, None) rather than guessed.
    """

    cells, blocks = content
    elements = [
        e for e in root.iter(etree.Element) if element_tag(e) in _HTML_FACT_TAGS
    ]
    if len(elements) != total:
        return [(None, None)] * len(positions)
    return [_locate(elements[position - 1], cells, blocks) for position in positions]

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arelle.formula.XPathContext import FunctionArgType, XPathException
from arelle.FunctionIxt import ixtFunctionNotAvailable
from arelle.ModelValue import InvalidValue
from arelle.ModelXbrl import ModelXbrl

from src.data_pipeline.contracts.sec_filings.xbrl_facts import FactKind
from src.data_pipeline.transforms.sec_filings.xbrl_sources import INLINE_XBRL

_INLINE_FACT_TAGS = frozenset(
    f"{{{INLINE_XBRL}}}{name}" for name in ("nonNumeric", "nonFraction", "fraction")
)
_HIDDEN = f"{{{INLINE_XBRL}}}hidden"
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
_CONTINUATION = f"{{{INLINE_XBRL}}}continuation"
# Errors of Inline transformations and number parsing on a fact's text; the
# fact is kept with its raw value and no normalized value.
_VALUE_ERRORS = (
    ixtFunctionNotAvailable,
    FunctionArgType,
    XPathException,
    ValueError,
    ArithmeticError,
)


class XbrlProcessorError(ValueError):
    """The XBRL processor could not load or interpret an instance."""


@dataclass(frozen=True)
class LoadedConcept:
    """A concept of the loaded DTS; QName-valued fields in Clark notation."""

    namespace_uri: str
    local_name: str
    data_type: str | None
    substitution_group: str | None
    period_type: str | None
    balance: str | None
    is_abstract: bool
    is_nillable: bool
    source_schema_uri: str
    is_extension: bool


@dataclass(frozen=True)
class LoadedFact:
    """A fact occurrence as the processor read it, in document and DOM order.

    `position` is its 1-based rank among all Inline fact elements of its
    document (None for XML instances), used to find it in another parse.
    """

    document_id: str
    position: int | None
    namespace_uri: str
    local_name: str
    context_ref: str
    unit_ref: str | None
    source_fact_id: str | None
    fact_kind: FactKind
    raw_value: str | None
    normalized_value: str | None
    normalization_error: str | None
    is_nil: bool
    is_hidden: bool
    decimals: str | None
    precision: str | None
    scale: int | None
    sign: str | None
    format: str | None
    escape: bool | None
    xml_language: str | None
    continuation_refs: tuple[str, ...]
    source_xpath: str


def _clark(qname: Any) -> str | None:
    return None if qname is None else str(qname.clarkNotation)


def read_concepts(model: ModelXbrl, directory: Path) -> list[LoadedConcept]:
    """Items, tuples and typed-dimension domains of the DTS, with schema lineage.

    Concepts defined by files under `directory`, where the filing's own
    documents were written, are the filing's extension.
    """

    typed_domains = {
        concept.typedDomainElement
        for concept in model.qnameConcepts.values()
        if concept.isTypedDimension and concept.typedDomainElement is not None
    }
    concepts = []
    for concept in model.qnameConcepts.values():
        # xbrli:item and xbrli:tuple head the substitution groups; they are
        # the only items or tuples without a group of their own.
        is_concept = (
            concept.isItem or concept.isTuple
        ) and concept.substitutionGroupQname is not None
        if not (is_concept or concept in typed_domains):
            continue
        uri = str(concept.modelDocument.uri)
        is_extension = Path(uri).is_relative_to(directory)
        concepts.append(
            LoadedConcept(
                namespace_uri=concept.qname.namespaceURI,
                local_name=concept.qname.localName,
                data_type=_clark(concept.typeQname),
                substitution_group=_clark(concept.substitutionGroupQname),
                period_type=concept.periodType,
                balance=concept.balance,
                is_abstract=bool(concept.isAbstract),
                is_nillable=bool(concept.isNillable),
                source_schema_uri=Path(uri).relative_to(directory).as_posix()
                if is_extension
                else uri,
                is_extension=is_extension,
            )
        )
    return concepts


def _continuations(
    fact: Any, chains: dict[str, Any], used: set[str]
) -> tuple[str, ...]:
    """IDs of a fact's `ix:continuation` chain in order; missing, reused or cyclic fails."""

    refs: list[str] = []
    next_id = fact.get("continuedAt")
    while next_id:
        if next_id in refs or next_id in used:
            raise XbrlProcessorError(f"continuation {next_id!r} is reused or cyclic")
        if next_id not in chains:
            raise XbrlProcessorError(f"continuation {next_id!r} does not exist")
        refs.append(next_id)
        next_id = chains[next_id].get("continuedAt")
    used.update(refs)
    return tuple(refs)


def _normalized(values: Sequence[Any]) -> tuple[str | None, str | None]:
    """The processor's value of a fact, or its fraction terms joined by "/".

    Returns (value, None), or (None, reason) when a transformation failed.
    """

    normalized = []
    for value_object in values:
        try:
            value = value_object.value
        except _VALUE_ERRORS as failure:
            return None, f"{type(failure).__name__}: {failure}"
        # A transformation that failed while loading is cached as a sentinel.
        if isinstance(value, InvalidValue):
            return None, f"transformation {value_object.format} failed"
        normalized.append(str(value))
    return "/".join(normalized), None


def _values(element: Any, *, inline: bool) -> tuple[str | None, str | None, str | None]:
    """(raw, normalized, normalization error) of a fact; nil facts have none.

    A fraction's values are "numerator/denominator": Inline terms carry their
    own format, scale and sign, XML terms are already normalized.
    """

    if element.isNil:
        return None, None, None
    if element.isFraction:
        raw = "/".join(element.fractionValue)
        return (raw, *_normalized(list(element))) if inline else (raw, raw, None)
    return (
        element.rawValue if inline else element.textValue,
        *_normalized([element]),
    )


def _xml_language(element: Any) -> str | None:
    """The xml:lang in scope at the fact in its source document.

    Inline facts inherit it from their HTML ancestors, which the processor's
    own lookup does not follow; an empty value means no language.
    """

    for node in (element, *element.iterancestors()):
        language = node.get(_XML_LANG)
        if language is not None:
            return language.strip() or None
    return None


def _fact(
    element: Any, document_id: str, position: int | None, refs: tuple[str, ...]
) -> LoadedFact:
    inline = position is not None
    kind: FactKind = (
        "fraction"
        if element.isFraction
        else "numeric"
        if element.isNumeric
        else "non_numeric"
    )
    raw, normalized, error = _values(element, inline=inline)
    scale = element.get("scale") if inline else None
    return LoadedFact(
        document_id=document_id,
        position=position,
        namespace_uri=element.qname.namespaceURI,
        local_name=element.qname.localName,
        context_ref=element.contextID,
        unit_ref=element.unitID or None,
        source_fact_id=element.get("id") or None,
        fact_kind=kind,
        raw_value=raw,
        normalized_value=normalized,
        normalization_error=error,
        is_nil=bool(element.isNil),
        is_hidden=inline and any(a.tag == _HIDDEN for a in element.iterancestors()),
        decimals=element.get("decimals"),
        precision=element.get("precision"),
        scale=None if scale is None else int(scale),
        sign=element.get("sign") if inline else None,
        format=_clark(element.format) if inline else None,
        escape=element.get("escape") in ("true", "1")
        if inline and kind == "non_numeric"
        else None,
        # Language only qualifies text; numbers have none.
        xml_language=_xml_language(element) if kind == "non_numeric" else None,
        continuation_refs=refs,
        source_xpath=str(element.getroottree().getpath(element)),
    )


def read_facts(
    model: ModelXbrl, entries: Sequence[tuple[str, Path]], *, inline: bool
) -> tuple[list[LoadedFact], dict[str, int], list[Any]]:
    """The instance's facts in entry order, then DOM order within each document.

    `entries` are the (document_id, path) of the instance's own documents.
    Also returns the number of Inline fact elements of each entry document,
    and the processor's fact objects in the same order, so relationships
    can name a fact by its position. Facts of concepts missing from the DTS,
    tuples and broken continuation chains raise XbrlProcessorError.
    """

    # The processor leaves such facts out of the instance: XML ones as
    # undefined facts, Inline ones as facts without a concept.
    undefined = [
        *model.undefinedFacts,
        *(fact for fact in model.facts if fact.concept is None),
    ]
    if undefined:
        names = sorted({str(fact.qname) for fact in undefined})
        raise XbrlProcessorError(f"fact concepts are not in the DTS: {names[:10]}")
    loaded = set(model.factsInInstance)
    if any(fact.isTuple for fact in loaded):
        raise XbrlProcessorError(
            "unsupported_xbrl_tuple: tuple facts are not supported"
        )
    documents = {Path(doc.filepath): doc for doc in model.urlDocs.values()}
    roots = [(doc_id, documents[path].xmlRootElement) for doc_id, path in entries]
    chains = {
        element.get("id"): element
        for _, root in roots
        for element in root.iter(_CONTINUATION)
    }
    facts: list[LoadedFact] = []
    elements: list[Any] = []
    used: set[str] = set()
    totals = {}
    for document_id, root in roots:
        position = 0
        for element in root.iter():
            is_inline_fact = inline and element.tag in _INLINE_FACT_TAGS
            position += is_inline_fact
            if element in loaded:
                refs = _continuations(element, chains, used) if inline else ()
                facts.append(
                    _fact(element, document_id, position if inline else None, refs)
                )
                elements.append(element)
        totals[document_id] = position
    if len(facts) != len(loaded):
        raise XbrlProcessorError(
            f"{len(loaded) - len(facts)} facts are outside the entry documents"
        )
    return facts, totals, elements

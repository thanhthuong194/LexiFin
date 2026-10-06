from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from arelle import XbrlConst
from arelle.ModelDtsObject import ModelConcept
from arelle.ModelXbrl import ModelXbrl

from src.data_pipeline.transforms.sec_filings.xbrl_model_enrichment import QNameParts
from src.data_pipeline.transforms.sec_filings.xbrl_model_reader import (
    XbrlProcessorError,
)

# Extended link elements of the networks, by the linkbase type they make.
LINKBASE_TYPES = {
    XbrlConst.qnLinkPresentationLink: "presentation",
    XbrlConst.qnLinkCalculationLink: "calculation",
    XbrlConst.qnLinkDefinitionLink: "definition",
}
# XBRL Dimensions attributes and the arcroles they apply to; Arelle also
# answers them on arcs where they do not apply.
_CLOSED_ARCROLES = frozenset({XbrlConst.all, XbrlConst.notAll})


@dataclass(frozen=True)
class LoadedLabel:
    """An effective concept-label relationship and its label resource.

    `document` identifies the DTS document of the resource: the filing's
    document_id for its own files, else the published URL.
    """

    concept: QNameParts
    document: str
    source_xpath: str
    text: str
    role_uri: str
    language: str | None
    label: str | None
    linkrole_uri: str
    arc_xpath: str


@dataclass(frozen=True)
class LoadedArc:
    """An effective presentation, calculation or definition relationship.

    Attributes that do not apply to the arc are None; `document` and
    `source_xpath` locate the arc as `LoadedLabel.document` does.
    """

    linkbase_type: str
    role_uri: str
    source: QNameParts
    target: QNameParts
    arcrole_uri: str
    order: Decimal
    weight: Decimal | None
    preferred_label_role: str | None
    target_role_uri: str | None
    context_element: str | None
    is_closed: bool | None
    is_usable: bool | None
    document: str
    source_xpath: str


def _parts(concept: ModelConcept) -> QNameParts:
    return str(concept.qname.namespaceURI), str(concept.qname.localName)


def _xpath(element: Any) -> str:
    return str(element.getroottree().getpath(element))


def _document(document: Any, documents: Mapping[Path, str]) -> str:
    """The filing document_id of a materialized file, else the document's URL."""

    document_id = documents.get(Path(document.filepath))
    if document_id is not None:
        return document_id
    uri = str(document.uri)
    if "://" not in uri:
        raise XbrlProcessorError(f"DTS document {uri} has no stable identity")
    return uri


def _relationships(
    model: ModelXbrl, links: Collection[Any]
) -> Iterator[tuple[Any, Any]]:
    """(extended link QName, relationship) of the effective relationship sets.

    Each set is one fully named base set (arcrole, link role, link, arc), so
    prohibited and overridden arcs are already resolved away.
    """

    keys = [key for key in model.baseSets if all(key) and key[2] in links]
    for arcrole, linkrole, link, arc in sorted(keys, key=str):
        relationships = model.relationshipSet(arcrole, linkrole, link, arc)
        for relationship in relationships.modelRelationships:
            yield link, relationship


def _finite(value: Decimal, what: str) -> Decimal:
    if not value.is_finite():
        raise XbrlProcessorError(f"{what} is not a finite decimal")
    return value


def read_labels(model: ModelXbrl, documents: Mapping[Path, str]) -> list[LoadedLabel]:
    """Effective concept-label relationships of the DTS, all roles and languages.

    `documents` maps materialized file paths to their document_id. A label
    without a role has the standard label role (XBRL 2.1).
    """

    labels = []
    for _, relationship in _relationships(model, {XbrlConst.qnLinkLabelLink}):
        concept, resource = relationship.fromModelObject, relationship.toModelObject
        where = f"{relationship.arcrole} relationship in {relationship.linkrole}"
        if not isinstance(concept, ModelConcept) or resource is None:
            raise XbrlProcessorError(f"{where} does not join a concept to a resource")
        if resource.qname != XbrlConst.qnLinkLabel:
            raise XbrlProcessorError(f"{where} ends at {resource.qname}, not a label")
        labels.append(
            LoadedLabel(
                concept=_parts(concept),
                document=_document(resource.modelDocument, documents),
                source_xpath=_xpath(resource),
                text=resource.stringValue,
                role_uri=resource.role or XbrlConst.standardLabel,
                language=resource.xmlLang,
                label=resource.xlinkLabel,
                linkrole_uri=str(relationship.linkrole),
                arc_xpath=_xpath(relationship.arcElement),
            )
        )
    return labels


def _arc(link: Any, relationship: Any, documents: Mapping[Path, str]) -> LoadedArc:
    source, target = relationship.fromModelObject, relationship.toModelObject
    where = f"{relationship.arcrole} relationship in {relationship.linkrole}"
    if not (isinstance(source, ModelConcept) and isinstance(target, ModelConcept)):
        raise XbrlProcessorError(f"{where} does not join two concepts")
    linkbase_type = LINKBASE_TYPES[link]
    weight = relationship.weightDecimal
    if weight is not None:
        weight = _finite(weight, f"{where}: weight")
    elif relationship.arcrole in XbrlConst.summationItems:
        raise XbrlProcessorError(f"{where} has no weight")
    arc = relationship.arcElement
    definition = linkbase_type == "definition"
    return LoadedArc(
        linkbase_type=linkbase_type,
        role_uri=str(relationship.linkrole),
        source=_parts(source),
        target=_parts(target),
        arcrole_uri=str(relationship.arcrole),
        # The processor gives the XBRL default order 1 when none is written.
        order=_finite(relationship.orderDecimal, f"{where}: order"),
        weight=weight if linkbase_type == "calculation" else None,
        preferred_label_role=relationship.preferredLabel
        if linkbase_type == "presentation"
        else None,
        target_role_uri=relationship.targetRole if definition else None,
        context_element=relationship.contextElement if definition else None,
        is_closed=relationship.isClosed
        if relationship.arcrole in _CLOSED_ARCROLES
        else None,
        is_usable=relationship.isUsable if relationship.usable is not None else None,
        document=_document(arc.modelDocument, documents),
        source_xpath=_xpath(arc),
    )


def read_arcs(model: ModelXbrl, documents: Mapping[Path, str]) -> list[LoadedArc]:
    """Effective presentation, calculation and definition relationships of the DTS.

    Every arcrole of these extended links is read, so calculation 1.1 and
    dimensional arcs come with the others. Both ends must be concepts.
    """

    return [
        _arc(link, relationship, documents)
        for link, relationship in _relationships(model, LINKBASE_TYPES)
    ]


def read_role_definitions(
    model: ModelXbrl, role_uris: Collection[str]
) -> dict[str, str | None]:
    """The definition of each role type in the DTS, None when it has none.

    Only the role type's own definition is read, never a name guessed from
    the URI; two different definitions of one role raise XbrlProcessorError.
    """

    definitions = {}
    for role_uri in role_uris:
        found = {
            role_type.definition for role_type in model.roleTypes.get(role_uri, ())
        }
        if len(found) > 1:
            raise XbrlProcessorError(f"role {role_uri} has definitions {sorted(found)}")
        definitions[role_uri] = next(iter(found), None)
    return definitions

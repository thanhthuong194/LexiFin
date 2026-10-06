from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arelle import XbrlConst, XmlUtil
from arelle.ModelXbrl import ModelXbrl
from arelle.PrototypeDtsObject import ArcPrototype

from src.data_pipeline.transforms.sec_filings.xbrl_model_reader import (
    XbrlProcessorError,
)
from src.data_pipeline.transforms.sec_filings.xbrl_resources import (
    canonical_xml,
    context_definition,
)

# The processor's name for every relationship of the footnote links, whatever
# its arcrole, so relationships other than fact-footnote cannot pass unseen.
_FOOTNOTE_LINK_RELATIONSHIPS = "XBRL-footnotes"
QNameParts = tuple[str, str]


@dataclass(frozen=True)
class LoadedDimension:
    """A dimension reported in a context, as (namespace, local name) QNames.

    Explicit dimensions have a `member`; typed ones the Clark QName, text
    content and canonical XML of their one member element.
    """

    context_ref: str
    context_element: str
    dimension: QNameParts
    member: QNameParts | None
    typed_member_qname: str | None
    typed_member_text: str | None
    typed_member_xml: str | None
    source_xpath: str


@dataclass(frozen=True)
class LoadedFootnote:
    """An effective fact-footnote relationship of the instance.

    `fact_index` is the fact's position in the instance's facts; the
    footnote is located by `document_id` and `source_xpath`. `text` is its
    text content and `html` its inner markup, None when it has no element.
    """

    fact_index: int
    document_id: str
    source_xpath: str
    text: str
    html: str | None
    role_uri: str
    arcrole_uri: str
    linkrole_uri: str
    language: str | None
    label: str | None
    arc_xpath: str | None


def _parts(qname: Any) -> QNameParts:
    return str(qname.namespaceURI), str(qname.localName)


def _xpath(element: Any) -> str:
    return str(element.getroottree().getpath(element))


def read_context_hashes(model: ModelXbrl) -> dict[str, str]:
    """The `context_hash` of every context, computed as XBRL core does."""

    return {
        ref: context_definition(context).sha256()
        for ref, context in model.contexts.items()
    }


def _dimension(context_ref: str, value: Any) -> LoadedDimension:
    owner = f"context {context_ref!r}, dimension {value.dimensionQname}"
    xpath = _xpath(value)
    if value.isExplicit:
        if value.member is None:
            raise XbrlProcessorError(
                f"{owner}: member {value.memberQname} is not in the DTS"
            )
        member: QNameParts | None = _parts(value.memberQname)
        typed: tuple[str | None, str | None, str | None] = (None, None, None)
    else:
        children = [child for child in value if isinstance(child.tag, str)]
        if len(children) != 1:
            raise XbrlProcessorError(
                f"{owner} at {xpath}: typed member has {len(children)} elements"
            )
        member, child = None, children[0]
        typed = (
            str(child.tag),
            XmlUtil.innerText(child, strip=False),
            canonical_xml(child),
        )
    return LoadedDimension(
        context_ref=context_ref,
        context_element=str(value.contextElement),
        dimension=_parts(value.dimensionQname),
        member=member,
        typed_member_qname=typed[0],
        typed_member_text=typed[1],
        typed_member_xml=typed[2],
        source_xpath=xpath,
    )


def read_dimensions(model: ModelXbrl) -> list[LoadedDimension]:
    """Dimensions reported in the contexts' segments and scenarios.

    Taxonomy defaults are not reported, so they are not read. A dimension
    the processor rejected (unknown, or repeated in a context element), or
    reported in both segment and scenario, raises XbrlProcessorError.
    """

    dimensions: list[LoadedDimension] = []
    for ref, context in model.contexts.items():
        if context.errorDimValues:
            names = sorted(
                str(value.dimensionQname) for value in context.errorDimValues
            )
            raise XbrlProcessorError(f"context {ref!r}: invalid dimensions {names}")
        segment, scenario = context.dimValues("segment"), context.dimValues("scenario")
        if segment.keys() & scenario.keys():
            raise XbrlProcessorError(
                f"context {ref!r}: a dimension is in both segment and scenario"
            )
        dimensions.extend(
            _dimension(ref, value) for value in [*segment.values(), *scenario.values()]
        )
    return dimensions


def _footnote(
    relationship: Any, fact_index: int, documents: Mapping[Path, str]
) -> LoadedFootnote:
    resource = relationship.toModelObject
    where = f"{relationship.arcrole} relationship in {relationship.linkrole}"
    if resource is None or resource.qname != XbrlConst.qnLinkFootnote:
        raise XbrlProcessorError(f"{where} ends at {type(resource).__name__}")
    document_id = documents.get(Path(resource.modelDocument.filepath))
    if document_id is None:
        raise XbrlProcessorError(f"{where}: footnote is outside the entry documents")
    if not (resource.role and relationship.linkrole):
        raise XbrlProcessorError(f"{where}: footnote role or link role is missing")
    arc = relationship.arcElement
    # An Inline relationship is a prototype of its ix:relationship element.
    source_arc = arc.arcElement if isinstance(arc, ArcPrototype) else arc
    html = XmlUtil.innerText(
        resource,
        ixExclude=True,
        ixEscape="html",
        ixContinuation=True,
        ixResolveUris=False,
        strip=True,
    )
    return LoadedFootnote(
        fact_index=fact_index,
        document_id=document_id,
        source_xpath=_xpath(resource),
        text=XmlUtil.innerText(
            resource, ixExclude=True, ixContinuation=True, strip=False
        ),
        # Escaped text never holds "<", so one marks an element.
        html=html if "<" in html else None,
        role_uri=str(resource.role),
        arcrole_uri=str(relationship.arcrole),
        linkrole_uri=str(relationship.linkrole),
        language=resource.xmlLang,
        label=resource.xlinkLabel,
        arc_xpath=None if source_arc is None else _xpath(source_arc),
    )


def read_footnotes(
    model: ModelXbrl, fact_objects: Sequence[Any], documents: Mapping[Path, str]
) -> list[LoadedFootnote]:
    """Effective relationships of the instance's footnote links.

    Each must be a fact-footnote relationship from one of `fact_objects` (the
    instance's facts in order) to a footnote of an entry document (`documents`
    maps their paths to document IDs); anything else raises XbrlProcessorError.
    """

    positions = {fact: index for index, fact in enumerate(fact_objects)}
    footnotes = []
    relationships = model.relationshipSet(_FOOTNOTE_LINK_RELATIONSHIPS)
    for relationship in relationships.modelRelationships:
        where = f"{relationship.arcrole} relationship in {relationship.linkrole}"
        if relationship.arcrole != XbrlConst.factFootnote:
            raise XbrlProcessorError(f"unsupported {where}")
        fact = relationship.fromModelObject
        if fact not in positions:
            raise XbrlProcessorError(
                f"{where} starts at {type(fact).__name__}, not a fact of the instance"
            )
        footnotes.append(_footnote(relationship, positions[fact], documents))
    return footnotes

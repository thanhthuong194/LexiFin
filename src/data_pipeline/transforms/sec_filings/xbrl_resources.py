from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Literal

from lxml import etree

from src.data_pipeline.contracts.sec_filings.xbrl_core import (
    ContextDefinition,
    PeriodType,
    unit_expression,
)
from src.data_pipeline.transforms.sec_filings.xbrl_sources import (
    XBRLI,
    XbrlConflict,
    expand_qname,
)

XBRLDI = "http://xbrl.org/2006/xbrldi"
_EXPLICIT_MEMBER = f"{{{XBRLDI}}}explicitMember"
_DIMENSION_MEMBERS = frozenset({_EXPLICIT_MEMBER, f"{{{XBRLDI}}}typedMember"})


@dataclass(frozen=True)
class UnitDefinition:
    """What a unit means: sorted numerator and denominator measures (Clark notation)."""

    unit_kind: Literal["simple", "divide"]
    numerator_measures: tuple[str, ...]
    denominator_measures: tuple[str, ...]

    @property
    def expression(self) -> str:
        """The unit's normalized expression, e.g. `({iso4217}USD)/({xbrli}shares)`."""

        return unit_expression(self.numerator_measures, self.denominator_measures)


def _xbrli(element: etree._Element, name: str) -> etree._Element | None:
    return element.find(f"{{{XBRLI}}}{name}")


def _required(element: etree._Element, name: str, owner: str) -> etree._Element:
    child = _xbrli(element, name)
    if child is None:
        raise ValueError(f"{owner} has no xbrli:{name}")
    return child


def _qname_values(element: etree._Element) -> list[tuple[str, str | None]]:
    """(dimension, explicit member) QNames of a dimension member, expanded."""

    if element.tag not in _DIMENSION_MEMBERS:
        return []
    member = element.text if element.tag == _EXPLICIT_MEMBER else None
    return [
        (expand_qname(element.get("dimension"), element), None)
        if member is None
        else (
            expand_qname(element.get("dimension"), element),
            expand_qname(member, element),
        )
    ]


def _namespaces(element: etree._Element) -> set[str]:
    """Namespaces of element and attribute names, and of QName values, in a subtree."""

    uris = set()
    for node in element.iter(etree.Element):
        uris.add(etree.QName(node).namespace)
        uris.update(etree.QName(name).namespace for name in node.attrib)
        for dimension, member in _qname_values(node):
            uris.update(etree.QName(q).namespace for q in (dimension, member) if q)
    return uris - {None}


def _member_key(element: etree._Element) -> tuple[int, str]:
    values = _qname_values(element)
    return (0, values[0][0]) if values else (1, "")


def _canonical_copy(
    source: etree._Element, parent: etree._Element | None, prefixes: Mapping[str, str]
) -> etree._Element:
    def prefixed(clark: str) -> str:
        qname = etree.QName(clark)
        return f"{prefixes[qname.namespace]}:{qname.localname}"

    if parent is None:
        nsmap = {prefix: uri for uri, prefix in prefixes.items()}
        copy = etree.Element(source.tag, nsmap=nsmap)
    else:
        copy = etree.SubElement(parent, source.tag)
    values = _qname_values(source)
    for name, value in sorted(source.attrib.items()):
        copy.set(
            name, prefixed(values[0][0]) if values and name == "dimension" else value
        )
    children = [child for child in source if isinstance(child.tag, str)]
    if values and values[0][1] is not None:
        copy.text = prefixed(values[0][1])
    elif not children or (source.text and source.text.strip()):
        copy.text = source.text
    # Dimension members are unordered (XBRL Dimensions); other children keep order.
    for child in sorted(children, key=_member_key):
        child_copy = _canonical_copy(child, copy, prefixes)
        child_copy.tail = child.tail if child.tail and child.tail.strip() else None
    return copy


def canonical_xml(element: etree._Element) -> str:
    """Serialize a segment or scenario so equal content reads alike in any document.

    Prefixes become ns0, ns1, ... by namespace URI, also inside QName values
    (dimension attributes, explicit members); whitespace-only text between
    elements and comments are dropped; dimension members are ordered by their
    expanded dimension; the result is inclusive C14N of a standalone element,
    so it declares exactly the namespaces it uses. Nothing else is removed.
    """

    prefixes = {uri: f"ns{i}" for i, uri in enumerate(sorted(_namespaces(element)))}
    copy = _canonical_copy(element, None, prefixes)
    return str(etree.tostring(copy, method="c14n").decode("utf-8"))


def _date(element: etree._Element, owner: str) -> date:
    text = (element.text or "").strip()
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{owner} date {text!r} is not YYYY-MM-DD") from error


def context_definition(context: etree._Element) -> ContextDefinition:
    """Read an `xbrli:context`: entity, lexical period dates, segment, scenario.

    Raises ValueError when a required part is missing or a date is not a
    calendar date.
    """

    owner = f"context {context.get('id')!r}"
    entity = _required(context, "entity", owner)
    identifier = _required(entity, "identifier", owner)
    period = _required(context, "period", owner)
    instant, start, end = (
        _xbrli(period, n) for n in ("instant", "startDate", "endDate")
    )
    dates: tuple[date | None, date | None, date | None]
    period_type: PeriodType
    if instant is not None:
        period_type, dates = "instant", (None, None, _date(instant, owner))
    elif start is not None and end is not None:
        period_type, dates = "duration", (_date(start, owner), _date(end, owner), None)
    elif _xbrli(period, "forever") is not None:
        period_type, dates = "forever", (None, None, None)
    else:
        raise ValueError(f"{owner} has no instant, duration or forever period")
    segment, scenario = _xbrli(entity, "segment"), _xbrli(context, "scenario")
    return ContextDefinition(
        entity_identifier_scheme=(identifier.get("scheme") or "").strip(),
        entity_identifier=(identifier.text or "").strip(),
        period_type=period_type,
        start_date=dates[0],
        end_date=dates[1],
        instant_date=dates[2],
        segment_xml=None if segment is None else canonical_xml(segment),
        scenario_xml=None if scenario is None else canonical_xml(scenario),
    )


def _measures(parent: etree._Element) -> tuple[str, ...]:
    return tuple(
        sorted(expand_qname(m.text, m) for m in parent.findall(f"{{{XBRLI}}}measure"))
    )


def unit_definition(unit: etree._Element) -> UnitDefinition:
    """Read an `xbrli:unit` as sorted Clark-notation measures.

    Raises ValueError for a unit without measures or a measure whose prefix is
    not in scope.
    """

    owner = f"unit {unit.get('id')!r}"
    divide = _xbrli(unit, "divide")
    if divide is None:
        definition = UnitDefinition("simple", _measures(unit), ())
    else:
        definition = UnitDefinition(
            "divide",
            _measures(_required(divide, "unitNumerator", owner)),
            _measures(_required(divide, "unitDenominator", owner)),
        )
    if not definition.numerator_measures or (
        divide is not None and not definition.denominator_measures
    ):
        raise ValueError(f"{owner} has no measure")
    return definition


def _merge[Definition](
    kind: str, definitions: Iterable[tuple[str, Definition]]
) -> dict[str, Definition]:
    merged: dict[str, Definition] = {}
    for ref, definition in definitions:
        if not ref:
            raise ValueError(f"an xbrli:{kind} has no id")
        if merged.setdefault(ref, definition) != definition:
            raise XbrlConflict(f"{kind} {ref!r} is defined twice, differently")
    return merged


def read_contexts(roots: Iterable[etree._Element]) -> dict[str, ContextDefinition]:
    """Every context of the given documents by ID; repeats must be identical."""

    return _merge(
        "context",
        (
            (context.get("id"), context_definition(context))
            for root in roots
            for context in root.iter(f"{{{XBRLI}}}context")
        ),
    )


def read_units(roots: Iterable[etree._Element]) -> dict[str, UnitDefinition]:
    """Every unit of the given documents by ID; repeats must be identical."""

    return _merge(
        "unit",
        (
            (unit.get("id"), unit_definition(unit))
            for root in roots
            for unit in root.iter(f"{{{XBRLI}}}unit")
        ),
    )

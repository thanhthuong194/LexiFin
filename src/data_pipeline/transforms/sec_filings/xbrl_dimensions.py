from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from lxml import etree

from src.data_pipeline.contracts.sec_filings.xbrl_concepts import concept_id_for
from src.data_pipeline.contracts.sec_filings.xbrl_dimensions import (
    XbrlDimensionRecord,
    dimension_hash,
    dimension_id_for,
)
from src.data_pipeline.transforms.sec_filings.html_dom import normalize_text
from src.data_pipeline.transforms.sec_filings.xbrl_model_enrichment import (
    LoadedDimension,
    QNameParts,
)
from src.data_pipeline.transforms.sec_filings.xbrl_resources import XBRLDI

_MEMBERS = frozenset({f"{{{XBRLDI}}}explicitMember", f"{{{XBRLDI}}}typedMember"})
_ELEMENT_ORDER = {"segment": 0, "scenario": 1}
# The stored segment and scenario XML is this project's own canonical output.
_PARSER = etree.XMLParser(resolve_entities=False, no_network=True)


def expected_dimension_count(segment_xml: str | None, scenario_xml: str | None) -> int:
    """How many dimension members a context's stored segment and scenario report."""

    return sum(
        sum(child.tag in _MEMBERS for child in etree.fromstring(xml.encode(), _PARSER))
        for xml in (segment_xml, scenario_xml)
        if xml is not None
    )


def _clark(parts: QNameParts) -> str:
    return f"{{{parts[0]}}}{parts[1]}"


def _order(dimension: LoadedDimension) -> tuple[int, str, str, str]:
    """Context element, dimension QName, member type, then member QName or XML hash."""

    if dimension.member is not None:
        return (
            _ELEMENT_ORDER[dimension.context_element],
            _clark(dimension.dimension),
            "explicit",
            _clark(dimension.member),
        )
    xml = (dimension.typed_member_xml or "").encode()
    return (
        _ELEMENT_ORDER[dimension.context_element],
        _clark(dimension.dimension),
        "typed",
        hashlib.sha256(xml).hexdigest(),
    )


def dimension_records(
    context_key: str, dimensions: Sequence[LoadedDimension], processed_at: datetime
) -> list[XbrlDimensionRecord]:
    """The `xbrl_dimensions` rows of one context, in canonical order."""

    ordered = sorted(dimensions, key=_order)
    records = []
    for ordinal, dimension in enumerate(ordered, start=1):
        text = dimension.typed_member_text
        values: dict[str, Any] = {
            "dimension_id": dimension_id_for(context_key, ordinal),
            "context_key": context_key,
            "dimension_ordinal": ordinal,
            "context_dimension_count": len(ordered),
            "context_element": dimension.context_element,
            "dimension_concept_id": concept_id_for(*dimension.dimension),
            "member_type": "typed" if dimension.member is None else "explicit",
            "member_concept_id": None
            if dimension.member is None
            else concept_id_for(*dimension.member),
            "typed_member_qname": dimension.typed_member_qname,
            "typed_member_value": None if text is None else normalize_text(text),
            "typed_member_xml": dimension.typed_member_xml,
            "source_xpath": dimension.source_xpath,
            "processed_at": processed_at,
        }
        values["dimension_hash"] = dimension_hash(values)
        records.append(XbrlDimensionRecord.model_validate(values))
    return records

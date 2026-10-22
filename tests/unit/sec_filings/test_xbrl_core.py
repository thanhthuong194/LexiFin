from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from lxml import etree
from pydantic import ValidationError

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlContextRecord
from src.data_pipeline.transforms.sec_filings.xbrl_core import (
    LogicalInstance,
    SourceDocument,
    build_logical_instances,
    context_records,
    instance_record,
    unit_records,
)
from src.data_pipeline.transforms.sec_filings.xbrl_sources import (
    XbrlConflict,
    parse_xbrl_source,
)

pytestmark = pytest.mark.unit

FILING_ID = "sec:0000320193-25-000079"
PROCESSED_AT = datetime(2026, 10, 6, 4, 0, tzinfo=UTC)
ISO = "{http://www.xbrl.org/2003/iso4217}"
XBRLI = "{http://www.xbrl.org/2003/instance}"
EX = "{http://example.com/2024}"
NAMESPACES = (
    'xmlns:link="http://www.xbrl.org/2003/linkbase" '
    'xmlns:xlink="http://www.w3.org/1999/xlink" '
    'xmlns:xbrldi="http://xbrl.org/2006/xbrldi" '
    'xmlns:iso4217="http://www.xbrl.org/2003/iso4217" xmlns:ex="http://example.com/2024"'
)
SCHEMA_REF = '<link:schemaRef xlink:type="simple" xlink:href=" ex-2024.xsd "/>'
DURATION = (
    "<xbrli:startDate>2024-01-01</xbrli:startDate>"
    "<xbrli:endDate>2024-12-31</xbrli:endDate>"
)
INSTANT = "<xbrli:instant>2024-02-29</xbrli:instant>"
USD = '<xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>'


def _context(
    context_id: str, period: str = DURATION, segment: str = "", scenario: str = ""
) -> str:
    return (
        f'<xbrli:context id="{context_id}"><xbrli:entity>'
        '<xbrli:identifier scheme="http://www.sec.gov/CIK">0000320193</xbrli:identifier>'
        f"{segment}</xbrli:entity><xbrli:period>{period}</xbrli:period>{scenario}"
        "</xbrli:context>"
    )  # fmt: skip


def _fact(
    concept: str, context: str, unit: str | None = None, target: str | None = None
) -> str:
    attributes = f' name="{concept}" contextRef="{context}"'
    attributes += f' target="{target}"' if target else ""
    if unit is None:
        return f"<ix:nonNumeric{attributes}>text</ix:nonNumeric>"
    return f'<ix:nonFraction{attributes} unitRef="{unit}" decimals="0">5</ix:nonFraction>'  # fmt: skip


def _inline(
    facts: str,
    resources: str,
    references: str = f"<ix:references>{SCHEMA_REF}</ix:references>",
) -> bytes:
    return (
        '\n<XBRL>\n<?xml version="1.0"?>\n<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:ix="http://www.xbrl.org/2013/inlineXBRL" '
        f'xmlns:xbrli="http://www.xbrl.org/2003/instance" {NAMESPACES}><body>'
        f"<ix:header>{references}<ix:resources>{resources}</ix:resources></ix:header>"
        f"<p>{facts}</p></body></html>\n</XBRL>\n"
    ).encode()  # fmt: skip


def _xml(body: str) -> bytes:
    """An XML instance written the way EDGAR renders Inline XBRL."""

    return (
        '\n<XML>\n<?xml version="1.0"?>\n<xbrl xmlns="http://www.xbrl.org/2003/instance" '
        f'xmlns:xbrli="http://www.xbrl.org/2003/instance" {NAMESPACES}>{SCHEMA_REF}'
        f"{body}</xbrl>\n</XML>\n"
    ).encode()


def _document(
    ordinal: int, content: bytes, document_type: str = "10-K"
) -> DocumentRecord:
    is_xml = content.lstrip().startswith(b"<XML>")
    return DocumentRecord.model_validate(
        {
            "document_id": f"{FILING_ID}:document:{ordinal:04d}",
            "filing_id": FILING_ID,
            "filing_date": date(2025, 10, 31),
            "document_ordinal": ordinal,
            "sequence": str(ordinal),
            "document_type": "XML" if is_xml and document_type == "10-K" else document_type,
            "filename": f"doc{ordinal}.{'xml' if is_xml else 'htm'}",
            "description": None,
            "role": "xbrl_instance" if is_xml else "primary_document",
            "content_kind": "xml" if is_xml else "html",
            "source_path": "bronze/sec_filings/x/full-submission.txt",
            "source_submission_sha256": "a" * 64,
            "document_start_byte": 0,
            "document_end_byte": len(content) + 20,
            "content_start_byte": 10,
            "content_end_byte": 10 + len(content),
            "content_size_bytes": len(content),
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "processed_at": PROCESSED_AT,
        }
    )  # fmt: skip


def _instances(
    inline: list[bytes], xml: list[bytes] = (), xml_type: str = "10-K"
) -> list[LogicalInstance]:
    def source(ordinal: int, content: bytes, kind: str = "10-K") -> SourceDocument:
        return SourceDocument(
            _document(ordinal, content, kind), parse_xbrl_source(content)
        )

    return build_logical_instances(
        FILING_ID,
        [source(i, content) for i, content in enumerate(inline, start=1)],
        [source(i, content, xml_type) for i, content in enumerate(xml, start=10)],
    )


def test_default_target_owns_every_context_and_unit() -> None:
    (instance,) = _instances(
        [_inline(_fact("ex:Revenue", "FY", "usd"), _context("FY") + _context("AsOf", INSTANT) + USD)]
    )  # fmt: skip
    record = instance_record(instance, PROCESSED_AT)

    assert record.instance_id == f"{FILING_ID}:xbrl:inline:default"
    assert (record.target_name, record.is_inline) == (None, True)
    assert (record.fact_count, record.context_count, record.unit_count) == (1, 2, 1)
    assert record.schema_refs == ["ex-2024.xsd"]
    assert record.namespace_map[""] == "http://www.w3.org/1999/xhtml"
    assert list(record.namespace_map) == sorted(record.namespace_map)
    assert record.source_document_ids == [f"{FILING_ID}:document:0001"]


def test_named_targets_split_facts_and_share_contexts() -> None:
    references = (
        f"<ix:references>{SCHEMA_REF}</ix:references>"
        f'<ix:references target="parent">{SCHEMA_REF.replace("ex-2024", "p-2024")}</ix:references>'
    )  # fmt: skip
    facts = _fact("ex:Revenue", "FY", "usd") + _fact("ex:Notes", "FY", target="parent")

    default, named = _instances([_inline(facts, _context("FY") + USD, references)])

    target_hash = hashlib.sha256(b"parent").hexdigest()
    assert named.instance_id == f"{FILING_ID}:xbrl:inline:{target_hash}"
    assert (default.target_name, named.target_name) == (None, "parent")
    assert named.schema_refs == ("p-2024.xsd",)
    assert (set(default.contexts), set(named.contexts)) == ({"FY"}, {"FY"})
    assert (set(default.units), set(named.units)) == ({"usd"}, set())
    keys = {
        c.context_key
        for i in (default, named)
        for c in context_records(i, PROCESSED_AT)
    }
    assert len(keys) == 2


def test_resource_unused_by_every_target_is_a_conflict() -> None:
    facts = _fact("ex:Revenue", "FY", "usd") + _fact("ex:Notes", "FY", target="parent")

    with pytest.raises(XbrlConflict, match="unassigned_inline_resource"):
        _instances([_inline(facts, _context("FY") + _context("Spare") + USD)])


def test_standalone_xml_instance_is_its_own_instance() -> None:
    xml = _xml(
        _context("FY").replace("xbrli:", "") + USD.replace("xbrli:", "")
        + '<ex:Revenue contextRef="FY" unitRef="usd" decimals="0">5</ex:Revenue>'
    )  # fmt: skip

    (instance,) = _instances([], [xml], xml_type="EX-101.INS")
    record = instance_record(instance, PROCESSED_AT)

    expected = hashlib.sha256(f"{FILING_ID}:document:0010".encode()).hexdigest()
    assert record.instance_id == f"{FILING_ID}:xbrl:xml:{expected}"
    assert (record.is_inline, record.target_name, record.fact_count) == (False, None, 1)
    assert record.namespace_map[""] == "http://www.xbrl.org/2003/instance"


def test_generated_xml_matching_inline_is_merged_into_one_instance() -> None:
    segment = (
        '<xbrli:segment><xbrldi:explicitMember dimension="ex:AAxis">ex:AMember'
        '</xbrldi:explicitMember><xbrldi:explicitMember dimension="ex:BAxis">'
        "ex:BMember</xbrldi:explicitMember></xbrli:segment>"
    )
    eur = '<xbrli:unit id="eur"><xbrli:measure>iso4217:EUR</xbrli:measure></xbrli:unit>'
    inline = _inline(
        _fact("ex:Revenue", "S", "usd"), _context("S", segment=segment) + USD + eur
    )
    # EDGAR: default namespace, other prefixes and whitespace, members reordered,
    # and the unused unit left out.
    reordered = (
        "<segment>\n  <xbrldi:explicitMember dimension='ex:BAxis'>ex:BMember"
        "</xbrldi:explicitMember>\n  <xbrldi:explicitMember dimension='ex:AAxis'> "
        "ex:AMember </xbrldi:explicitMember>\n</segment>"
    )
    xml = _xml(
        _context("S").replace("xbrli:", "").replace("</entity>", f"{reordered}</entity>")
        + USD.replace("xbrli:", "")
        + '<ex:Revenue contextRef="S" unitRef="usd" decimals="0">5</ex:Revenue>'
    )  # fmt: skip

    (instance,) = _instances([inline], [xml])

    assert [d.document_id for d in instance.sources] == [
        f"{FILING_ID}:document:0001",
        f"{FILING_ID}:document:0010",
    ]
    assert set(instance.units) == {"usd", "eur"}


@pytest.mark.parametrize(
    ("xml_value", "merged"), [("Cover  Agreements", True), ("Other", False)]
)
def test_generated_xml_typed_member_whitespace_still_matches(
    xml_value: str, merged: bool
) -> None:
    def segment(value: str) -> str:
        return (
            '<xbrli:segment><xbrldi:typedMember dimension="ex:RegionAxis">'
            f"<ex:Region>{value}</ex:Region></xbrldi:typedMember></xbrli:segment>"
        )

    inline = _inline(
        _fact("ex:Revenue", "S", "usd"),
        _context("S", segment=segment(" Cover Agreements ")) + USD,
    )
    # EDGAR trims and collapses whitespace in typed member values.
    xml = _xml(
        _context("S", segment=segment(xml_value)).replace("xbrli:", "")
        + USD.replace("xbrli:", "")
        + '<ex:Revenue contextRef="S" unitRef="usd" decimals="0">5</ex:Revenue>'
    )

    if not merged:
        with pytest.raises(XbrlConflict, match="does not represent"):
            _instances([inline], [xml])
        return
    (instance,) = _instances([inline], [xml])

    assert len(instance.sources) == 2
    # The Inline definition, as written, is the one kept.
    assert "<ns0:Region> Cover Agreements </ns0:Region>" in (
        instance.contexts["S"].segment_xml or ""
    )


def test_nested_default_namespace_rebinding_keeps_the_outer_binding() -> None:
    # The resources rebind the default namespace (XHTML at the root) to xbrli.
    inline = _inline(_fact("ex:Revenue", "FY", "usd"), _context("FY") + USD).replace(
        b"<ix:resources>",
        b'<ix:resources xmlns="http://www.xbrl.org/2003/instance">',
    )

    (instance,) = _instances([inline])

    assert instance.namespace_map[""] == "http://www.w3.org/1999/xhtml"
    assert set(instance.contexts) == {"FY"}


def test_generated_xml_with_other_facts_is_a_conflict() -> None:
    inline = _inline(_fact("ex:Revenue", "FY", "usd"), _context("FY") + USD)
    xml = _xml(_context("FY").replace("xbrli:", "") + USD.replace("xbrli:", ""))

    with pytest.raises(XbrlConflict, match="does not represent"):
        _instances([inline], [xml])


def test_periods_keep_their_lexical_dates() -> None:
    resources = (
        _context("D") + _context("I", INSTANT) + _context("F", "<xbrli:forever/>")
    )
    facts = _fact("ex:A", "D") + _fact("ex:B", "I") + _fact("ex:C", "F")

    (instance,) = _instances([_inline(facts, resources)])

    periods = {
        c.context_ref: (c.period_type, c.start_date, c.end_date, c.instant_date)
        for c in context_records(instance, PROCESSED_AT)
    }
    assert periods == {
        "D": ("duration", date(2024, 1, 1), date(2024, 12, 31), None),
        "I": ("instant", None, None, date(2024, 2, 29)),
        "F": ("forever", None, None, None),
    }


def test_segment_and_scenario_keep_explicit_and_typed_dimensions() -> None:
    segment = (
        '<xbrli:segment><xbrldi:typedMember dimension="ex:RegionAxis">'
        "<ex:Region>North</ex:Region></xbrldi:typedMember>"
        '<xbrldi:explicitMember dimension="ex:ProductAxis">ex:WidgetsMember'
        "</xbrldi:explicitMember><ex:Note>kept</ex:Note></xbrli:segment>"
    )
    scenario = (
        '<xbrli:scenario><xbrldi:explicitMember dimension="ex:ScenarioAxis">'
        "ex:ForecastMember</xbrldi:explicitMember></xbrli:scenario>"
    )

    (instance,) = _instances([_inline(_fact("ex:A", "S"), _context("S", segment=segment, scenario=scenario))])  # fmt: skip
    (context,) = context_records(instance, PROCESSED_AT)

    assert context.segment_xml is not None and context.scenario_xml is not None
    parsed = etree.fromstring(context.segment_xml)
    members = parsed.findall("{http://xbrl.org/2006/xbrldi}explicitMember")
    assert [m.get("dimension") for m in members] == ["ns0:ProductAxis"]
    assert parsed.nsmap["ns0"] == "http://example.com/2024"
    assert "North" in context.segment_xml and "kept" in context.segment_xml
    assert "ForecastMember" in context.scenario_xml


def test_units_expand_and_sort_measures() -> None:
    units = (
        USD
        + '<xbrli:unit id="perShare"><xbrli:divide><xbrli:unitNumerator>'
        "<xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unitNumerator>"
        "<xbrli:unitDenominator><xbrli:measure>xbrli:shares</xbrli:measure>"
        "</xbrli:unitDenominator></xbrli:divide></xbrli:unit>"
        '<xbrli:unit id="mixed" xmlns:cur="http://www.xbrl.org/2003/iso4217"><xbrli:divide>'
        "<xbrli:unitNumerator><xbrli:measure>ex:Widget</xbrli:measure>"
        "<xbrli:measure>cur:EUR</xbrli:measure></xbrli:unitNumerator><xbrli:unitDenominator>"
        "<xbrli:measure>xbrli:pure</xbrli:measure><xbrli:measure>xbrli:pure</xbrli:measure>"
        "</xbrli:unitDenominator></xbrli:divide></xbrli:unit>"
    )  # fmt: skip
    facts = "".join(_fact("ex:A", "FY", unit) for unit in ("usd", "perShare", "mixed"))

    (instance,) = _instances([_inline(facts, _context("FY") + units)])

    found = {
        u.unit_ref: (u.unit_kind, u.numerator_measures, u.denominator_measures, u.normalized_expression)
        for u in unit_records(instance, PROCESSED_AT)
    }  # fmt: skip
    assert found == {
        "usd": ("simple", [f"{ISO}USD"], [], f"{ISO}USD"),
        "perShare": ("divide", [f"{ISO}USD"], [f"{XBRLI}shares"], f"({ISO}USD)/({XBRLI}shares)"),
        "mixed": (
            "divide",
            [f"{EX}Widget", f"{ISO}EUR"],
            [f"{XBRLI}pure", f"{XBRLI}pure"],
            f"({EX}Widget*{ISO}EUR)/({XBRLI}pure*{XBRLI}pure)",
        ),
    }  # fmt: skip


def test_nonnumeric_only_instance_has_no_units() -> None:
    (instance,) = _instances(
        [_inline(_fact("ex:A", "FY") + _fact("ex:B", "FY"), _context("FY"))]
    )

    record = instance_record(instance, PROCESSED_AT)

    assert (record.fact_count, record.unit_count) == (2, 0)
    assert unit_records(instance, PROCESSED_AT) == []


@pytest.mark.parametrize(
    "second",
    [
        _context("FY", INSTANT),
        _context("FY") + '<xbrli:unit id="usd"><xbrli:measure>iso4217:EUR</xbrli:measure></xbrli:unit>',
    ],
    ids=["context", "unit"],
)  # fmt: skip
def test_same_id_defined_differently_is_a_conflict(second: str) -> None:
    first = _inline(_fact("ex:A", "FY", "usd"), _context("FY") + USD)

    with pytest.raises(XbrlConflict, match="defined twice"):
        _instances([first, _inline("", second)])


@pytest.mark.parametrize(
    ("facts", "message"),
    [
        (_fact("ex:A", "Missing"), "undefined contexts"),
        (_fact("ex:A", "FY", "missing"), "undefined units"),
        (
            '<ix:nonFraction name="ex:A" contextRef="FY">5</ix:nonFraction>',
            "no unitRef",
        ),
    ],
    ids=["context", "unit", "numeric-without-unit"],
)
def test_unresolved_references_fail(facts: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _instances([_inline(facts, _context("FY") + USD)])


def test_records_and_hashes_are_deterministic() -> None:
    content = _inline(
        _fact("ex:A", "A") + _fact("ex:B", "B"), _context("A") + _context("B")
    )

    first, again = _instances([content]), _instances([content])
    contexts = context_records(first[0], PROCESSED_AT)

    assert instance_record(first[0], PROCESSED_AT) == instance_record(
        again[0], PROCESSED_AT
    )
    assert contexts == context_records(again[0], PROCESSED_AT)
    # Two refs with one definition share the semantic hash, not the key.
    assert contexts[0].context_hash == contexts[1].context_hash
    assert contexts[0].context_key != contexts[1].context_key


def test_external_entities_are_not_resolved(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET")
    doctype = f'<!DOCTYPE html [<!ENTITY leak SYSTEM "file://{secret}">]>'
    content = _inline(
        _fact("ex:A", "FY"), _context("FY").replace("0000320193", "&leak;")
    )
    content = content.replace(b"<html ", doctype.encode() + b"<html ", 1)

    parsed = parse_xbrl_source(content)

    assert b"TOP-SECRET" not in etree.tostring(parsed.root)


def test_inline_tuples_are_rejected_explicitly() -> None:
    content = _inline(
        '<ix:tuple name="ex:T">' + _fact("ex:A", "FY") + "</ix:tuple>", _context("FY")
    )

    with pytest.raises(ValueError, match="unsupported_inline_tuple"):
        _instances([content])


def test_context_contract_enforces_period_shape() -> None:
    (instance,) = _instances([_inline(_fact("ex:A", "FY"), _context("FY"))])
    (context,) = context_records(instance, PROCESSED_AT)

    with pytest.raises(ValidationError, match="do not fit a instant period"):
        XbrlContextRecord.model_validate(
            {**context.model_dump(), "period_type": "instant"}
        )

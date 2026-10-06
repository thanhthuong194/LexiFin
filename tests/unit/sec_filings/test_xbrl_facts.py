from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from lxml import etree
from pydantic import ValidationError

from src.data_pipeline.contracts.sec_filings.xbrl_concepts import (
    ConceptDefinition,
    concept_id_for,
)
from src.data_pipeline.contracts.sec_filings.xbrl_core import (
    context_key_for,
    unit_key_for,
)
from src.data_pipeline.contracts.sec_filings.xbrl_facts import (
    XbrlFactRecord,
    fact_sha256,
)
from src.data_pipeline.transforms.sec_filings.html_dom import parse_html
from src.data_pipeline.transforms.sec_filings.xbrl_facts import (
    XbrlCoreMismatch,
    align_document,
    concept_definition,
    concept_record,
    fact_records,
)
from src.data_pipeline.transforms.sec_filings.xbrl_model_reader import (
    LoadedConcept,
    XbrlProcessorError,
)
from src.data_pipeline.transforms.sec_filings.xbrl_processor import (
    LoadedInstance,
    ProcessorSettings,
    SourceFile,
    load_instance,
)

pytestmark = pytest.mark.unit

PROCESSED_AT = datetime(2026, 10, 6, 4, 0, tzinfo=UTC)
INSTANCE_ID = "sec:0000320193-25-000079:xbrl:inline:default"
EX = "http://example.com/unit"
XBRLI = "{http://www.xbrl.org/2003/instance}"
XBRLDT = "{http://xbrl.org/2005/xbrldt}"
IXT = "{http://www.xbrl.org/inlineXBRL/transformation/2020-02-12}"
IXT_SEC = "{http://www.sec.gov/inlineXBRL/transformation/2015-08-31}"
# (name, type, period type, substitution group, other attributes)
CONCEPTS = (
    ("Revenue", "xbrli:monetaryItemType", "duration", "xbrli:item",
     'xbrli:balance="credit"'),
    ("NetLoss", "xbrli:monetaryItemType", "duration", "xbrli:item",
     'xbrli:balance="debit"'),
    ("Ratio", "xbrli:pureItemType", "instant", "xbrli:item", ""),
    ("Share", "xbrli:fractionItemType", "duration", "xbrli:item", ""),
    ("Description", "xbrli:stringItemType", "duration", "xbrli:item", ""),
    ("Flag", "xbrli:booleanItemType", "duration", "xbrli:item", ""),
    ("Street", "xbrli:stringItemType", "duration", "xbrli:item", ""),
    ("StatementAbstract", "xbrli:stringItemType", "duration", "xbrli:item",
     'abstract="true"'),
    ("RegionAxis", "xbrli:stringItemType", "duration", "xbrldt:dimensionItem",
     'abstract="true" xbrldt:typedDomainRef="#ex_Region"'),
)  # fmt: skip
# A self-contained extension taxonomy: it imports only the XBRL 2.1 and
# dimension schemas, which the processor ships with.
SCHEMA = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" '
    'xmlns:xbrli="http://www.xbrl.org/2003/instance" '
    'xmlns:xbrldt="http://xbrl.org/2005/xbrldt" xmlns:ex="http://example.com/unit" '
    'targetNamespace="http://example.com/unit" elementFormDefault="qualified">'
    '<xs:import namespace="http://www.xbrl.org/2003/instance" '
    'schemaLocation="http://www.xbrl.org/2003/xbrl-instance-2003-12-31.xsd"/>'
    '<xs:import namespace="http://xbrl.org/2005/xbrldt" '
    'schemaLocation="http://www.xbrl.org/2005/xbrldt-2005.xsd"/>'
    + "".join(
        f'<xs:element id="ex_{name}" name="{name}" type="{item_type}" '
        f'substitutionGroup="{group}" xbrli:periodType="{period}" nillable="true" '
        f"{attributes}/>"
        for name, item_type, period, group, attributes in CONCEPTS
    )
    + '<xs:element id="ex_Region" name="Region" type="xs:string"/>'
    '<xs:element id="ex_Address" name="Address" substitutionGroup="xbrli:tuple" '
    'nillable="true"><xs:complexType><xs:complexContent>'
    '<xs:restriction base="xs:anyType"><xs:sequence>'
    '<xs:element ref="ex:Street"/></xs:sequence></xs:restriction>'
    "</xs:complexContent></xs:complexType></xs:element></xs:schema>\n"
)
IDENTIFIER = (
    '<xbrli:entity><xbrli:identifier scheme="http://www.sec.gov/CIK">'
    "0000320193</xbrli:identifier></xbrli:entity>"
)
RESOURCES = (
    f'<xbrli:context id="FY">{IDENTIFIER}<xbrli:period>'
    "<xbrli:startDate>2024-01-01</xbrli:startDate>"
    "<xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period></xbrli:context>"
    f'<xbrli:context id="AsOf">{IDENTIFIER}<xbrli:period>'
    "<xbrli:instant>2024-12-31</xbrli:instant></xbrli:period></xbrli:context>"
    '<xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>'
    '<xbrli:unit id="pure"><xbrli:measure>xbrli:pure</xbrli:measure></xbrli:unit>'
)
REFS = ({"FY", "AsOf"}, {"usd", "pure"})


def _inline(body: str, *, hidden: str = "") -> str:
    """An Inline XBRL document in English whose header holds `hidden` facts."""

    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:ix="http://www.xbrl.org/2013/inlineXBRL" '
        'xmlns:ixt="http://www.xbrl.org/inlineXBRL/transformation/2020-02-12" '
        'xmlns:ixt-sec="http://www.sec.gov/inlineXBRL/transformation/2015-08-31" '
        'xmlns:xbrli="http://www.xbrl.org/2003/instance" '
        'xmlns:link="http://www.xbrl.org/2003/linkbase" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xmlns:iso4217="http://www.xbrl.org/2003/iso4217" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xmlns:ex="http://example.com/unit" xmlns:alt="http://example.com/unit" '
        'xml:lang="en-US"><head><title>report</title></head><body>'
        f'<div style="display:none"><ix:header><ix:hidden>{hidden}</ix:hidden>'
        '<ix:references><link:schemaRef xlink:type="simple" xlink:href="schema.xsd"/>'
        f"</ix:references><ix:resources>{RESOURCES}</ix:resources></ix:header></div>"
        f"{body}</body></html>"
    )


# One fact of each kind, in DOM order; FACT_NAMES lists them as loaded.
RICH = _inline(
    '<div>Revenue was $<ix:nonFraction name="ex:Revenue" contextRef="FY" '
    'unitRef="usd" decimals="-5" scale="6" format="ixt:num-dot-decimal">1,234.5'
    "</ix:nonFraction> million.</div>"
    '<table><tr><td>Net loss</td><td>(<ix:nonFraction name="alt:NetLoss" '
    'contextRef="FY" unitRef="usd" decimals="0" sign="-" '
    'format="ixt:num-dot-decimal">200</ix:nonFraction>)</td></tr></table>'
    '<p><ix:nonFraction name="ex:Ratio" contextRef="AsOf" unitRef="pure" '
    'decimals="INF">0.1234567890123456789012345</ix:nonFraction></p>'
    '<p><ix:nonFraction name="ex:Revenue" contextRef="FY" unitRef="usd" '
    'xsi:nil="true"/></p>'
    '<p xml:lang="fr"><ix:nonNumeric name="ex:Description" contextRef="FY" '
    'id="greeting">Bonjour</ix:nonNumeric> <ix:nonNumeric name="ex:Description" '
    'contextRef="FY">Bonjour</ix:nonNumeric></p>'
    '<p><ix:nonNumeric name="ex:Description" contextRef="FY" continuedAt="c1" '
    'escape="true">First</ix:nonNumeric></p>'
    '<p><ix:continuation id="c1" continuedAt="c2"> second</ix:continuation></p>'
    '<p><ix:continuation id="c2"> third</ix:continuation></p>'
    '<p><ix:fraction name="ex:Share" contextRef="FY" unitRef="pure">'
    "<ix:numerator>1</ix:numerator><ix:denominator>3</ix:denominator>"
    "</ix:fraction></p>"
    '<p><ix:nonFraction name="ex:Ratio" contextRef="AsOf" unitRef="pure" '
    'decimals="0" format="ixt-sec:numwordsen">no</ix:nonFraction></p>'
    '<p><ix:nonFraction name="alt:Revenue" contextRef="FY" unitRef="usd" '
    'decimals="0">5</ix:nonFraction></p>',
    hidden='<ix:nonNumeric name="ex:Flag" contextRef="FY">true</ix:nonNumeric>',
)
FACT_NAMES = [
    "Flag", "Revenue", "NetLoss", "Ratio", "Revenue", "Description",
    "Description", "Description", "Share", "Ratio", "Revenue",
]  # fmt: skip


def _load(directory: Path, html: str) -> LoadedInstance:
    files = [
        SourceFile("doc-0001", "report.htm", html.encode(), is_entry=True),
        SourceFile("doc-0003", "schema.xsd", SCHEMA.encode(), is_entry=False),
    ]
    settings = ProcessorSettings(
        cache_dir=directory / "cache", allow_network=False, http_user_agent=None
    )
    return load_instance(files, is_inline=True, target_name=None, settings=settings)


def _records(
    loaded: LoadedInstance,
    refs: tuple[set[str], set[str]] = REFS,
    location: tuple[str | None, str | None] = (None, None),
    processed_at: datetime = PROCESSED_AT,
) -> list[XbrlFactRecord]:
    records, _ = fact_records(
        INSTANCE_ID, loaded.facts, refs, [location] * len(loaded.facts), processed_at
    )
    return records


@pytest.fixture(scope="module")
def rich(tmp_path_factory: pytest.TempPathFactory) -> LoadedInstance:
    return _load(tmp_path_factory.mktemp("rich"), RICH)


@pytest.fixture(scope="module")
def records(rich: LoadedInstance) -> list[XbrlFactRecord]:
    return _records(rich)


def _concept(loaded: LoadedInstance, namespace: str, name: str) -> LoadedConcept:
    return next(
        concept
        for concept in loaded.concepts
        if (concept.namespace_uri, concept.local_name) == (namespace, name)
    )


def test_prefixes_of_one_namespace_share_a_concept_id(
    records: list[XbrlFactRecord],
) -> None:
    # Facts 2 and 11 name ex:Revenue and alt:Revenue, both bound to EX.
    assert records[1].concept_id == records[10].concept_id
    assert records[1].concept_id == concept_id_for(EX, "Revenue")


class _RecordList(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_processor_log_stays_out_of_the_application_log(tmp_path: Path) -> None:
    # The application logs through the root logger (pytest's caplog would
    # also capture loggers that do not propagate).
    application_log = _RecordList()
    logging.getLogger().addHandler(application_log)
    try:
        _load(tmp_path, RICH)
    finally:
        logging.getLogger().removeHandler(application_log)

    assert not [r for r in application_log.records if r.name == "arelle"]


def test_concept_metadata_is_read_from_the_dts(rich: LoadedInstance) -> None:
    assert _concept(rich, EX, "Revenue") == LoadedConcept(
        namespace_uri=EX,
        local_name="Revenue",
        data_type=f"{XBRLI}monetaryItemType",
        substitution_group=f"{XBRLI}item",
        period_type="duration",
        balance="credit",
        is_abstract=False,
        is_nillable=True,
        source_schema_uri="schema.xsd",
        is_extension=True,
    )
    axis = _concept(rich, EX, "RegionAxis")
    assert (axis.substitution_group, axis.is_abstract) == (
        f"{XBRLDT}dimensionItem",
        True,
    )
    assert _concept(rich, EX, "StatementAbstract").is_abstract
    assert _concept(rich, EX, "Address").substitution_group == f"{XBRLI}tuple"
    # The typed dimension's domain is a concept although it is no item.
    region = _concept(rich, EX, "Region")
    assert (region.data_type, region.substitution_group, region.period_type) == (
        "{http://www.w3.org/2001/XMLSchema}string",
        None,
        None,
    )
    names = {(c.namespace_uri, c.local_name) for c in rich.concepts}
    # Substitution group heads and XML Schema types are not concepts.
    assert ("http://www.xbrl.org/2003/instance", "item") not in names
    assert ("http://www.xbrl.org/2003/instance", "tuple") not in names
    assert all(
        c.namespace_uri != "http://www.w3.org/2001/XMLSchema" for c in rich.concepts
    )


def test_extension_concepts_are_classified_by_schema_lineage(
    rich: LoadedInstance,
) -> None:
    standard = _concept(rich, "http://xbrl.org/2005/xbrldt", "dimensionItem")

    assert (standard.is_extension, standard.source_schema_uri) == (
        False,
        "http://www.xbrl.org/2005/xbrldt-2005.xsd",
    )
    assert all(
        (c.is_extension, c.source_schema_uri) == (True, "schema.xsd")
        for c in rich.concepts
        if c.namespace_uri == EX
    )


def test_concept_hash_is_deterministic_and_ignores_lineage(
    rich: LoadedInstance,
) -> None:
    revenue = _concept(rich, EX, "Revenue")
    record = concept_record(revenue, PROCESSED_AT)
    elsewhere = concept_record(
        replace(revenue, source_schema_uri="https://example.com/r.xsd"), PROCESSED_AT
    )

    assert record.concept_id == concept_id_for(EX, "Revenue")
    assert record.concept_definition_sha256 == elsewhere.concept_definition_sha256
    assert (
        record.concept_definition_sha256
        == ConceptDefinition(
            namespace_uri=EX,
            local_name="Revenue",
            data_type=f"{XBRLI}monetaryItemType",
            substitution_group=f"{XBRLI}item",
            period_type="duration",
            balance="credit",
            is_abstract=False,
            is_nillable=True,
        ).sha256()
    )
    assert (
        concept_definition(replace(revenue, balance="debit")).sha256()
        != record.concept_definition_sha256
    )


@pytest.mark.parametrize(
    ("order", "raw", "normalized", "attributes"),
    [
        # Plain number, as written.
        (11, "5", "5", {"decimals": "0", "scale": None, "sign": None, "format": None}),
        # Formatted, then scaled by 10^6.
        (2, "1,234.5", "1234500000",
         {"decimals": "-5", "scale": 6, "sign": None,
          "format": f"{IXT}num-dot-decimal"}),
        # Shown without its sign, which the sign attribute restores.
        (3, "200", "-200",
         {"decimals": "0", "scale": None, "sign": "-",
          "format": f"{IXT}num-dot-decimal"}),
    ],
)  # fmt: skip
def test_numeric_facts_keep_raw_text_and_processor_values(
    records: list[XbrlFactRecord],
    order: int,
    raw: str,
    normalized: str,
    attributes: dict[str, object],
) -> None:
    fact = records[order - 1]

    assert fact.fact_kind == "numeric"
    assert (fact.raw_value, fact.normalized_value, fact.numeric_value) == (
        raw,
        normalized,
        normalized,
    )
    assert {name: getattr(fact, name) for name in attributes} == attributes
    assert (fact.escape, fact.xml_language) == (None, None)


def test_nonnumeric_fact_has_no_unit_or_number(records: list[XbrlFactRecord]) -> None:
    fact = records[5]

    assert (fact.fact_kind, fact.unit_key, fact.numeric_value) == (
        "non_numeric",
        None,
        None,
    )
    assert (fact.raw_value, fact.normalized_value) == ("Bonjour", "Bonjour")
    assert (fact.source_fact_id, fact.escape, fact.format) == ("greeting", False, None)


def test_continuation_chain_is_joined_in_order(records: list[XbrlFactRecord]) -> None:
    fact = records[7]

    assert fact.raw_value == "First second third"
    assert fact.continuation_refs == ["c1", "c2"]
    assert fact.escape is True
    assert all(not r.continuation_refs for r in records if r is not fact)


MISSING_CONTINUATION = (
    '<p><ix:nonNumeric name="ex:Description" contextRef="FY" '
    'continuedAt="missing">a</ix:nonNumeric></p>'
)
CONTINUATION_CYCLE = (
    '<p><ix:nonNumeric name="ex:Description" contextRef="FY" '
    'continuedAt="c1">a</ix:nonNumeric></p><p><ix:continuation id="c1" '
    'continuedAt="c2">b</ix:continuation><ix:continuation id="c2" '
    'continuedAt="c1">c</ix:continuation></p>'
)
TUPLE = (
    '<p><ix:tuple name="ex:Address" tupleID="t1"/><ix:nonNumeric '
    'name="ex:Street" contextRef="FY" tupleRef="t1">Main St</ix:nonNumeric></p>'
)
UNDEFINED_CONCEPT = (
    '<p><ix:nonNumeric name="ex:Unknown" contextRef="FY">x</ix:nonNumeric></p>'
)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (MISSING_CONTINUATION, "continuation 'missing' does not exist"),
        (CONTINUATION_CYCLE, "continuation 'c1' is reused or cyclic"),
        (TUPLE, "unsupported_xbrl_tuple"),
        (UNDEFINED_CONCEPT, r"fact concepts are not in the DTS: \['ex:Unknown'\]"),
    ],
    ids=["missing-continuation", "continuation-cycle", "tuple", "undefined-concept"],
)
def test_unsupported_or_broken_facts_fail_the_instance(
    tmp_path: Path, body: str, message: str
) -> None:
    with pytest.raises(XbrlProcessorError, match=message):
        _load(tmp_path, _inline(body))


def test_nil_fact_has_no_value_but_keeps_its_unit(
    records: list[XbrlFactRecord],
) -> None:
    fact = records[4]

    assert fact.is_nil
    assert (fact.raw_value, fact.normalized_value, fact.numeric_value) == (
        None,
        None,
        None,
    )
    assert fact.unit_key == unit_key_for(INSTANCE_ID, "usd")


def test_only_facts_in_ix_hidden_are_hidden(records: list[XbrlFactRecord]) -> None:
    assert [fact.is_hidden for fact in records] == [True] + [False] * 10


def test_decimals_and_precision_stay_lexical(records: list[XbrlFactRecord]) -> None:
    assert [fact.decimals for fact in records[1:4]] == ["-5", "0", "INF"]
    assert all(fact.precision is None for fact in records)


def test_nonnumeric_facts_keep_the_language_in_scope(
    records: list[XbrlFactRecord],
) -> None:
    languages = {
        fact.fact_order: fact.xml_language
        for fact in records
        if fact.fact_kind == "non_numeric"
    }

    # Inherited from the html element, or from the enclosing paragraph.
    assert languages == {1: "en-US", 6: "fr", 7: "fr", 8: "en-US"}
    assert all(f.xml_language is None for f in records if f.fact_kind != "non_numeric")


def test_fraction_fact_keeps_numerator_and_denominator(
    records: list[XbrlFactRecord],
) -> None:
    fact = records[8]

    assert fact.fact_kind == "fraction"
    assert (fact.raw_value, fact.normalized_value, fact.numeric_value) == (
        "1/3",
        "1/3",
        None,
    )
    assert fact.unit_key == unit_key_for(INSTANCE_ID, "pure")


def test_equal_facts_stay_separate_occurrences(records: list[XbrlFactRecord]) -> None:
    first, second = records[5], records[6]

    assert (first.concept_id, first.context_key, first.normalized_value) == (
        second.concept_id,
        second.context_key,
        second.normalized_value,
    )
    assert first.fact_id != second.fact_id
    assert first.source_xpath != second.source_xpath


def test_fact_order_and_ids_follow_dom_order_across_loads(
    tmp_path: Path, rich: LoadedInstance, records: list[XbrlFactRecord]
) -> None:
    again = _records(_load(tmp_path, RICH))

    assert [fact.local_name for fact in rich.facts] == FACT_NAMES
    assert [fact.fact_id for fact in records] == [
        f"{INSTANCE_ID}:fact:{order:08d}" for order in range(1, 12)
    ]
    assert [r.model_dump() for r in again] == [r.model_dump() for r in records]


def test_context_and_unit_refs_become_instance_keys(
    records: list[XbrlFactRecord],
) -> None:
    ratio = records[3]

    assert ratio.context_key == context_key_for(INSTANCE_ID, "AsOf")
    assert ratio.unit_key == unit_key_for(INSTANCE_ID, "pure")
    assert records[0].context_key == context_key_for(INSTANCE_ID, "FY")


@pytest.mark.parametrize(
    ("refs", "message"),
    [
        (({"FY"}, {"usd", "pure"}), "fact 4: context 'AsOf' is not in xbrl_contexts"),
        (({"FY", "AsOf"}, {"usd"}), "fact 4: unit 'pure' is not in xbrl_units"),
    ],
    ids=["missing-context", "missing-unit"],
)
def test_refs_missing_from_xbrl_core_are_a_mismatch(
    rich: LoadedInstance, refs: tuple[set[str], set[str]], message: str
) -> None:
    with pytest.raises(XbrlCoreMismatch, match=message):
        _records(rich, refs)


def test_numeric_value_is_exact_and_untransformable_values_stay_raw(
    rich: LoadedInstance, records: list[XbrlFactRecord]
) -> None:
    ratio, sec_words = records[3], records[9]
    _, unusable = fact_records(
        INSTANCE_ID, rich.facts, REFS, [(None, None)] * 11, PROCESSED_AT
    )

    assert ratio.numeric_value == "0.1234567890123456789012345"
    assert Decimal(ratio.numeric_value) == Decimal(ratio.raw_value or "")
    # The processor has no SEC transformation registry: the raw text is kept.
    assert sec_words.format == f"{IXT_SEC}numwordsen"
    assert (sec_words.raw_value, sec_words.normalized_value) == ("no", None)
    assert sec_words.numeric_value is None
    assert unusable == 1


def test_fact_hash_covers_source_fields_but_not_enrichment(
    rich: LoadedInstance, records: list[XbrlFactRecord]
) -> None:
    later = _records(
        rich,
        location=("block-1", None),
        processed_at=datetime(2027, 1, 1, tzinfo=UTC),
    )

    assert [r.fact_sha256 for r in later] == [r.fact_sha256 for r in records]
    assert all(fact_sha256(r.model_dump()) == r.fact_sha256 for r in records)
    assert len({r.fact_sha256 for r in records}) == len(records)
    with pytest.raises(ValidationError, match="fact_sha256 does not match"):
        XbrlFactRecord.model_validate({**records[1].model_dump(), "raw_value": "1"})


# A primary document as the narrative stage parses it: a paragraph block, a
# table, a fact wrapping two blocks, a paragraph stored as two blocks, and a
# hidden header fact.
ALIGNMENT_HTML = (
    b'<html><body><div style="display:none"><ix:header><ix:hidden>'
    b'<ix:nonNumeric name="ex:Flag">true</ix:nonNumeric></ix:hidden></ix:header></div>'
    b'<div>We make <b><ix:nonNumeric name="ex:Product">widgets</ix:nonNumeric></b>.</div>'
    b"<table><tr><td>Revenue</td><td>$<ix:nonFraction "
    b'name="ex:Revenue">1,000</ix:nonFraction></td></tr></table>'
    b'<ix:nonNumeric name="ex:Policy"><div>One.</div><div>Two.</div></ix:nonNumeric>'
    b'<p>Split <ix:nonNumeric name="ex:Note">note</ix:nonNumeric></p></body></html>'
)


def _alignment_maps(
    root: etree._Element,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Stored cells and blocks keyed by the XPath of their parsed element."""

    tree = root.getroottree()
    divs, cells, paragraph = (
        root.findall(".//div"),
        root.findall(".//td"),
        root.find(".//p"),
    )
    return (
        {tree.getpath(cell): f"cell-{i}" for i, cell in enumerate(cells, start=1)},
        {
            tree.getpath(divs[1]): ["block-product"],
            tree.getpath(divs[2]): ["block-one"],
            tree.getpath(divs[3]): ["block-two"],
            tree.getpath(root.find(".//table")): ["block-table"],
            tree.getpath(paragraph): ["block-split-1", "block-split-2"],
        },
    )


def test_alignment_maps_cells_and_blocks_only_when_unambiguous() -> None:
    root = parse_html(ALIGNMENT_HTML)

    locations = align_document(root, [1, 2, 3, 4, 5], 5, _alignment_maps(root))

    assert locations == [
        (None, None),  # hidden
        ("block-product", None),  # nearest block container, through inline tags
        (None, "cell-2"),  # the cell wins over the table's block
        (None, None),  # spans two blocks
        (None, None),  # its paragraph is stored as two blocks
    ]


def test_alignment_leaves_facts_unaligned_without_a_stored_match() -> None:
    root = parse_html(ALIGNMENT_HTML)
    cells, blocks = _alignment_maps(root)

    # A cell that was not stored does not fall back to the table's block.
    assert align_document(root, [3], 5, ({}, blocks)) == [(None, None)]
    assert align_document(root, [2], 5, (cells, {})) == [(None, None)]
    # The facts cannot be found again when the parse sees another number.
    assert align_document(root, [2, 3], 4, (cells, blocks)) == [(None, None)] * 2


def test_a_fact_is_never_in_both_a_block_and_a_cell(
    records: list[XbrlFactRecord],
) -> None:
    root = parse_html(ALIGNMENT_HTML)
    locations = align_document(root, [1, 2, 3, 4, 5], 5, _alignment_maps(root))

    assert all(None in location for location in locations)
    with pytest.raises(ValidationError, match="a block or a cell, not both"):
        XbrlFactRecord.model_validate(
            {**records[2].model_dump(), "block_id": "block-1", "cell_id": "cell-1"}
        )

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.data_pipeline.contracts.sec_filings.xbrl_concepts import concept_id_for
from src.data_pipeline.contracts.sec_filings.xbrl_dimensions import (
    XbrlDimensionRecord,
)
from src.data_pipeline.contracts.sec_filings.xbrl_fact_footnotes import (
    XbrlFactFootnoteRecord,
)
from src.data_pipeline.quality.sec_filings.xbrl_enrichment import (
    dimension_group_problem,
)
from src.data_pipeline.transforms.sec_filings.xbrl_dimensions import (
    dimension_records,
    expected_dimension_count,
)
from src.data_pipeline.transforms.sec_filings.xbrl_footnotes import footnote_records
from src.data_pipeline.transforms.sec_filings.xbrl_model_reader import (
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
FOOTNOTE_ROLE = "http://www.xbrl.org/2003/role/footnote"
STANDARD_LINK = "http://www.xbrl.org/2003/role/link"
# (name, substitution group, other attributes) of the dimension concepts.
DIMENSION_CONCEPTS = (
    ("ProductAxis", "xbrldt:dimensionItem", ""),
    ("WidgetsMember", "xbrli:item", ""),
    ("ScenarioAxis", "xbrldt:dimensionItem", ""),
    ("PlanMember", "xbrli:item", ""),
    ("DefaultAxis", "xbrldt:dimensionItem", ""),
    ("TotalMember", "xbrli:item", ""),
    ("RegionAxis", "xbrldt:dimensionItem", 'xbrldt:typedDomainRef="#ex_Region"'),
    ("AddressAxis", "xbrldt:dimensionItem", 'xbrldt:typedDomainRef="#ex_Address"'),
)
SCHEMA = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" '
    'xmlns:xbrli="http://www.xbrl.org/2003/instance" '
    'xmlns:xbrldt="http://xbrl.org/2005/xbrldt" '
    'xmlns:link="http://www.xbrl.org/2003/linkbase" '
    'xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:ex="http://example.com/unit" '
    'targetNamespace="http://example.com/unit" elementFormDefault="qualified">'
    "<xs:annotation><xs:appinfo>"
    '<link:linkbaseRef xlink:type="simple" xlink:href="definition.xml" '
    'xlink:role="http://www.xbrl.org/2003/role/definitionLinkbaseRef" '
    'xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>'
    "</xs:appinfo></xs:annotation>"
    '<xs:import namespace="http://www.xbrl.org/2003/instance" '
    'schemaLocation="http://www.xbrl.org/2003/xbrl-instance-2003-12-31.xsd"/>'
    '<xs:import namespace="http://xbrl.org/2005/xbrldt" '
    'schemaLocation="http://www.xbrl.org/2005/xbrldt-2005.xsd"/>'
    '<xs:element id="ex_Revenue" name="Revenue" type="xbrli:monetaryItemType" '
    'substitutionGroup="xbrli:item" xbrli:periodType="duration" nillable="true"/>'
    '<xs:element id="ex_Note" name="Note" type="xbrli:stringItemType" '
    'substitutionGroup="xbrli:item" xbrli:periodType="duration" nillable="true"/>'
    + "".join(
        f'<xs:element id="ex_{name}" name="{name}" type="xbrli:stringItemType" '
        f'substitutionGroup="{group}" xbrli:periodType="duration" abstract="true" '
        f'nillable="true" {attributes}/>'
        for name, group, attributes in DIMENSION_CONCEPTS
    )
    + '<xs:element id="ex_Region" name="Region" type="xs:string"/>'
    '<xs:element id="ex_Address" name="Address"><xs:complexType><xs:sequence>'
    '<xs:element name="Street" type="xs:string"/>'
    '<xs:element name="City" type="xs:string"/></xs:sequence>'
    '<xs:attribute name="kind" type="xs:string"/></xs:complexType></xs:element>'
    "</xs:schema>\n"
)
# DefaultAxis defaults to TotalMember: a context that does not report it
# still "has" it in XBRL Dimensions terms, but it is not reported.
DEFINITION_LINKBASE = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<link:linkbase xmlns:link="http://www.xbrl.org/2003/linkbase" '
    'xmlns:xlink="http://www.w3.org/1999/xlink">'
    '<link:arcroleRef arcroleURI="http://xbrl.org/int/dim/arcrole/dimension-default" '
    'xlink:type="simple" '
    'xlink:href="http://www.xbrl.org/2005/xbrldt-2005.xsd#dimension-default"/>'
    '<link:definitionLink xlink:type="extended" '
    'xlink:role="http://www.xbrl.org/2003/role/link">'
    '<link:loc xlink:type="locator" xlink:href="schema.xsd#ex_DefaultAxis" '
    'xlink:label="axis"/>'
    '<link:loc xlink:type="locator" xlink:href="schema.xsd#ex_TotalMember" '
    'xlink:label="total"/>'
    '<link:definitionArc xlink:type="arc" xlink:from="axis" xlink:to="total" '
    'xlink:arcrole="http://xbrl.org/int/dim/arcrole/dimension-default"/>'
    "</link:definitionLink></link:linkbase>\n"
)
IDENTIFIER = (
    '<xbrli:identifier scheme="http://www.sec.gov/CIK">0000320193</xbrli:identifier>'
)
PERIOD = (
    "<xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate>"
    "<xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period>"
)


def _member(axis: str, member: str) -> str:
    return f'<xbrldi:explicitMember dimension="ex:{axis}">ex:{member}</xbrldi:explicitMember>'


def _typed(axis: str, content: str) -> str:
    return f'<xbrldi:typedMember dimension="ex:{axis}">{content}</xbrldi:typedMember>'


def _context(context_id: str, segment: str = "", scenario: str = "") -> str:
    entity = f"<xbrli:entity>{IDENTIFIER}"
    if segment:
        entity += f"<xbrli:segment>{segment}</xbrli:segment>"
    scenario = f"<xbrli:scenario>{scenario}</xbrli:scenario>" if scenario else ""
    return (
        f'<xbrli:context id="{context_id}">{entity}</xbrli:entity>{PERIOD}{scenario}'
        "</xbrli:context>"
    )


CONTEXTS = (
    _context("Plain")
    + _context(
        "Seg",
        _typed("RegionAxis", "<ex:Region>  North&#160; America </ex:Region>")
        + _member("ProductAxis", "WidgetsMember"),
    )
    + _context("Scen", scenario=_member("ScenarioAxis", "PlanMember"))
    + _context(
        "Nested",
        _typed(
            "AddressAxis",
            '<ex:Address kind="hq">\n  <ex:Street>1 Main St</ex:Street> '
            "<ex:City>Springfield</ex:City>\n</ex:Address>",
        ),
    )
    + _context("RegionEx", _typed("RegionAxis", "<ex:Region>West</ex:Region>"))
    + _context("RegionAlt", _typed("RegionAxis", "<alt:Region>West</alt:Region>"))
)


def _inline(body: str, *, contexts: str = CONTEXTS, relationships: str = "") -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:ix="http://www.xbrl.org/2013/inlineXBRL" '
        'xmlns:xbrli="http://www.xbrl.org/2003/instance" '
        'xmlns:xbrldi="http://xbrl.org/2006/xbrldi" '
        'xmlns:link="http://www.xbrl.org/2003/linkbase" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xmlns:iso4217="http://www.xbrl.org/2003/iso4217" '
        'xmlns:ex="http://example.com/unit" xmlns:alt="http://example.com/unit">'
        '<head><title>report</title></head><body><div style="display:none">'
        "<ix:header><ix:references>"
        '<link:schemaRef xlink:type="simple" xlink:href="schema.xsd"/>'
        f"</ix:references><ix:resources>{contexts}"
        '<xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>'
        f"{relationships}</ix:resources></ix:header></div>{body}</body></html>"
    )


# Facts 1 and 2 are equal; fact 2 and fact 3 share the marked-up footnote,
# and fact 3 also has a plain, a custom-role and an empty footnote.
FACTS = (
    '<p>Revenue <ix:nonFraction id="f1" name="ex:Revenue" contextRef="Plain" '
    'unitRef="usd" decimals="0">100</ix:nonFraction> and again '
    '<ix:nonFraction id="f2" name="ex:Revenue" contextRef="Plain" unitRef="usd" '
    'decimals="0">100</ix:nonFraction><sup>1</sup>.</p>'
    '<p><ix:nonNumeric id="f3" name="ex:Note" contextRef="Plain">A note'
    "</ix:nonNumeric></p><p><sup>1</sup> A narrative footnote, not XBRL.</p>"
)
# An Inline target holds only the contexts its facts use.
CONTEXT_FACTS = "".join(
    f'<p><ix:nonNumeric name="ex:Note" contextRef="{ref}">x</ix:nonNumeric></p>'
    for ref in ("Seg", "Scen", "Nested", "RegionEx", "RegionAlt")
)
FOOTNOTES = (
    '<div><ix:footnote id="fn-markup" xml:lang=" EN-us ">Includes <b>one-time</b>'
    "&#160;items.</ix:footnote>"
    '<ix:footnote id="fn-plain">Plain note.</ix:footnote>'
    '<ix:footnote id="fn-custom" footnoteRole="http://example.com/role/custom">'
    "Custom.</ix:footnote>"
    '<ix:footnote id="fn-empty"/></div>'
)
RELATIONSHIPS = (
    '<ix:relationship fromRefs="f2 f3" toRefs="fn-markup"/>'
    '<ix:relationship fromRefs="f3" toRefs="fn-plain fn-empty"/>'
    '<ix:relationship fromRefs="f3" toRefs="fn-custom" '
    'linkRole="http://example.com/role/other"/>'
)
DOCUMENT = _inline(FACTS + CONTEXT_FACTS + FOOTNOTES, relationships=RELATIONSHIPS)
UNSUPPORTED_RELATIONSHIP = (
    '<ix:relationship fromRefs="f1" toRefs="f2" '
    'arcrole="http://www.xbrl.org/2009/arcrole/fact-explanatoryFact"/>'
)


def _load(directory: Path, html: str) -> LoadedInstance:
    files = [
        SourceFile("doc-0001", "report.htm", html.encode(), is_entry=True),
        SourceFile("doc-0003", "schema.xsd", SCHEMA.encode(), is_entry=False),
        SourceFile(
            "doc-0004", "definition.xml", DEFINITION_LINKBASE.encode(), is_entry=False
        ),
    ]
    settings = ProcessorSettings(
        cache_dir=directory / "cache", allow_network=False, http_user_agent=None
    )
    return load_instance(files, is_inline=True, target_name=None, settings=settings)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> LoadedInstance:
    return _load(tmp_path_factory.mktemp("enrichment"), DOCUMENT)


def _dimensions(loaded: LoadedInstance, context_ref: str) -> list[XbrlDimensionRecord]:
    group = [d for d in loaded.dimensions if d.context_ref == context_ref]
    return dimension_records(
        f"{INSTANCE_ID}:context:{context_ref}", group, PROCESSED_AT
    )


@pytest.fixture(scope="module")
def footnotes(loaded: LoadedInstance) -> list[XbrlFactFootnoteRecord]:
    return footnote_records(INSTANCE_ID, loaded.footnotes, PROCESSED_AT)


def _fact(order: int) -> str:
    return f"{INSTANCE_ID}:fact:{order:08d}"


def test_explicit_dimensions_keep_their_context_element(loaded: LoadedInstance) -> None:
    [scenario] = _dimensions(loaded, "Scen")
    segment = next(d for d in _dimensions(loaded, "Seg") if d.member_type == "explicit")

    assert (segment.context_element, scenario.context_element) == (
        "segment",
        "scenario",
    )
    assert (segment.dimension_concept_id, segment.member_concept_id) == (
        concept_id_for(EX, "ProductAxis"),
        concept_id_for(EX, "WidgetsMember"),
    )
    assert scenario.member_concept_id == concept_id_for(EX, "PlanMember")
    assert (segment.typed_member_qname, segment.typed_member_value) == (None, None)
    assert segment.typed_member_xml is None
    assert segment.source_xpath.endswith("xbrli:segment/xbrldi:explicitMember")


def test_typed_dimension_keeps_qname_normalized_value_and_xml(
    loaded: LoadedInstance,
) -> None:
    typed = next(d for d in _dimensions(loaded, "Seg") if d.member_type == "typed")

    assert typed.member_concept_id is None
    assert typed.dimension_concept_id == concept_id_for(EX, "RegionAxis")
    assert typed.typed_member_qname == f"{{{EX}}}Region"
    assert typed.typed_member_value == "North America"
    assert typed.typed_member_xml == (
        '<ns0:Region xmlns:ns0="http://example.com/unit">  North\u00a0 America </ns0:Region>'
    )


def test_typed_dimension_keeps_nested_xml_and_attributes(
    loaded: LoadedInstance,
) -> None:
    [address] = _dimensions(loaded, "Nested")

    assert address.typed_member_xml == (
        '<ns0:Address xmlns:ns0="http://example.com/unit" kind="hq">'
        "<ns0:Street>1 Main St</ns0:Street><ns0:City>Springfield</ns0:City>"
        "</ns0:Address>"
    )
    # The XPath string value of the member: its descendant text, normalized.
    assert address.typed_member_value == "1 Main St Springfield"


def test_typed_member_xml_and_hash_do_not_depend_on_prefixes(
    loaded: LoadedInstance,
) -> None:
    [ex], [alt] = _dimensions(loaded, "RegionEx"), _dimensions(loaded, "RegionAlt")

    assert ex.typed_member_xml == alt.typed_member_xml
    assert ex.dimension_hash == alt.dimension_hash
    assert ex.source_xpath != alt.source_xpath


def test_contexts_without_reported_dimensions_have_no_rows(
    loaded: LoadedInstance,
) -> None:
    # DefaultAxis defaults to TotalMember in the taxonomy; no context reports it.
    default_axis = concept_id_for(EX, "DefaultAxis")
    rows = [
        d
        for ref in ("Plain", "Seg", "Scen", "Nested")
        for d in _dimensions(loaded, ref)
    ]

    assert _dimensions(loaded, "Plain") == []
    assert all(d.dimension_concept_id != default_axis for d in rows)
    assert expected_dimension_count(None, None) == 0


def test_dimensions_are_ordered_numbered_and_hashed_deterministically(
    tmp_path: Path, loaded: LoadedInstance
) -> None:
    rows = _dimensions(loaded, "Seg")
    again = _dimensions(_load(tmp_path, DOCUMENT), "Seg")

    # Ordered by dimension QName: ProductAxis before RegionAxis.
    assert [d.dimension_concept_id for d in rows] == [
        concept_id_for(EX, "ProductAxis"),
        concept_id_for(EX, "RegionAxis"),
    ]
    assert [(d.dimension_ordinal, d.context_dimension_count) for d in rows] == [
        (1, 2),
        (2, 2),
    ]
    assert [d.dimension_id for d in rows] == [
        f"{INSTANCE_ID}:context:Seg:dimension:0001",
        f"{INSTANCE_ID}:context:Seg:dimension:0002",
    ]
    assert [r.model_dump() for r in again] == [r.model_dump() for r in rows]


def test_stored_segment_xml_tells_the_expected_dimension_count() -> None:
    segment = (
        '<ns1:segment xmlns:ns0="http://example.com/unit" '
        'xmlns:ns1="http://www.xbrl.org/2003/instance" '
        'xmlns:ns2="http://xbrl.org/2006/xbrldi">'
        '<ns2:explicitMember dimension="ns0:ProductAxis">ns0:WidgetsMember'
        '</ns2:explicitMember><ns2:typedMember dimension="ns0:RegionAxis">'
        "<ns0:Region>West</ns0:Region></ns2:typedMember><ns0:Other/></ns1:segment>"
    )

    assert expected_dimension_count(segment, None) == 2


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"typed_member_qname": "{x}y"}, "an explicit dimension has a member concept"),
        ({"member_concept_id": None}, "an explicit dimension has a member concept"),
        ({"member_type": "typed"}, "a typed dimension has a typed member only"),
    ],
)
def test_explicit_and_typed_nullability_is_enforced(
    loaded: LoadedInstance, changes: dict[str, object], message: str
) -> None:
    [explicit] = _dimensions(loaded, "Scen")

    with pytest.raises(ValidationError, match=message):
        XbrlDimensionRecord.model_validate({**explicit.model_dump(), **changes})


def test_missing_dimension_or_member_concepts_are_reported(
    loaded: LoadedInstance,
) -> None:
    rows = _dimensions(loaded, "Seg")
    concepts = {d.dimension_concept_id for d in rows} | {
        d.member_concept_id for d in rows if d.member_concept_id
    }
    member = concept_id_for(EX, "WidgetsMember")
    axis = concept_id_for(EX, "RegionAxis")

    assert dimension_group_problem(rows, 2, concepts) is None
    assert member in str(dimension_group_problem(rows, 2, concepts - {member}))
    assert axis in str(dimension_group_problem(rows, 2, concepts - {axis}))
    assert dimension_group_problem(rows, 3, concepts) == (
        "2 dimensions; the context reports 3"
    )


@pytest.mark.parametrize(
    ("contexts", "message"),
    [
        (
            _context(
                "Twice",
                _member("ProductAxis", "WidgetsMember")
                + _member("ProductAxis", "WidgetsMember"),
            ),
            "invalid dimensions",
        ),
        (
            _context(
                "Both",
                _member("ProductAxis", "WidgetsMember"),
                _member("ProductAxis", "WidgetsMember"),
            ),
            "both segment and scenario",
        ),
        (
            _context(
                "TwoChildren",
                _typed(
                    "RegionAxis", "<ex:Region>a</ex:Region><ex:Region>b</ex:Region>"
                ),
            ),
            "typed member has 2 elements",
        ),
    ],
    ids=["repeated-dimension", "segment-and-scenario", "typed-two-children"],
)
def test_invalid_reported_dimensions_fail_the_instance(
    tmp_path: Path, contexts: str, message: str
) -> None:
    ref = contexts.split('"', 2)[1]
    fact = f'<ix:nonNumeric name="ex:Note" contextRef="{ref}">x</ix:nonNumeric>'

    with pytest.raises(XbrlProcessorError, match=message):
        _load(tmp_path, _inline(f"<p>{fact}</p>", contexts=contexts))


def _resource(rows: list[XbrlFactFootnoteRecord], label: str) -> XbrlFactFootnoteRecord:
    return next(row for row in rows if row.resource_label == label)


def test_each_relationship_is_one_row_per_fact(
    footnotes: list[XbrlFactFootnoteRecord],
) -> None:
    by_fact = {
        fact: [f.resource_label for f in footnotes if f.fact_id == fact]
        for fact in (_fact(1), _fact(2), _fact(3))
    }

    # Fact 1 equals fact 2 but has no footnote: facts map by position.
    assert by_fact[_fact(1)] == []
    assert by_fact[_fact(2)] == ["fn-markup"]
    assert sorted(by_fact[_fact(3)]) == [
        "fn-custom",
        "fn-empty",
        "fn-markup",
        "fn-plain",
    ]


def test_a_footnote_noting_several_facts_keeps_one_resource_id(
    footnotes: list[XbrlFactFootnoteRecord],
) -> None:
    shared = [f for f in footnotes if f.resource_label == "fn-markup"]

    assert [f.fact_id for f in shared] == [_fact(2), _fact(3)]
    assert len({f.footnote_resource_id for f in shared}) == 1
    assert len({f.footnote_content_sha256 for f in shared}) == 1
    assert len({f.footnote_resource_id for f in footnotes}) == 4


def test_footnote_text_html_roles_and_language(
    footnotes: list[XbrlFactFootnoteRecord],
) -> None:
    markup, plain = _resource(footnotes, "fn-markup"), _resource(footnotes, "fn-plain")
    custom, empty = _resource(footnotes, "fn-custom"), _resource(footnotes, "fn-empty")

    assert (markup.footnote_text, markup.footnote_html) == (
        "Includes one-time items.",
        "Includes <b>one-time</b>\u00a0items.",
    )
    assert (plain.footnote_text, plain.footnote_html) == ("Plain note.", None)
    assert (markup.language, plain.language) == ("en-us", None)
    assert (markup.role_uri, markup.arcrole_uri, markup.linkrole_uri) == (
        FOOTNOTE_ROLE,
        "http://www.xbrl.org/2003/arcrole/fact-footnote",
        STANDARD_LINK,
    )
    assert (custom.role_uri, custom.linkrole_uri) == (
        "http://example.com/role/custom",
        "http://example.com/role/other",
    )
    # An empty footnote is kept, with empty text and no markup.
    assert (empty.footnote_text, empty.footnote_html) == ("", None)
    assert markup.source_xpath.endswith("ix:footnote[1]")
    assert markup.arc_xpath is not None
    assert markup.arc_xpath.endswith("ix:resources/ix:relationship[1]")


def test_footnote_rows_are_ordered_numbered_and_hashed_deterministically(
    tmp_path: Path, footnotes: list[XbrlFactFootnoteRecord]
) -> None:
    again = footnote_records(
        INSTANCE_ID,
        _load(tmp_path, DOCUMENT).footnotes,
        datetime(2027, 1, 1, tzinfo=UTC),
    )
    fact_three = [f for f in footnotes if f.fact_id == _fact(3)]

    assert [f.footnote_relation_id for f in fact_three] == [
        f"{_fact(3)}:footnote:{n:04d}" for n in range(1, 5)
    ]
    assert {f.fact_footnote_count for f in fact_three} == {4}
    assert [f.footnote_resource_id for f in fact_three] == sorted(
        f.footnote_resource_id for f in fact_three
    )
    assert [f.model_dump(exclude={"processed_at"}) for f in again] == [
        f.model_dump(exclude={"processed_at"}) for f in footnotes
    ]


def test_without_xbrl_footnotes_narrative_footnotes_yield_no_rows(
    tmp_path: Path,
) -> None:
    # The document keeps its <sup> markers and narrative note.
    loaded = _load(tmp_path, _inline(FACTS))

    assert loaded.footnotes == []
    assert footnote_records(INSTANCE_ID, loaded.footnotes, PROCESSED_AT) == []


def test_relationships_other_than_fact_footnote_fail_the_instance(
    tmp_path: Path,
) -> None:
    document = _inline(FACTS + FOOTNOTES, relationships=UNSUPPORTED_RELATIONSHIP)

    with pytest.raises(XbrlProcessorError, match="unsupported .*fact-explanatoryFact"):
        _load(tmp_path, document)

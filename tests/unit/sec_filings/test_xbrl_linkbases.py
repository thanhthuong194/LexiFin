from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from src.data_pipeline.contracts.sec_filings.xbrl_concepts import concept_id_for
from src.data_pipeline.contracts.sec_filings.xbrl_labels import XbrlLabelRecord
from src.data_pipeline.contracts.sec_filings.xbrl_networks import (
    XbrlNetworkRecord,
    XbrlRelationshipRecord,
    network_id_for,
    plain_decimal,
)
from src.data_pipeline.quality.sec_filings.xbrl_linkbases import (
    network_problem,
    relationship_attributes_problem,
)
from src.data_pipeline.transforms.sec_filings.xbrl_linkbases import (
    label_records,
    network_records,
)
from src.data_pipeline.transforms.sec_filings.xbrl_processor import (
    LoadedInstance,
    ProcessorSettings,
    SourceFile,
    load_instance,
)

pytestmark = pytest.mark.unit

PROCESSED_AT = datetime(2026, 10, 7, 4, 0, tzinfo=UTC)
INSTANCE_ID = "sec:0000320193-25-000079:xbrl:inline:default"
FILING = "sec:0000320193-25-000079:document:"
EX = "http://example.com/unit"
ROLE_A, ROLE_B, ROLE_EMPTY = (
    f"http://example.com/role/{name}" for name in ("A", "B", "Empty")
)
STANDARD, TERSE = (
    f"http://www.xbrl.org/2003/role/{name}" for name in ("label", "terseLabel")
)
PARENT_CHILD = "http://www.xbrl.org/2003/arcrole/parent-child"
SUMMATION = "http://www.xbrl.org/2003/arcrole/summation-item"
SUMMATION_11 = "https://xbrl.org/2023/arcrole/summation-item"
DIMENSION = "http://xbrl.org/int/dim/arcrole/"
# (name, type, period type, substitution group, abstract)
CONCEPTS = (
    ("Assets", "xbrli:monetaryItemType", "instant", "xbrli:item", "false"),
    ("AssetsCurrent", "xbrli:monetaryItemType", "instant", "xbrli:item", "false"),
    ("AssetsNoncurrent", "xbrli:monetaryItemType", "instant", "xbrli:item", "false"),
    ("Revenue", "xbrli:monetaryItemType", "duration", "xbrli:item", "false"),
    ("StatementAbstract", "xbrli:stringItemType", "duration", "xbrli:item", "true"),
    (
        "ProductTable",
        "xbrli:stringItemType",
        "duration",
        "xbrldt:hypercubeItem",
        "true",
    ),
    ("ProductAxis", "xbrli:stringItemType", "duration", "xbrldt:dimensionItem", "true"),
    ("WidgetsMember", "xbrli:stringItemType", "duration", "xbrli:item", "true"),
    ("GadgetsMember", "xbrli:stringItemType", "duration", "xbrli:item", "true"),
)
LINKBASES = (
    "labels.xml",
    "presentation.xml",
    "calculation.xml",
    "definition.xml",
    "override.xml",
)
NAMESPACES = (
    'xmlns:link="http://www.xbrl.org/2003/linkbase" '
    'xmlns:xlink="http://www.w3.org/1999/xlink" '
    'xmlns:xbrldt="http://xbrl.org/2005/xbrldt"'
)


def _role_type(role: str, definition: str | None) -> str:
    text = (
        "" if definition is None else f"<link:definition>{definition}</link:definition>"
    )
    return (
        f'<link:roleType roleURI="{role}" id="{role.rsplit("/", 1)[1]}">{text}'
        "<link:usedOn>link:presentationLink</link:usedOn>"
        "<link:usedOn>link:calculationLink</link:usedOn>"
        "<link:usedOn>link:definitionLink</link:usedOn></link:roleType>"
    )


SCHEMA = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" '
    f'xmlns:xbrli="http://www.xbrl.org/2003/instance" {NAMESPACES} '
    'targetNamespace="http://example.com/unit" elementFormDefault="qualified">'
    "<xs:annotation><xs:appinfo>"
    + "".join(
        f'<link:linkbaseRef xlink:type="simple" xlink:href="{name}" '
        'xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>'
        for name in LINKBASES
    )
    + _role_type(ROLE_A, "100 - Statement - Assets")
    + _role_type(ROLE_B, None)
    + _role_type(ROLE_EMPTY, "Only prohibited arcs")
    + "</xs:appinfo></xs:annotation>"
    '<xs:import namespace="http://www.xbrl.org/2003/instance" '
    'schemaLocation="http://www.xbrl.org/2003/xbrl-instance-2003-12-31.xsd"/>'
    '<xs:import namespace="http://xbrl.org/2005/xbrldt" '
    'schemaLocation="http://www.xbrl.org/2005/xbrldt-2005.xsd"/>'
    + "".join(
        f'<xs:element id="ex_{name}" name="{name}" type="{item_type}" '
        f'substitutionGroup="{group}" xbrli:periodType="{period}" '
        f'abstract="{abstract}" nillable="true"/>'
        for name, item_type, period, group, abstract in CONCEPTS
    )
    + "</xs:schema>\n"
)


def _loc(name: str) -> str:
    return (
        f'<link:loc xlink:type="locator" xlink:href="schema.xsd#ex_{name}" '
        f'xlink:label="{name}"/>'
    )


def _arc(
    kind: str, arcrole: str, source: str, target: str, attributes: str = ""
) -> str:
    return (
        f'<link:{kind}Arc xlink:type="arc" xlink:arcrole="{arcrole}" '
        f'xlink:from="{source}" xlink:to="{target}" {attributes}/>'
    )


def _link(kind: str, role: str, names: tuple[str, ...], arcs: str) -> str:
    return (
        f'<link:{kind}Link xlink:type="extended" xlink:role="{role}">'
        + "".join(map(_loc, names))
        + arcs
        + f"</link:{kind}Link>"
    )


def _linkbase(body: str, refs: str = "") -> str:
    return (
        f'<?xml version="1.0" encoding="utf-8"?>\n<link:linkbase {NAMESPACES}>'
        f"{refs}{body}</link:linkbase>\n"
    )


def _label(name: str, text: str, attributes: str) -> str:
    return f'<link:label xlink:type="resource" xlink:label="{name}" {attributes}>{text}</link:label>'


CONCEPT_LABEL = "http://www.xbrl.org/2003/arcrole/concept-label"
# Assets: the same text as standard and terse label, and a French standard
# label; AssetsCurrent: a label with no role and no language whose text
# needs normalizing; one resource labels both AssetsCurrent and AssetsNoncurrent.
LABELS = _linkbase(
    _link(
        "label",
        "http://www.xbrl.org/2003/role/link",
        ("Assets", "AssetsCurrent", "AssetsNoncurrent"),
        _label("assets", "Assets", f'xlink:role="{STANDARD}" xml:lang="en-US"')
        + _label("assets_terse", "Assets", f'xlink:role="{TERSE}" xml:lang="en-US"')
        + _label("assets_fr", "Actifs", f'xlink:role="{STANDARD}" xml:lang="fr"')
        + _label("current", "  Current\r\nassets e&#769; ", "")
        + _label("shared", "Shared", f'xlink:role="{TERSE}" xml:lang="en"')
        + "".join(
            _arc("label", CONCEPT_LABEL, concept, label)
            for concept, label in (
                ("Assets", "assets"),
                ("Assets", "assets_terse"),
                ("Assets", "assets_fr"),
                ("AssetsCurrent", "current"),
                ("AssetsCurrent", "shared"),
                ("AssetsNoncurrent", "shared"),
            )
        ),
    )
)
PRESENTATION = _linkbase(
    _link(
        "presentation",
        ROLE_A,
        ("StatementAbstract", "Assets", "Revenue"),
        _arc("presentation", PARENT_CHILD, "StatementAbstract", "Assets",
             f'order="1" preferredLabel="{TERSE}"')
        + _arc("presentation", PARENT_CHILD, "StatementAbstract", "Revenue", 'order="2.0"'),
    )
    + _link(
        "presentation",
        ROLE_B,
        ("StatementAbstract", "Revenue"),
        _arc("presentation", PARENT_CHILD, "StatementAbstract", "Revenue", 'order="1"'),
    )
)  # fmt: skip
CALCULATION = _linkbase(
    _link(
        "calculation",
        ROLE_A,
        ("Assets", "AssetsCurrent", "AssetsNoncurrent", "Revenue"),
        _arc("calculation", SUMMATION, "Assets", "AssetsCurrent", 'order="1" weight="1.000"')
        + _arc("calculation", SUMMATION, "Assets", "AssetsNoncurrent",
               'order="2" weight="0.1000000000000000000001"')
        + _arc("calculation", SUMMATION, "Assets", "Revenue", 'order="3" weight="-1"'),
    )
    + _link(
        "calculation",
        ROLE_B,
        ("Assets", "AssetsCurrent"),
        _arc("calculation", SUMMATION_11, "Assets", "AssetsCurrent", 'order="1" weight="1"'),
    )
    + _link(
        "calculation",
        ROLE_EMPTY,
        ("Assets", "Revenue"),
        _arc("calculation", SUMMATION, "Assets", "Revenue", 'order="1" weight="1"'),
    ),
    '<link:arcroleRef arcroleURI="https://xbrl.org/2023/arcrole/summation-item" '
    'xlink:type="simple" '
    'xlink:href="https://xbrl.org/2023/calculation-1.1.xsd#summation-item"/>',
)  # fmt: skip
DEFINITION = _linkbase(
    _link(
        "definition",
        ROLE_A,
        ("Assets", "ProductTable", "ProductAxis", "WidgetsMember", "GadgetsMember"),
        _arc("definition", f"{DIMENSION}all", "Assets", "ProductTable",
             f'xbrldt:closed="true" xbrldt:contextElement="segment" '
             f'xbrldt:targetRole="{ROLE_B}"')
        + _arc("definition", f"{DIMENSION}hypercube-dimension", "ProductTable", "ProductAxis")
        + _arc("definition", f"{DIMENSION}dimension-domain", "ProductAxis", "WidgetsMember",
               'xbrldt:usable="false"')
        + _arc("definition", f"{DIMENSION}domain-member", "WidgetsMember", "GadgetsMember"),
    )
)  # fmt: skip
# Prohibits Assets -> Revenue in role A and every arc of role Empty, and
# overrides the presentation of Revenue in role A with a higher priority.
OVERRIDE = _linkbase(
    _link(
        "calculation",
        ROLE_A,
        ("Assets", "Revenue"),
        _arc("calculation", SUMMATION, "Assets", "Revenue",
             'order="3" weight="-1" use="prohibited" priority="1"'),
    )
    + _link(
        "calculation",
        ROLE_EMPTY,
        ("Assets", "Revenue"),
        _arc("calculation", SUMMATION, "Assets", "Revenue",
             'order="1" weight="1" use="prohibited" priority="1"'),
    )
    + _link(
        "presentation",
        ROLE_A,
        ("StatementAbstract", "Revenue"),
        _arc("presentation", PARENT_CHILD, "StatementAbstract", "Revenue",
             'order="2.0" priority="1"'),
    )
)  # fmt: skip
INSTANCE = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<html xmlns="http://www.w3.org/1999/xhtml" '
    'xmlns:ix="http://www.xbrl.org/2013/inlineXBRL" '
    'xmlns:xbrli="http://www.xbrl.org/2003/instance" '
    'xmlns:link="http://www.xbrl.org/2003/linkbase" '
    'xmlns:xlink="http://www.w3.org/1999/xlink" '
    'xmlns:iso4217="http://www.xbrl.org/2003/iso4217" '
    'xmlns:ex="http://example.com/unit"><head><title>report</title></head><body>'
    '<div style="display:none"><ix:header><ix:references>'
    '<link:schemaRef xlink:type="simple" xlink:href="schema.xsd"/></ix:references>'
    '<ix:resources><xbrli:context id="AsOf"><xbrli:entity>'
    '<xbrli:identifier scheme="http://www.sec.gov/CIK">0000320193</xbrli:identifier>'
    "</xbrli:entity><xbrli:period><xbrli:instant>2024-12-31</xbrli:instant>"
    '</xbrli:period></xbrli:context><xbrli:unit id="usd"><xbrli:measure>'
    "iso4217:USD</xbrli:measure></xbrli:unit></ix:resources></ix:header></div>"
    '<p><ix:nonFraction name="ex:Assets" contextRef="AsOf" unitRef="usd" '
    'decimals="0">100</ix:nonFraction></p></body></html>'
)
FILES = {
    "report.htm": INSTANCE,
    "schema.xsd": SCHEMA,
    "labels.xml": LABELS,
    "presentation.xml": PRESENTATION,
    "calculation.xml": CALCULATION,
    "definition.xml": DEFINITION,
    "override.xml": OVERRIDE,
}
DOCUMENT_IDS = {name: f"{FILING}{n:04d}" for n, name in enumerate(FILES, start=1)}


def _load(directory: Path) -> LoadedInstance:
    files = [
        SourceFile(
            DOCUMENT_IDS[name], name, text.encode(), is_entry=name == "report.htm"
        )
        for name, text in FILES.items()
    ]
    settings = ProcessorSettings(
        cache_dir=directory / "cache", allow_network=False, http_user_agent=None
    )
    return load_instance(files, is_inline=True, target_name=None, settings=settings)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> LoadedInstance:
    return _load(tmp_path_factory.mktemp("linkbases"))


@pytest.fixture(scope="module")
def labels(loaded: LoadedInstance) -> list[XbrlLabelRecord]:
    return label_records(loaded.labels, PROCESSED_AT)


@pytest.fixture(scope="module")
def built(
    loaded: LoadedInstance,
) -> tuple[list[XbrlNetworkRecord], list[XbrlRelationshipRecord]]:
    return network_records(
        INSTANCE_ID, loaded.arcs, loaded.role_definitions, PROCESSED_AT
    )


def _labels_of(
    labels: list[XbrlLabelRecord], name: str
) -> dict[tuple[str, str | None], XbrlLabelRecord]:
    concept_id = concept_id_for(EX, name)
    return {(l.role_uri, l.language): l for l in labels if l.concept_id == concept_id}


def test_every_label_role_and_language_is_its_own_row(
    labels: list[XbrlLabelRecord],
) -> None:
    assets = _labels_of(labels, "Assets")

    assert set(assets) == {(STANDARD, "en-us"), (TERSE, "en-us"), (STANDARD, "fr")}
    # The same text under two roles stays two labels.
    assert (
        assets[STANDARD, "en-us"].label_text
        == assets[TERSE, "en-us"].label_text
        == "Assets"
    )
    assert len({label.label_id for label in assets.values()}) == 3
    assert assets[STANDARD, "fr"].label_text == "Actifs"


def test_label_without_role_or_language_gets_the_standard_role_and_no_language(
    labels: list[XbrlLabelRecord],
) -> None:
    current = _labels_of(labels, "AssetsCurrent")[STANDARD, None]

    # CRLF becomes LF, e + combining acute is composed (NFC), ends are trimmed.
    assert current.label_text == "Current\nassets \u00e9"
    assert current.resource_label == "current"
    assert current.source_document_id == DOCUMENT_IDS["labels.xml"]


def test_a_resource_labelling_two_concepts_gives_two_labels(
    labels: list[XbrlLabelRecord],
) -> None:
    shared = [label for label in labels if label.resource_label == "shared"]

    assert {label.concept_id for label in shared} == {
        concept_id_for(EX, "AssetsCurrent"),
        concept_id_for(EX, "AssetsNoncurrent"),
    }
    assert len({label.label_resource_id for label in shared}) == 1
    assert len({label.label_id for label in shared}) == 2


def test_label_ids_are_stable_between_loads(
    tmp_path: Path, labels: list[XbrlLabelRecord]
) -> None:
    again = label_records(_load(tmp_path).labels, datetime(2027, 1, 1, tzinfo=UTC))

    assert sorted(l.label_id for l in again) == sorted(l.label_id for l in labels)
    assert all(l.label_id.startswith("xbrl:label:") for l in labels)


def _network(
    networks: list[XbrlNetworkRecord], linkbase_type: str, role: str
) -> XbrlNetworkRecord:
    return next(
        n for n in networks if (n.linkbase_type, n.role_uri) == (linkbase_type, role)
    )


def _relationships(
    relationships: list[XbrlRelationshipRecord], network: XbrlNetworkRecord
) -> list[XbrlRelationshipRecord]:
    return [r for r in relationships if r.network_id == network.network_id]


def test_each_linkbase_type_and_role_is_one_network(
    built: tuple[list[XbrlNetworkRecord], list[XbrlRelationshipRecord]],
) -> None:
    networks, relationships = built

    # Role Empty holds only prohibited arcs, so it makes no network.
    assert [(n.linkbase_type, n.role_uri) for n in networks] == [
        ("presentation", ROLE_A),
        ("presentation", ROLE_B),
        ("calculation", ROLE_A),
        ("calculation", ROLE_B),
        ("definition", ROLE_A),
    ]
    assert [n.network_ordinal for n in networks] == [1, 2, 3, 4, 5]
    assert {n.instance_network_count for n in networks} == {5}
    assert all(
        n.relationship_count == len(_relationships(relationships, n)) >= 1
        for n in networks
    )
    assert (
        _network(networks, "presentation", ROLE_A).role_definition
        == "100 - Statement - Assets"
    )
    assert _network(networks, "presentation", ROLE_B).role_definition is None


def test_network_ids_and_hashes_are_deterministic(
    loaded: LoadedInstance,
    built: tuple[list[XbrlNetworkRecord], list[XbrlRelationshipRecord]],
) -> None:
    networks, _ = built
    again, _ = network_records(
        INSTANCE_ID, loaded.arcs, loaded.role_definitions, PROCESSED_AT
    )
    renamed, _ = network_records(
        INSTANCE_ID,
        loaded.arcs,
        {**loaded.role_definitions, ROLE_A: "other"},
        PROCESSED_AT,
    )

    assert [n.model_dump() for n in again] == [n.model_dump() for n in networks]
    assert network_id_for(INSTANCE_ID, "presentation", ROLE_A) != network_id_for(
        "sec:other:xbrl:inline:default", "presentation", ROLE_A
    )
    assert network_id_for(INSTANCE_ID, "presentation", ROLE_A) != network_id_for(
        INSTANCE_ID, "calculation", ROLE_A
    )
    assert renamed[0].network_hash != networks[0].network_hash
    assert renamed[0].network_id == networks[0].network_id


def test_presentation_keeps_order_and_preferred_label(
    built: tuple[list[XbrlNetworkRecord], list[XbrlRelationshipRecord]],
) -> None:
    networks, relationships = built
    rows = _relationships(relationships, _network(networks, "presentation", ROLE_A))
    by_target = {r.target_concept_id: r for r in rows}

    assets = by_target[concept_id_for(EX, "Assets")]
    revenue = by_target[concept_id_for(EX, "Revenue")]
    assert (assets.order_value, assets.preferred_label_role) == ("1", TERSE)
    # The overriding arc (priority 1) is the one kept, with its location.
    assert (revenue.order_value, revenue.preferred_label_role) == ("2", None)
    assert revenue.source_document_id == DOCUMENT_IDS["override.xml"]
    assert len(rows) == 2
    assert all(r.weight is None and r.target_role_uri is None for r in rows)


def test_calculation_keeps_exact_weights_and_both_arcroles(
    built: tuple[list[XbrlNetworkRecord], list[XbrlRelationshipRecord]],
) -> None:
    networks, relationships = built
    rows = _relationships(relationships, _network(networks, "calculation", ROLE_A))
    weights = {r.target_concept_id: r.weight for r in rows}
    [calculation_11] = _relationships(
        relationships, _network(networks, "calculation", ROLE_B)
    )

    # Assets -> Revenue is prohibited, so it is not a relationship.
    assert weights == {
        concept_id_for(EX, "AssetsCurrent"): "1",
        concept_id_for(EX, "AssetsNoncurrent"): "0.1000000000000000000001",
    }
    assert calculation_11.arcrole_uri == SUMMATION_11
    assert [r.relationship_ordinal for r in rows] == [1, 2]
    assert all(r.preferred_label_role is None and r.is_closed is None for r in rows)


def test_definition_keeps_dimensional_attributes_where_they_apply(
    built: tuple[list[XbrlNetworkRecord], list[XbrlRelationshipRecord]],
) -> None:
    networks, relationships = built
    rows = {
        r.arcrole_uri.removeprefix(DIMENSION): r
        for r in _relationships(relationships, _network(networks, "definition", ROLE_A))
    }

    assert (rows["all"].target_role_uri, rows["all"].context_element) == (
        ROLE_B,
        "segment",
    )
    assert (rows["all"].is_closed, rows["all"].is_usable) == (True, None)
    assert (rows["dimension-domain"].is_usable, rows["dimension-domain"].is_closed) == (
        False,
        None,
    )
    # XBRL Dimensions makes domain-member arcs usable unless they say otherwise.
    assert rows["domain-member"].is_usable is True
    hypercube = rows["hypercube-dimension"]
    assert (hypercube.is_closed, hypercube.is_usable, hypercube.context_element) == (
        None,
        None,
        None,
    )


def test_relationships_are_ordered_numbered_and_never_merged(
    built: tuple[list[XbrlNetworkRecord], list[XbrlRelationshipRecord]],
) -> None:
    networks, relationships = built
    for network in networks:
        rows = _relationships(relationships, network)
        assert network_problem(network, rows) is None
        assert all(relationship_attributes_problem(network, r) is None for r in rows)
        assert len({r.relationship_id for r in rows}) == len(rows)
        keys = [
            (r.source_concept_id, Decimal(r.order_value), r.target_concept_id)
            for r in rows
        ]
        assert keys == sorted(keys)


def test_network_checks_catch_truncated_groups_and_misplaced_attributes(
    built: tuple[list[XbrlNetworkRecord], list[XbrlRelationshipRecord]],
) -> None:
    networks, relationships = built
    calculation = _network(networks, "calculation", ROLE_A)
    rows = _relationships(relationships, calculation)
    presentation = _network(networks, "presentation", ROLE_A)

    assert (
        network_problem(calculation, rows[1:])
        == "1 relationships but declared counts [2]"
    )
    assert relationship_attributes_problem(presentation, rows[0]) == (
        "presentation relationship has ['weight']"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1.000", "1"), ("0.500", "0.5"), ("-1.0", "-1"), ("-0", "0"), ("1E+2", "100")],
)
def test_order_and_weight_use_plain_decimals(value: str, expected: str) -> None:
    assert plain_decimal(Decimal(value)) == expected

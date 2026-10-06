from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

SEC_HEADER_TEMPLATE = """<SEC-DOCUMENT>{accession}.txt : {filed}
<SEC-HEADER>{accession}.hdr.sgml : {filed}
<ACCEPTANCE-DATETIME>{filed}160102
ACCESSION NUMBER:\t\t{accession}
CONFORMED SUBMISSION TYPE:\t{form}
PUBLIC DOCUMENT COUNT:\t\t{document_count}
CONFORMED PERIOD OF REPORT:\t20241231
FILED AS OF DATE:\t\t{filed}

{entities}</SEC-HEADER>
"""

# One FILER block laid out like real EDGAR headers: tab-indented sections,
# including the FORMER COMPANY name history that is not an entity of its own.
SEC_FILER_TEMPLATE = """FILER:

\tCOMPANY DATA:\t
\t\tCOMPANY CONFORMED NAME:\t\t\t{name}
\t\tCENTRAL INDEX KEY:\t\t\t{cik}
\t\tSTANDARD INDUSTRIAL CLASSIFICATION:\tELECTRONIC COMPUTERS [3571]
\t\tORGANIZATION NAME:           \t06 Technology
\t\tIRS NUMBER:\t\t\t\t942404110
\t\tSTATE OF INCORPORATION:\t\t\tCA
\t\tFISCAL YEAR END:\t\t\t0927

\tFILING VALUES:
\t\tFORM TYPE:\t\t{form}
\t\tSEC ACT:\t\t1934 Act
\t\tSEC FILE NUMBER:\t001-36743
\t\tFILM NUMBER:\t\t251451234

\tBUSINESS ADDRESS:\t
\t\tSTREET 1:\t\tONE EXAMPLE WAY
\t\tCITY:\t\t\tCUPERTINO
\t\tSTATE:\t\t\tCA
\t\tZIP:\t\t\t95014
\t\tBUSINESS PHONE:\t\t(408) 996-1010

\tMAIL ADDRESS:\t
\t\tSTREET 1:\t\tONE EXAMPLE WAY
\t\tCITY:\t\t\tCUPERTINO
\t\tSTATE:\t\t\tCA
\t\tZIP:\t\t\t95014

\tFORMER COMPANY:\t
\t\tFORMER CONFORMED NAME:\tEXAMPLE COMPUTER INC
\t\tDATE OF NAME CHANGE:\t19970808

"""

_XBRL_NAMESPACES = (
    'xmlns:link="http://www.xbrl.org/2003/linkbase" '
    'xmlns:xlink="http://www.w3.org/1999/xlink" '
    'xmlns:xbrldi="http://xbrl.org/2006/xbrldi" '
    'xmlns:iso4217="http://www.xbrl.org/2003/iso4217" '
    'xmlns:ex="http://example.com/2024"'
)
_SCHEMA_REF = '<link:schemaRef xlink:type="simple" xlink:href="ex-2024.xsd"/>'
# (name, type, period type, substitution group, other attributes) of the
# extension concepts: items used by facts, abstract members, and an explicit
# and a typed dimension whose domain is the non-item element ex:Region.
_CONCEPTS = (
    ("AmendmentFlag", "xbrli:booleanItemType", "duration", "xbrli:item", ""),
    ("DocumentType", "xbrli:stringItemType", "duration", "xbrli:item", ""),
    ("Product", "xbrli:stringItemType", "duration", "xbrli:item", ""),
    ("Revenue", "xbrli:monetaryItemType", "duration", "xbrli:item",
     'xbrli:balance="credit"'),
    ("EarningsPerShare", "xbrli:decimalItemType", "duration", "xbrli:item", ""),
    ("SharesOutstanding", "xbrli:sharesItemType", "instant", "xbrli:item", ""),
    ("Cash", "xbrli:monetaryItemType", "instant", "xbrli:item",
     'xbrli:balance="debit"'),
    ("ProductAxis", "xbrli:stringItemType", "duration", "xbrldt:dimensionItem",
     'abstract="true"'),
    ("WidgetsMember", "xbrli:stringItemType", "duration", "xbrli:item",
     'abstract="true"'),
    ("RegionAxis", "xbrli:stringItemType", "duration", "xbrldt:dimensionItem",
     'abstract="true" xbrldt:typedDomainRef="#ex_Region"'),
    ("StatementAbstract", "xbrli:stringItemType", "duration", "xbrli:item",
     'abstract="true"'),
    ("ProductTable", "xbrli:stringItemType", "duration", "xbrldt:hypercubeItem",
     'abstract="true"'),
)  # fmt: skip
STATEMENT_ROLE = "http://example.com/role/Statement"


def _loc(name: str) -> str:
    return (
        f'<link:loc xlink:type="locator" xlink:href="#ex_{name}" xlink:label="{name}"/>'
    )


def _arc(
    kind: str, arcrole: str, source: str, target: str, attributes: str = ""
) -> str:
    return (
        f'<link:{kind}Arc xlink:type="arc" xlink:arcrole="{arcrole}" '
        f'xlink:from="{source}" xlink:to="{target}" {attributes}/>'
    )


def _label(name: str, role: str, language: str, text: str) -> str:
    return (
        f'<link:label xlink:type="resource" xlink:label="{name}" '
        f'xlink:role="http://www.xbrl.org/2003/role/{role}" xml:lang="{language}">'
        f"{text}</link:label>"
    )


_DIMENSION_ARCROLE = "http://xbrl.org/int/dim/arcrole/"
# Linkbases embedded in the extension schema: Revenue's standard, terse and
# French labels, and one statement role with a presentation, a calculation
# and a definition network (a closed hypercube on ProductAxis).
_LINKBASES = (
    "<xs:annotation><xs:appinfo>"
    f'<link:roleType roleURI="{STATEMENT_ROLE}" id="Statement">'
    "<link:definition>100 - Statement - Revenue</link:definition>"
    "<link:usedOn>link:presentationLink</link:usedOn>"
    "<link:usedOn>link:calculationLink</link:usedOn>"
    "<link:usedOn>link:definitionLink</link:usedOn></link:roleType>"
    "<link:linkbase>"
    '<link:labelLink xlink:type="extended" '
    'xlink:role="http://www.xbrl.org/2003/role/link">'
    + _loc("Revenue")
    + _label("revenue_standard", "label", "en-US", "Revenue")
    + _label("revenue_terse", "terseLabel", "en-US", "Revenues")
    + _label("revenue_french", "label", "fr", "Chiffre d&#8217;affaires")
    + "".join(
        _arc("label", "http://www.xbrl.org/2003/arcrole/concept-label", "Revenue", label)
        for label in ("revenue_standard", "revenue_terse", "revenue_french")
    )
    + "</link:labelLink>"
    f'<link:presentationLink xlink:type="extended" xlink:role="{STATEMENT_ROLE}">'
    + _loc("StatementAbstract")
    + _loc("Revenue")
    + _loc("ProductTable")
    + _arc("presentation", "http://www.xbrl.org/2003/arcrole/parent-child",
           "StatementAbstract", "Revenue",
           'order="1" preferredLabel="http://www.xbrl.org/2003/role/terseLabel"')
    + _arc("presentation", "http://www.xbrl.org/2003/arcrole/parent-child",
           "StatementAbstract", "ProductTable", 'order="2"')
    + "</link:presentationLink>"
    f'<link:calculationLink xlink:type="extended" xlink:role="{STATEMENT_ROLE}">'
    + _loc("Revenue")
    + _loc("Cash")
    + _arc("calculation", "http://www.xbrl.org/2003/arcrole/summation-item",
           "Revenue", "Cash", 'order="1" weight="1.000"')
    + "</link:calculationLink>"
    f'<link:definitionLink xlink:type="extended" xlink:role="{STATEMENT_ROLE}">'
    + _loc("Revenue")
    + _loc("ProductTable")
    + _loc("ProductAxis")
    + _loc("WidgetsMember")
    + _arc("definition", f"{_DIMENSION_ARCROLE}all", "Revenue", "ProductTable",
           'xbrldt:closed="true" xbrldt:contextElement="segment"')
    + _arc("definition", f"{_DIMENSION_ARCROLE}hypercube-dimension",
           "ProductTable", "ProductAxis")
    + _arc("definition", f"{_DIMENSION_ARCROLE}dimension-domain",
           "ProductAxis", "WidgetsMember")
    + "</link:definitionLink></link:linkbase></xs:appinfo></xs:annotation>"
)  # fmt: skip
# The filing's extension schema, with its linkbases embedded. It imports only
# the XBRL 2.1 and dimension schemas, which the XBRL processor ships with, so
# instances load offline.
EXTENSION_SCHEMA = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" '
    'xmlns:xbrli="http://www.xbrl.org/2003/instance" '
    'xmlns:xbrldt="http://xbrl.org/2005/xbrldt" '
    'xmlns:link="http://www.xbrl.org/2003/linkbase" '
    'xmlns:xlink="http://www.w3.org/1999/xlink" '
    'targetNamespace="http://example.com/2024" elementFormDefault="qualified">'
    + _LINKBASES
    + '<xs:import namespace="http://www.xbrl.org/2003/instance" '
    'schemaLocation="http://www.xbrl.org/2003/xbrl-instance-2003-12-31.xsd"/>'
    '<xs:import namespace="http://xbrl.org/2005/xbrldt" '
    'schemaLocation="http://www.xbrl.org/2005/xbrldt-2005.xsd"/>'
    + "".join(
        f'<xs:element id="ex_{name}" name="{name}" type="{item_type}" '
        f'substitutionGroup="{group}" xbrli:periodType="{period}" nillable="true" '
        f"{attributes}/>"
        for name, item_type, period, group, attributes in _CONCEPTS
    )
    + '<xs:element id="ex_Region" name="Region" type="xs:string"/></xs:schema>\n'
)
# Contexts and units by ID, written with an `xbrli` prefix as in Inline XBRL;
# GENERATED_XML rewrites them the way EDGAR does (default namespace, other
# whitespace and member order), so both representations must still match.
_CONTEXTS = {
    "FY": "<xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate>"
    "<xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period>",
    "AsOf": "<xbrli:period><xbrli:instant>2024-12-31</xbrli:instant></xbrli:period>",
    "FY_Widgets": "<xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate>"
    "<xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period>",
}
_SEGMENT = (
    "<xbrli:segment>\n  "
    '<xbrldi:explicitMember dimension="ex:ProductAxis">ex:WidgetsMember'
    "</xbrldi:explicitMember>\n  "
    '<xbrldi:typedMember dimension="ex:RegionAxis"><ex:Region>North</ex:Region>'
    "</xbrldi:typedMember>\n</xbrli:segment>"
)
_UNITS = {
    "usd": "<xbrli:measure>iso4217:USD</xbrli:measure>",
    "shares": "<xbrli:measure>xbrli:shares</xbrli:measure>",
    "usdPerShare": "<xbrli:divide><xbrli:unitNumerator><xbrli:measure>iso4217:USD"
    "</xbrli:measure></xbrli:unitNumerator><xbrli:unitDenominator><xbrli:measure>"
    "xbrli:shares</xbrli:measure></xbrli:unitDenominator></xbrli:divide>",
    "eur": "<xbrli:measure>iso4217:EUR</xbrli:measure>",
}
# Hidden facts as (element, concept, context, unit, value) per form: the 10-K
# also has visible facts in its body and an unused EUR unit; the 8-K has only
# nonnumeric facts, so no unit.
_FACTS = {
    "10-K": [
        ("nonNumeric", "ex:AmendmentFlag", "FY", None, "false"),
        ("nonFraction", "ex:Revenue", "FY", "usd", "1000"),
        ("nonFraction", "ex:EarningsPerShare", "FY_Widgets", "usdPerShare", "2.5"),
        ("nonFraction", "ex:SharesOutstanding", "AsOf", "shares", "400"),
    ],
    "10-Q": [
        ("nonNumeric", "ex:AmendmentFlag", "FY", None, "false"),
        ("nonFraction", "ex:Cash", "AsOf", "usd", "70"),
    ],
    "8-K": [
        ("nonNumeric", "ex:AmendmentFlag", "FY", None, "false"),
        ("nonNumeric", "ex:DocumentType", "FY", None, "8-K"),
    ],
}
# Visible facts of the primary documents as (concept, context, unit, value in
# the XML rendering): the 10-K names its product in a paragraph and repeats
# the hidden revenue fact, formatted, in a table cell.
_VISIBLE_FACTS = {
    "10-K": [("ex:Product", "FY", None, "widgets"), ("ex:Revenue", "FY", "usd", "1000")]
}
_RESOURCES = {
    "10-K": (["FY", "AsOf", "FY_Widgets"], ["usd", "shares", "usdPerShare", "eur"]),
    "10-Q": (["FY", "AsOf"], ["usd"]),
    "8-K": (["FY"], []),
}


def _context(context_id: str) -> str:
    segment = _SEGMENT if context_id == "FY_Widgets" else ""
    return (
        f'<xbrli:context id="{context_id}"><xbrli:entity><xbrli:identifier '
        f'scheme="http://www.sec.gov/CIK">0000320193</xbrli:identifier>{segment}'
        f"</xbrli:entity>{_CONTEXTS[context_id]}</xbrli:context>"
    )


# Fact-footnote relationships of the 10-K (hidden facts have IDs hidden-1...):
# the hidden and the visible revenue share a footnote with markup; the visible
# one also has a plain footnote with its own role, in another link role.
_RELATIONSHIPS = {
    "10-K": '<ix:relationship fromRefs="hidden-2 revenue-cell" toRefs="fn-revenue"/>'
    '<ix:relationship fromRefs="revenue-cell" toRefs="fn-estimate" '
    'linkRole="http://example.com/role/notes"/>'
}
# The 10-K footnotes, in a hidden div so that narrative content is unchanged.
_FOOTNOTES = (
    '<div style="display: none"><ix:footnote id="fn-revenue" xml:lang="en-US">'
    "Excludes <b>discontinued</b>&#160;operations.</ix:footnote>"
    '<ix:footnote id="fn-estimate" footnoteRole="http://example.com/role/estimate">'
    "Estimated.</ix:footnote></div>"
)


def _inline_header(form: str) -> str:
    contexts, units = _RESOURCES[form]
    facts = "".join(
        f'<ix:{kind} id="hidden-{n}" name="{concept}" contextRef="{context}"'
        + (f' unitRef="{unit}" decimals="0"' if unit else "")
        + f">{value}</ix:{kind}>"
        for n, (kind, concept, context, unit, value) in enumerate(_FACTS[form], 1)
    )
    resources = (
        "".join(_context(c) for c in contexts)
        + "".join(f'<xbrli:unit id="{u}">{_UNITS[u]}</xbrli:unit>' for u in units)
        + _RELATIONSHIPS.get(form, "")
    )
    return (
        f'<div style="display: none"><ix:header><ix:hidden>{facts}</ix:hidden>'
        f"<ix:references>{_SCHEMA_REF}</ix:references>"
        f"<ix:resources>{resources}</ix:resources></ix:header></div>"
    )


def _generated_xml(form: str) -> str:
    """EDGAR's XML rendering of the form's Inline XBRL, unused units left out."""

    contexts, _ = _RESOURCES[form]
    facts = [fact[1:] for fact in _FACTS[form]] + _VISIBLE_FACTS.get(form, [])
    used = [unit for _, _, unit, _ in facts if unit]
    body = (
        "".join(_context(c) for c in contexts)
        .replace("xbrli:", "")
        .replace(
            "<segment>\n  "
            '<xbrldi:explicitMember dimension="ex:ProductAxis">ex:WidgetsMember'
            "</xbrldi:explicitMember>\n  ",
            "<segment>\n        ",
        )
        .replace(
            "</xbrldi:typedMember>\n</segment>",
            "</xbrldi:typedMember>\n        "
            '<xbrldi:explicitMember dimension="ex:ProductAxis"> ex:WidgetsMember '
            "</xbrldi:explicitMember>\n      </segment>",
        )
    )
    body += "".join(
        f'<unit id="{u}">{_UNITS[u]}</unit>'.replace("xbrli:measure", "measure")
        .replace("xbrli:divide", "divide")
        .replace("xbrli:unit", "unit")
        for u in dict.fromkeys(used)
    )
    body += "".join(
        f'<{concept} contextRef="{context}"'
        + (f' unitRef="{unit}" decimals="0"' if unit else "")
        + f">{value}</{concept}>"
        for concept, context, unit, value in facts
    )
    return (
        '\n<XML>\n<?xml version="1.0" encoding="utf-8"?>\n'
        '<xbrl xmlns="http://www.xbrl.org/2003/instance" '
        f'xmlns:xbrli="http://www.xbrl.org/2003/instance" {_XBRL_NAMESPACES}>'
        f"{_SCHEMA_REF}{body}</xbrl>\n</XML>\n"
    )


# Inline XBRL primary documents as EDGAR stores them: wrapped in <XBRL> with an
# XML prolog, an ix:header (hidden facts, schema reference, contexts, units), a
# linked table of contents, headings in divs or table cells, visible ix facts
# in a paragraph and a table cell, a narrative ITEM mention and a table; the
# 10-K also links two of its facts to hidden footnotes. Links wrap existing text, so they leave
# blocks and cells unchanged: the 10-K links its own exhibit by SEC URL and an
# external site, the 8-K its exhibit, a missing file and another filing's
# exhibit of the same name.
_INLINE_XBRL = (
    "\n<XBRL>\n<?xml version='1.0' encoding='ASCII'?>\n"
    '<html xmlns="http://www.w3.org/1999/xhtml" '
    'xmlns:ix="http://www.xbrl.org/2013/inlineXBRL" '
    'xmlns:xbrli="http://www.xbrl.org/2003/instance" '
    'xmlns:ixt="http://www.xbrl.org/inlineXBRL/transformation/2020-02-12" '
    f"{_XBRL_NAMESPACES}><head><title>example</title></head><body>"
    "{header}{body}</body></html>\n</XBRL>\n"
)
PRIMARY_HTML = {
    "10-K": _INLINE_XBRL.format(
        header=_inline_header("10-K"),
        body='<div><a href="https://www.sec.gov/Archives/edgar/data/320193/'
        '000032019324000123/ex31.htm">Example Corp</a> Annual Report</div>'
        "<div>TABLE OF CONTENTS</div>"
        '<table><tr><td><a href="#p1">Part I</a></td></tr>'
        '<tr><td><a href="#i1">Item 1.</a></td><td>Business</td></tr>'
        '<tr><td><a href="#i1a">Item 1A.</a></td><td>Risk Factors</td></tr>'
        '<tr><td><a href="#p2">Part II</a></td></tr>'
        '<tr><td><a href="#i7">Item 7.</a></td><td>MD&amp;A</td></tr></table>'
        '<div id="p1">PART I</div>'
        '<div id="i1"><span style="font-weight:700">Item 1. Business</span></div>'
        '<div>We make <a href="https://www.example.com/products">'
        '<ix:nonNumeric name="ex:Product" contextRef="FY">widgets</ix:nonNumeric></a>.</div>'
        '<div id="i1a">Item 1A. Risk Factors</div>'
        '<div>As discussed in <a href="#i7">Item 7</a>, demand may fall.</div>'
        '<div id="p2">PART II</div>'
        '<div id="i7">Item 7. Management&#8217;s Discussion and Analysis</div>'
        '<table><tr><td>Revenue</td><td>$<ix:nonFraction id="revenue-cell" '
        'name="ex:Revenue" '
        'contextRef="FY" unitRef="usd" decimals="0" format="ixt:num-dot-decimal">'
        "1,000</ix:nonFraction></td></tr></table>"
        '<div><a name="signatures"></a>SIGNATURES</div><div>/s/ Jane Doe</div>'
        + _FOOTNOTES,
    ),
    "10-Q": _INLINE_XBRL.format(
        header=_inline_header("10-Q"),
        body="<div>PART I &#8212; FINANCIAL INFORMATION</div>"
        "<div>Item 1. Financial Statements</div><div>Net income rose.</div>"
        "<div>Item 2. Management&#8217;s Discussion and Analysis</div>"
        "<p>Results improved.</p>"
        "<div>PART II &#8212; OTHER INFORMATION</div>"
        "<div>Item 1. Legal Proceedings</div><div>None.</div>"
        "<div>Item 1A. Risk Factors</div><div>No material changes.</div>"
        "<div>SIGNATURES</div><div>/s/ Jane Doe</div>",
    ),
    "8-K": _INLINE_XBRL.format(
        header=_inline_header("8-K"),
        body="<div>Item 2.02 Results of Operations and Financial Condition.</div>"
        '<div>Example Corp reported <a href="https://www.sec.gov/Archives/edgar/'
        'data/789019/000078901924000001/ex31.htm">results</a>.</div>'
        "<div>Item 9.01 Financial Statements and Exhibits.</div>"
        "<div>EXHIBIT INDEX</div>"
        '<table><tr><td><a href="ex31.htm">99.1</a></td>'
        '<td><a href="press-release.htm">Press release</a></td></tr></table>'
        "<div>SIGNATURE</div><div>/s/ Jane Doe</div>",
    ),
}

# (TYPE, SEQUENCE, FILENAME, DESCRIPTION, <TEXT> content) of the documents in
# every fake submission, shaped like real EDGAR filings: the form's primary
# document (its <TEXT> is PRIMARY_HTML of the folder form), an exhibit, XBRL
# schema and extracted instance, an uuencoded graphic without DESCRIPTION, and
# an EDGAR-rendered R page (sequence gap). "{form}" in TYPE is replaced by the
# submission's form.
SEC_DOCUMENTS = (
    ("{form}", "1", "primary.htm", "{form}", ""),
    ("EX-31.1", "2", "ex31.htm", "CERTIFICATION", "\n<html>certify</html>\n"),
    (
        "EX-101.SCH",
        "3",
        "ex-2024.xsd",
        "XBRL TAXONOMY EXTENSION SCHEMA",
        f"\n<XBRL>\n{EXTENSION_SCHEMA}</XBRL>\n",
    ),
    (
        "XML",
        "4",
        "primary_htm.xml",
        "IDEA: XBRL DOCUMENT",
        '\n<XBRL>\n<?xml version="1.0"?>\n<xbrl xmlns="http://www.xbrl.org/2003/instance"/>\n</XBRL>\n',
    ),
    ("GRAPHIC", "5", "logo.jpg", None, "\nbegin 644 logo.jpg\nM_]C_X``0\nend\n"),
    ("XML", "7", "R1.htm", "IDEA: XBRL DOCUMENT", "\n<html>cover page</html>\n"),
)

SecFilingWriter = Callable[..., Path]


def _document_block(form: str, document: tuple[str, str, str, str | None, str]) -> str:
    document_type, sequence, filename, description, text = document
    lines = [
        "<DOCUMENT>",
        f"<TYPE>{document_type.format(form=form)}",
        f"<SEQUENCE>{sequence}",
        f"<FILENAME>{filename}",
    ]
    if description is not None:
        lines.append(f"<DESCRIPTION>{description.format(form=form)}")
    lines.extend([f"<TEXT>{text}</TEXT>", "</DOCUMENT>"])
    return "\n".join(lines) + "\n"


@pytest.fixture
def write_sec_filing() -> SecFilingWriter:
    """Return a helper that writes one accession folder into SEC Bronze.

    The SEC header repeats the folder's CIK, form and accession unless a
    `header_*` override is given, and holds one FILER block, plus a second
    FILER block when `co_registrant_cik` is given (as in multi-registrant
    8-Ks). The submission holds SEC_DOCUMENTS with the primary document typed
    as the header form; `primary_text` replaces its <TEXT> content, and the
    extracted XBRL instance is EDGAR's XML rendering of the form's Inline
    XBRL, left out when `xbrl_instance` is false. Returns
    the accession directory.
    """

    def write(
        data_dir: Path,
        *,
        cik: str,
        form: str,
        accession: str,
        filed: str,
        header_cik: str | None = None,
        header_form: str | None = None,
        header_accession: str | None = None,
        co_registrant_cik: str | None = None,
        primary_text: str | None = None,
        xbrl_instance: bool = True,
        primary_html: bytes | None = b"<html>primary</html>",
    ) -> Path:
        directory = (
            data_dir
            / "bronze"
            / "sec_filings"
            / "sec-edgar-filings"
            / cik
            / form
            / accession
        )
        directory.mkdir(parents=True)
        submission_form = header_form or form
        filers = [("Example Corp", header_cik or cik)]
        if co_registrant_cik is not None:
            filers.append(("Example Subsidiary LLC", co_registrant_cik))
        header = SEC_HEADER_TEMPLATE.format(
            accession=header_accession or accession,
            form=submission_form,
            filed=filed,
            document_count=len(SEC_DOCUMENTS),
            entities="".join(
                SEC_FILER_TEMPLATE.format(
                    name=name, cik=filer_cik, form=submission_form
                )
                for name, filer_cik in filers
            ),
        )
        primary, *others = SEC_DOCUMENTS
        default_text = PRIMARY_HTML.get(form, "\n<html><body>report</body></html>\n")
        primary = (*primary[:4], primary_text or default_text)
        if form in _FACTS:
            others = [
                (*doc[:4], _generated_xml(form)) if doc[2] == "primary_htm.xml" else doc
                for doc in others
            ]
        if not xbrl_instance:
            others = [doc for doc in others if doc[2] != "primary_htm.xml"]
        documents = "".join(
            _document_block(submission_form, document)
            for document in (primary, *others)
        )
        (directory / "full-submission.txt").write_text(
            f"{header}{documents}</SEC-DOCUMENT>\n", encoding="ascii"
        )
        if primary_html is not None:
            (directory / "primary-document.html").write_bytes(primary_html)
        return directory

    return write

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

# Inline XBRL primary documents as EDGAR stores them: wrapped in <XBRL> with an
# XML prolog, hidden ix:header facts, a linked table of contents, headings in
# divs or table cells, visible ix facts, a narrative ITEM mention and a table.
# Links wrap existing text, so they leave blocks and cells unchanged: the 10-K
# links its own exhibit by SEC URL and an external site, the 8-K its exhibit,
# a missing file and another filing's exhibit of the same name.
_INLINE_XBRL = (
    "\n<XBRL>\n<?xml version='1.0' encoding='ASCII'?>\n<html><head>"
    "<title>example</title></head><body>"
    '<div style="display: none"><ix:header><ix:hidden>'
    '<ix:nonNumeric name="dei:AmendmentFlag">false</ix:nonNumeric>'
    "</ix:hidden></ix:header></div>{body}</body></html>\n</XBRL>\n"
)
PRIMARY_HTML = {
    "10-K": _INLINE_XBRL.format(
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
        '<ix:nonNumeric name="ex:Product">widgets</ix:nonNumeric></a>.</div>'
        '<div id="i1a">Item 1A. Risk Factors</div>'
        '<div>As discussed in <a href="#i7">Item 7</a>, demand may fall.</div>'
        '<div id="p2">PART II</div>'
        '<div id="i7">Item 7. Management&#8217;s Discussion and Analysis</div>'
        "<table><tr><td>Revenue</td><td>$1,000</td></tr></table>"
        '<div><a name="signatures"></a>SIGNATURES</div><div>/s/ Jane Doe</div>'
    ),
    "10-Q": _INLINE_XBRL.format(
        body="<div>PART I &#8212; FINANCIAL INFORMATION</div>"
        "<div>Item 1. Financial Statements</div><div>Net income rose.</div>"
        "<div>Item 2. Management&#8217;s Discussion and Analysis</div>"
        "<p>Results improved.</p>"
        "<div>PART II &#8212; OTHER INFORMATION</div>"
        "<div>Item 1. Legal Proceedings</div><div>None.</div>"
        "<div>Item 1A. Risk Factors</div><div>No material changes.</div>"
        "<div>SIGNATURES</div><div>/s/ Jane Doe</div>"
    ),
    "8-K": _INLINE_XBRL.format(
        body="<div>Item 2.02 Results of Operations and Financial Condition.</div>"
        '<div>Example Corp reported <a href="https://www.sec.gov/Archives/edgar/'
        'data/789019/000078901924000001/ex31.htm">results</a>.</div>'
        "<div>Item 9.01 Financial Statements and Exhibits.</div>"
        "<div>EXHIBIT INDEX</div>"
        '<table><tr><td><a href="ex31.htm">99.1</a></td>'
        '<td><a href="press-release.htm">Press release</a></td></tr></table>'
        "<div>SIGNATURE</div><div>/s/ Jane Doe</div>"
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
        "report.xsd",
        "XBRL TAXONOMY EXTENSION SCHEMA",
        '\n<XBRL>\n<?xml version="1.0"?>\n<xs:schema/>\n</XBRL>\n',
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
    as the header form; `primary_text` replaces its <TEXT> content. Returns
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

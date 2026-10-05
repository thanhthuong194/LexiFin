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

# (TYPE, SEQUENCE, FILENAME, DESCRIPTION, <TEXT> content) of the documents in
# every fake submission, shaped like real EDGAR filings: the form's primary
# document, an exhibit, XBRL schema and extracted instance, an uuencoded
# graphic without DESCRIPTION, and an EDGAR-rendered R page (sequence gap).
# "{form}" in TYPE is replaced by the submission's form.
SEC_DOCUMENTS = (
    ("{form}", "1", "primary.htm", "{form}", "\n<html><body>report</body></html>\n"),
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
    as the header form. Returns the accession directory.
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
        documents = "".join(
            _document_block(submission_form, document) for document in SEC_DOCUMENTS
        )
        (directory / "full-submission.txt").write_text(
            f"{header}{documents}</SEC-DOCUMENT>\n", encoding="ascii"
        )
        if primary_html is not None:
            (directory / "primary-document.html").write_bytes(primary_html)
        return directory

    return write

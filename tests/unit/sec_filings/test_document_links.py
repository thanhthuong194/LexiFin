from __future__ import annotations

import hashlib
import unicodedata
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.data_pipeline.contracts.sec_filings.document_links import DocumentLinkRecord
from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.storage.sec_filings.documents import read_document_content
from src.data_pipeline.transforms.sec_filings.document_links import (
    build_document_links,
    extract_links,
    index_filenames,
)
from src.data_pipeline.transforms.sec_filings.html_dom import parse_html

pytestmark = pytest.mark.unit

FILING_ID = "sec:0000320193-25-000079"
OTHER_FILING_ID = "sec:0000320193-24-000123"
SOURCE_ID = f"{FILING_ID}:document:0001"
EXHIBIT_ID = f"{FILING_ID}:document:0002"
PROCESSED_AT = datetime(2026, 10, 6, 4, 0, tzinfo=UTC)
SEC_FOLDER = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079"


def _document(
    filing_id: str,
    ordinal: int,
    filename: str,
    role: str = "exhibit",
    content: bytes = b"<html></html>",
) -> DocumentRecord:
    return DocumentRecord.model_validate(
        {
            "document_id": f"{filing_id}:document:{ordinal:04d}",
            "filing_id": filing_id,
            "filing_date": date(2025, 10, 31),
            "document_ordinal": ordinal,
            "sequence": str(ordinal),
            "document_type": "10-K" if role == "primary_document" else "EX-10.1",
            "filename": filename,
            "description": None,
            "role": role,
            "content_kind": "html",
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
    )


SOURCE = _document(FILING_ID, 1, "aapl-20250927.htm", role="primary_document")
INVENTORY = [
    SOURCE,
    _document(FILING_ID, 2, "ex10-1.htm"),
    _document(FILING_ID, 3, "logo.jpg", role="graphic"),
    # Same filename as this filing's exhibit, and one only the other filing has.
    _document(OTHER_FILING_ID, 2, "ex10-1.htm"),
    _document(OTHER_FILING_ID, 3, "ex20.htm"),
]


def _links(
    body: str, inventory: list[DocumentRecord] = INVENTORY
) -> list[DocumentLinkRecord]:
    root = parse_html(f"<html><body>{body}</body></html>".encode())
    filenames = index_filenames(inventory).get(FILING_ID, {})
    return build_document_links(SOURCE, extract_links(root), filenames, PROCESSED_AT)


def test_every_anchor_with_href_is_one_row_in_dom_order() -> None:
    body = (
        '<p><a name="top"></a><a href="#a">First</a></p>'
        '<table><tr><td><a href="ex10-1.htm">Exhibit</a></td></tr></table>'
        '<p><a href="#a">First</a> <a href="">Self</a></p>'
    )

    links = _links(body)

    assert [(r.link_ordinal, r.href, r.source_xpath) for r in links] == [
        (1, "#a", "/html/body/p[1]/a[2]"),
        (2, "ex10-1.htm", "/html/body/table/tr/td/a"),
        (3, "#a", "/html/body/p[2]/a[1]"),
        (4, "", "/html/body/p[2]/a[2]"),
    ]
    assert [r.link_id for r in links] == [
        f"{SOURCE_ID}:link:00000{n}" for n in (1, 2, 3, 4)
    ]
    assert {r.document_link_count for r in links} == {4}
    assert {r.source_document_sha256 for r in links} == {SOURCE.content_sha256}
    assert _links(body) == links


def test_link_text_is_normalized_visible_text() -> None:
    links = _links(
        '<a href="#a" title="  Risk&#160; factors ">Item&#160;1A.&#8203;  <b>Risk '
        "<i>Factors</i></b></a>"
        '<a href="#b">Net income was <ix:nonFraction name="us-gaap:NetIncome">5'
        "</ix:nonFraction> million</a>"
        '<a href="#c" title=" "><img src="logo.jpg" alt="Logo"></a>'
        '<a href="#d" style="display:none">hidden</a>'
        '<div hidden><a href="#e">also hidden</a></div>'
    )

    assert [(r.link_text, r.title) for r in links] == [
        ("Item 1A. Risk Factors", "Risk factors"),
        ("Net income was 5 million", None),
        ("", None),
        ("", None),
        ("", None),
    ]


@pytest.mark.parametrize(
    ("href", "kind", "status", "target", "fragment"),
    [
        ("#item1a", "fragment", "resolved", SOURCE_ID, "item1a"),
        ("#", "fragment", "resolved", SOURCE_ID, None),
        ("", "relative_document", "resolved", SOURCE_ID, None),
        ("?page=2", "relative_document", "resolved", SOURCE_ID, None),
        ("ex10-1.htm", "relative_document", "resolved", EXHIBIT_ID, None),
        ("./ex10-1.htm?v=1#sec2", "relative_document", "resolved", EXHIBIT_ID, "sec2"),
        ("docs/ex10%2D1.htm", "relative_document", "resolved", EXHIBIT_ID, None),
        ("aapl-20250927.htm#p1", "relative_document", "resolved", SOURCE_ID, "p1"),
        ("missing.htm", "relative_document", "unresolved", None, None),
        ("ex20.htm", "relative_document", "unresolved", None, None),
        ("EX10-1.HTM", "relative_document", "unresolved", None, None),
        (f"{SEC_FOLDER}/ex10-1.htm", "sec_url", "resolved", EXHIBIT_ID, None),
        (
            "/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm#p2",
            "sec_url",
            "resolved",
            SOURCE_ID,
            "p2",
        ),
        (
            "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/ex10-1.htm",
            "sec_url",
            "unresolved",
            None,
            None,
        ),
        ("HTTPS://WWW.SEC.GOV/cgi-bin/browse-edgar?action=getcompany", "sec_url", "unresolved", None, None),
        ("//efts.sec.gov/LATEST/search-index?q=x", "sec_url", "unresolved", None, None),
        ("https://www.apple.com/investor/#top", "external_url", "not_applicable", None, None),
        ("https://sec.gov.example.com/x.htm", "external_url", "not_applicable", None, None),
        ("//example.com/ex10-1.htm", "external_url", "not_applicable", None, None),
        ("mailto:ir@example.com", "email", "not_applicable", None, None),
        ("tel:+1-408-996-1010", "telephone", "not_applicable", None, None),
        ("JavaScript:void(0);", "javascript", "not_applicable", None, None),
        ("data:text/plain,hi", "data", "not_applicable", None, None),
        ("ftp://example.com/ex10-1.htm", "other", "unresolved", None, None),
        ("http://[::1/ex10-1.htm", "other", "unresolved", None, None),
    ],
)  # fmt: skip
def test_links_are_classified_and_resolved_within_the_filing(
    href: str, kind: str, status: str, target: str | None, fragment: str | None
) -> None:
    (link,) = _links(f'<a href="{href}">x</a>')

    assert (
        link.link_kind,
        link.resolution_status,
        link.target_document_id,
        link.target_fragment,
    ) == (kind, status, target, fragment)


def test_a_filename_shared_by_two_documents_is_ambiguous() -> None:
    inventory = [*INVENTORY, _document(FILING_ID, 4, "ex10-1.htm")]

    (link,) = _links('<a href="ex10-1.htm">Exhibit</a>', inventory)

    assert (link.resolution_status, link.target_document_id) == ("ambiguous", None)


def test_href_keeps_its_value_and_is_only_trimmed_and_nfc_normalized() -> None:
    (link,) = _links('<a href=" Ex10-1.htm?Q=A&amp;b=2#Se&#769;c ">x</a>')

    decomposed = unicodedata.normalize("NFD", "S\N{LATIN SMALL LETTER E WITH ACUTE}c")
    assert link.href == f" Ex10-1.htm?Q=A&b=2#{decomposed} "
    assert (
        link.normalized_href
        == "Ex10-1.htm?Q=A&b=2#S\N{LATIN SMALL LETTER E WITH ACUTE}c"
    )
    assert link.target_fragment == "S\N{LATIN SMALL LETTER E WITH ACUTE}c"


def test_malformed_html_still_yields_its_links() -> None:
    links = _links(
        "<div><p>Open <A HREF=ex10-1.htm>exhibit <b>ten</div>"
        "</span><a href='#x'>next</a><table><td><a href=#y>cell"
    )

    assert [(r.href, r.link_text, r.resolution_status) for r in links] == [
        ("ex10-1.htm", "exhibit ten", "resolved"),
        ("#x", "next", "resolved"),
        ("#y", "cell", "resolved"),
    ]


def test_contract_enforces_the_target_relation() -> None:
    (link,) = _links('<a href="ex10-1.htm">x</a>')
    row = link.model_dump()

    for changes, message in [
        ({"target_document_id": None}, "exactly when resolved"),
        ({"resolution_status": "unresolved"}, "exactly when resolved"),
        ({"resolution_status": "not_applicable"}, "cannot be not_applicable"),
        ({"link_kind": "fragment"}, "targets its own source"),
        ({"target_fragment": "#x"}, "must not start with '#'"),
        ({"link_id": f"{SOURCE_ID}:link:1"}, "derived from source document"),
    ]:
        with pytest.raises(ValidationError, match=message):
            DocumentLinkRecord.model_validate({**row, **changes})


def test_source_content_must_match_its_recorded_checksum(tmp_path: Path) -> None:
    content = b'<html><a href="#a">x</a></html>'
    source = _document(FILING_ID, 1, "aapl.htm", "primary_document", content)
    path = tmp_path / source.source_path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"0123456789" + content + b"0123456789")

    assert read_document_content(source, SecBronze.under(tmp_path)) == content
    path.write_bytes(b"0123456789" + content.replace(b"#a", b"#b") + b"0123456789")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        read_document_content(source, SecBronze.under(tmp_path))

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pytest

from src.data_pipeline.contracts.sec_filings.content_blocks import (
    CONTENT_BLOCKS_ARROW_SCHEMA,
    ContentBlockRecord,
)
from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.sections import (
    SECTIONS_ARROW_SCHEMA,
    SectionRecord,
)
from src.data_pipeline.quality.sec_filings.narrative import (
    narrative_problem,
    sections_problem,
)
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.storage.sec_filings.documents import read_document_content
from src.data_pipeline.transforms.sec_filings.html_blocks import (
    HtmlBlock,
    extract_blocks,
)
from src.data_pipeline.transforms.sec_filings.html_dom import (
    normalize_text,
    parse_html,
)
from src.data_pipeline.transforms.sec_filings.narrative import (
    DocumentNarrative,
    build_document_narrative,
    detect_section_headings,
)

pytestmark = pytest.mark.unit

DOCUMENT_ID = "sec:0000320193-25-000079:document:0001"
PROCESSED_AT = datetime(2026, 10, 5, 4, 0, tzinfo=UTC)
TEN_K_BODY = (
    "<div>Example Corp Annual Report</div><div>TABLE OF CONTENTS</div>"
    '<table><tr><td><a href="#p1">Part I</a></td></tr>'
    '<tr><td><a href="#i1">Item 1.</a></td><td>Business</td></tr>'
    '<tr><td><a href="#i1a">Item 1A.</a></td><td>Risk Factors</td></tr>'
    '<tr><td><a href="#p2">Part II</a></td></tr>'
    '<tr><td><a href="#i7">Item 7.</a></td><td>MD&amp;A</td></tr></table>'
    "<div>PART I</div>"
    "<div><span>Item 1. Business</span></div>"
    "<div>We make widgets.</div>"
    "<div>Item 1A. Risk Factors</div>"
    "<div>As discussed in Item 7, demand may fall.</div>"
    "<div>PART II</div>"
    "<div>Item 7. Management&#8217;s Discussion and Analysis</div>"
    "<table><tr><td>Revenue</td><td>$1,000</td></tr></table>"
    "<div>SIGNATURES</div><div>/s/ Jane Doe</div>"
)


def _html(body: str) -> bytes:
    return f"<html><head><title>t</title></head><body>{body}</body></html>".encode()


def _blocks(body: str) -> list[HtmlBlock]:
    return extract_blocks(parse_html(_html(body))).blocks


def _texts(body: str) -> list[str]:
    return [block.text for block in _blocks(body)]


def _parent(form: str = "10-K") -> DocumentRecord:
    return DocumentRecord.model_validate(
        {
            "document_id": DOCUMENT_ID,
            "filing_id": "sec:0000320193-25-000079",
            "filing_date": date(2025, 10, 31),
            "document_ordinal": 1,
            "sequence": "1",
            "document_type": form,
            "filename": "primary.htm",
            "description": form,
            "role": "primary_document",
            "content_kind": "html",
            "source_path": "bronze/sec_filings/x/full-submission.txt",
            "source_submission_sha256": "a" * 64,
            "document_start_byte": 0,
            "document_end_byte": 100,
            "content_start_byte": 10,
            "content_end_byte": 90,
            "content_size_bytes": 80,
            "content_sha256": "b" * 64,
            "processed_at": PROCESSED_AT,
        }
    )


def _narrative(body: str, form: str = "10-K") -> DocumentNarrative:
    blocks = _blocks(body)
    headings = detect_section_headings(form, blocks)
    return build_document_narrative(_parent(form), blocks, headings, PROCESSED_AT)


def _codes(body: str, form: str = "10-K") -> list[str | None]:
    return [section.section_code for section in _narrative(body, form).sections[1:]]


def test_nested_inline_tags_do_not_duplicate_text() -> None:
    body = "<div><span>Revenue <b>grew <i>10%</i></b></span> in 2025</div>"

    assert _texts(body) == ["Revenue grew 10% in 2025"]


def test_visible_inline_xbrl_is_kept_and_hidden_content_dropped() -> None:
    body = (
        '<div style="DISPLAY : None"><ix:header><ix:hidden>'
        '<ix:nonNumeric name="dei:Flag">false</ix:nonNumeric></ix:hidden></ix:header></div>'
        "<IX:HIDDEN>also hidden</IX:HIDDEN>"
        "<script>var x = 1;</script><style>p {}</style><noscript>n</noscript>"
        "<template>t</template><div hidden>h</div><div aria-hidden=' TRUE '>a</div>"
        '<div style="color:red; visibility: hidden">v</div>'
        '<div>Net income was <ix:nonFraction name="us-gaap:NetIncome">5</ix:nonFraction>'
        ' million in <ix:nonNumeric name="dei:Period">fiscal 2025</ix:nonNumeric>.</div>'
    )

    assert _texts(body) == ["Net income was 5 million in fiscal 2025."]


def test_blocks_keep_document_order_types_and_split_parent_text() -> None:
    blocks = _blocks(
        "<h2>Overview</h2><p>First paragraph.</p><ul><li>One</li><li>Two</li></ul>"
        "<pre>  code   line </pre><blockquote>Quoted.</blockquote>"
        "<div>Intro<p>Inner</p>Outro</div>"
    )

    assert [(b.text, b.block_type, b.source_tag, b.heading_level) for b in blocks] == [
        ("Overview", "heading", "h2", 2),
        ("First paragraph.", "paragraph", "p", None),
        ("One", "list_item", "li", None),
        ("Two", "list_item", "li", None),
        ("code line", "preformatted", "pre", None),
        ("Quoted.", "paragraph", "blockquote", None),
        ("Intro", "paragraph", "div", None),
        ("Inner", "paragraph", "p", None),
        ("Outro", "paragraph", "div", None),
    ]


def test_each_table_is_one_block_and_nested_tables_are_their_own() -> None:
    body = (
        "<div>Before</div><table><tr><th>Item</th><td></td><td><p>$1,000</p>"
        "<table><tr><td>inner</td></tr></table></td></tr></table><div>After</div>"
    )

    blocks = _blocks(body)

    assert [
        (b.text, b.block_type, b.source_tag, b.source_xpath, b.table_xpath)
        for b in blocks
    ] == [
        ("Before", "paragraph", "div", "/html/body/div[1]", None),
        ("Item\n$1,000", "table", "table", "/html/body/table", None),
        (
            "inner",
            "table",
            "table",
            "/html/body/table/tr/td[2]/table",
            "/html/body/table",
        ),
        ("After", "paragraph", "div", "/html/body/div[2]", None),
    ]
    assert all(b.block_type != "table" or b.heading_level is None for b in blocks)
    assert _blocks(body) == blocks


def test_table_with_only_empty_cells_is_an_empty_text_block() -> None:
    narrative = _narrative(
        "<div>Text.</div><table><tr><td> </td><td></td></tr></table>"
    )

    table = narrative.blocks[1]
    assert (table.block_type, table.text, table.text_length) == ("table", "", 0)
    assert narrative_problem(_parent(), narrative.sections, narrative.blocks) is None


def test_table_markup_without_cells_creates_no_table_block() -> None:
    extracted = extract_blocks(
        parse_html(_html("<table><tr></tr></table><table></table><div>x</div>"))
    )

    assert [b.block_type for b in extracted.blocks] == ["paragraph"]
    assert extracted.empty_tables == 2


def test_normalize_text_collapses_spacing_and_keeps_content() -> None:
    raw = "\u00a0 Net\u200b  Income\n\t$ (1,234) \u2014 -5% cafe\u0301 \ufeff"

    assert normalize_text(raw) == "Net Income $ (1,234) \u2014 -5% caf\u00e9"


def test_empty_and_spacer_elements_create_no_blocks() -> None:
    assert _texts("<div>&#160;</div><div> </div><p></p><div><span> </span></div>") == []


def test_parse_html_skips_edgar_wrapper_and_rejects_empty_input() -> None:
    wrapped = (
        b"\n<XBRL>\n<?xml version='1.0'?>\n" + _html("<div>Body</div>") + b"\n</XBRL>\n"
    )

    assert [block.text for block in extract_blocks(parse_html(wrapped)).blocks] == [
        "Body"
    ]
    with pytest.raises(ValueError, match="HTML document is empty"):
        parse_html(b"   ")


def test_ten_k_sections_skip_toc_links_and_narrative_mentions() -> None:
    assert _codes(TEN_K_BODY) == [
        "TABLE OF CONTENTS",
        "PART I",
        "ITEM 1",
        "ITEM 1A",
        "PART II",
        "ITEM 7",
        "SIGNATURES",
    ]


def test_toc_table_lists_codes_but_single_heading_tables_count() -> None:
    body = (
        "<table><tr><td>PART I</td></tr><tr><td>Item 1.</td><td>Business</td></tr>"
        "<tr><td>Item 1A.</td><td>Risk Factors</td></tr></table>"
        "<div>PART I</div><table><tr><td>Item 1.</td><td>Business</td></tr></table>"
        "<div>Text.</div><table><tr><td>Item 1A.</td><td>Risk Factors</td></tr></table>"
    )

    narrative = _narrative(body)

    assert [(s.section_code, s.title) for s in narrative.sections[1:]] == [
        ("PART I", "PART I"),
        ("ITEM 1", "Item 1. Business"),
        ("ITEM 1A", "Item 1A. Risk Factors"),
    ]
    assert narrative.blocks[2].text == "Item 1.\nBusiness"
    assert [s.start_block_ordinal for s in narrative.sections[1:]] == [2, 3, 5]
    assert [b.block_type for b in narrative.blocks] == [
        "table",
        "heading",
        "table",
        "paragraph",
        "table",
    ]


def test_running_page_headers_do_not_create_sections() -> None:
    body = (
        "<p>PART I</p><p>Item 1</p><p>Cover text.</p>"
        "<p>PART I</p><p>ITEM 1. BUSINESS</p><p>We sell.</p>"
        "<p>PART I</p><p>Item 1</p><p>More.</p>"
        "<p>Item 1B, 2, 3, 4</p><p>ITEM 1B. UNRESOLVED STAFF COMMENTS</p>"
    )

    narrative = _narrative(body)

    assert [(s.section_code, s.title) for s in narrative.sections[1:]] == [
        ("PART I", "PART I"),
        ("ITEM 1", "ITEM 1. BUSINESS"),
        ("ITEM 1B", "ITEM 1B. UNRESOLVED STAFF COMMENTS"),
    ]
    assert narrative.duplicate_section_codes == 0


def test_ten_q_items_repeat_under_each_part() -> None:
    body = (
        "<div>PART I &#8212; FINANCIAL INFORMATION</div><div>Item 1. Financial Statements</div>"
        "<div>Item 2. MD&amp;A</div><div>PART II. Other Information</div>"
        "<div>Item 1. Legal Proceedings</div><div>Item 1A. Risk Factors</div>"
    )

    sections = _narrative(body, "10-Q").sections
    by_id = {s.section_id: s for s in sections}

    assert [
        (
            s.section_code,
            s.level,
            by_id[s.parent_section_id or s.section_id].section_code,
        )
        for s in sections[1:]
    ] == [
        ("PART I", 1, None),
        ("ITEM 1", 2, "PART I"),
        ("ITEM 2", 2, "PART I"),
        ("PART II", 1, None),
        ("ITEM 1", 2, "PART II"),
        ("ITEM 1A", 2, "PART II"),
    ]


def test_eight_k_detects_decimal_items_signature_and_exhibit_index() -> None:
    body = (
        "<div>Item 2.02 Results of Operations and Financial Condition.</div>"
        "<div>Results.</div><div>ITEM 9.01. FINANCIAL STATEMENTS AND EXHIBITS.</div>"
        "<div>Index to Exhibits</div><table><tr><td>99.1</td><td>Press release</td></tr></table>"
        "<div>Signature</div><div>/s/ Jane Doe</div>"
    )

    narrative = _narrative(body, "8-K")

    assert [(s.section_kind, s.section_code, s.level) for s in narrative.sections] == [
        ("document", None, 0),
        ("item", "ITEM 2.02", 1),
        ("item", "ITEM 9.01", 1),
        ("exhibit_index", "EXHIBIT INDEX", 1),
        ("signature", "SIGNATURES", 1),
    ]
    # 8-Ks have no PARTs: items, the exhibit index and signatures hang off root.
    root = narrative.sections[0].section_id
    assert {s.parent_section_id for s in narrative.sections[1:]} == {root}
    assert _codes("<div>Item 2.02 Results of Operations.</div>") == []


@pytest.mark.parametrize(
    "text",
    [
        "As discussed in Item 1A, the company faces risks.",
        "Item 1A of this report describes our risks.",
        "See Item 7 for additional information.",
        "Part II of this report",
    ],
)
def test_narrative_item_mentions_are_not_sections(text: str) -> None:
    assert _codes(f"<div>{text}</div>") == []


def test_ranges_hierarchy_and_most_specific_assignment() -> None:
    narrative = _narrative(TEN_K_BODY)
    sections = {s.section_code: s for s in narrative.sections}
    owners = {b.block_ordinal: b.section_id for b in narrative.blocks}

    assert [
        (s.section_code, s.level, s.start_block_ordinal, s.end_block_ordinal)
        for s in narrative.sections
    ] == [
        (None, 0, 1, 14),
        ("TABLE OF CONTENTS", 1, 2, 4),
        ("PART I", 1, 4, 9),
        ("ITEM 1", 2, 5, 7),
        ("ITEM 1A", 2, 7, 9),
        ("PART II", 1, 9, 12),
        ("ITEM 7", 2, 10, 12),
        ("SIGNATURES", 1, 12, 14),
    ]
    for section in narrative.sections[1:]:
        assert owners[section.start_block_ordinal] == section.section_id
    assert owners[6] == sections["ITEM 1"].section_id
    assert owners[1] == sections[None].section_id
    assert narrative.blocks[5].text == "We make widgets."
    assert [
        b.block_type for b in narrative.blocks if b.block_ordinal in (3, 4, 5, 11)
    ] == [
        "table",
        "heading",
        "heading",
        "table",
    ]
    assert narrative.blocks[10].text == "Revenue\n$1,000"


def test_parent_part_hash_covers_child_items_and_counts_are_declared() -> None:
    narrative = _narrative(TEN_K_BODY)
    texts = [block.text for block in narrative.blocks]
    part_one = narrative.sections[2]

    assert part_one.section_code == "PART I"
    assert (
        part_one.section_text_sha256
        == hashlib.sha256("\n".join(texts[3:8]).encode()).hexdigest()
    )
    assert not any(
        block.section_id == part_one.section_id for block in narrative.blocks[4:8]
    )
    assert {s.document_section_count for s in narrative.sections} == {8}
    assert {b.document_block_count for b in narrative.blocks} == {13}
    assert sections_problem(_parent(), narrative.sections) is None
    assert narrative_problem(_parent(), narrative.sections, narrative.blocks) is None


def test_root_can_own_no_block_and_root_only_documents_are_valid() -> None:
    eight_k = _narrative("<div>Item 8.01 Other Events.</div><div>Text.</div>", "8-K")
    root_only = _narrative("<div>Just a letter.</div><p>Thanks.</p>", "8-K")

    assert {b.section_id for b in eight_k.blocks} == {eight_k.sections[1].section_id}
    assert narrative_problem(_parent("8-K"), eight_k.sections, eight_k.blocks) is None
    assert len(root_only.sections) == 1
    assert root_only.sections[0].title == "8-K Document"
    assert {b.section_id for b in root_only.blocks} == {
        root_only.sections[0].section_id
    }
    assert (
        narrative_problem(_parent("8-K"), root_only.sections, root_only.blocks) is None
    )


def test_ids_are_stable_and_no_block_lacks_a_section() -> None:
    first, second = _narrative(TEN_K_BODY), _narrative(TEN_K_BODY)

    assert first == second
    assert first.sections[0].section_id == f"{DOCUMENT_ID}:section:0001"
    assert first.blocks[0].block_id == f"{DOCUMENT_ID}:block:000001"
    section_ids = {section.section_id for section in first.sections}
    assert all(block.section_id in section_ids for block in first.blocks)
    assert all(block.document_id == DOCUMENT_ID for block in first.blocks)


def test_document_without_visible_text_fails() -> None:
    with pytest.raises(ValueError, match="no visible text blocks"):
        _narrative('<div style="display:none">hidden</div>')


def test_truncated_and_inconsistent_groups_are_detected() -> None:
    narrative = _narrative(TEN_K_BODY)
    parent = _parent()
    wrong_parent = narrative.sections[3].model_copy(
        update={"parent_section_id": narrative.sections[5].section_id}
    )

    assert "declared counts" in str(
        narrative_problem(parent, narrative.sections, narrative.blocks[:-1])
    )
    assert "declared counts" in str(sections_problem(parent, narrative.sections[:-1]))
    assert "range lies outside" in str(
        sections_problem(
            parent, [*narrative.sections[:3], wrong_parent, *narrative.sections[4:]]
        )
    )
    item_one, part_one = narrative.sections[3], narrative.sections[2]
    other_document = item_one.model_copy(
        update={
            "parent_section_id": f"{_parent().filing_id}:document:0002:section:0001"
        }
    )
    self_parent = item_one.model_copy(update={"parent_section_id": item_one.section_id})
    cycle = part_one.model_copy(update={"parent_section_id": item_one.section_id})
    for broken, problem in (
        ((other_document,), "parent is not a section of this document"),
        ((self_parent,), "level is not below its parent's"),
        ((cycle, item_one), "level is not below its parent's"),
    ):
        replaced = {section.section_id: section for section in broken}
        sections = [replaced.get(s.section_id, s) for s in narrative.sections]
        assert problem in str(sections_problem(parent, sections))
    other_source = parent.model_copy(update={"content_sha256": "c" * 64})
    assert "differs from the parent" in str(
        sections_problem(other_source, narrative.sections)
    )


def test_records_match_exact_arrow_schemas() -> None:
    narrative = _narrative(TEN_K_BODY)

    sections = pa.Table.from_pylist(
        [s.model_dump() for s in narrative.sections], schema=SECTIONS_ARROW_SCHEMA
    )
    blocks = pa.Table.from_pylist(
        [b.model_dump() for b in narrative.blocks], schema=CONTENT_BLOCKS_ARROW_SCHEMA
    )

    assert list(SectionRecord.model_fields) == SECTIONS_ARROW_SCHEMA.names
    assert list(ContentBlockRecord.model_fields) == CONTENT_BLOCKS_ARROW_SCHEMA.names
    assert sections.schema.equals(SECTIONS_ARROW_SCHEMA)
    assert blocks.schema.equals(CONTENT_BLOCKS_ARROW_SCHEMA)


def test_read_document_content_checks_offsets_size_and_checksum(tmp_path: Path) -> None:
    bronze = SecBronze(tmp_path / "moved")
    source = bronze.root / "full-submission.txt"
    source.parent.mkdir()
    source.write_bytes(b"0123456789<html>body</html>tail")
    content = b"<html>body</html>"
    parent = _parent().model_copy(
        update={
            "source_path": "bronze/sec_filings/full-submission.txt",
            "content_start_byte": 10,
            "content_end_byte": 27,
            "content_size_bytes": 17,
            "content_sha256": hashlib.sha256(content).hexdigest(),
        }
    )

    assert read_document_content(parent, bronze) == content
    with pytest.raises(ValueError, match="outside"):
        read_document_content(
            parent.model_copy(update={"content_end_byte": 99}), bronze
        )
    with pytest.raises(ValueError, match="SHA-256 differs"):
        read_document_content(
            parent.model_copy(update={"content_sha256": "0" * 64}), bronze
        )
    with pytest.raises(ValueError, match="escapes"):
        read_document_content(
            parent.model_copy(update={"source_path": "bronze/sec_filings/../x.txt"}),
            bronze,
        )
    with pytest.raises(ValueError, match="Not an SEC Bronze path"):
        read_document_content(
            parent.model_copy(update={"source_path": "elsewhere/x.txt"}), bronze
        )

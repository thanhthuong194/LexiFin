from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest

from src.data_pipeline.transforms.native_rag.blocks import SourceBlock, SourceSection
from src.data_pipeline.transforms.native_rag.context import (
    FilingContext,
    chunk_context,
    period_label,
    short_title,
)
from src.data_pipeline.transforms.native_rag.noise import drop_reasons
from src.data_pipeline.transforms.native_rag.section_topics import (
    SectionHead,
    section_topic,
)
from src.data_pipeline.transforms.native_rag.sentences import split_sentences
from src.data_pipeline.transforms.native_rag.tokenizer import (
    TOKENIZER_SHA256,
    QwenTokenCounter,
    tokenizer_path,
)

pytestmark = pytest.mark.unit

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Revenue rose 6.5% to $1.2 billion. Apple Inc. sells phones. Margins fell.",
            [
                "Revenue rose 6.5% to $1.2 billion. ",
                "Apple Inc. sells phones. ",
                "Margins fell.",
            ],
        ),
        (
            "Sales in the U.S. Grew, e.g. Retail. See Note 4. It was “strong.” Next one.",
            [
                "Sales in the U.S. Grew, e.g. Retail. ",
                "See Note 4. ",
                "It was “strong.” ",
                "Next one.",
            ],
        ),
        ("J. Smith joined. He leads.", ["J. Smith joined. ", "He leads."]),
        ("no boundary here", ["no boundary here"]),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected
    assert "".join(split_sentences(text)) == text


@pytest.mark.parametrize(
    ("form", "document_type", "head", "topic"),
    [
        (
            "10-K",
            "10-K",
            SectionHead("item", "ITEM 7", "Item 7. MD&A", "PART II"),
            "mdna",
        ),
        (
            "10-K",
            "10-K",
            SectionHead("item", "ITEM 1A", "Risk Factors"),
            "risk_factors",
        ),
        ("10-K", "10-K", SectionHead("item", "ITEM 9B", "Other"), "other"),
        ("10-Q", "10-Q", SectionHead("item", "ITEM 2", "MD&A", "PART I"), "mdna"),
        (
            "10-Q",
            "10-Q",
            SectionHead("item", "ITEM 2", "Unregistered", "PART II"),
            "market_equity",
        ),
        (
            "10-Q",
            "10-Q",
            SectionHead("item", "ITEM 1", "Legal Proceedings"),
            "legal_proceedings",
        ),
        ("8-K", "8-K", SectionHead("item", "ITEM 2.02", "Results"), "8k_event"),
        ("8-K", "8-K", SectionHead("item", "ITEM 9.01", "Exhibits"), "exhibits"),
        ("8-K", "EX-99.1", SectionHead("document", None, "EX-99.1"), "ex99"),
        ("10-K", "10-K", SectionHead("document", None, "10-K Document"), "cover"),
        (
            "10-K",
            "10-K",
            SectionHead("signature", "SIGNATURES", "SIGNATURES"),
            "signatures",
        ),
        ("10-K", "10-K", SectionHead("part", "PART I", "PART I"), "other"),
    ],
)
def test_section_topic(
    form: str, document_type: str, head: SectionHead, topic: str
) -> None:
    assert section_topic(form, document_type, head) == topic


def _section(
    section_id: str, kind: str, start: int, parent: str | None = "root"
) -> SourceSection:
    return SourceSection(section_id, 1, parent, kind, None, kind, start, (), "other")


def test_drop_reasons() -> None:
    sections = {
        "root": _section("root", "document", 1, None),
        "toc": _section("toc", "table_of_contents", 2),
        "a": _section("a", "item", 4),
        "b": _section("b", "item", 20),
        "c": _section("c", "item", 30),
    }
    texts = [
        ("toc", "Item 1. Business 3"),
        ("toc", "This report contains forward-looking statements " * 5),
        ("toc", "x"),
        ("a", "Item 1. Business"),  # opens section a
        ("a", "12"),
        ("a", "Page 3"),
        ("a", "F-12"),
        ("a", "xii"),
        ("a", "23."),
        ("a", "* * *"),
        ("a", "None."),
        ("a", "Mild"),
        ("a", "Example Inc. | 2025 Form 10-K | 4"),
        ("b", "Example Inc. | 2025 Form 10-K | 5"),
        ("b", "None."),
        ("c", "Example Inc. | 2025 Form 10-K | 6"),
        ("c", "Example Inc. | 2025 Form 10-K | 7"),
        ("c", "None."),
        ("c", "None."),
    ]
    # Section a opens at its heading block (ordinal 4); others start elsewhere.
    blocks = [
        SourceBlock(f"b{i}", 4 if i == 3 else 100 + i, section, "paragraph", text)
        for i, (section, text) in enumerate(texts)
    ]
    reasons = drop_reasons(blocks, sections)
    by_text = {texts[int(k[1:])][1]: v for k, v in reasons.items()}
    assert by_text["Item 1. Business 3"] == "table_of_contents"
    assert by_text["x"] == "page_number"  # a TOC section keeps only non-entries
    assert by_text["Item 1. Business"] == "section_title"
    for page in ("12", "Page 3", "F-12", "xii", "23."):
        assert by_text[page] == "page_number"
    assert by_text["* * *"] == "decoration"
    assert by_text["Example Inc. | 2025 Form 10-K | 4"] == "running_header"
    kept = {texts[int(b.block_id[1:])][1] for b in blocks if b.block_id not in reasons}
    assert kept == {
        "This report contains forward-looking statements " * 5,
        "None.",
        "Mild",
    }


def test_context_and_period() -> None:
    ten_q = FilingContext(
        "Apple Inc.", "10-Q", date(2025, 5, 2), date(2025, 3, 29), 2025, "Q2"
    )
    assert period_label(ten_q) == "FY2025 Q2 (period ended 2025-03-29)"
    ten_k = FilingContext(
        "Apple Inc.", "10-K", date(2025, 11, 1), date(2025, 9, 27), 2025, "FY"
    )
    assert period_label(ten_k) == "FY2025 (period ended 2025-09-27)"
    eight_k = FilingContext(
        "Apple Inc.", "8-K", date(2025, 1, 30), date(2025, 1, 28), None, None
    )
    assert period_label(eight_k) == "filed 2025-01-30 (event 2025-01-28)"
    assert (
        chunk_context(
            ten_k, ("PART II", "Item 7. MD&A"), subheading="Net Sales", table="Sales"
        )
        == "Apple Inc. | 10-K | FY2025 (period ended 2025-09-27) | PART II › Item 7. MD&A › "
        "Net Sales | Table: Sales"
    )
    assert chunk_context(ten_k, (), table="").endswith("| Table")
    long_title = "Item 7. " + "word " * 30
    assert len(short_title(long_title)) <= 80
    assert short_title(long_title).endswith("…")


def test_tokenizer_file_is_verified(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="curl"):
        QwenTokenCounter(tmp_path / "missing.json")
    wrong = tmp_path / "tokenizer.json"
    wrong.write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        QwenTokenCounter(wrong)


def test_qwen_tokenizer_counts_and_splits() -> None:
    path = tokenizer_path(REPOSITORY_ROOT / "data")
    if (
        not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != TOKENIZER_SHA256
    ):
        pytest.skip("pinned tokenizer file not downloaded")
    counter = QwenTokenCounter(path)
    text = "Net sales increased 6% to $391,035 million. " * 20
    assert counter.count("Net sales") == 3  # two tokens + <|endoftext|>
    pieces = counter.split(text, 50)
    assert "".join(pieces) == text
    assert all(counter.count(piece) - 1 <= 50 for piece in pieces)


def test_toc_tables_and_long_opening_blocks_are_not_noise() -> None:
    long_title = (
        "Item 2.02. Results of Operations. "
        + "The information here is furnished. " * 10
    )
    sections = {
        "root": _section("root", "document", 1, None),
        "toc": _section("toc", "table_of_contents", 2),
        "item": SourceSection(
            "item",
            3,
            "root",
            "item",
            "ITEM 2.02",
            long_title,
            10,
            ("Item",),
            "8k_event",
        ),
    }
    blocks = [
        SourceBlock("t", 3, "toc", "table", "Net earnings\n1,234\n1,100"),
        SourceBlock("o", 10, "item", "table", long_title),
    ]
    assert drop_reasons(blocks, sections) == {}


def test_toc_section_drops_only_toc_entries() -> None:
    sections = {
        "root": _section("root", "document", 1, None),
        "toc": _section("toc", "table_of_contents", 2),
    }
    lines = [
        "Item 1. Business 3",
        "Table of Contents",
        "Hurricane Ida",
        "Operating Revenues",
    ]
    lines += [f"Subheading {chr(65 + i % 26)}{i // 26}x" for i in range(110)]
    blocks = [
        SourceBlock(f"b{i}", 10 + i, "toc", "paragraph", t) for i, t in enumerate(lines)
    ]
    reasons = drop_reasons(blocks, sections)
    assert set(reasons) == {"b0", "b1"}
    # Short prose before the first item (v5 dropped it in TOCs of up to 100
    # short lines): forward-looking bullets, "Where You Can Find..." stay.
    small = drop_reasons(
        [*blocks[:2], SourceBlock("w", 99, "toc", "paragraph",
                                  "•fluctuations in our effective tax rate;")],
        sections,
    )  # fmt: skip
    assert set(small) == {"b0", "b1"}


def test_running_headers_need_page_breaks_or_a_counter() -> None:
    sections = {
        "root": _section("root", "document", 1, None),
        **{s: _section(s, "item", 1000 + i) for i, s in enumerate("abc")},
    }
    texts: list[tuple[str, str]] = []
    for page, section in enumerate("aabbcc", start=10):
        body = [(section, "Body text of the page goes on for a while here.")] * 4
        texts += [
            *body,
            (section, "US, Canada - Named Storm"),  # a table row label, mid-page
            *body,
            (section, "Example Corp and Subsidiaries"),  # page header
            (section, str(page)),
            (section, f"Example Corp | Form 10-K | {page}"),  # page footer
        ]
    blocks = [
        SourceBlock(f"b{i}", 100 + i, section, "paragraph", text)
        for i, (section, text) in enumerate(texts)
    ]
    reasons = drop_reasons(blocks, sections)
    dropped = {
        texts[int(k[1:])][1] for k, r in reasons.items() if r == "running_header"
    }
    assert "Example Corp and Subsidiaries" in dropped
    assert any(t.startswith("Example Corp | Form 10-K") for t in dropped)
    assert "US, Canada - Named Storm" not in dropped


def test_repeated_subheading_away_from_page_breaks_stays() -> None:
    sections = {
        "root": _section("root", "document", 1, None),
        **{s: _section(s, "item", 1000 + i) for i, s in enumerate("abc")},
    }
    texts: list[tuple[str, str]] = []
    for page, section in enumerate("aabbccaabb", start=10):
        texts += [
            (section, "Body text of the page goes on for a while here."),
            (section, "More Personal Computing"),  # a segment heading
            (section, "(a)"),
            (section, "Our segment consists of devices and gaming."),
            (section, "Body text of the page goes on for a while here, again."),
            (section, str(page)),
        ]
    blocks = [
        SourceBlock(f"b{i}", 10 + i, s, "paragraph", t)
        for i, (s, t) in enumerate(texts)
    ]
    reasons = drop_reasons(blocks, sections)
    dropped = {
        texts[int(k[1:])][1] for k, v in reasons.items() if v == "running_header"
    }
    # Within 3 blocks of a page number, but real text lies between (v5
    # dropped both); markers never count.
    assert dropped == set()

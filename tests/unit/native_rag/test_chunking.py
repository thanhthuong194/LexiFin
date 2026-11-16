from __future__ import annotations

from datetime import UTC, date, datetime
from itertools import pairwise

import pytest

from src.data_pipeline.contracts.native_rag.chunks import (
    ChunkRecord,
    chunk_id_for,
    chunk_sha256,
)
from src.data_pipeline.transforms.native_rag.blocks import SourceBlock, SourceSection
from src.data_pipeline.transforms.native_rag.context import embedded_text
from src.data_pipeline.transforms.native_rag.document_chunks import (
    ChunkParams,
    DocumentInput,
    chunk_document,
)
from src.data_pipeline.transforms.native_rag.table_chunks import (
    Cell,
    RenderedTable,
    classify_table,
    is_value,
    layout_paragraphs,
    render_table,
    split_rows,
)
from src.data_pipeline.transforms.native_rag.text_chunks import (
    Paragraph,
    Sizing,
    chunk_paragraphs,
    is_title_like,
)
from tests.native_rag_support import WordCounter

pytestmark = pytest.mark.unit

COUNTER = WordCounter()


def _sizing(cap: int, min_tokens: int = 4, prefix_words: int = 2) -> Sizing:
    def fits(subheading: str | None, text: str) -> bool:
        return COUNTER.count(text) + prefix_words <= cap

    return Sizing(
        size=lambda text: len(text.split()),
        fits=fits,
        budget=lambda subheading: cap - prefix_words - 1,
        split_tokens=COUNTER.split,
        min_tokens=min_tokens,
    )


def _paragraphs(*texts: str) -> list[Paragraph]:
    return [Paragraph(f"b{i}", t, is_title_like(t)) for i, t in enumerate(texts)]


def test_value_cells_exclude_years_and_dates() -> None:
    for value in ("1,234", "(1,234)", "$", "12.5", "—", "39%", "(3)", "$1,000"):
        assert is_value(value) or value == "$", value
    for label in ("2025", "FY 2025", "March 31, 2025", "Apr 27, 2025", "Revenue", ""):
        assert not is_value(label), label


def test_title_like() -> None:
    assert is_title_like("Gross Margin")
    assert is_title_like("Amazon.com, Inc.")
    assert not is_title_like("Net sales increased 6%.")
    assert not is_title_like("The following table shows net sales:")
    assert not is_title_like("12")
    assert not is_title_like("word " * 13)


def _apple_table() -> list[Cell]:
    # Header row in columns 3..5 (each spanning 2 grid columns), a group
    # label row, and value rows where `$` and `%` sit in their own cells.
    rows = [
        [(3, 2, "2021"), (5, 2, "Change"), (7, 2, "2020")],
        [(1, 1, "Net sales by category:")],
        [(1, 1, "iPhone"), (3, 1, "$"), (4, 1, "191,973"), (5, 1, "39"), (6, 1, "%"),
         (7, 1, "$"), (8, 1, "137,781")],
        [(1, 1, "Mac"), (4, 1, "35,190"), (5, 1, "23"), (6, 1, "%"), (8, 1, "28,622")],
        [(1, 1, ""), (2, 1, "")],
    ]  # fmt: skip
    return [
        Cell(row, column, span, text)
        for row, cells in enumerate(rows, start=1)
        for column, span, text in cells
    ]


def test_render_data_table() -> None:
    cells = _apple_table()
    assert classify_table(cells) == "data"
    table = render_table(cells)
    assert table.header == (" | 2021 | Change | 2020",)
    assert table.rows == (
        "Net sales by category:",
        "iPhone | $191,973 | 39% | $137,781",
        "Mac | 35,190 | 23% | 28,622",
    )


def test_render_multi_level_header_and_units() -> None:
    cells = [
        Cell(1, 3, 4, "Three Months Ended"),
        Cell(2, 3, 2, "Apr 27, 2025"),
        Cell(2, 5, 2, "Apr 28, 2024"),
        Cell(3, 3, 4, "($ in millions)"),
        Cell(4, 1, 2, "R&D"),
        Cell(4, 3, 1, "$"),
        Cell(4, 4, 1, "3,989"),
        Cell(4, 5, 1, "$"),
        Cell(4, 6, 1, "2,720"),
    ]
    table = render_table(cells)
    assert table.header == (
        "($ in millions)",
        " | Three Months Ended Apr 27, 2025 | Three Months Ended Apr 28, 2024",
    )
    assert table.rows == ("R&D | $3,989 | $2,720",)


def test_classify_layout_toc_signature() -> None:
    bullets = [
        Cell(1, 1, 1, "•"),
        Cell(1, 2, 1, "integration risk;"),
        Cell(2, 2, 1, "x"),
    ]
    assert classify_table(bullets) == "layout"
    assert layout_paragraphs(bullets) == ["• integration risk;", "x"]
    toc = [
        Cell(r, c, 1, t)
        for r, (a, b) in enumerate(
            [
                ("Item 1.", "Business 3"),
                ("Item 1A.", "Risk Factors 9"),
                ("Item 2.", "Properties 20"),
            ],
            start=1,
        )
        for c, t in ((1, a), (2, b))
    ]
    assert classify_table(toc) == "toc"
    signature = [
        Cell(1, 1, 1, "By:"),
        Cell(1, 2, 1, "/s/ Jane Doe"),
        Cell(1, 3, 1, "2025"),
        Cell(2, 3, 1, "100"),
    ]
    assert classify_table(signature) == "signature"


def test_split_rows_repeats_header() -> None:
    table = RenderedTable(
        (" | 2025 | 2024",), tuple(f"row{i} | {i} | {i}" for i in range(9))
    )

    def fits(text: str) -> bool:
        return COUNTER.count(text) <= 25

    parts = split_rows(table, "Title", fits)
    assert len(parts) > 1
    assert all(p.startswith("Title\n | 2025 | 2024\n") for p in parts)
    assert all(fits(p) for p in parts)
    rows = [line for p in parts for line in p.splitlines()[2:]]
    assert rows == list(table.rows)


def test_units_merge_short_and_split_balanced() -> None:
    paragraphs = _paragraphs(
        "Fiscal Highlights",  # heading only: merges forward
        "Sales grew strongly this year across all regions and products.",
        "Gross Margin",
        " ".join(["margin"] * 30),
        " ".join(["costs"] * 30),
        "Outlook",
        "Short.",  # last unit, short: merges backward
    )
    chunks = chunk_paragraphs(paragraphs, None, _sizing(cap=40))
    assert chunks[0].subheading == "Fiscal Highlights"
    assert chunks[0].text.startswith("Fiscal Highlights\nSales grew")
    margin = [c for c in chunks if c.subheading == "Gross Margin"]
    assert len(margin) == 2  # 62 words split into two balanced parts
    assert margin[0].block_ids == ("b2", "b3") and margin[1].block_ids == ("b4",)
    assert chunks[-1].text.endswith("Outlook\nShort.")
    covered = [b for c in chunks for b in c.block_ids]
    assert covered == [p.block_id for p in paragraphs]


def test_long_paragraph_split_by_sentences_with_overlap() -> None:
    sentences = [f"Sentence {i} has exactly six words." for i in range(12)]
    paragraph = Paragraph("p", " ".join(sentences), False)
    chunks = chunk_paragraphs([paragraph], "Risks", _sizing(cap=30))
    assert len(chunks) > 1
    assert {c.block_ids for c in chunks} == {("p",)}
    assert [c.part for c in chunks] == list(range(1, len(chunks) + 1))
    assert {c.part_count for c in chunks} == {len(chunks)}
    for previous, current in pairwise(chunks):
        last_sentence = previous.text.rsplit("Sentence", 1)[1]
        assert current.text.startswith("Sentence" + last_sentence)
    assert all(COUNTER.count(c.text) + 2 <= 30 for c in chunks)


def _document() -> DocumentInput:
    sections = {
        "root": SourceSection(
            "root", 1, None, "document", None, "Doc", 1, ("Cover page",), "cover"
        ),
        "s7": SourceSection(
            "s7",
            2,
            "root",
            "item",
            "ITEM 7",
            "Item 7. MD&A",
            2,
            ("Item 7. MD&A",),
            "mdna",
        ),
    }
    texts = [
        (
            "root",
            "paragraph",
            "Example Corp annual report cover text with several words here.",
        ),
        ("s7", "paragraph", "Item 7. MD&A"),
        ("s7", "paragraph", "Net Sales"),
        ("s7", "paragraph", "The following table shows net sales (in millions):"),
        ("s7", "table", "2025\n2024\nProducts\n100\n90"),
        ("s7", "paragraph", "(1) Includes services."),
        (
            "s7",
            "paragraph",
            "Sales rose because demand for products increased in every region.",
        ),
        ("s7", "paragraph", "7"),
    ]
    blocks = [
        SourceBlock(f"d:block:{i:06d}", i, section, kind, text)
        for i, (section, kind, text) in enumerate(texts, start=1)
    ]
    cells = {
        "d:block:000005:table": [
            Cell(1, 2, 1, "2025"),
            Cell(1, 3, 1, "2024"),
            Cell(2, 1, 1, "Products"),
            Cell(2, 2, 1, "100"),
            Cell(2, 3, 1, "90"),
        ]
    }

    def context(
        path: tuple[str, ...], subheading: str | None, table: str | None
    ) -> str:
        parts = ["Example", *path, *([subheading] if subheading else [])]
        if table is not None:
            parts.append(f"Table: {table}")
        return " | ".join(parts)

    return DocumentInput(blocks, sections, cells, context)


def test_chunk_document_covers_every_block() -> None:
    document = _document()
    params = ChunkParams(max_tokens=60, min_unit_tokens=4, small_table_chars=5)
    result = chunk_document(document, COUNTER, params)
    kinds = [c.chunk_kind for c in result.chunks]
    assert kinds == ["text", "table", "text"]
    table = result.chunks[1]
    assert (
        table.table_title
        == "Net Sales The following table shows net sales (in millions):"
    )
    assert table.subheading == "Net Sales"
    assert table.text == " | 2025 | 2024\nProducts | 100 | 90\n(1) Includes services."
    assert table.block_ids == (
        "d:block:000003",
        "d:block:000004",
        "d:block:000005",
        "d:block:000006",
    )
    assert result.chunks[2].subheading == "Net Sales"
    assert result.dropped == {
        "d:block:000002": "section_title",
        "d:block:000008": "page_number",
    }
    placed = [b for c in result.chunks for b in c.block_ids] + list(result.dropped)
    assert sorted(placed) == sorted(b.block_id for b in document.blocks)
    for chunk in result.chunks:
        assert chunk.token_count == COUNTER.count(
            embedded_text(chunk.context, chunk.text)
        )
        assert chunk.token_count <= params.max_tokens


def _record(**changes: object) -> dict[str, object]:
    context, text = "Example | 10-K", "Body text"
    row: dict[str, object] = {
        "chunk_id": chunk_id_for("doc", 1),
        "filing_id": "sec:1",
        "document_id": "doc",
        "section_id": "doc:section:0001",
        "table_id": None,
        "block_ids": ["doc:block:000001"],
        "filing_date": date(2025, 1, 2),
        "cik": "0000320193",
        "tickers": ["AAPL"],
        "form": "10-K",
        "fiscal_year": 2024,
        "fiscal_period": "FY",
        "period_end_date": None,
        "section_topic": "mdna",
        "document_type": "10-K",
        "chunk_kind": "text",
        "company_name": "Apple Inc.",
        "accession_number": "0000320193-25-000001",
        "sec_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000001/a.htm",
        "section_title": "Item 7",
        "subheading": None,
        "table_title": None,
        "chunk_ordinal": 1,
        "document_chunk_count": 1,
        "part": None,
        "part_count": None,
        "context": context,
        "text": text,
        "token_count": 5,
        "chunk_sha256": chunk_sha256(context, text),
        "processed_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    row.update(changes)
    return row


def test_chunk_contract() -> None:
    ChunkRecord.model_validate(_record())
    for changes in (
        {"chunk_id": "doc:chunk:1"},
        {"chunk_ordinal": 2},
        {"part": 1},
        {"part": 3, "part_count": 2},
        {"chunk_kind": "table"},
        {"text": "changed"},
        {"section_topic": "unknown"},
    ):
        with pytest.raises(ValueError):
            ChunkRecord.model_validate(_record(**changes))


def test_small_number_data_table_is_not_toc() -> None:
    cells = [
        Cell(1, 1, 1, "Cash"),
        Cell(1, 2, 1, "$"),
        Cell(1, 3, 1, "512"),
        Cell(2, 1, 1, "Receivables"),
        Cell(2, 3, 1, "120"),
        Cell(3, 1, 1, "Total"),
        Cell(3, 3, 1, "632"),
    ]
    assert classify_table(cells) == "data"
    index = [
        Cell(r, c, 1, t)
        for r, (a, b) in enumerate(
            [
                ("Consolidated Statements of Operations", "45"),
                ("Consolidated Balance Sheets and Notes", "47"),
                ("Consolidated Statements of Cash Flows", "49"),
            ],
            start=1,
        )
        for c, t in ((1, a), (2, b))
    ]
    assert classify_table(index) == "toc"


def test_spanning_label_and_check_mark_cells() -> None:
    cells = [
        Cell(1, 3, 1, "2025"),
        Cell(1, 4, 1, "2024"),
        Cell(1, 5, 1, "Filed"),
        Cell(2, 1, 4, "Current assets:"),
        Cell(3, 1, 1, "Cash"),
        Cell(3, 3, 1, "10"),
        Cell(3, 4, 1, "9"),
        Cell(3, 5, 1, "X"),
    ]
    table = render_table(cells)
    assert table.rows == ("Current assets:", "Cash | 10 | 9 | X")


def test_exhibit_numbers_stay_labels() -> None:
    cells = [
        Cell(1, 1, 1, "Exhibit"),
        Cell(1, 2, 1, "Description"),
        Cell(1, 3, 1, "Form"),
        Cell(1, 4, 1, "Exhibit"),
        Cell(2, 1, 1, "10.1"),
        Cell(2, 2, 1, "Credit Agreement"),
        Cell(2, 3, 1, "8-K"),
        Cell(2, 4, 1, "10.2"),
    ]
    table = render_table(cells)
    assert table.header == (" | Description | Form | Exhibit",)
    assert table.rows == ("10.1 | Credit Agreement | 8-K | 10.2",)


def test_paragraph_without_sentences_stays_under_cap() -> None:
    paragraph = Paragraph("p", " ".join(f"{i}.5%" for i in range(200)), False)
    chunks = chunk_paragraphs([paragraph], None, _sizing(cap=40))
    assert len(chunks) > 1
    assert all(COUNTER.count(c.text) + 2 <= 40 for c in chunks)
    assert [c.part for c in chunks] == list(range(1, len(chunks) + 1))

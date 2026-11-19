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
from tests.native_rag_support import DigitCounter, WordCounter

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
    assert table.header == ("Exhibit | Description | Form | Exhibit",)
    assert table.rows == ("10.1 | Credit Agreement | 8-K | 10.2",)


def test_paragraph_without_sentences_stays_under_cap() -> None:
    paragraph = Paragraph("p", " ".join(f"{i}.5%" for i in range(200)), False)
    chunks = chunk_paragraphs([paragraph], None, _sizing(cap=40))
    assert len(chunks) > 1
    assert all(COUNTER.count(c.text) + 2 <= 40 for c in chunks)
    assert [c.part for c in chunks] == list(range(1, len(chunks) + 1))


def test_wide_table_parts_fit_with_their_real_labels() -> None:
    # A header so wide that each part holds one row; some rows are cut by
    # tokens, so there are more parts than rows and `(part i/n)` grows. Rows
    # of every length near the cap must still fit with their real label.
    columns = 12
    header = [
        Cell(1, c + 2, 1, f"Three months ended period {c}") for c in range(columns)
    ]
    sections = {
        "root": SourceSection(
            "root", 1, None, "document", None, "Doc", 1, ("Cover",), "cover"
        )
    }

    def context(
        path: tuple[str, ...], subheading: str | None, table: str | None
    ) -> str:
        return " | ".join(["Example", *path, f"Table: {table}"])

    params = ChunkParams(max_tokens=120, min_unit_tokens=4)
    for extra in range(40):
        rows = [
            Cell(r, c + 2, 1, f"{r},{c}00")
            for r in range(2, 11)
            for c in range(columns)
        ] + [
            Cell(
                r, 1, 1, "Long segment label " * 30 if r % 4 == 0 else "S" + "y" * extra
            )
            for r in range(2, 11)
        ]
        blocks = [
            SourceBlock("d:block:000001", 1, "root", "paragraph", "Segment results"),
            SourceBlock("d:block:000002", 2, "root", "table", "x" * 200),
        ]
        cells = {"d:block:000002:table": header + rows}
        result = chunk_document(
            DocumentInput(blocks, sections, cells, context), DigitCounter(), params
        )
        tables = [c for c in result.chunks if c.chunk_kind == "table"]
        assert len(tables) > 9, extra
        assert all(c.token_count <= params.max_tokens for c in tables), extra


def test_word_fragmented_document_is_joined() -> None:
    words = ("Revenue increased due to strong demand in every region. " * 40).split()
    sections = {
        "root": SourceSection(
            "root", 1, None, "document", None, "Doc", 1, ("Cover",), "cover"
        )
    }
    blocks = [
        SourceBlock(f"d:block:{i:06d}", i, "root", "paragraph", w)
        for i, w in enumerate(words, start=1)
    ]

    def context(
        path: tuple[str, ...], subheading: str | None, table: str | None
    ) -> str:
        return "Example"

    result = chunk_document(
        DocumentInput(blocks, sections, {}, context),
        COUNTER,
        ChunkParams(max_tokens=200),
    )
    assert result.dropped == {}
    assert all(c.subheading is None for c in result.chunks)
    assert result.chunks[0].text.startswith("Revenue increased due to strong demand")
    placed = [b for c in result.chunks for b in c.block_ids]
    assert sorted(placed) == sorted(b.block_id for b in blocks)


def _two_tables(after_first: list[str]) -> DocumentInput:
    sections = {
        "root": SourceSection(
            "root", 1, None, "document", None, "Doc", 1, ("Cover",), "cover"
        )
    }
    table = "Revenue\n1,000\n900\nCost\n400\n300"
    texts = [
        ("table", table),
        *(("paragraph", t) for t in after_first),
        ("table", table),
    ]
    blocks = [
        SourceBlock(f"d:block:{i:06d}", i, "root", kind, text)
        for i, (kind, text) in enumerate(texts, start=1)
    ]
    grid = [
        Cell(1, 2, 1, "2025"),
        Cell(1, 3, 1, "2024"),
        Cell(2, 1, 1, "Revenue"),
        Cell(2, 2, 1, "1,000"),
        Cell(2, 3, 1, "900"),
        Cell(3, 1, 1, "Cost"),
        Cell(3, 2, 1, "400"),
        Cell(3, 3, 1, "300"),
    ]
    cells = {f"{b.block_id}:table": grid for b in blocks if b.block_type == "table"}

    def context(
        path: tuple[str, ...], subheading: str | None, table: str | None
    ) -> str:
        return "Example"

    return DocumentInput(blocks, sections, cells, context)


def test_statement_footer_joins_its_table() -> None:
    params = ChunkParams(max_tokens=200, min_unit_tokens=16, small_table_chars=5)
    footer = "See accompanying Notes to Financial Statements."
    result = chunk_document(
        _two_tables([footer, "Consolidated Balance Sheets"]), COUNTER, params
    )
    assert [c.chunk_kind for c in result.chunks] == ["table", "table"]
    assert result.chunks[0].text.endswith(footer)
    assert result.chunks[1].table_title == "Consolidated Balance Sheets"
    # A short first sentence that leads into more prose stays text.
    prose = [
        "Net sales increased.",
        (
            "Growth came from every region and product line, led by services, "
            "wearables and a strong holiday quarter in the Americas and Europe."
        ),
    ]
    result = chunk_document(_two_tables(prose), COUNTER, params)
    assert [c.chunk_kind for c in result.chunks] == ["table", "text", "table"]


def test_next_tables_title_is_not_taken() -> None:
    params = ChunkParams(max_tokens=200, min_unit_tokens=16, small_table_chars=5)
    lead_in = "Revenue by geography is based on customer location."
    result = chunk_document(_two_tables([lead_in]), COUNTER, params)
    assert [c.table_title for c in result.chunks] == [None, lead_in]
    # A footer straight before the next table still closes the table above.
    footer = "See accompanying notes."
    result = chunk_document(_two_tables([footer]), COUNTER, params)
    assert result.chunks[0].text.endswith(footer)
    assert result.chunks[1].table_title is None
    # A colon lead-in after the footer: the footer still joins the table above.
    result = chunk_document(
        _two_tables([footer, "Revenue by segment was as follows:"]), COUNTER, params
    )
    assert result.chunks[0].text.endswith(footer)
    assert result.chunks[1].table_title == "Revenue by segment was as follows:"


def test_note_before_a_shown_below_lead_in_joins_the_table() -> None:
    params = ChunkParams(max_tokens=200, min_unit_tokens=16, small_table_chars=5)
    note = "Gross unrealized losses were not significant."
    lead_in = "The fair values of our securities are shown below by maturity."
    result = chunk_document(_two_tables([note, lead_in]), COUNTER, params)
    assert [c.chunk_kind for c in result.chunks] == ["table", "table"]
    assert result.chunks[0].text.endswith(note)
    assert result.chunks[1].table_title == lead_in


def _notes_between_tables(rows: list[tuple[str, str]]) -> DocumentInput:
    """Two data tables with a layout table of `rows` (marker, text) between."""

    document = _two_tables([])
    first, last = document.blocks
    notes = SourceBlock("d:block:000009", 2, "root", "table", "notes")
    cells = {
        **document.cells_by_table,
        "d:block:000009:table": [
            Cell(r, c, 1, text)
            for r, row in enumerate(rows, start=1)
            for c, text in enumerate(row, start=1)
            if text
        ],
    }
    blocks = [first, notes, SourceBlock(last.block_id, 3, "root", "table", last.text)]
    return DocumentInput(blocks, document.sections, cells, document.context)


def test_a_layout_table_of_notes_is_never_split() -> None:
    params = ChunkParams(max_tokens=200, min_unit_tokens=16, small_table_chars=5)
    # 500-company v6: rows past `max_footnotes` and a title row after notes
    # went to other chunks than the rest of their layout table.
    six = [(f"({m})", f"Note {m}.") for m in "abcdef"]
    result = chunk_document(_notes_between_tables(six), COUNTER, params)
    assert [c.block_ids for c in result.chunks] == [
        ("d:block:000001", "d:block:000009"),
        ("d:block:000002",),
    ]
    assert result.chunks[0].text.endswith("(e) Note e.\n(f) Note f.")
    mixed = [("(2)", "Comparable basis excludes items."), ("EPS Guidance", "")]
    result = chunk_document(_notes_between_tables(mixed), COUNTER, params)
    assert [c.chunk_kind for c in result.chunks] == ["table", "text", "table"]
    assert result.chunks[1].block_ids == ("d:block:000009",)
    assert result.chunks[2].table_title is None


def test_title_only_unit_joins_the_next_unit_under_its_subheading() -> None:
    paragraphs = _paragraphs(
        "Legal and Regulatory Risks",  # category heading over a risk heading
        "We Are Subject to Payments Risks",
        " ".join(["payments"] * 30),
        " ".join(["fraud"] * 30),
    )
    chunks = chunk_paragraphs(paragraphs, None, _sizing(cap=40, min_tokens=8))
    # v5 left "Legal and Regulatory Risks" as a chunk of its own.
    assert chunks[0].text.startswith(
        "Legal and Regulatory Risks\nWe Are Subject to Payments Risks"
    )
    assert {c.subheading for c in chunks} == {"We Are Subject to Payments Risks"}
    sentences = " ".join(f"Sentence {i} has exactly six words." for i in range(12))
    split = chunk_paragraphs(_paragraphs("Exhibit 99.1", sentences), None, _sizing(30))
    assert split[0].text.startswith("Exhibit 99.1\nSentence 0")
    assert split[0].part == 1 and split[0].block_ids == ("b0", "b1")


def test_markers_are_not_titles() -> None:
    assert not is_title_like("(c)") and not is_title_like("*")
    assert is_title_like("Income Taxes")


def test_lower_header_row_splits_the_columns_it_falls_under() -> None:
    cells = [
        Cell(0, 1, 2, "Common Stock"),
        Cell(0, 3, 1, "Additional Paid-in"),
        Cell(0, 4, 1, "Total"),
        Cell(1, 1, 1, "Shares"),
        Cell(1, 2, 1, "Amount"),
        Cell(2, 0, 1, "Balances"),
        Cell(2, 1, 1, "24,388"),
        Cell(2, 2, 1, "24"),
        Cell(2, 3, 1, "100"),
        Cell(2, 4, 1, "124"),
    ]
    table = render_table(cells)
    assert table.header == (
        " | Common Stock Shares | Common Stock Amount | Additional Paid-in | Total",
    )
    assert table.rows == ("Balances | 24,388 | 24 | 100 | 124",)


def test_8k_contents_table_is_toc() -> None:
    rows = [("ITEM 2.02 Results of Operations", "3"), ("SIGNATURES", "4"),
            ("EXHIBIT 99.1", "")]  # fmt: skip
    cells = [
        Cell(r, c, 1, text)
        for r, row in enumerate(rows)
        for c, text in enumerate(row)
        if text
    ]
    assert classify_table(cells) == "toc"
    bare = [Cell(0, 0, 1, "ITEM 8.01 Other Events"), Cell(0, 1, 1, "3"),
            *(Cell(r, 0, 1, f"EXHIBIT 4.{r}") for r in range(1, 5))]  # fmt: skip
    assert classify_table(bare) == "toc"


def test_company_line_is_never_a_subheading() -> None:
    from src.data_pipeline.transforms.native_rag.document_chunks import (
        _subheading_after,
    )

    run = _paragraphs("Debt", "NVIDIA CORPORATION AND SUBSIDIARIES")
    assert _subheading_after(run, None) == "Debt"
    assert _subheading_after(_paragraphs("Apple Inc."), "Notes") == "Notes"


def test_lower_header_row_with_label_text_stays_header() -> None:
    cells = [
        Cell(0, 1, 2, "Less than 12 Months"),
        Cell(0, 3, 2, "12 Months or Greater"),
        Cell(1, 0, 1, "(In millions)"),
        Cell(1, 1, 1, "Fair Value"),
        Cell(1, 2, 1, "Unrealized Losses"),
        Cell(1, 3, 1, "Fair Value"),
        Cell(1, 4, 1, "Unrealized Losses"),
        Cell(2, 0, 1, "U.S. government securities"),
        Cell(2, 1, 1, "2,569"),
        Cell(2, 2, 1, "(51)"),
        Cell(2, 3, 1, "34,608"),
        Cell(2, 4, 1, "(1,411)"),
    ]
    table = render_table(cells)
    assert table.rows == (
        "U.S. government securities | 2,569 | (51) | 34,608 | (1,411)",
    )
    assert table.header[0].startswith("(In millions) | Less than 12 Months Fair Value")
    two_rows = [Cell(0, 0, 1, "ITEM 5.02 Departure of Directors"), Cell(0, 1, 1, "3"),
                Cell(1, 0, 1, "SIGNATURES"), Cell(1, 1, 1, "4")]  # fmt: skip
    assert classify_table(two_rows) == "toc"


def test_text_rows_mapping_onto_columns_stay_data() -> None:
    cells = [
        Cell(0, 0, 1, "Title of each class"),
        Cell(0, 1, 1, "Trading Symbol(s)"),
        Cell(0, 2, 1, "Name of exchange"),
        Cell(1, 0, 1, "Class A Common Stock"),
        Cell(1, 1, 1, "GOOGL"),
        Cell(1, 2, 1, "Nasdaq"),
        Cell(2, 0, 1, "Class C Capital Stock"),
        Cell(2, 1, 1, "GOOG"),
        Cell(2, 2, 1, "Nasdaq"),
    ]
    table = render_table(cells)
    assert table.rows == (
        "Class A Common Stock | GOOGL | Nasdaq",
        "Class C Capital Stock | GOOG | Nasdaq",
    )


def test_exhibit_index_rows_under_a_spanning_header_stay_data() -> None:
    cells = [
        Cell(0, 0, 1, "Exhibit Number"),
        Cell(0, 1, 1, "Description"),
        Cell(0, 2, 2, "Incorporated by reference herein"),
        Cell(1, 0, 1, "10.06.1"),
        Cell(1, 1, 1, "Amended Stock Plan"),
        Cell(1, 2, 1, "Annual Report on Form 10-K"),
        Cell(1, 3, 1, "February 4, 2020"),
        Cell(2, 0, 1, "31.1"),
        Cell(2, 1, 1, "Certification"),
        Cell(2, 2, 1, "Filed herewith"),
        Cell(2, 3, 1, "1"),
    ]
    table = render_table(cells)
    assert any(row.startswith("10.06.1 | Amended Stock Plan") for row in table.rows)


def test_our_company_is_a_subheading() -> None:
    from src.data_pipeline.transforms.native_rag.document_chunks import (
        _subheading_after,
    )

    assert _subheading_after(_paragraphs("Our Company"), None) == "Our Company"


def test_the_company_names_stay_company_lines() -> None:
    from src.data_pipeline.transforms.native_rag.document_chunks import _COMPANY_LINE

    for line in ("The Coca-Cola Company", "THE HOME DEPOT, INC. AND SUBSIDIARIES"):
        assert _COMPANY_LINE.search(line), line
    for heading in ("The Company", "Our Company", "Risks Specific to our Company"):
        assert not _COMPANY_LINE.search(heading), heading


def test_year_row_spanning_column_pairs_prefixes_them() -> None:
    cells = [
        Cell(0, 0, 1, "(In millions)"),
        Cell(0, 1, 1, "Shares"),
        Cell(0, 2, 1, "Amount"),
        Cell(0, 3, 1, "Shares"),
        Cell(0, 4, 1, "Amount"),
        Cell(1, 0, 1, "Year Ended June 30,"),
        Cell(1, 1, 2, "2024"),
        Cell(1, 3, 2, "2023"),
        Cell(2, 0, 1, "First Quarter"),
        Cell(2, 1, 1, "7"),
        Cell(2, 2, 1, "2,800"),
        Cell(2, 3, 1, "6"),
        Cell(2, 4, 1, "2,500"),
    ]
    table = render_table(cells)
    assert table.header[-1].endswith(
        "| 2024 Shares | 2024 Amount | 2023 Shares | 2023 Amount"
    )
    assert table.rows == ("First Quarter | 7 | 2,800 | 6 | 2,500",)


def test_one_wide_units_cell_is_not_a_grouping_row() -> None:
    cells = [
        Cell(0, 1, 1, "Oct 31, 2021"),
        Cell(0, 2, 1, "Jan 31, 2021"),
        Cell(1, 0, 1, "Inventories:"),
        Cell(1, 1, 2, "(In millions)"),
        Cell(2, 0, 1, "Raw materials"),
        Cell(2, 1, 1, "925"),
        Cell(2, 2, 1, "632"),
    ]
    table = render_table(cells)
    assert table.header[-1] == " | Oct 31, 2021 | Jan 31, 2021"
    assert table.rows[-1] == "Raw materials | 925 | 632"
    from src.data_pipeline.transforms.native_rag.document_chunks import _COMPANY_LINE

    assert not _COMPANY_LINE.search("Overview of the Company")

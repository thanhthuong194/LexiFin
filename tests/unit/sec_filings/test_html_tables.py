from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from lxml import etree
from pydantic import ValidationError

from src.data_pipeline.contracts.sec_filings.content_blocks import (
    ContentBlockRecord,
    sha256_text,
)
from src.data_pipeline.contracts.sec_filings.document_tables import (
    DocumentTableRecord,
)
from src.data_pipeline.transforms.sec_filings.document_tables import (
    build_document_tables,
)
from src.data_pipeline.transforms.sec_filings.html_blocks import extract_blocks
from src.data_pipeline.transforms.sec_filings.html_dom import parse_html
from src.data_pipeline.transforms.sec_filings.html_tables import (
    TableLayout,
    build_grid,
    extract_table,
    table_content_sha256,
)

pytestmark = pytest.mark.unit

DOCUMENT_ID = "sec:0000320193-25-000079:document:0001"
PROCESSED_AT = datetime(2026, 10, 6, 4, 0, tzinfo=UTC)
SOURCE_SHA256 = "b" * 64


def _root(body: str) -> etree._Element:
    return parse_html(f"<html><body>{body}</body></html>".encode())


def _layout(body: str, index: int = 0) -> TableLayout:
    return build_grid(extract_table(list(_root(body).iter("table"))[index]))


def _grid(layout: TableLayout) -> list[tuple[int, int, int, int, int, str]]:
    return [
        (
            c.row_ordinal,
            c.row_cell_ordinal,
            c.column_ordinal,
            c.rowspan,
            c.colspan,
            c.text,
        )
        for c in layout.cells
    ]


def _table_block(body: str) -> tuple[ContentBlockRecord, TableLayout]:
    """The first table block of `body` as a stored record, and its layout."""

    root = _root(body)
    html = next(b for b in extract_blocks(root).blocks if b.block_type == "table")
    tree = root.getroottree()
    table = next(t for t in root.iter("table") if tree.getpath(t) == html.source_xpath)
    block = ContentBlockRecord.model_validate(
        {
            "block_id": f"{DOCUMENT_ID}:block:000002",
            "filing_id": "sec:0000320193-25-000079",
            "document_id": DOCUMENT_ID,
            "filing_date": date(2025, 10, 31),
            "section_id": f"{DOCUMENT_ID}:section:0001",
            "block_ordinal": 2,
            "document_block_count": 3,
            "block_type": "table",
            "text": html.text,
            "text_length": len(html.text),
            "text_sha256": sha256_text(html.text),
            "source_tag": "table",
            "source_xpath": html.source_xpath,
            "heading_level": None,
            "inside_table": False,
            "source_table_xpath": None,
            "source_document_sha256": SOURCE_SHA256,
            "processed_at": PROCESSED_AT,
        }
    )
    return block, build_grid(extract_table(table))


def test_simple_table_with_caption_header_and_data_cells() -> None:
    layout = _layout(
        "<table><caption> Revenue  by year </caption>"
        "<tr><th scope=' COL '>Year</th><th>Revenue</th></tr>"
        "<tr><td>2025</td><td>$1,000</td></tr></table>"
    )

    assert (layout.caption, layout.row_count, layout.column_count) == (
        "Revenue by year",
        2,
        2,
    )
    assert [(c.source_tag, c.is_header, c.scope, c.text) for c in layout.cells] == [
        ("th", True, "col", "Year"),
        ("th", True, None, "Revenue"),
        ("td", False, None, "2025"),
        ("td", False, None, "$1,000"),
    ]
    assert [c.source_xpath for c in layout.cells][:2] == [
        "/html/body/table/tr[1]/th[1]",
        "/html/body/table/tr[1]/th[2]",
    ]


def test_one_html_table_is_one_table_block_without_cell_blocks() -> None:
    blocks = extract_blocks(
        _root("<p>Before</p><table><tr><td>a</td><td><p>b</p></td></tr></table>")
    ).blocks

    assert [(b.block_type, b.source_tag, b.text) for b in blocks] == [
        ("paragraph", "p", "Before"),
        ("table", "table", "a\nb"),
    ]


def test_empty_cells_are_kept() -> None:
    layout = _layout(
        "<table><tr><td>a</td><td> </td><td></td><td>&#160;</td></tr></table>"
    )

    assert [c.text for c in layout.cells] == ["a", "", "", ""]
    assert layout.column_count == 4


def test_rowspan_and_colspan_place_cells_on_logical_columns() -> None:
    layout = _layout(
        "<table>"
        '<tr><td rowspan="2">A</td><td colspan="2">B</td><td>C</td></tr>'
        "<tr><td>D</td><td>E</td><td>F</td></tr>"
        '<tr><td colspan="3">G</td><td>H</td></tr>'
        "</table>"
    )

    assert _grid(layout) == [
        (1, 1, 1, 2, 1, "A"),
        (1, 2, 2, 1, 2, "B"),
        (1, 3, 4, 1, 1, "C"),
        (2, 1, 2, 1, 1, "D"),
        (2, 2, 3, 1, 1, "E"),
        (2, 3, 4, 1, 1, "F"),
        (3, 1, 1, 1, 3, "G"),
        (3, 2, 4, 1, 1, "H"),
    ]
    assert (layout.row_count, layout.column_count, layout.invalid_spans) == (3, 4, 0)


def test_invalid_and_zero_spans_follow_html_rules() -> None:
    layout = _layout(
        "<table><tbody>"
        '<tr><td rowspan="0">A</td><td colspan="x">B</td><td colspan="0">C</td>'
        '<td rowspan="-1">D</td></tr>'
        '<tr><td rowspan="9">E</td></tr>'
        "</tbody><tbody><tr><td>F</td><td>G</td></tr></tbody></table>"
    )

    assert _grid(layout) == [
        (1, 1, 1, 2, 1, "A"),
        (1, 2, 2, 1, 1, "B"),
        (1, 3, 3, 1, 1, "C"),
        (1, 4, 4, 1, 1, "D"),
        (2, 1, 2, 1, 1, "E"),
        (3, 1, 1, 1, 1, "F"),
        (3, 2, 2, 1, 1, "G"),
    ]
    assert layout.invalid_spans == 3


def test_nested_table_is_a_separate_table_with_its_own_cells() -> None:
    body = (
        "<table><tr><td>outer<table><tr><td>inner 1</td><td>inner 2</td></tr>"
        "</table></td><td>right</td></tr></table>"
    )

    outer, inner = _layout(body, 0), _layout(body, 1)
    blocks = extract_blocks(_root(body)).blocks

    assert [c.text for c in outer.cells] == ["outer", "right"]
    assert [c.text for c in inner.cells] == ["inner 1", "inner 2"]
    assert [(b.text, b.table_xpath) for b in blocks] == [
        ("outer\nright", None),
        ("inner 1\ninner 2", "/html/body/table"),
    ]
    assert "parent_table_id" not in DocumentTableRecord.model_fields


def test_row_groups_follow_the_row_container() -> None:
    layout = _layout(
        "<table><thead><tr><th>h</th></tr></thead><tbody><tr><td>b</td></tr></tbody>"
        "<tfoot><tr><td>f</td></tr></tfoot><tr><td>direct</td></tr>"
        "<div><tr><td>odd</td></tr></div></table>"
    )

    assert [(c.text, c.row_group) for c in layout.cells] == [
        ("h", "header"),
        ("b", "body"),
        ("f", "footer"),
        ("direct", "body"),
        ("odd", "unknown"),
    ]


def test_table_markup_without_cells_is_skipped() -> None:
    extracted = extract_blocks(_root("<table></table><table><tr></tr></table><p>x</p>"))

    assert [b.block_type for b in extracted.blocks] == ["paragraph"]
    assert extracted.empty_tables == 2


def test_cell_outside_any_row_of_its_table_fails() -> None:
    with pytest.raises(ValueError, match="has no row of its own table"):
        extract_blocks(_root("<table><td>a</td></table>"))


def test_multiple_direct_captions_are_joined() -> None:
    layout = _layout(
        "<table><caption>First</caption><caption>Second</caption>"
        "<tr><td>a</td></tr></table>"
    )

    assert layout.caption == "First\nSecond"


def test_records_have_deterministic_ids_and_hashes() -> None:
    block, layout = _table_block(
        "<p>x</p><table><tr><th>Item</th><td></td><td>$5</td></tr></table>"
    )

    tables, cells = build_document_tables([block], [layout], PROCESSED_AT)

    assert build_document_tables([block], [layout], PROCESSED_AT) == (tables, cells)
    (table,) = tables
    assert table.table_id == f"{block.block_id}:table"
    assert [cell.cell_id for cell in cells] == [
        f"{table.table_id}:cell:00000{n}" for n in (1, 2, 3)
    ]
    assert (table.cell_count, table.header_cell_count, table.column_count) == (3, 1, 3)
    assert table.table_content_sha256 == table_content_sha256(None, 1, 3, layout.cells)
    assert table.table_content_sha256 == table_content_sha256(None, 1, 3, cells)
    assert cells[1].text_sha256 == sha256_text("")
    assert block.text == "Item\n$5"


def test_empty_text_is_only_valid_for_table_blocks() -> None:
    block, _ = _table_block("<table><tr><td></td></tr></table>")

    assert (block.text, block.text_length) == ("", 0)
    with pytest.raises(ValidationError, match="only a table block"):
        ContentBlockRecord.model_validate(
            {
                **block.model_dump(),
                "block_type": "paragraph",
                "source_tag": "p",
            }
        )


def test_a_table_needs_at_least_one_cell() -> None:
    block, layout = _table_block("<table><tr><td>a</td></tr></table>")
    (table,), _ = build_document_tables([block], [layout], PROCESSED_AT)

    with pytest.raises(ValidationError):
        DocumentTableRecord.model_validate({**table.model_dump(), "cell_count": 0})

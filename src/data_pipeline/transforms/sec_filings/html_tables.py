from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from lxml import etree

from src.data_pipeline.contracts.sec_filings.document_table_cells import (
    DocumentTableCellRecord,
    RowGroup,
)
from src.data_pipeline.transforms.sec_filings.html_dom import (
    element_tag,
    is_hidden,
    visible_text,
)

_ROW_GROUP_BY_PARENT: dict[str, RowGroup] = {
    "thead": "header",
    "tbody": "body",
    "tfoot": "footer",
    "table": "body",
}
_DIGITS = re.compile(r"\d+")
_MAX_COLSPAN = 1000  # the HTML limit, which also bounds the logical grid


@dataclass(frozen=True)
class PhysicalCell:
    """A visible `<td>`/`<th>` owned by a table, with its row (0-based) and text."""

    element: etree._Element
    row_index: int
    text: str
    link_text: str


@dataclass(frozen=True)
class RawCell:
    """A physical cell's values as read from the DOM, spans still unparsed."""

    row_index: int
    source_tag: str
    text: str
    source_xpath: str
    rowspan: str | None
    colspan: str | None
    scope: str | None


@dataclass(frozen=True)
class RawTable:
    """What one `<table>` holds, read from the DOM (no element references kept).

    `row_group_keys` holds the XPath of each row's parent element: rows that
    share it form one row group.
    """

    captions: tuple[str, ...]
    row_groups: tuple[RowGroup, ...]
    row_group_keys: tuple[str, ...]
    cells: tuple[RawCell, ...]


@dataclass(frozen=True)
class GridCell:
    """A physical cell placed on the table's logical grid (1-based ordinals)."""

    cell_ordinal: int
    row_ordinal: int
    row_group: RowGroup
    row_cell_ordinal: int
    column_ordinal: int
    rowspan: int
    colspan: int
    is_header: bool
    scope: str | None
    text: str
    source_tag: str
    source_xpath: str


@dataclass(frozen=True)
class TableLayout:
    """A table's caption, size and placed cells; `invalid_spans` counts fixes."""

    caption: str | None
    row_count: int
    column_count: int
    cells: tuple[GridCell, ...]
    invalid_spans: int


# Cell fields covered by the table content hash, in both layouts and stored rows.
_HASHED_CELL_FIELDS = (
    "cell_ordinal",
    "row_ordinal",
    "row_group",
    "row_cell_ordinal",
    "column_ordinal",
    "rowspan",
    "colspan",
    "is_header",
    "scope",
    "text",
)


def _nearest_table(element: etree._Element) -> etree._Element | None:
    return next(element.iterancestors("table"), None)


def _hidden_below(element: etree._Element, table: etree._Element) -> bool:
    """Whether the element, or an ancestor inside `table`, is hidden."""

    node: etree._Element | None = element
    while node is not None and node is not table:
        if is_hidden(node, element_tag(node)):
            return True
        node = node.getparent()
    return False


def _owned(element: etree._Element, table: etree._Element) -> bool:
    return _nearest_table(element) is table and not _hidden_below(element, table)


def _rows_and_cells(
    table: etree._Element,
) -> tuple[list[etree._Element], list[PhysicalCell]]:
    """Visible rows and cells whose nearest table is `table`, in DOM order.

    Rows and cells of nested tables belong to those tables. Raises ValueError
    for a cell without a row of its own table.
    """

    rows = [row for row in table.iter("tr") if _owned(row, table)]
    row_index = {row: index for index, row in enumerate(rows)}
    cells = []
    for cell in table.iter("td", "th"):
        if not _owned(cell, table):
            continue
        row = next(cell.iterancestors("tr"), None)
        if row not in row_index:
            path = cell.getroottree().getpath(cell)
            raise ValueError(f"Table cell {path} has no row of its own table")
        text, link_text = visible_text(cell)
        cells.append(PhysicalCell(cell, row_index[row], text, link_text))
    return rows, cells


def physical_cells(table: etree._Element) -> list[PhysicalCell]:
    """Return the visible cells owned by `table` in document order.

    A table with no such cell is markup only and is not represented.
    """

    return _rows_and_cells(table)[1]


def extract_table(table: etree._Element) -> RawTable:
    """Read a table's direct captions, owned rows and cells from the DOM."""

    rows, cells = _rows_and_cells(table)
    captions = tuple(
        visible_text(child)[0]
        for child in table
        if element_tag(child) == "caption" and not is_hidden(child, "caption")
    )
    tree = table.getroottree()
    return RawTable(
        captions=captions,
        row_groups=tuple(
            _ROW_GROUP_BY_PARENT.get(element_tag(row.getparent()), "unknown")
            for row in rows
        ),
        row_group_keys=tuple(tree.getpath(row.getparent()) for row in rows),
        cells=tuple(
            RawCell(
                row_index=cell.row_index,
                source_tag=element_tag(cell.element),
                text=cell.text,
                source_xpath=tree.getpath(cell.element),
                rowspan=cell.element.get("rowspan"),
                colspan=cell.element.get("colspan"),
                scope=cell.element.get("scope"),
            )
            for cell in cells
        ),
    )


def _span(raw: str | None) -> tuple[int, bool]:
    """Return (value, well_formed) of a span attribute; missing means 1."""

    if raw is None:
        return 1, True
    value = raw.strip()
    if _DIGITS.fullmatch(value) is None:
        return 1, False
    return int(value), True


def _rows_left(raw: RawTable) -> list[int]:
    """Rows from each row to the end of its row group, the row itself included."""

    left = [0] * len(raw.row_group_keys)
    for index in reversed(range(len(left))):
        same_group = (
            index + 1 < len(left)
            and raw.row_group_keys[index + 1] == raw.row_group_keys[index]
        )
        left[index] = 1 + (left[index + 1] if same_group else 0)
    return left


def build_grid(raw: RawTable) -> TableLayout:
    """Place every physical cell on the logical grid, honouring row/col spans.

    Each cell takes the first column at or after the previous cell of its row
    whose slots are all free, so no two cells share a slot. Malformed spans
    and colspan="0" become 1 (counted in `invalid_spans`); rowspan="0" and
    longer row spans end with their row group, as in HTML.
    """

    rows_left = _rows_left(raw)
    occupied: set[tuple[int, int]] = set()
    next_column = [1] * len(rows_left)
    row_cell_counts = [0] * len(rows_left)
    invalid = 0
    cells: list[GridCell] = []
    for ordinal, cell in enumerate(raw.cells, start=1):
        row = cell.row_index
        colspan, colspan_ok = _span(cell.colspan)
        rowspan, rowspan_ok = _span(cell.rowspan)
        invalid += (not colspan_ok or colspan == 0) + (not rowspan_ok)
        colspan = min(max(colspan, 1), _MAX_COLSPAN)
        rowspan = rows_left[row] if rowspan == 0 else min(rowspan, rows_left[row])
        column = next_column[row]
        while any(
            (row + r, column + c) in occupied
            for r in range(rowspan)
            for c in range(colspan)
        ):
            column += 1
        occupied.update(
            (row + r, column + c) for r in range(rowspan) for c in range(colspan)
        )
        next_column[row] = column + colspan
        row_cell_counts[row] += 1
        cells.append(
            GridCell(
                cell_ordinal=ordinal,
                row_ordinal=row + 1,
                row_group=raw.row_groups[row],
                row_cell_ordinal=row_cell_counts[row],
                column_ordinal=column,
                rowspan=rowspan,
                colspan=colspan,
                is_header=cell.source_tag == "th",
                scope=(cell.scope or "").strip().lower() or None,
                text=cell.text,
                source_tag=cell.source_tag,
                source_xpath=cell.source_xpath,
            )
        )
    return TableLayout(
        caption="\n".join(text for text in raw.captions if text) or None,
        row_count=len(rows_left),
        column_count=max(c.column_ordinal + c.colspan - 1 for c in cells),
        cells=tuple(cells),
        invalid_spans=invalid,
    )


def table_content_sha256(
    caption: str | None,
    row_count: int,
    column_count: int,
    cells: Sequence[GridCell] | Sequence[DocumentTableCellRecord],
) -> str:
    """Hash a table's caption, size and ordered cells as canonical JSON.

    Built layouts and stored cell rows hash alike, so stored tables can be
    re-checked without the source.
    """

    payload = {
        "caption": caption,
        "row_count": row_count,
        "column_count": column_count,
        "cells": [
            {name: getattr(cell, name) for name in _HASHED_CELL_FIELDS}
            for cell in sorted(cells, key=lambda cell: cell.cell_ordinal)
        ],
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

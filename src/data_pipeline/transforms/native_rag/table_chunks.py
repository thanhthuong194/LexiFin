from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

TableKind = Literal["data", "layout", "toc", "signature"]

# A value cell: optional sign/currency/brackets around digits, or a dash used
# for zero/none. Years and dates are labels, not values.
_NUMBER = re.compile(
    r"^[\s(\-–—+]*[$€£¥]?\s*\(?\s*\d[\d,]*(\.\d+)?\s*\)?\s*(%|x|bps)?\s*\)?$",
    re.IGNORECASE,
)
_DASH = re.compile(r"^[\-–—]+$")
_YEAR = re.compile(r"^(fiscal\s+|fy\s*)?(19|20)\d{2}$", re.IGNORECASE)
_MONTH = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b",
    re.IGNORECASE,
)
# Cells glued to their neighbour: currency signs before, closers after.
_LEADING = frozenset({"$", "€", "£", "¥", "("})
_TRAILING = frozenset({"%", ")", ")%", "%)", "x", "bps"})
_TOC_ROW = re.compile(r"^(part|item)\s+[ivx\d]", re.IGNORECASE)
_PAGE = re.compile(r"^([a-z]-)?\d{1,3}$|^[ivxl]{1,6}$", re.IGNORECASE)
_SIGNATURE = re.compile(r"^(/s/|by:)", re.IGNORECASE)


@dataclass(frozen=True)
class Cell:
    """A visible cell placed on the table grid (ordinals from Silver)."""

    row: int
    column: int
    colspan: int
    text: str

    @property
    def last_column(self) -> int:
        return self.column + self.colspan - 1


@dataclass(frozen=True)
class RenderedTable:
    """A data table as text lines: header lines repeat in every part."""

    header: tuple[str, ...]
    rows: tuple[str, ...]


def is_value(text: str) -> bool:
    """Whether a cell holds a number (or a dash for none), not a label."""

    stripped = text.strip()
    if not stripped or _YEAR.match(stripped) or _MONTH.search(stripped):
        return False
    return bool(_NUMBER.match(stripped) or _DASH.match(stripped))


def _rows(cells: Sequence[Cell]) -> list[list[Cell]]:
    by_row: dict[int, list[Cell]] = defaultdict(list)
    for cell in cells:
        if cell.text.strip():
            by_row[cell.row].append(cell)
    return [sorted(by_row[row], key=lambda c: c.column) for row in sorted(by_row)]


def classify_table(cells: Sequence[Cell]) -> TableKind:
    """How a table is read.

    `toc`: no `$` cell, and most non-empty rows (at least 3) are TOC rows
    (`_is_toc_row`): a table of contents outside a TOC section, so a small
    data table whose values look like page numbers stays data. `signature`:
    a cell starts
    with `/s/` or `By:`. `layout`: non-empty cells start in at most two
    columns, or no cell is a value (bullet lists, text laid out in a grid).
    Otherwise `data`.
    """

    rows = _rows(cells)
    if not rows:
        return "layout"
    has_currency = any(c.text.strip() in _LEADING for row in rows for c in row)
    toc_rows = sum(_is_toc_row(row) for row in rows)
    if len(rows) >= 3 and not has_currency and toc_rows * 2 > len(rows):
        return "toc"
    if any(_SIGNATURE.match(cell.text.strip()) for row in rows for cell in row):
        return "signature"
    columns = {cell.column for row in rows for cell in row}
    if len(columns) <= 2 or not any(is_value(c.text) for row in rows for c in row):
        return "layout"
    return "data"


TOC_LABEL_MIN_CHARS = 20


def _is_toc_row(row: Sequence[Cell]) -> bool:
    """A row starting with PART/ITEM, or a long text label (at least 20
    chars) followed only by a page number (`Consolidated Balance Sheets | 45`)."""

    first = row[0].text.strip()
    if _TOC_ROW.match(first):
        return True
    return (
        len(row) >= 2
        and _PAGE.match(row[-1].text.strip()) is not None
        and len(first) >= TOC_LABEL_MIN_CHARS
        and not any(is_value(cell.text) for cell in row[:-1])
    )


def layout_paragraphs(cells: Sequence[Cell]) -> list[str]:
    """A layout table's text: one paragraph per non-empty row."""

    return [" ".join(cell.text.strip() for cell in row) for row in _rows(cells)]


def _glue(row: Sequence[Cell]) -> list[tuple[int, str]]:
    """Merge `$`, `(` into the next cell and `%`, `)` into the previous one;
    return (first column, text) per merged value."""

    merged: list[tuple[int, str]] = []
    pending: tuple[int, str] | None = None
    previous_end = 0
    for cell in row:
        text = cell.text.strip()
        if text in _LEADING:
            pending = (cell.column, (pending[1] if pending else "") + text)
            continue
        if _glues_back(text, merged, cell.column - previous_end) and pending is None:
            column, previous = merged[-1]
            merged[-1] = (column, previous + text)
            previous_end = cell.last_column + 1
            continue
        previous_end = cell.last_column + 1
        if pending is not None:
            merged.append((pending[0], pending[1] + text))
            pending = None
        else:
            merged.append((cell.column, text))
    if pending is not None:
        merged.append(pending)
    return merged


def _glues_back(text: str, merged: Sequence[tuple[int, str]], gap: int) -> bool:
    """`%`, `)` always join the value before; lowercase `x` (multiple) and
    `bps` only right after a value; an uppercase `X` is a check mark cell."""

    if not merged or text.lower() not in _TRAILING or text == "X":
        return False
    if text in ("x", "bps"):
        return gap <= 1 and is_value(merged[-1][1])
    return True


def render_table(cells: Sequence[Cell]) -> RenderedTable:
    """Render a data table as `label | value | value` lines.

    Header rows are the top rows, before the first row with a value, that have
    text right of the label column (leading rows with text only in the label
    column are kept as plain lines before the header). Each cell of the last
    header row defines an output column; upper header cells covering it are
    prefixed (`Three Months Ended March 31, 2025`). Body values go to the
    output column whose span holds their first grid column (else the nearest
    one); padding cells disappear.
    """

    rows = _rows(cells)
    label_column = min(cell.column for row in rows for cell in row)
    leading, header_rows = _header(rows, label_column)
    groups, notes = _groups(
        header_rows, label_column, _first_value_column(rows, label_column)
    )
    header_lines = [*leading, *notes]
    if groups:
        header_lines.append(" | ".join(["", *(label for _, _, label in groups)]))
    body = rows[len(leading) + len(header_rows) :]
    return RenderedTable(
        tuple(header_lines),
        tuple(_render_row(row, groups, label_column) for row in body),
    )


def _header(
    rows: Sequence[list[Cell]], label_column: int
) -> tuple[list[str], list[list[Cell]]]:
    leading: list[str] = []
    header_rows: list[list[Cell]] = []
    for row in rows:
        if any(is_value(cell.text) for cell in row):
            break
        has_label = any(cell.column == label_column for cell in row)
        if header_rows and has_label:
            break
        if all(cell.column == label_column for cell in row):
            leading.append(" ".join(cell.text.strip() for cell in row))
        else:
            header_rows.append(row)
    if not header_rows:
        leading = []
    return leading, header_rows


def _groups(
    header_rows: Sequence[list[Cell]], label_column: int, first_value: int
) -> tuple[list[tuple[int, int, str]], list[str]]:
    """Output columns from the header row with the most cells (the last on
    ties), prefixed by the header cells above that cover them; single-cell
    rows below it (units such as `($ in millions)`) become notes."""

    if not header_rows:
        return [], []
    position = max(range(len(header_rows)), key=lambda i: (len(header_rows[i]), i))
    notes = [
        " ".join(cell.text.strip() for cell in row)
        for row in header_rows[position + 1 :]
    ]
    groups: list[tuple[int, int, str]] = []
    for cell in header_rows[position]:
        if cell.column == label_column and cell.last_column < first_value:
            continue
        parts = [
            upper.text.strip()
            for header in header_rows[:position]
            for upper in header
            if upper.column != label_column
            and upper.column <= cell.column <= upper.last_column
        ]
        groups.append(
            (cell.column, cell.last_column, " ".join([*parts, cell.text.strip()]))
        )
    return groups, notes


def _is_label(column: int, text: str, first: int, label_column: int) -> bool:
    """Left of the first output column: text, or anything in the label column."""

    return column < first and (column == label_column or not is_value(text))


def _render_row(
    row: Sequence[Cell], groups: Sequence[tuple[int, int, str]], label_column: int
) -> str:
    values = _glue(row)
    if not groups:
        return " | ".join(text for _, text in values)
    first = groups[0][0]
    label = " ".join(
        c.text.strip() for c in row if _is_label(c.column, c.text, first, label_column)
    )
    slots = [""] * len(groups)
    for column, text in values:
        if _is_label(column, text, first, label_column):
            continue
        slot = _slot(groups, column)
        slots[slot] = f"{slots[slot]} {text}".strip()
    cells_out = [label, *slots]
    while len(cells_out) > 1 and not cells_out[-1]:
        cells_out.pop()
    return " | ".join(cells_out)


def _first_value_column(rows: Sequence[Sequence[Cell]], label_column: int) -> int:
    """First grid column holding a value, the label column excepted (exhibit
    numbers such as `10.1` there are labels)."""

    columns = [
        c.column
        for row in rows
        for c in row
        if c.column != label_column and is_value(c.text)
    ]
    return min(columns) if columns else 1 << 30


def _slot(groups: Sequence[tuple[int, int, str]], column: int) -> int:
    for position, (first, last, _) in enumerate(groups):
        if first <= column <= last:
            return position
    return min(
        range(len(groups)),
        key=lambda p: (min(abs(column - groups[p][0]), abs(column - groups[p][1])), p),
    )


def split_rows(
    table: RenderedTable, lead: str, fits: Callable[[str], bool]
) -> list[str]:
    """Cut a rendered table into texts that each `fits`, by whole rows.

    Every part is `lead` (title, may be empty), the header lines and a run of
    body rows; parts are balanced (same number of parts, smallest largest
    part). A single row that does not fit is still its own part (the caller
    splits it further).
    """

    head = [line for line in (lead, *table.header) if line]
    rows = list(table.rows)

    def text(chunk: Sequence[str]) -> str:
        return "\n".join([*head, *chunk])

    if fits(text(rows)) or len(rows) <= 1:
        return [text(rows)]
    parts = 2
    while parts < len(rows):
        size = -(-len(rows) // parts)
        pieces = [rows[i : i + size] for i in range(0, len(rows), size)]
        if all(fits(text(piece)) for piece in pieces):
            return [text(piece) for piece in pieces]
        parts += 1
    return [text([row]) for row in rows]

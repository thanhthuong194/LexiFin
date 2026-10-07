from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from src.data_pipeline.contracts.sec_filings.exhibits import (
    ExhibitRecord,
    exhibit_id_for,
    exhibit_text_sha256,
)
from src.data_pipeline.transforms.sec_filings.html_dom import normalize_text

# Header cells of the exhibit number and description columns.
_NUMBER_HEADER = re.compile(
    r"(?:exhibit|exh\.?)(?:\s*(?:no\.?|number|#))?|no\.?|number", re.IGNORECASE
)
_DESCRIPTION_HEADER = re.compile(
    r"(?:exhibit\s+)?description(?:\s+of\s+(?:exhibits?|documents?))?|document|title",
    re.IGNORECASE,
)
# An exhibit number (3.1, 10.1A, 101.INS, 104) and footnote markers after it.
_EXHIBIT_NUMBER = re.compile(
    r"(\d{1,3}(?:\.\d+)*(?:\.[A-Z]{2,3})?[A-Za-z]?)[\s*†‡#+^,]*"
)


@dataclass(frozen=True)
class GridCell:
    """A stored table cell placed in its table's grid."""

    row_ordinal: int
    column_ordinal: int
    colspan: int
    text: str
    source_xpath: str


@dataclass(frozen=True)
class CellLink:
    """A stored link and its target document in the same filing, if resolved."""

    source_xpath: str
    href: str
    target_document_id: str | None


@dataclass(frozen=True)
class ExhibitEntry:
    """One row of an Exhibit Index table.

    `href` is the row's link when all its links agree, None otherwise.
    """

    exhibit_number: str
    description: str | None
    href: str | None
    resolved_document_id: str | None
    source_xpath: str
    has_conflicting_links: bool


@dataclass
class TableEntries:
    """The entries of one table, and its rows that are no entry."""

    entries: list[ExhibitEntry]
    rejected_rows: int


def is_exhibit_region(
    form: str, section_kind: str, section_code: str | None, parent_code: str | None
) -> bool:
    """Whether a section is where a filing discloses its Exhibit Index.

    That is an EXHIBIT INDEX section, the exhibits item of the form (10-K
    ITEM 15, 10-Q PART II ITEM 6, 8-K ITEM 9.01), or nothing else.
    """

    if section_kind == "exhibit_index":
        return True
    if section_kind != "item":
        return False
    return (form, section_code, parent_code) in {
        ("10-K", "ITEM 15", "PART IV"),
        ("10-K", "ITEM 15", None),
        ("10-Q", "ITEM 6", "PART II"),
        ("8-K", "ITEM 9.01", None),
    }


def _columns(cell: GridCell) -> range:
    return range(cell.column_ordinal, cell.column_ordinal + cell.colspan)


def _texts(cells: Sequence[GridCell], header: GridCell) -> str:
    columns = set(_columns(header))
    return normalize_text(" ".join(c.text for c in cells if columns & set(_columns(c))))


def _header(rows: dict[int, list[GridCell]]) -> tuple[int, GridCell, GridCell] | None:
    """The first row naming both an exhibit number and a description column."""

    for row in sorted(rows):
        numbers = [
            c for c in rows[row] if _NUMBER_HEADER.fullmatch(normalize_text(c.text))
        ]
        descriptions = [
            c
            for c in rows[row]
            if _DESCRIPTION_HEADER.fullmatch(normalize_text(c.text))
        ]
        if numbers and descriptions:
            return row, numbers[0], descriptions[0]
    return None


def _link(
    row: Sequence[GridCell], links: Sequence[CellLink]
) -> tuple[CellLink | None, bool]:
    """The row's link when its links agree on one href; whether they disagree."""

    found = [
        link
        for link in links
        if any(link.source_xpath.startswith(f"{cell.source_xpath}/") for cell in row)
    ]
    if len({link.href for link in found}) > 1:
        return None, True
    if not found:
        return None, False
    targets = {link.target_document_id for link in found}
    return CellLink(
        found[0].source_xpath,
        found[0].href,
        targets.pop() if len(targets) == 1 else None,
    ), False


def _entry(
    number: re.Match[str],
    number_cell: GridCell,
    description: str,
    row: Sequence[GridCell],
    links: Sequence[CellLink],
) -> ExhibitEntry:
    link, conflicting = _link(row, links)
    return ExhibitEntry(
        exhibit_number=number.group(1),
        description=description or None,
        href=None if link is None else link.href,
        resolved_document_id=None if link is None else link.target_document_id,
        source_xpath=number_cell.source_xpath.rsplit("/", 1)[0],
        has_conflicting_links=conflicting,
    )


def _headed_entries(
    rows: dict[int, list[GridCell]],
    header: tuple[int, GridCell, GridCell],
    links: Sequence[CellLink],
) -> TableEntries:
    header_row, number_header, description_header = header
    result = TableEntries([], 0)
    for row in sorted(r for r in rows if r > header_row):
        number = _EXHIBIT_NUMBER.fullmatch(_texts(rows[row], number_header))
        if number is None:
            result.rejected_rows += any(c.text.strip() for c in rows[row])
            continue
        number_cell = next(
            c for c in rows[row] if set(_columns(c)) & set(_columns(number_header))
        )
        description = _texts(rows[row], description_header)
        result.entries.append(
            _entry(number, number_cell, description, rows[row], links)
        )
    return result


def _headerless_entries(
    rows: dict[int, list[GridCell]], links: Sequence[CellLink]
) -> TableEntries | None:
    result = TableEntries([], 0)
    for row in sorted(rows):
        filled = sorted(
            (c for c in rows[row] if c.text.strip()), key=lambda c: c.column_ordinal
        )
        if not filled:
            continue
        number = _EXHIBIT_NUMBER.fullmatch(normalize_text(filled[0].text))
        if number is not None and len(filled) > 1:
            description = normalize_text(" ".join(c.text for c in filled[1:]))
            result.entries.append(
                _entry(number, filled[0], description, rows[row], links)
            )
        elif number is None and len(filled) == 1:
            result.rejected_rows += 1
        else:
            return None
    return result if result.entries else None


def table_entries(
    cells: Sequence[GridCell], links: Sequence[CellLink]
) -> TableEntries | None:
    """The Exhibit Index entries of a table, or None if it is no exhibit table.

    A table qualifies by a header row naming an exhibit number and a
    description column; every later row whose number cell holds an exhibit
    number is an entry, and other non-empty rows (repeated headers, notes,
    rows without a number) are counted as rejected, never guessed into
    entries. A table without such a header (as in most 8-K Item 9.01 lists,
    or a list continued from a headed table) qualifies only if it has an
    entry and every other non-empty row is a one-cell note: any row with
    another shape, or a number without a description, means the table is no
    exhibit table.
    """

    rows: dict[int, list[GridCell]] = defaultdict(list)
    for cell in cells:
        rows[cell.row_ordinal].append(cell)
    header = _header(rows)
    if header is None:
        return _headerless_entries(rows, links)
    return _headed_entries(rows, header, links)


def exhibit_records(
    filing_id: str,
    filing_date: date,
    source_document_id: str,
    entries: Sequence[ExhibitEntry],
    processed_at: datetime,
) -> list[ExhibitRecord]:
    """The `exhibits` rows of a filing's entries, numbered in document order."""

    records = []
    for ordinal, entry in enumerate(entries, start=1):
        values: dict[str, Any] = {
            "exhibit_id": exhibit_id_for(filing_id, ordinal),
            "filing_id": filing_id,
            "source_document_id": source_document_id,
            "filing_date": filing_date,
            "exhibit_ordinal": ordinal,
            "filing_exhibit_count": len(entries),
            "exhibit_number": entry.exhibit_number,
            "description": entry.description,
            "href": entry.href,
            "resolved_document_id": entry.resolved_document_id,
            "source_xpath": entry.source_xpath,
            "processed_at": processed_at,
        }
        values["source_text_sha256"] = exhibit_text_sha256(values)
        records.append(ExhibitRecord.model_validate(values))
    return records

from __future__ import annotations

from datetime import UTC, date, datetime

import pyarrow as pa
import pytest
from pydantic import ValidationError

from src.data_pipeline.contracts.sec_filings.exhibits import (
    EXHIBITS_ARROW_SCHEMA,
    ExhibitRecord,
)
from src.data_pipeline.transforms.sec_filings.exhibits import (
    CellLink,
    ExhibitEntry,
    GridCell,
    exhibit_records,
    is_exhibit_region,
    table_entries,
)

pytestmark = pytest.mark.unit

FILING_ID = "sec:0000320193-25-000079"
SOURCE_ID = f"{FILING_ID}:document:0001"
EXHIBIT_ID = f"{FILING_ID}:document:0002"
PROCESSED_AT = datetime(2026, 10, 7, 4, 0, tzinfo=UTC)
TABLE = "/html/body/table[1]"
OLD_EXHIBIT = (
    "https://www.sec.gov/Archives/edgar/data/320193/000119312520213158/dex31.htm"
)


def _cells(*rows: tuple[str, ...]) -> list[GridCell]:
    """One cell per text, each one column wide, placed row by row."""

    return [
        GridCell(row, column, 1, text, f"{TABLE}/tr[{row}]/td[{column}]")
        for row, texts in enumerate(rows, start=1)
        for column, text in enumerate(texts, start=1)
    ]


def _link(row: int, column: int, href: str, target: str | None = None) -> CellLink:
    return CellLink(f"{TABLE}/tr[{row}]/td[{column}]/a[1]", href, target)


def test_headed_table_takes_number_and_description_columns() -> None:
    cells = _cells(
        ("EXHIBIT INDEX", "", ""),
        ("Exhibit No.", "", "Description of Exhibit"),
        ("3.1", "", "Restated Articles of Incorporation"),
        ("10.1*, **", "", "Employment  Agreement"),
        ("101.INS", "", "Inline XBRL Instance Document"),
        ("31.1", "", ""),
        ("Exhibit No.", "", "Description of Exhibit"),
        ("* Management contract.", "", ""),
        ("", "", ""),
    )

    found = table_entries(cells, [])

    assert found is not None
    assert [(e.exhibit_number, e.description) for e in found.entries] == [
        ("3.1", "Restated Articles of Incorporation"),
        ("10.1", "Employment Agreement"),
        ("101.INS", "Inline XBRL Instance Document"),
        ("31.1", None),
    ]
    assert [e.source_xpath for e in found.entries] == [
        f"{TABLE}/tr[{row}]" for row in (3, 4, 5, 6)
    ]
    assert found.rejected_rows == 2


def test_header_columns_cover_spanning_and_padding_cells() -> None:
    cells = [
        GridCell(1, 1, 2, "Exhibit Number", f"{TABLE}/tr[1]/td[1]"),
        GridCell(1, 3, 1, "Description", f"{TABLE}/tr[1]/td[2]"),
        GridCell(2, 1, 1, "", f"{TABLE}/tr[2]/td[1]"),
        GridCell(2, 2, 1, "4.2", f"{TABLE}/tr[2]/td[2]"),
        GridCell(2, 3, 1, "Indenture", f"{TABLE}/tr[2]/td[3]"),
    ]

    found = table_entries(cells, [])

    assert found is not None
    assert [(e.exhibit_number, e.description) for e in found.entries] == [
        ("4.2", "Indenture")
    ]


def test_row_links_resolve_within_the_filing_or_stay_unresolved() -> None:
    cells = _cells(
        ("Exhibit", "Description"),
        ("99.1", "Press release"),
        ("3.1", "Articles, incorporated by reference"),
        ("104", "Cover Page Interactive Data File"),
        ("10.2", "Credit agreement"),
        ("10.3", "Lease"),
    )
    links = [
        _link(2, 2, "ex991.htm", EXHIBIT_ID),
        _link(3, 2, OLD_EXHIBIT),
        _link(5, 1, "ex102.htm", EXHIBIT_ID),
        _link(5, 2, "ex102a.htm"),
        _link(6, 1, "ex103.htm", EXHIBIT_ID),
        _link(6, 2, "ex103.htm", EXHIBIT_ID),
        # A link of another table's row with the same number is not this row's.
        CellLink("/html/body/table[2]/tr[4]/td[1]/a[1]", "other.htm", EXHIBIT_ID),
    ]

    found = table_entries(cells, links)

    assert found is not None
    assert [
        (e.href, e.resolved_document_id, e.has_conflicting_links) for e in found.entries
    ] == [
        ("ex991.htm", EXHIBIT_ID, False),
        (OLD_EXHIBIT, None, False),
        (None, None, False),
        (None, None, True),
        ("ex103.htm", EXHIBIT_ID, False),
    ]


def test_headerless_table_qualifies_if_its_rows_are_entries_or_notes() -> None:
    entries = table_entries(
        _cells(
            ("", "", ""),
            ("99.1", "", "Press release, dated January 30, 2024"),
            ("104", "Cover Page Interactive Data File", "(embedded)"),
            ("As permitted by Item 601(b)(4)(iii)(A), omitted.", "", ""),
        ),
        [],
    )

    assert entries is not None
    assert [(e.exhibit_number, e.description) for e in entries.entries] == [
        ("99.1", "Press release, dated January 30, 2024"),
        ("104", "Cover Page Interactive Data File (embedded)"),
    ]
    assert entries.rejected_rows == 1


@pytest.mark.parametrize(
    "rows",
    [
        (("99.1", "Press release"), ("Note", "x")),
        (("99.1", "Press release"), ("104", "")),
        (("Exhibits are listed below.",),),
    ],
)
def test_headerless_table_with_another_row_shape_is_not_exhibit_table(
    rows: tuple[tuple[str, ...], ...],
) -> None:
    assert table_entries(_cells(*rows), []) is None


@pytest.mark.parametrize(
    "rows",
    [
        (("Revenue", "$1,000"),),
        (("Item 9.01.", "Financial Statements and Exhibits"),),
        (("Exhibit", "Filed herewith"), ("10.1", "x")),
        (("", ""),),
    ],
)
def test_tables_without_exhibit_rows_are_not_exhibit_tables(
    rows: tuple[tuple[str, ...], ...],
) -> None:
    assert table_entries(_cells(*rows), []) is None


@pytest.mark.parametrize(
    ("form", "kind", "code", "parent", "expected"),
    [
        ("10-K", "exhibit_index", "EXHIBIT INDEX", None, True),
        ("8-K", "exhibit_index", "EXHIBIT INDEX", None, True),
        ("10-K", "item", "ITEM 15", "PART IV", True),
        ("10-K", "item", "ITEM 15", None, True),
        ("10-Q", "item", "ITEM 6", "PART II", True),
        ("8-K", "item", "ITEM 9.01", None, True),
        ("10-Q", "item", "ITEM 6", "PART I", False),
        ("10-K", "item", "ITEM 6", "PART II", False),
        ("8-K", "item", "ITEM 2.02", None, False),
        ("10-K", "part", "PART IV", None, False),
        ("10-K", "signature", "SIGNATURES", None, False),
    ],
)
def test_exhibit_regions(
    form: str, kind: str, code: str, parent: str | None, expected: bool
) -> None:
    assert is_exhibit_region(form, kind, code, parent) is expected


def _entry(
    number: str, href: str | None = None, target: str | None = None
) -> ExhibitEntry:
    return ExhibitEntry(number, f"Exhibit {number}", href, target, TABLE, False)


def test_records_number_entries_in_order_with_stable_ids_and_hashes() -> None:
    entries = [_entry("3.1"), _entry("99.1", "ex991.htm", EXHIBIT_ID)]

    first = exhibit_records(
        FILING_ID, date(2025, 10, 31), SOURCE_ID, entries, PROCESSED_AT
    )
    again = exhibit_records(
        FILING_ID, date(2025, 10, 31), SOURCE_ID, entries, datetime.now(UTC)
    )

    assert [
        (r.exhibit_id, r.exhibit_ordinal, r.filing_exhibit_count) for r in first
    ] == [
        (f"{FILING_ID}:exhibit:0001", 1, 2),
        (f"{FILING_ID}:exhibit:0002", 2, 2),
    ]
    assert [r.model_dump(exclude={"processed_at"}) for r in first] == [
        r.model_dump(exclude={"processed_at"}) for r in again
    ]
    assert first[0].source_text_sha256 != first[1].source_text_sha256
    table = pa.Table.from_pylist(
        [r.model_dump() for r in first], schema=EXHIBITS_ARROW_SCHEMA
    )
    assert table.schema.names == list(ExhibitRecord.model_fields)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"exhibit_id": f"{FILING_ID}:exhibit:0009"}, "derived from filing"),
        ({"exhibit_ordinal": 2, "exhibit_id": f"{FILING_ID}:exhibit:0002"}, "exceeds"),
        (
            {"source_document_id": "sec:0000320193-24-000123:document:0001"},
            "of the filing",
        ),
        (
            {"resolved_document_id": "sec:0000320193-24-000123:document:0002"},
            "of the filing",
        ),
        ({"href": None}, "only an entry with a link"),
        ({"description": "Other"}, "does not match"),
        ({"description": ""}, "at least 1 character"),
    ],
)
def test_contract_rejects_inconsistent_entries(
    changes: dict[str, object], message: str
) -> None:
    (record,) = exhibit_records(
        FILING_ID,
        date(2025, 10, 31),
        SOURCE_ID,
        [_entry("99.1", "ex991.htm", EXHIBIT_ID)],
        PROCESSED_AT,
    )

    with pytest.raises(ValidationError, match=message):
        ExhibitRecord.model_validate({**record.model_dump(), **changes})

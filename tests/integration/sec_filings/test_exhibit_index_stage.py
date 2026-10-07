from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from pyiceberg.expressions import EqualTo
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.exhibits import (
    EXHIBITS_ARROW_SCHEMA,
    ExhibitRecord,
)
from src.data_pipeline.pipelines.sec_filings.bronze_to_silver import (
    run_sec_filings_silver,
)
from src.data_pipeline.storage.iceberg import append_rows, open_silver_catalog
from src.data_pipeline.transforms.sec_filings.exhibits import (
    ExhibitEntry,
    exhibit_records,
)
from tests.conftest import PRIMARY_HTML

pytestmark = pytest.mark.integration

EXHIBITS = "sec_filings.exhibits"
SECTIONS = "sec_filings.sections"
BLOCKS = "sec_filings.content_blocks"
TEN_K = ("0000320193", "10-K", "0000320193-24-000123", "20241101")
TEN_Q = ("0000320193", "10-Q", "0000320193-25-000057", "20250502")
EIGHT_K = ("0000789019", "8-K", "0000789019-25-000010", "20250129")
TEN_K_ID = "sec:0000320193-24-000123"
EIGHT_K_ID = "sec:0000789019-25-000010"
EIGHT_K_PRIMARY = f"{EIGHT_K_ID}:document:0001"
EIGHT_K_EXHIBIT = f"{EIGHT_K_ID}:document:0002"
OLD_EXHIBIT = (
    "https://www.sec.gov/Archives/edgar/data/789019/000078901924000001/ex31.htm"
)
# The fixture 8-K lists its exhibit in a one-row table without a header.
HEADERLESS_TABLE = (
    '<table><tr><td><a href="ex31.htm">99.1</a></td>'
    '<td><a href="press-release.htm">Press release</a></td></tr></table>'
)
HEADED_TABLE = (
    "<table><tr><td>Exhibit No.</td><td>Description</td></tr>"
    '<tr><td>99.1</td><td><a href="ex31.htm">Press release</a></td></tr>'
    f'<tr><td>10.1*</td><td><a href="{OLD_EXHIBIT}">Credit agreement</a></td></tr>'
    "<tr><td>104</td><td>Cover Page Interactive Data File</td></tr>"
    "<tr><td>* Management contract.</td><td></td></tr></table>"
)
EXPECTED_HEADED = [
    ("99.1", "Press release", "ex31.htm", EIGHT_K_EXHIBIT),
    ("10.1", "Credit agreement", OLD_EXHIBIT, None),
    ("104", "Cover Page Interactive Data File", None, None),
]


def _write(data_dir: Path, write_sec_filing, *, exhibit_table: str) -> None:
    assert HEADERLESS_TABLE in PRIMARY_HTML["8-K"]
    eight_k = PRIMARY_HTML["8-K"].replace(HEADERLESS_TABLE, exhibit_table)
    for cik, form, accession, filed in (TEN_K, TEN_Q, EIGHT_K):
        write_sec_filing(
            data_dir,
            cik=cik,
            form=form,
            accession=accession,
            filed=filed,
            primary_text=eight_k if form == "8-K" else None,
        )


def _table(data_dir: Path, identifier: str) -> Table:
    with open_silver_catalog(data_dir) as catalog:
        return catalog.load_table(identifier)


def _snapshot(data_dir: Path, identifier: str) -> int | None:
    snapshot = _table(data_dir, identifier).current_snapshot()
    return None if snapshot is None else snapshot.snapshot_id


def _damage(data_dir: Path, change: Callable[[Table], None]) -> None:
    """Change the stored entries through PyIceberg while the catalog is open."""

    with open_silver_catalog(data_dir) as catalog:
        change(catalog.load_table(EXHIBITS))


def _entries(data_dir: Path) -> list[dict[str, Any]]:
    rows = _table(data_dir, EXHIBITS).scan().to_arrow().to_pylist()
    return sorted(rows, key=lambda row: row["exhibit_id"])


def _shown(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            row["exhibit_number"],
            row["description"],
            row["href"],
            row["resolved_document_id"],
        )
        for row in rows
    ]


def test_exhibit_index_entries_reference_their_filing_and_rerun_skips(
    tmp_path: Path, write_sec_filing
) -> None:
    _write(tmp_path, write_sec_filing, exhibit_table=HEADED_TABLE)

    first = run_sec_filings_silver(tmp_path)

    assert first.completed
    summary = first.exhibit_index
    assert summary is not None
    assert (
        summary.filings_considered,
        summary.filings_with_exhibit_index,
        summary.filings_inserted,
        summary.filings_without_exhibit_table,
        summary.exhibit_tables,
        summary.rejected_rows,
        summary.entries_inserted,
        summary.entries_with_href,
        summary.entries_resolved,
        summary.conflicting_links,
    ) == (3, 1, 1, 0, 1, 1, 3, 2, 1, 0)
    rows = _entries(tmp_path)
    assert _shown(rows) == EXPECTED_HEADED
    assert {
        (
            row["filing_id"],
            row["source_document_id"],
            row["filing_date"],
            row["filing_exhibit_count"],
        )
        for row in rows
    } == {(EIGHT_K_ID, EIGHT_K_PRIMARY, date(2025, 1, 29), 3)}
    assert [row["exhibit_ordinal"] for row in rows] == [1, 2, 3]
    assert all(ExhibitRecord.model_validate(row) for row in rows)
    documents = _table(tmp_path, "sec_filings.documents").scan().to_arrow()
    assert EIGHT_K_EXHIBIT in documents["document_id"].to_pylist()
    assert [f.name for f in _table(tmp_path, EXHIBITS).spec().fields] == ["filing_year"]

    snapshots = {t: _snapshot(tmp_path, t) for t in (EXHIBITS, SECTIONS, BLOCKS)}
    rerun = run_sec_filings_silver(tmp_path).exhibit_index

    assert rerun is not None
    assert (rerun.status, rerun.filings_skipped, rerun.entries_inserted) == (
        "completed",
        1,
        0,
    )
    assert rerun.snapshot_before == rerun.snapshot_after == snapshots[EXHIBITS]
    assert {t: _snapshot(tmp_path, t) for t in snapshots} == snapshots
    assert _shown(_entries(tmp_path)) == EXPECTED_HEADED


def test_headerless_list_is_an_entry_and_disagreeing_links_give_no_href(
    tmp_path: Path, write_sec_filing
) -> None:
    _write(tmp_path, write_sec_filing, exhibit_table=HEADERLESS_TABLE)

    summary = run_sec_filings_silver(tmp_path).exhibit_index

    assert summary is not None
    assert (summary.status, summary.entries_inserted, summary.conflicting_links) == (
        "completed",
        1,
        1,
    )
    assert _shown(_entries(tmp_path)) == [("99.1", "Press release", None, None)]


def test_unparsed_exhibit_index_writes_nothing_for_the_filing(
    tmp_path: Path, write_sec_filing
) -> None:
    _write(
        tmp_path,
        write_sec_filing,
        exhibit_table="<table><tr><td>See the exhibits.</td></tr></table>",
    )

    summary = run_sec_filings_silver(tmp_path).exhibit_index

    assert summary is not None
    assert (
        summary.status,
        summary.filings_with_exhibit_index,
        summary.filings_without_exhibit_table,
        summary.snapshot_after,
    ) == ("completed", 0, 1, None)
    assert _entries(tmp_path) == []


def test_deleted_filing_entries_are_rebuilt_unchanged(
    tmp_path: Path, write_sec_filing
) -> None:
    _write(tmp_path, write_sec_filing, exhibit_table=HEADED_TABLE)
    run_sec_filings_silver(tmp_path)
    before = _entries(tmp_path)
    _damage(tmp_path, lambda table: table.delete(EqualTo("filing_id", EIGHT_K_ID)))

    summary = run_sec_filings_silver(tmp_path).exhibit_index

    assert summary is not None
    assert (summary.filings_inserted, summary.entries_inserted) == (1, 3)
    assert [
        {k: v for k, v in row.items() if k != "processed_at"}
        for row in _entries(tmp_path)
    ] == [{k: v for k, v in row.items() if k != "processed_at"} for row in before]


def _ten_k_record(**changes: Any) -> ExhibitRecord:
    entry = ExhibitEntry(
        "3.1", "Articles", None, None, "/html/body/table[1]/tr[2]", False
    )
    (record,) = exhibit_records(
        TEN_K_ID,
        date(2024, 11, 1),
        f"{TEN_K_ID}:document:0001",
        [entry],
        datetime.now(UTC),
    )
    return record.model_copy(update=changes)


def _drop_last_entry(table: Table) -> None:
    table.delete(EqualTo("exhibit_id", f"{EIGHT_K_ID}:exhibit:0003"))


def _add_stale_filing(table: Table) -> None:
    append_rows(table, [_ten_k_record()], EXHIBITS_ARROW_SCHEMA)


def _add_invalid_entry(table: Table) -> None:
    append_rows(table, [_ten_k_record(description="Bylaws")], EXHIBITS_ARROW_SCHEMA)


@pytest.mark.parametrize(
    "case",
    [
        (_drop_last_entry, EIGHT_K_ID, "differ"),
        (_add_stale_filing, TEN_K_ID, "differ"),
        (_add_invalid_entry, TEN_K_ID, "stored entry is invalid"),
    ],
)
def test_changed_stored_entries_reject_the_batch_before_any_write(
    tmp_path: Path,
    write_sec_filing,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    case: tuple[Callable[[Table], None], str, str],
) -> None:
    damage, filing_id, reason = case
    _write(tmp_path, write_sec_filing, exhibit_table=HEADED_TABLE)
    run_sec_filings_silver(tmp_path)
    _damage(tmp_path, damage)
    damaged = _snapshot(tmp_path, EXHIBITS)
    capsys.readouterr()
    caplog.clear()

    result = run_sec_filings_silver(tmp_path)

    assert result.exhibit_index is not None
    assert (result.exhibit_index.status, result.exhibit_index.filings_conflicted) == (
        "rejected",
        1,
    )
    assert result.xbrl_core is None
    assert not result.completed
    assert _snapshot(tmp_path, EXHIBITS) == damaged
    logs = capsys.readouterr().out + caplog.text
    conflict = next(
        line for line in logs.splitlines() if "exhibit_index_filing_conflict" in line
    )
    assert filing_id in conflict
    assert reason in conflict

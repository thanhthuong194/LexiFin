from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import pytest
from pyiceberg.expressions import EqualTo
from pyiceberg.table import Table

from scripts.sec_filings.build_silver import REPOSITORY_ROOT, parse_args
from src.data_pipeline.contracts.sec_filings.documents import DOCUMENTS_ARROW_SCHEMA
from src.data_pipeline.contracts.sec_filings.filings import FILINGS_ARROW_SCHEMA
from src.data_pipeline.pipelines.sec_filings import documents as documents_stage
from src.data_pipeline.pipelines.sec_filings.bronze_to_silver import (
    run_sec_filings_silver,
)
from src.data_pipeline.storage.bronze import sha256_file
from src.data_pipeline.storage.iceberg import open_silver_catalog

pytestmark = pytest.mark.integration

FILINGS = "sec_filings.filings"
DOCUMENTS = "sec_filings.documents"
# Every fake submission from the write_sec_filing fixture holds six documents.
DOCUMENTS_PER_FILING = 6
CIK = "0000320193"
OTHER_CIK = "0000789019"
INITIAL_FILINGS = [
    (CIK, "10-K", "0000320193-24-000123", "20241101"),
    (CIK, "10-Q", "0000320193-25-000057", "20250502"),
    (OTHER_CIK, "8-K", "0000789019-25-000010", "20250129"),
]
MANIFEST_RUN_ID = "3c0c83bd38034ccbbbd9a746a215e6b1"


def _write_bronze(data_dir: Path, write_sec_filing) -> None:
    for cik, form, accession, filed in INITIAL_FILINGS:
        write_sec_filing(data_dir, cik=cik, form=form, accession=accession, filed=filed)
    # The Bronze manifest records which ingestion run wrote the 10-K artifacts.
    cik, form, accession, _ = INITIAL_FILINGS[0]
    folder = f"bronze/sec_filings/sec-edgar-filings/{cik}/{form}/{accession}"
    artifacts = [
        {
            "role": role,
            "relative_path": f"{folder}/{name}",
            "sha256": sha256_file(data_dir / folder / name),
            "size_bytes": (data_dir / folder / name).stat().st_size,
        }
        for role, name in [
            ("full_submission", "full-submission.txt"),
            ("primary_document", "primary-document.html"),
        ]
    ]
    manifest = (
        data_dir / "bronze" / "sec_filings" / "_metadata" / "filing_manifest.jsonl"
    )
    manifest.parent.mkdir(parents=True)
    entry = {
        "run_id": MANIFEST_RUN_ID,
        "filing_id": f"{cik}:{form}:{accession}",
        "cik": cik,
        "accession_number": accession,
        "form": form,
        "filing_date": "2024-11-01",
        "status": "success",
        "artifacts": artifacts,
        "error": None,
        "recorded_at": "2026-09-23T07:58:19Z",
    }
    manifest.write_text(json.dumps(entry) + "\n")


def _write_new_year_filing(data_dir: Path, write_sec_filing) -> None:
    write_sec_filing(
        data_dir,
        cik=CIK,
        form="8-K",
        accession="0000320193-26-000001",
        filed="20260105",
    )


def _table(data_dir: Path, identifier: str = FILINGS) -> Table:
    with open_silver_catalog(data_dir) as catalog:
        return catalog.load_table(identifier)


def _rows_per_year(table: Table) -> Counter[int]:
    """Count rows per partition year from the table's current data files."""

    counts: Counter[int] = Counter()
    for data_file in table.inspect.files().to_pylist():
        # Iceberg's year transform stores years since 1970.
        counts[1970 + data_file["partition"]["filing_year"]] += data_file[
            "record_count"
        ]
    return counts


def _partition_columns(table: Table) -> list[tuple[str, str, str | None]]:
    return [
        (
            field.name,
            str(field.transform),
            table.schema().find_column_name(field.source_id),
        )
        for field in table.spec().fields
    ]


def test_first_run_writes_partitioned_rows_and_rerun_skips_them(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)

    first = run_sec_filings_silver(tmp_path)

    assert first.completed
    assert (first.filings.discovered, first.filings.parsed) == (3, 3)
    assert (first.filings.inserted, first.filings.skipped) == (3, 0)
    assert (first.filings.conflicts, first.filings.failed) == (0, 0)
    assert first.filings.snapshot_before is None
    assert first.filings.snapshot_after is not None

    table = _table(tmp_path)
    assert table.location() == str(tmp_path / "silver" / "sec_filings" / "filings")
    assert _partition_columns(table) == [("filing_year", "year", "filing_date")]
    rows = table.scan().to_arrow()
    assert rows.schema.equals(FILINGS_ARROW_SCHEMA)
    by_id = {row["filing_id"]: row for row in rows.to_pylist()}
    assert sorted(by_id) == sorted(f"sec:{acc}" for _, _, acc, _ in INITIAL_FILINGS)
    ten_k = by_id["sec:0000320193-24-000123"]
    assert ten_k["form"] == "10-K"
    assert ten_k["cik"] == CIK
    assert ten_k["filing_date"] == date(2024, 11, 1)
    assert ten_k["report_date"] == date(2024, 12, 31)
    assert ten_k["accepted_at"] == datetime.fromisoformat("2024-11-01T16:01:02")
    assert ten_k["public_document_count"] == DOCUMENTS_PER_FILING
    assert ten_k["full_submission_sha256"] == sha256_file(
        tmp_path / ten_k["full_submission_path"]
    )
    assert ten_k["ingestion_run_id"] == MANIFEST_RUN_ID
    assert by_id["sec:0000789019-25-000010"]["ingestion_run_id"] is None
    assert len({row["processed_at"] for row in by_id.values()}) == 1
    assert _rows_per_year(table) == {2024: 1, 2025: 2}

    rerun = run_sec_filings_silver(tmp_path)

    assert rerun.completed
    assert (rerun.filings.inserted, rerun.filings.skipped) == (0, 3)
    assert rerun.filings.conflicts == 0
    assert (
        rerun.filings.snapshot_before
        == rerun.filings.snapshot_after
        == first.filings.snapshot_after
    )
    assert _table(tmp_path).scan().to_arrow().num_rows == 3


def test_documents_inventory_every_block_with_exact_offsets_and_reruns_skip(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)

    first = run_sec_filings_silver(tmp_path)

    documents = first.documents
    assert documents is not None
    assert documents.status == "completed"
    assert documents.filings_considered == 3
    assert (documents.filings_inserted, documents.filings_skipped) == (3, 0)
    assert (documents.documents_inserted, documents.documents_skipped) == (18, 0)
    assert (documents.filings_conflicted, documents.documents_failed) == (0, 0)
    assert documents.document_count_mismatches == 0
    assert documents.snapshot_before is None
    assert documents.snapshot_after is not None

    table = _table(tmp_path, DOCUMENTS)
    assert table.location() == str(tmp_path / "silver" / "sec_filings" / "documents")
    assert _partition_columns(table) == [("filing_year", "year", "filing_date")]
    rows = table.scan().to_arrow()
    assert rows.schema.equals(DOCUMENTS_ARROW_SCHEMA)
    assert rows.num_rows == 3 * DOCUMENTS_PER_FILING
    assert _rows_per_year(table) == {2024: 6, 2025: 12}
    parents = {
        row["filing_id"]: row for row in _table(tmp_path).scan().to_arrow().to_pylist()
    }
    by_filing: dict[str, list[dict]] = {}
    for row in rows.to_pylist():
        by_filing.setdefault(row["filing_id"], []).append(row)
    assert set(by_filing) == set(parents)
    for filing_id, group in by_filing.items():
        parent = parents[filing_id]
        assert sorted(row["document_ordinal"] for row in group) == list(range(1, 7))
        assert {row["filing_date"] for row in group} == {parent["filing_date"]}
        assert {row["processed_at"] for row in group} == {parent["processed_at"]}
        assert Counter(row["role"] for row in group) == {
            "primary_document": 1,
            "exhibit": 1,
            "xbrl_schema": 1,
            "xbrl_instance": 1,
            "graphic": 1,
            "generated": 1,
        }
        raw = (tmp_path / parent["full_submission_path"]).read_bytes()
        for row in group:
            block = raw[row["document_start_byte"] : row["document_end_byte"]]
            assert block.startswith(b"<DOCUMENT>")
            assert block.endswith(b"</DOCUMENT>")
            content = raw[row["content_start_byte"] : row["content_end_byte"]]
            assert len(content) == row["content_size_bytes"]
            assert hashlib.sha256(content).hexdigest() == row["content_sha256"]
    primary = next(
        row
        for row in by_filing["sec:0000320193-24-000123"]
        if row["role"] == "primary_document"
    )
    assert primary["document_id"] == "sec:0000320193-24-000123:document:0001"
    assert (primary["document_type"], primary["filename"]) == ("10-K", "primary.htm")
    raw = (tmp_path / primary["source_path"]).read_bytes()
    content = raw[primary["content_start_byte"] : primary["content_end_byte"]]
    assert content == b"\n<html><body>report</body></html>\n"

    rerun = run_sec_filings_silver(tmp_path)

    assert rerun.completed
    assert rerun.documents is not None
    assert (rerun.documents.filings_skipped, rerun.documents.documents_skipped) == (
        3,
        18,
    )
    assert rerun.documents.documents_inserted == 0
    assert (
        rerun.documents.snapshot_before
        == rerun.documents.snapshot_after
        == documents.snapshot_after
    )
    assert _table(tmp_path, DOCUMENTS).scan().to_arrow().num_rows == 18


def test_new_year_and_backfill_append_into_their_year_partitions(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    first = run_sec_filings_silver(tmp_path)

    _write_new_year_filing(tmp_path, write_sec_filing)
    new_year = run_sec_filings_silver(tmp_path)

    assert (new_year.filings.inserted, new_year.filings.skipped) == (1, 3)
    assert new_year.filings.snapshot_before == first.filings.snapshot_after
    assert new_year.filings.snapshot_after != new_year.filings.snapshot_before
    assert _rows_per_year(_table(tmp_path)) == {2024: 1, 2025: 2, 2026: 1}
    assert new_year.documents is not None
    assert (
        new_year.documents.filings_inserted,
        new_year.documents.filings_skipped,
    ) == (
        1,
        3,
    )
    assert new_year.documents.documents_inserted == DOCUMENTS_PER_FILING
    document_filings = Counter(
        _table(tmp_path, DOCUMENTS)
        .scan(selected_fields=("filing_id",))
        .to_arrow()["filing_id"]
        .to_pylist()
    )
    assert len(document_filings) == 4
    assert document_filings["sec:0000320193-26-000001"] == DOCUMENTS_PER_FILING
    assert _rows_per_year(_table(tmp_path, DOCUMENTS)) == {2024: 6, 2025: 12, 2026: 6}

    write_sec_filing(
        tmp_path,
        cik=OTHER_CIK,
        form="10-K",
        accession="0000789019-24-000099",
        filed="20240730",
    )
    backfill = run_sec_filings_silver(tmp_path)

    assert (backfill.filings.inserted, backfill.filings.skipped) == (1, 4)
    assert _rows_per_year(_table(tmp_path)) == {2024: 2, 2025: 2, 2026: 1}
    assert _table(tmp_path).scan().to_arrow().num_rows == 5
    assert _rows_per_year(_table(tmp_path, DOCUMENTS)) == {
        2024: 12,
        2025: 12,
        2026: 6,
    }


def test_filing_checksum_conflict_rejects_both_stages(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    first = run_sec_filings_silver(tmp_path)
    cik, form, accession, _ = INITIAL_FILINGS[1]
    changed = (
        tmp_path / "bronze/sec_filings/sec-edgar-filings" / cik / form / accession
    ) / "primary-document.html"
    changed.write_bytes(b"<html>different bytes</html>")
    _write_new_year_filing(tmp_path, write_sec_filing)

    result = run_sec_filings_silver(tmp_path)

    assert not result.completed
    assert result.filings.status == "rejected"
    assert (result.filings.inserted, result.filings.skipped) == (0, 2)
    assert result.filings.conflicts == 1
    assert (
        result.filings.snapshot_after
        == result.filings.snapshot_before
        == first.filings.snapshot_after
    )
    assert result.documents is None
    table = _table(tmp_path)
    assert table.current_snapshot().snapshot_id == first.filings.snapshot_after
    assert table.scan().to_arrow().num_rows == 3
    assert first.documents is not None
    documents = _table(tmp_path, DOCUMENTS)
    assert documents.current_snapshot().snapshot_id == first.documents.snapshot_after
    assert documents.scan().to_arrow().num_rows == 18


def test_partial_document_group_is_a_conflict_without_partial_append(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    run_sec_filings_silver(tmp_path)
    with open_silver_catalog(tmp_path) as catalog:
        documents = catalog.load_table(DOCUMENTS)
        documents.delete(
            EqualTo("document_id", "sec:0000320193-25-000057:document:0003")
        )
        damaged_snapshot = documents.current_snapshot().snapshot_id
    _write_new_year_filing(tmp_path, write_sec_filing)

    result = run_sec_filings_silver(tmp_path)

    assert not result.completed
    assert (result.filings.status, result.filings.inserted) == ("completed", 1)
    assert result.documents is not None
    assert result.documents.status == "rejected"
    assert result.documents.filings_conflicted == 1
    assert result.documents.filings_skipped == 2
    assert result.documents.documents_inserted == 0
    table = _table(tmp_path, DOCUMENTS)
    assert table.current_snapshot().snapshot_id == damaged_snapshot
    assert table.scan().to_arrow().num_rows == 17
    assert "sec:0000320193-26-000001" not in set(
        table.scan(selected_fields=("filing_id",)).to_arrow()["filing_id"].to_pylist()
    )


def test_rerun_builds_documents_missing_after_a_failed_documents_stage(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bronze(tmp_path, write_sec_filing)

    def fail_append(*_args: object) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(documents_stage, "append_rows", fail_append)
        with pytest.raises(OSError, match="disk full"):
            run_sec_filings_silver(tmp_path)
    assert _table(tmp_path).scan().to_arrow().num_rows == 3
    assert _table(tmp_path, DOCUMENTS).current_snapshot() is None

    recovered = run_sec_filings_silver(tmp_path)

    assert recovered.completed
    assert (recovered.filings.inserted, recovered.filings.skipped) == (0, 3)
    assert recovered.documents is not None
    assert recovered.documents.filings_inserted == 3
    assert recovered.documents.documents_inserted == 18


def test_invalid_filing_rejects_batch_before_any_write(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    write_sec_filing(
        tmp_path,
        cik=CIK,
        form="8-K",
        accession="0000320193-26-000001",
        filed="20260105",
        header_form="8-K/A",
    )

    result = run_sec_filings_silver(tmp_path)

    assert result.filings.status == "rejected"
    assert (result.filings.discovered, result.filings.parsed) == (4, 3)
    assert (result.filings.failed, result.filings.inserted) == (1, 0)
    assert result.documents is None
    assert not (tmp_path / "silver" / "sec_filings" / "filings").exists()
    assert not (tmp_path / "silver" / "sec_filings" / "documents").exists()


def test_cli_default_data_dir_is_repository_relative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["build_silver.py"])

    assert parse_args().data_dir == REPOSITORY_ROOT / "data"

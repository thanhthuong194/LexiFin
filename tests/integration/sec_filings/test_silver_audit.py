from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyiceberg.expressions import EqualTo
from pyiceberg.table import DataScan

from scripts.sec_filings.audit_silver import main
from src.data_pipeline.contracts.sec_filings.xbrl_core import context_key_for
from src.data_pipeline.contracts.sec_filings.xbrl_networks import network_id_for
from src.data_pipeline.pipelines.sec_filings.bronze_to_silver import (
    run_sec_filings_silver,
)
from src.data_pipeline.quality.sec_filings import silver_audit, silver_audit_sql
from src.data_pipeline.quality.sec_filings.silver_audit import (
    AuditOptions,
    audit_silver,
)
from src.data_pipeline.quality.sec_filings.silver_audit_report import (
    SAMPLE_IDS,
    SilverAuditReport,
)
from src.data_pipeline.quality.sec_filings.silver_model import SILVER_TABLES
from src.data_pipeline.storage.iceberg import (
    create_or_load_table,
    open_silver_catalog,
    silver_table_location,
)
from src.data_pipeline.storage.sec_filings.documents import DOCUMENTS_SPEC
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION
from tests.conftest import STATEMENT_ROLE

pytestmark = pytest.mark.integration

FILINGS = "sec_filings.filings"
DOCUMENTS = "sec_filings.documents"
SECTIONS = "sec_filings.sections"
BLOCKS = "sec_filings.content_blocks"
TABLES = "sec_filings.document_tables"
CELLS = "sec_filings.document_table_cells"
LINKS = "sec_filings.document_links"
FACTS = "sec_filings.xbrl_facts"
INSTANCES = "sec_filings.xbrl_instances"
CONTEXTS = "sec_filings.xbrl_contexts"
DIMENSIONS = "sec_filings.xbrl_dimensions"
NETWORKS = "sec_filings.xbrl_networks"
RELATIONSHIPS = "sec_filings.xbrl_relationships"
TEN_K_FILING = "sec:0000320193-24-000123"
TEN_Q_FILING = "sec:0000320193-25-000057"
EIGHT_K_FILING = "sec:0000789019-25-000010"
TEN_K = f"{TEN_K_FILING}:document:0001"
TEN_Q = f"{TEN_Q_FILING}:document:0001"
EIGHT_K = f"{EIGHT_K_FILING}:document:0001"
EIGHT_K_TABLE = f"{EIGHT_K}:block:000005:table"
TEN_K_INSTANCE = f"{TEN_K_FILING}:xbrl:inline:default"
PRESENTATION = network_id_for(TEN_K_INSTANCE, "presentation", STATEMENT_ROLE)
SCHEMAS = {table.identifier: table.spec.arrow_schema for table in SILVER_TABLES}


@pytest.fixture
def silver(tmp_path: Path, write_sec_filing) -> Path:
    """A complete Silver layer built by the pipeline from a 10-K, a 10-Q and an 8-K."""

    for cik, form, accession, filed in [
        ("0000320193", "10-K", "0000320193-24-000123", "20241101"),
        ("0000320193", "10-Q", "0000320193-25-000057", "20250502"),
        ("0000789019", "8-K", "0000789019-25-000010", "20250129"),
    ]:
        write_sec_filing(tmp_path, cik=cik, form=form, accession=accession, filed=filed)
    assert run_sec_filings_silver(tmp_path).completed
    return tmp_path


def _rows(
    data_dir: Path, identifier: str, column: str, value: str
) -> list[dict[str, Any]]:
    with open_silver_catalog(data_dir) as catalog:
        table = catalog.load_table(identifier)
        rows: list[dict[str, Any]] = (
            table.scan(row_filter=EqualTo(column, value)).to_arrow().to_pylist()
        )
        return rows


def _change(
    identifier: str,
    column: str,
    value: str,
    changes: dict[str, Any] | None = None,
    *,
    keep: bool = False,
) -> Callable[[Path], None]:
    """Replace the matching rows by copies with `changes` (none: delete them).

    With `keep`, the original rows stay and the copies are added. Rows are
    written as raw Arrow, so they may break the record contract.
    """

    def mutate(data_dir: Path) -> None:
        rows = _rows(data_dir, identifier, column, value)
        assert rows
        with open_silver_catalog(data_dir) as catalog:
            table = catalog.load_table(identifier)
            if not keep:
                table.delete(EqualTo(column, value))
            if changes is not None:
                copies = [{**row, **changes} for row in rows]
                table.append(pa.Table.from_pylist(copies, schema=SCHEMAS[identifier]))

    return mutate


def _cycle(data_dir: Path) -> None:
    """Make a 10-K section the parent of its own (non-root) parent."""

    sections = _rows(data_dir, SECTIONS, "document_id", TEN_K)
    levels = {section["section_id"]: section["level"] for section in sections}
    child = next(s for s in sections if levels.get(s["parent_section_id"], 0) > 0)
    parent = child["parent_section_id"]
    _change(SECTIONS, "section_id", parent, {"parent_section_id": child["section_id"]})(
        data_dir
    )


def _snapshots(data_dir: Path) -> dict[str, tuple[int | None, int]]:
    with open_silver_catalog(data_dir) as catalog:
        states = {}
        for table in SILVER_TABLES:
            loaded = catalog.load_table(table.identifier)
            snapshot = loaded.current_snapshot()
            states[table.identifier] = (
                snapshot and snapshot.snapshot_id,
                len(loaded.scan().to_arrow()),
            )
        return states


def _files(data_dir: Path) -> dict[str, bytes]:
    paths = [*(data_dir / "silver").rglob("*"), *(data_dir / "_catalog").rglob("*")]
    return {str(path): path.read_bytes() for path in sorted(paths) if path.is_file()}


def _statuses(report: SilverAuditReport, status: str) -> set[tuple[str, str | None]]:
    return {(c.check_name, c.table) for c in report.checks if c.status == status}


def test_healthy_silver_passes_and_traces_every_form_to_bronze(silver: Path) -> None:
    states, files = _snapshots(silver), _files(silver)

    report = audit_silver(silver, AuditOptions(batch_size=7))

    assert _statuses(report, "FAIL") == set()
    # The fixture's unresolved links are valid data, reported as a warning.
    assert _statuses(report, "WARN") == {("resolved_links", LINKS)}
    assert report.status == "WARN"
    assert list(report.tables) == sorted(table.identifier for table in SILVER_TABLES)
    assert report.rows_scanned == sum(t.row_count for t in report.tables.values())
    assert [(s.form, s.filing_id) for s in report.samples] == [
        ("10-K", TEN_K_FILING),
        ("10-Q", TEN_Q_FILING),
        ("8-K", EIGHT_K_FILING),
    ]
    sample_checks = [c for c in report.checks if c.check_name.startswith("sample_")]
    assert {c.check_name for c in sample_checks} >= {
        "sample_bronze_files",
        "sample_documents",
        "sample_entities",
        "sample_narrative",
        "sample_tables",
        "sample_xbrl",
    }
    assert {c.status for c in sample_checks} <= {"PASS", "SKIP"}
    ten_k = "\n".join(report.samples[0].lines)
    assert "SHA-256 verified" in ten_k and "fact 1:" in ten_k
    assert "PART" in ten_k and "ITEM" in ten_k
    assert report.snapshot_changes == {}
    assert _snapshots(silver) == states
    assert _files(silver) == files
    summary = report.summary()
    assert summary["tables_discovered"] == 19 and summary["tables_failed"] == 0
    assert summary["checks_failed"] == 0 and summary["samples_checked"] == 3


def test_null_link_targets_are_not_orphans(silver: Path) -> None:
    unresolved = _rows(silver, LINKS, "resolution_status", "unresolved")
    assert unresolved and all(row["target_document_id"] is None for row in unresolved)

    report = audit_silver(silver, AuditOptions(samples_per_form=0))

    (check,) = [
        c
        for c in report.checks
        if c.check_name == "orphan:document_links.target_document_id -> documents"
    ]
    assert (check.status, check.failed_count) == ("PASS", 0)


@pytest.mark.parametrize(
    ("mutate", "check_name", "table"),
    [
        (_change(DOCUMENTS, "document_id", f"{TEN_Q_FILING}:document:0003", {}, keep=True), "unique:document_id", DOCUMENTS),
        (_change(FILINGS, "filing_id", EIGHT_K_FILING), "orphan:documents.filing_id -> filings", DOCUMENTS),
        (_change(BLOCKS, "block_id", f"{TEN_Q}:block:000012"), "complete_group", BLOCKS),
        (_change(FACTS, "fact_id", f"{TEN_K_FILING}:xbrl:inline:default:fact:00000001"), "declared_count:fact_count", INSTANCES),
        (_change(BLOCKS, "block_id", f"{TEN_Q}:block:000012", {"block_id": f"{TEN_Q}:block:999999", "block_type": "table_cell"}, keep=True), "no_table_cell_blocks", BLOCKS),
        (_change(SECTIONS, "section_id", f"{TEN_K}:section:0002", {"parent_section_id": f"{TEN_K}:section:0002"}), "no_self_parent", SECTIONS),
        (_cycle, "parent_shallower", SECTIONS),
        (_change(DOCUMENTS, "document_id", f"{TEN_Q_FILING}:document:0002", {"content_start_byte": None, "content_end_byte": None, "content_sha256": None, "content_size_bytes": 0}), "sample_documents", DOCUMENTS),
        (_change(SECTIONS, "section_id", f"{TEN_K}:section:0008"), "complete_group", SECTIONS),
        (_change(SECTIONS, "section_id", f"{TEN_K}:section:0004", {"level": 1}), "parent_shallower", SECTIONS),
        (_change(TABLES, "table_id", EIGHT_K_TABLE), "table_block_has_table", BLOCKS),
        (_change(TABLES, "table_id", EIGHT_K_TABLE, {"table_id": f"{EIGHT_K}:block:000002:table", "block_id": f"{EIGHT_K}:block:000002"}), "table_on_table_block", TABLES),
        (_change(TABLES, "table_id", EIGHT_K_TABLE, {"header_cell_count": 1}), "header_cell_count", TABLES),
        (_change(TABLES, "table_id", EIGHT_K_TABLE, {"column_count": 9}), "cells_fill_grid", TABLES),
        (_change(CELLS, "cell_id", f"{EIGHT_K_TABLE}:cell:000002"), "complete_group", CELLS),
        (_change(CELLS, "table_id", EIGHT_K_TABLE), "declared_count:cell_count", TABLES),
        (_change(LINKS, "link_id", f"{EIGHT_K}:link:000003"), "complete_group", LINKS),
        (_change(LINKS, "link_id", f"{EIGHT_K}:link:000002", {"target_document_id": None}), "row_contract", LINKS),
        (_change(LINKS, "link_id", f"{EIGHT_K}:link:000002", {"target_document_id": f"{TEN_K_FILING}:document:0002"}), "lineage:document_links.target_document_id -> documents", LINKS),
        (_change(DIMENSIONS, "context_key", context_key_for(TEN_K_INSTANCE, "FY_Widgets")), "declared_count:expected_dimension_count", CONTEXTS),
        (_change(RELATIONSHIPS, "network_id", PRESENTATION), "declared_count:relationship_count", NETWORKS),
        (_change(RELATIONSHIPS, "network_id", PRESENTATION, {"weight": "1"}), "relationship_attributes", RELATIONSHIPS),
    ],
    ids=[
        "duplicate-document-id",
        "orphan-documents-of-a-deleted-filing",
        "block-missing-from-its-group",
        "instance-missing-a-fact",
        "legacy-table-cell-block",
        "section-is-its-own-parent",
        "section-parent-cycle",
        "document-span-differs-from-bronze",
        "truncated-sections",
        "section-level-not-below-parent",
        "table-block-without-document-table",
        "document-table-on-a-non-table-block",
        "header-count-differs-from-cells",
        "cells-do-not-span-the-columns",
        "truncated-cells",
        "table-without-cells",
        "truncated-links",
        "resolved-link-without-target",
        "resolved-target-in-another-filing",
        "context-without-its-dimensions",
        "network-without-relationships",
        "attribute-of-another-linkbase-type",
    ],
)  # fmt: skip
def test_broken_silver_fails_the_check_without_changes(
    silver: Path, mutate: Callable[[Path], None], check_name: str, table: str
) -> None:
    mutate(silver)
    states, files = _snapshots(silver), _files(silver)

    report = audit_silver(silver)

    assert report.status == "FAIL"
    assert (check_name, table) in _statuses(report, "FAIL")
    assert _snapshots(silver) == states
    assert _files(silver) == files


def test_missing_table_skips_the_row_checks(silver: Path) -> None:
    with open_silver_catalog(silver) as catalog:
        catalog.drop_table(LINKS)

    report = audit_silver(silver)

    assert ("catalog_has_defined_tables", None) in _statuses(report, "FAIL")
    assert ("row_checks", None) in _statuses(report, "SKIP")
    assert ("read_only", None) in _statuses(report, "PASS")
    assert report.rows_scanned == 0 and LINKS not in report.tables


def test_schema_mismatch_skips_the_row_checks(silver: Path) -> None:
    title = SCHEMAS[LINKS].get_field_index("title")
    with open_silver_catalog(silver) as catalog:
        catalog.drop_table(LINKS)
        create_or_load_table(
            catalog,
            LINKS,
            arrow_schema=SCHEMAS[LINKS].set(
                title, SCHEMAS[LINKS].field(title).with_nullable(False)
            ),
            location=silver_table_location(silver, LINKS),
            partition_by=FILING_YEAR_PARTITION,
        )

    report = audit_silver(silver)

    (check,) = [c for c in report.checks if c.status == "FAIL"]
    assert (check.check_name, check.table) == ("schema_and_partition", LINKS)
    assert f"column {title + 1}" in check.message and "title" in check.message
    assert ("row_checks", None) in _statuses(report, "SKIP")


def test_bronze_checksum_mismatch_fails_its_sample_and_the_full_rehash(
    silver: Path,
) -> None:
    submission = next(silver.glob("bronze/**/0000320193-25-000057/full-submission.txt"))
    submission.write_bytes(submission.read_bytes().replace(b"None.", b"Nope."))

    report = audit_silver(silver)

    (check,) = [c for c in report.checks if c.status == "FAIL"]
    assert (check.check_name, check.sample_ids) == (
        "sample_bronze_files",
        (TEN_Q_FILING,),
    )
    assert "SHA-256 differs" in check.message
    assert ("source_hashes", FILINGS) in _statuses(report, "SKIP")
    report = audit_silver(
        silver, AuditOptions(samples_per_form=0, verify_source_hashes=True)
    )
    (check,) = [c for c in report.checks if c.status == "FAIL"]
    assert (check.check_name, check.sample_ids) == ("source_hashes", (TEN_Q_FILING,))
    assert check.rows_checked == 3


def test_legacy_documents_identifier_is_reported(silver: Path) -> None:
    with open_silver_catalog(silver) as catalog:
        catalog.create_table("sec_filings.document", schema=DOCUMENTS_SPEC.arrow_schema)

    report = audit_silver(silver)

    (naming,) = [c for c in report.checks if c.check_name == "documents_identifier"]
    assert (naming.status, naming.sample_ids) == ("FAIL", ("catalog",))
    assert ("catalog_tables_managed", None) in _statuses(report, "WARN")
    assert "sec_filings.document" in report.tables


def test_failures_keep_a_bounded_number_of_sample_ids(silver: Path) -> None:
    _change(DOCUMENTS, "document_id", TEN_K)(silver)

    report = audit_silver(silver)

    (check,) = [
        c
        for c in report.checks
        if c.check_name == "orphan:content_blocks.document_id -> documents"
    ]
    assert check.status == "FAIL"
    assert check.failed_count > SAMPLE_IDS
    assert len(check.sample_ids) == SAMPLE_IDS


def test_a_write_during_the_audit_fails_it(
    silver: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    check_sample, written = silver_audit.check_sample, []

    def check_and_write(*args: Any) -> None:
        check_sample(*args)
        if not written:
            _change(LINKS, "link_id", f"{TEN_K}:link:000001")(silver)
            written.append(LINKS)

    monkeypatch.setattr(silver_audit, "check_sample", check_and_write)

    report = audit_silver(silver)

    assert ("read_only", None) in _statuses(report, "FAIL")
    assert set(report.snapshot_changes) == {LINKS}


def test_rows_are_read_in_batches_never_as_whole_tables(
    silver: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_sizes: list[int] = []
    iter_batches = pq.ParquetFile.iter_batches

    def spy_batches(self: pq.ParquetFile, *args: Any, **kwargs: Any) -> Any:
        for batch in iter_batches(self, *args, **kwargs):
            batch_sizes.append(batch.num_rows)
            yield batch

    def whole_table(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a scan was materialized")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", spy_batches)
    monkeypatch.setattr(DataScan, "to_arrow", whole_table)
    monkeypatch.setattr(DataScan, "to_arrow_batch_reader", whole_table)

    report = audit_silver(silver, AuditOptions(batch_size=3))

    assert report.status == "WARN"
    assert sum(batch_sizes) == report.rows_scanned
    assert max(batch_sizes) == 3
    assert len(report.samples) == 3


@pytest.mark.parametrize(
    ("damage", "exit_code"),
    [(None, 0), ("orphan", 1), ("missing-catalog", 2)],
)
def test_cli_exit_codes(
    silver: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    damage: str | None,
    exit_code: int,
) -> None:
    data_dir = silver / "missing" if damage == "missing-catalog" else silver
    if damage == "orphan":
        _change(FILINGS, "filing_id", EIGHT_K_FILING)(silver)
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_silver.py", "--data-dir", str(data_dir), "--batch-size", "50"],
    )

    assert main() == exit_code
    assert not (silver / "missing").exists()
    if exit_code != 2:
        assert "SEC Silver audit:" in capsys.readouterr().out


def test_cli_rejects_a_zero_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["audit_silver.py", "--batch-size", "0"])

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 2


def _outcomes(report: SilverAuditReport) -> list[tuple[Any, ...]]:
    return [
        (c.check_name, c.table, c.status, c.rows_checked, c.failed_count, c.sample_ids)
        for c in report.checks
    ]


def _scans(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The tables scan_table reads from now on."""

    scan_table, scanned = silver_audit.scan_table, []

    def spy(table: Any, identifier: str, *args: Any, **kwargs: Any) -> Any:
        scanned.append(identifier)
        return scan_table(table, identifier, *args, **kwargs)

    monkeypatch.setattr(silver_audit, "scan_table", spy)
    return scanned


def test_an_audit_stopped_in_its_checks_resumes_after_the_last_completed_one(
    silver: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh = audit_silver(silver, AuditOptions(batch_size=7))
    work_dir = tmp_path / "audit"
    options = AuditOptions(batch_size=7, work_dir=work_dir)
    with monkeypatch.context() as patch:
        patch.setattr(
            silver_audit_sql, "_check_groups", Mock(side_effect=OSError("power"))
        )
        with pytest.raises(OSError, match="power"):
            audit_silver(silver, options)
    assert (work_dir / "audit.duckdb").is_file()
    scanned = _scans(monkeypatch)
    orphans = Mock(wraps=silver_audit_sql._check_orphans)
    monkeypatch.setattr(silver_audit_sql, "_check_orphans", orphans)

    report = audit_silver(silver, options)

    assert scanned == []
    orphans.assert_not_called()  # completed before the stop
    assert _outcomes(report) == _outcomes(fresh)
    assert report.rows_scanned == fresh.rows_scanned
    assert [(s.form, s.filing_id) for s in report.samples] == [
        (s.form, s.filing_id) for s in fresh.samples
    ]
    assert all(s.lines for s in report.samples)
    assert not (work_dir / "audit.duckdb").exists()


def test_an_audit_stopped_in_a_scan_scans_that_table_again(
    silver: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh = audit_silver(silver, AuditOptions(batch_size=3))
    options = AuditOptions(batch_size=3, work_dir=tmp_path / "audit")
    load = silver_audit_sql.ScratchDatabase.load

    def stop_while_storing_cells(
        self: silver_audit_sql.ScratchDatabase, identifier: str, *args: Any
    ) -> None:
        load(self, identifier, *args)
        if identifier == CELLS:
            raise OSError("power")

    with monkeypatch.context() as patch:
        patch.setattr(
            silver_audit_sql.ScratchDatabase, "load", stop_while_storing_cells
        )
        with pytest.raises(OSError, match="power"):
            audit_silver(silver, options)
    scanned = _scans(monkeypatch)

    report = audit_silver(silver, options)

    order = [table.identifier for table in SILVER_TABLES]
    assert scanned == order[order.index(CELLS) :]
    assert _outcomes(report) == _outcomes(fresh)
    assert report.rows_scanned == fresh.rows_scanned


def test_a_changed_table_starts_the_stopped_audit_over(
    silver: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = AuditOptions(work_dir=tmp_path / "audit")
    with monkeypatch.context() as patch:
        patch.setattr(
            silver_audit_sql, "_check_groups", Mock(side_effect=OSError("power"))
        )
        with pytest.raises(OSError, match="power"):
            audit_silver(silver, options)
    _change(LINKS, "link_id", f"{TEN_K}:link:000001", {}, keep=True)(silver)
    scanned = _scans(monkeypatch)

    report = audit_silver(silver, options)

    assert scanned == [table.identifier for table in SILVER_TABLES]
    assert ("unique:link_id", LINKS) in _statuses(report, "FAIL")


def test_worker_processes_give_the_same_report(silver: Path) -> None:
    serial = audit_silver(silver, AuditOptions(batch_size=3))

    parallel = audit_silver(silver, AuditOptions(batch_size=3, workers=2))

    assert _outcomes(parallel) == _outcomes(serial)
    assert [s.lines for s in parallel.samples] == [s.lines for s in serial.samples]

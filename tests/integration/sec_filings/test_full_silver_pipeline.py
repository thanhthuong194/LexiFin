from __future__ import annotations

import csv
import io
import sys
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pytest
from pyiceberg.expressions import EqualTo

from scripts.sec_filings import query_silver
from scripts.sec_filings.build_silver import main
from src.data_pipeline.pipelines.sec_filings import xbrl_instance_loading
from src.data_pipeline.pipelines.sec_filings.bronze_to_silver import (
    run_sec_filings_silver,
)
from src.data_pipeline.quality.sec_filings.silver_audit import audit_silver
from src.data_pipeline.quality.sec_filings.silver_model import SILVER_TABLES
from src.data_pipeline.storage.iceberg import open_silver_catalog
from src.data_pipeline.storage.sec_filings.xbrl_facts import XBRL_FACTS_SPEC

pytestmark = pytest.mark.integration

FILINGS = [
    ("0000320193", "10-K", "0000320193-24-000123", "20241101"),
    ("0000320193", "10-Q", "0000320193-25-000057", "20250502"),
    ("0000789019", "8-K", "0000789019-25-000010", "20250129"),
]
TEN_K_ID = "sec:0000320193-24-000123"
TEN_K_DOCUMENT = f"{TEN_K_ID}:document:0001"
TEN_K_INSTANCE = f"{TEN_K_ID}:xbrl:inline:default"
EIGHT_K_ID = "sec:0000789019-25-000010"
NEW_FILING = ("0000320193", "10-Q", "0000320193-26-000001", "20260130")
NEW_ID = "sec:0000320193-26-000001"


def _write(data_dir: Path, write_sec_filing, filings=FILINGS) -> None:
    for cik, form, accession, filed in filings:
        write_sec_filing(data_dir, cik=cik, form=form, accession=accession, filed=filed)


@pytest.fixture
def silver(tmp_path: Path, write_sec_filing) -> Path:
    """Bronze for a 10-K, a 10-Q and an 8-K, and the Silver model built from it."""

    _write(tmp_path, write_sec_filing)
    assert run_sec_filings_silver(tmp_path).completed
    return tmp_path


def _cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["build_silver.py", *args])
    return main()


def _snapshots(data_dir: Path) -> dict[str, int | None]:
    with open_silver_catalog(data_dir) as catalog:
        return {
            t.identifier: (
                snapshot.snapshot_id
                if catalog.table_exists(t.identifier)
                and (snapshot := catalog.load_table(t.identifier).current_snapshot())
                else None
            )
            for t in SILVER_TABLES
        }


def _rows(data_dir: Path, identifier: str) -> list[dict[str, Any]]:
    with open_silver_catalog(data_dir) as catalog:
        rows: list[dict[str, Any]] = (
            catalog.load_table(identifier).scan().to_arrow().to_pylist()
        )
        return rows


def _files(data_dir: Path) -> dict[str, bytes]:
    return {
        str(path): path.read_bytes()
        for path in sorted(data_dir.rglob("*"))
        if path.is_file() and "bronze" not in path.parts
    }


def _query(data_dir: Path, sql: str, *params: object) -> list[tuple[Any, ...]]:
    """Run SQL over every Silver table, registered by its short name."""

    connection = duckdb.connect()
    with open_silver_catalog(data_dir) as catalog:
        for table in SILVER_TABLES:
            arrow = catalog.load_table(table.identifier).scan().to_arrow()
            connection.register(table.identifier.split(".")[1], arrow)
    rows: list[tuple[Any, ...]] = connection.execute(sql, list(params)).fetchall()
    return rows


def test_one_command_builds_the_whole_model_and_a_rerun_changes_nothing(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, write_sec_filing)

    assert _cli(monkeypatch, "--data-dir", str(tmp_path)) == 0

    with open_silver_catalog(tmp_path) as catalog:
        for table in SILVER_TABLES:
            loaded = catalog.load_table(table.identifier)
            assert table.spec.definition_problem(loaded) is None
    report = audit_silver(tmp_path)
    # The fixture's unresolved links are valid data, reported as a warning.
    assert report.status == "WARN"
    assert {c.status for c in report.checks if c.check_name.startswith(("unique:", "orphan:", "lineage:"))} == {"PASS"}  # fmt: skip
    rows = {name: table.row_count for name, table in report.tables.items()}
    assert all(rows[t.identifier] > 0 for t in SILVER_TABLES)
    snapshots = _snapshots(tmp_path)

    rerun = run_sec_filings_silver(tmp_path)

    assert rerun.completed
    assert rerun.stages_completed == rerun.stages_requested
    assert rerun.rows_inserted == {}
    assert rerun.rows_skipped == sum(rows.values())
    assert rerun.summary()["snapshot_changes"] == {}
    assert rerun.integrity is not None
    assert rerun.integrity.rows_scanned == report.rows_scanned == sum(rows.values())
    assert _snapshots(tmp_path) == snapshots


def test_representative_queries_follow_the_model_keys(silver: Path) -> None:
    company_filings = _query(
        silver,
        "select f.form, f.filing_date from filing_entities e "
        "join filings f on f.filing_id = e.filing_id "
        "where e.cik = ? order by f.filing_date",
        "0000320193",
    )
    risk_factors = _query(
        silver,
        "select b.text from filings f "
        "join documents d on d.filing_id = f.filing_id and d.role = 'primary_document' "
        "join sections s on s.document_id = d.document_id "
        "join content_blocks b on b.document_id = s.document_id "
        " and b.block_ordinal >= s.start_block_ordinal "
        " and b.block_ordinal < s.end_block_ordinal "
        "where f.filing_id = ? and s.section_code = 'ITEM 1A' order by b.block_ordinal",
        TEN_K_ID,
    )
    hierarchy = _query(
        silver,
        "select p.section_code, c.section_code from sections c "
        "join sections p on p.section_id = c.parent_section_id "
        "join sections r on r.section_id = p.parent_section_id "
        "where c.document_id = ? and r.parent_section_id is null "
        "order by c.section_ordinal",
        TEN_K_DOCUMENT,
    )
    grid = _query(
        silver,
        "select c.row_ordinal, c.column_ordinal, c.colspan, c.text, "
        "t.row_count, t.column_count from content_blocks b "
        "join document_tables t on t.block_id = b.block_id "
        "join document_table_cells c on c.table_id = t.table_id "
        "where b.document_id = ? and b.text like 'Revenue%' order by c.cell_ordinal",
        TEN_K_DOCUMENT,
    )

    assert company_filings == [("10-K", date(2024, 11, 1)), ("10-Q", date(2025, 5, 2))]
    assert risk_factors == [
        ("Item 1A. Risk Factors",),
        ("As discussed in Item 7, demand may fall.",),
    ]
    assert hierarchy == [
        ("PART I", "ITEM 1"),
        ("PART I", "ITEM 1A"),
        ("PART II", "ITEM 7"),
    ]
    assert grid == [(1, 1, 1, "Revenue", 1, 2), (1, 2, 1, "$1,000", 1, 2)]


def test_exhibit_and_xbrl_queries_follow_the_model_keys(silver: Path) -> None:
    exhibit_index = _query(
        silver,
        "select x.exhibit_number, x.description, d.filename from exhibits x "
        "left join documents d on d.document_id = x.resolved_document_id "
        "where x.filing_id = ? order by x.exhibit_ordinal",
        EIGHT_K_ID,
    )
    attachments = _query(
        silver, "select count(*) from documents where role = 'exhibit'"
    )
    fact = _query(
        silver,
        "select c.local_name, f.numeric_value, x.period_type, x.end_date, "
        "u.unit_kind, dc.local_name, coalesce(m.local_name, d.typed_member_value) "
        "from xbrl_facts f "
        "join xbrl_concepts c on c.concept_id = f.concept_id "
        "join xbrl_contexts x on x.context_key = f.context_key "
        "left join xbrl_units u on u.unit_key = f.unit_key "
        "left join xbrl_dimensions d on d.context_key = f.context_key "
        "left join xbrl_concepts dc on dc.concept_id = d.dimension_concept_id "
        "left join xbrl_concepts m on m.concept_id = d.member_concept_id "
        "where f.instance_id = ? and c.local_name = 'EarningsPerShare' "
        "order by d.dimension_ordinal",
        TEN_K_INSTANCE,
    )
    # Labels are a dictionary over every DTS file: an instance uses those of
    # its filing's own files and of published taxonomies (URLs).
    presentation = _query(
        silver,
        "with dts_labels as (select l.concept_id, l.label_text, d.filing_id "
        " from xbrl_labels l "
        " left join documents d on d.document_id = l.source_document_id "
        " where l.role_uri = 'http://www.xbrl.org/2003/role/label' "
        " and l.language = 'en-us') "
        "select s.local_name, t.local_name, l.label_text from xbrl_networks n "
        "join xbrl_instances i on i.instance_id = n.instance_id "
        "join xbrl_relationships r on r.network_id = n.network_id "
        "join xbrl_concepts s on s.concept_id = r.source_concept_id "
        "join xbrl_concepts t on t.concept_id = r.target_concept_id "
        "left join dts_labels l on l.concept_id = t.concept_id "
        " and (l.filing_id = i.filing_id or l.filing_id is null) "
        "where n.instance_id = ? and n.linkbase_type = 'presentation' "
        "qualify row_number() over "
        " (partition by r.relationship_id order by l.filing_id is null) = 1 "
        "order by t.local_name",
        TEN_K_INSTANCE,
    )

    assert exhibit_index == [("99.1", "Press release", None)]
    assert attachments == [(3,)]
    assert fact == [
        ("EarningsPerShare", "2.5", "duration", date(2024, 12, 31), "divide", "ProductAxis", "WidgetsMember"),
        ("EarningsPerShare", "2.5", "duration", date(2024, 12, 31), "divide", "RegionAxis", "North"),
    ]  # fmt: skip
    assert presentation == [
        ("StatementAbstract", "ProductTable", None),
        ("StatementAbstract", "Revenue", "Revenue"),
    ]


def test_rows_trace_back_to_their_bronze_files(silver: Path) -> None:
    block = _query(
        silver,
        "select d.filename, d.role, f.accession_number, f.full_submission_path "
        "from content_blocks b join documents d on d.document_id = b.document_id "
        "join filings f on f.filing_id = d.filing_id where b.text = 'We make widgets.'",
    )
    cell = _query(
        silver,
        "select d.filename, d.content_sha256 is not null from document_table_cells c "
        "join documents d on d.document_id = c.document_id where c.text = '$1,000'",
    )
    fact = _query(
        silver,
        "select d.filename, c.text from xbrl_facts f "
        "join documents d on d.document_id = f.document_id "
        "join document_table_cells c on c.cell_id = f.cell_id "
        "where f.instance_id = ?",
        TEN_K_INSTANCE,
    )
    relationship_files = _query(
        silver,
        "select distinct d.filename, d.role from xbrl_relationships r "
        "join xbrl_networks n on n.network_id = r.network_id "
        "join documents d on d.document_id = r.source_document_id "
        "where n.instance_id = ?",
        TEN_K_INSTANCE,
    )

    assert block == [
        (
            "primary.htm",
            "primary_document",
            "0000320193-24-000123",
            (
                "bronze/sec_filings/sec-edgar-filings/0000320193/10-K/"
                "0000320193-24-000123/full-submission.txt"
            ),
        )
    ]
    assert cell == [("primary.htm", True)]
    assert fact == [("primary.htm", "$1,000")]
    assert relationship_files == [("ex-2024.xsd", "xbrl_schema")]


def test_new_filing_appends_only_its_own_groups(silver: Path, write_sec_filing) -> None:
    before = {t.identifier: _rows(silver, t.identifier) for t in SILVER_TABLES}
    _write(silver, write_sec_filing, [NEW_FILING])

    result = run_sec_filings_silver(silver)

    assert result.completed
    instances = {r["instance_id"] for r in _rows(silver, "sec_filings.xbrl_instances")}
    new_instances = {i for i in instances if i.startswith(f"{NEW_ID}:")}
    for table in SILVER_TABLES:
        key = table.unique[0][0]
        after = {row[key]: row for row in _rows(silver, table.identifier)}
        old = before[table.identifier]
        # Stored rows are never rewritten.
        assert all(after.get(row[key]) == row for row in old), table.identifier
        added = [after[k] for k in after.keys() - {row[key] for row in old}]
        for row in added:
            owner = row.get("filing_id") or row.get("instance_id")
            assert owner in {NEW_ID, *new_instances} or owner is None, table.identifier
    assert result.rows_inserted["sec_filings.filings"] == 1
    assert len(new_instances) == 1


def test_corrupted_state_fails_the_integrity_gate(
    silver: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with open_silver_catalog(silver) as catalog:
        facts = catalog.load_table(XBRL_FACTS_SPEC.identifier)
        row = next(r for r in facts.scan().to_arrow().to_pylist() if r["block_id"])
        facts.delete(EqualTo("fact_id", row["fact_id"]))
        # An aligned block that does not exist: the block is no part of the
        # fact's hash, so the facts stage skips the instance; only the audit
        # follows the reference.
        moved = {**row, "block_id": f"{row['document_id']}:block:000999"}
        facts.append(pa.Table.from_pylist([moved], schema=XBRL_FACTS_SPEC.arrow_schema))

    result = run_sec_filings_silver(silver)

    assert result.failed_stage == "integrity"
    assert result.stages_completed == result.stages_requested[:-1]
    assert result.integrity is not None
    (failure,) = result.summary()["integrity_failures"]
    assert (failure["check_name"], failure["table"]) == (
        "orphan:xbrl_facts.block_id -> content_blocks",
        "sec_filings.xbrl_facts",
    )
    assert failure["sample_ids"] == (row["fact_id"],)
    assert _cli(monkeypatch, "--data-dir", str(silver)) == 1
    assert _cli(monkeypatch, "--data-dir", str(silver), "--check-only") == 1


def test_bronze_and_silver_can_live_in_separate_directories(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bronze written under one root, then moved and renamed apart from Silver.
    _write(tmp_path / "ingest", write_sec_filing)
    bronze = tmp_path / "bronze" / "sec_filings_3"
    bronze.parent.mkdir()
    (tmp_path / "ingest" / "bronze" / "sec_filings").rename(bronze)
    silver = tmp_path / "silver" / "sec_filings_3"

    assert (
        _cli(monkeypatch, "--data-dir", str(silver), "--bronze-dir", str(bronze)) == 0
    )

    assert not (silver / "bronze").exists()
    (filing, *_) = _rows(silver, "sec_filings.filings")
    assert filing["full_submission_path"].startswith(
        "bronze/sec_filings/sec-edgar-filings/"
    )
    assert (
        _cli(
            monkeypatch,
            "--data-dir",
            str(silver),
            "--bronze-dir",
            str(bronze),
            "--check-only",
        )
        == 0
    )
    # Without its Bronze directory the audit cannot find the files.
    assert _cli(monkeypatch, "--data-dir", str(silver), "--check-only") == 1


def test_xbrl_workers_build_the_same_rows(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "ingest", write_sec_filing)
    bronze = tmp_path / "ingest" / "bronze" / "sec_filings"
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"

    assert (
        _cli(monkeypatch, "--data-dir", str(serial), "--bronze-dir", str(bronze)) == 0
    )
    assert (
        _cli(
            monkeypatch,
            "--data-dir",
            str(parallel),
            "--bronze-dir",
            str(bronze),
            "--xbrl-workers",
            "2",
        )
        == 0
    )

    def content(data_dir: Path, identifier: str) -> list[str]:
        return sorted(
            repr(sorted((k, v) for k, v in row.items() if k != "processed_at"))
            for row in _rows(data_dir, identifier)
        )

    for table in SILVER_TABLES:
        assert content(parallel, table.identifier) == content(serial, table.identifier)
    assert _rows(serial, "sec_filings.xbrl_relationships")


def test_check_only_reads_without_writing(
    silver: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots, files = _snapshots(silver), _files(silver)

    assert _cli(monkeypatch, "--data-dir", str(silver), "--check-only") == 0

    assert _snapshots(silver) == snapshots
    assert _files(silver) == files


def test_from_stage_needs_the_tables_of_the_stages_it_skips(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, write_sec_filing)

    no_catalog = run_sec_filings_silver(tmp_path, from_stage="narrative")

    assert no_catalog.prerequisite_problems == (
        f"no Silver catalog under {tmp_path.resolve()}",
    )
    assert no_catalog.stages_skipped == no_catalog.stages_requested
    assert not (tmp_path / "_catalog").exists()
    assert run_sec_filings_silver(tmp_path, to_stage="documents").completed
    no_narrative = run_sec_filings_silver(
        tmp_path, from_stage="document_tables", to_stage="document_tables"
    )
    assert no_narrative.prerequisite_problems == (
        "sec_filings.sections does not exist",
        "sec_filings.content_blocks does not exist",
    )
    assert (
        _cli(monkeypatch, "--data-dir", str(tmp_path), "--from-stage", "narrative") == 1
    )
    snapshots = _snapshots(tmp_path)

    resumed = run_sec_filings_silver(tmp_path, from_stage="filing_entities")

    assert resumed.completed
    assert (resumed.filings, resumed.documents) == (None, None)
    for identifier in ("sec_filings.filings", "sec_filings.documents"):
        assert _snapshots(tmp_path)[identifier] == snapshots[identifier]


def test_to_stage_stops_after_it(tmp_path: Path, write_sec_filing) -> None:
    _write(tmp_path, write_sec_filing)

    result = run_sec_filings_silver(tmp_path, to_stage="document_links")

    assert result.completed
    assert result.stages_requested[-1] == "document_links"
    assert (result.exhibit_index, result.integrity) == (None, None)
    with open_silver_catalog(tmp_path) as catalog:
        assert catalog.table_exists("sec_filings.document_links")
        assert not catalog.table_exists("sec_filings.exhibits")
        assert not catalog.table_exists("sec_filings.xbrl_instances")


def test_a_rejected_stage_skips_every_later_stage(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, write_sec_filing)
    write_sec_filing(
        tmp_path,
        cik="0000320193",
        form="8-K",
        accession="0000320193-26-000002",
        filed="20260105",
        header_form="8-K/A",
    )

    result = run_sec_filings_silver(tmp_path)

    assert result.failed_stage == "filings"
    assert result.stages_completed == ()
    assert result.stages_skipped == result.stages_requested[1:]
    assert result.summary()["failed_stage_summary"]["failed"] == 1
    assert _cli(monkeypatch, "--data-dir", str(tmp_path)) == 1


def _sample_query(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    data_dir: Path,
    query: str,
    **params: str,
) -> tuple[int, list[dict[str, str]]]:
    path = Path(query_silver.__file__).parent / "queries" / query
    arguments = [f"--param={name}={value}" for name, value in params.items()]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "query_silver.py",
            str(path),
            "--data-dir",
            str(data_dir),
            "--csv",
            *arguments,
        ],
    )
    status = query_silver.main()
    return status, list(csv.DictReader(io.StringIO(capsys.readouterr().out)))


def test_sample_queries_read_silver_without_writing(
    silver: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _snapshots(silver)
    table_id = next(
        cell["table_id"]
        for cell in _rows(silver, "sec_filings.document_table_cells")
        if cell["filing_id"] == TEN_K_ID and cell["text"] == "Revenue"
    )

    revenue = _sample_query(
        monkeypatch, capsys, silver, "annual_values.sql", concept="Revenue"
    )
    risk = _sample_query(monkeypatch, capsys, silver, "item_1a.sql", filing_id=TEN_K_ID)
    table = _sample_query(
        monkeypatch, capsys, silver, "table_rows.sql", table_id=table_id
    )
    missing = _sample_query(monkeypatch, capsys, silver, "item_1a.sql")

    assert revenue[0] == 0
    assert [
        (r["cik"], r["start_date"], r["end_date"], r["value"]) for r in revenue[1]
    ] == [("0000320193", "2024-01-01", "2024-12-31", "1000.0")]
    assert risk[0] == 0
    assert [(r["block_type"], r["text"]) for r in risk[1]] == [
        ("heading", "Item 1A. Risk Factors"),
        ("paragraph", "As discussed in Item 7, demand may fall."),
    ]
    assert table == (0, [{"row_ordinal": "1", "cells": "Revenue | $1,000"}])
    assert missing[0] == 1
    assert _snapshots(silver) == before


def _content(data_dir: Path, identifier: str) -> list[str]:
    return sorted(
        repr(sorted((k, v) for k, v in row.items() if k != "processed_at"))
        for row in _rows(data_dir, identifier)
    )


def _count_loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The entry files of every instance the processor loads from now on."""

    load_instance, loaded = xbrl_instance_loading.load_instance, []

    def spy(files: Any, **kwargs: Any) -> Any:
        loaded.append(next(f.document_id for f in files if f.is_entry))
        return load_instance(files, **kwargs)

    monkeypatch.setattr(xbrl_instance_loading, "load_instance", spy)
    return loaded


def test_the_xbrl_stages_load_each_instance_once_and_build_the_same_rows(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "ingest", write_sec_filing)
    bronze = tmp_path / "ingest" / "bronze" / "sec_filings"
    cached, uncached = tmp_path / "cached", tmp_path / "uncached"
    with monkeypatch.context() as patch:
        patch.setattr(xbrl_instance_loading.InstanceCache, "get", lambda *_: None)
        reloads = _count_loads(patch)
        assert run_sec_filings_silver(uncached, bronze_dir=bronze).completed
    loads = _count_loads(monkeypatch)

    assert run_sec_filings_silver(cached, bronze_dir=bronze).completed

    # Facts, enrichment (unless nothing is missing) and linkbases each load.
    assert len(reloads) > len(set(reloads)) == len(FILINGS)
    assert sorted(loads) == sorted(set(reloads))
    for table in SILVER_TABLES:
        assert _content(cached, table.identifier) == _content(
            uncached, table.identifier
        )
    # Removed once the last stage that needs it completed.
    assert not (cached / "_cache" / "xbrl_instances").exists()


def test_loads_cached_by_a_stopped_build_serve_the_next_run(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, write_sec_filing)
    assert run_sec_filings_silver(tmp_path, to_stage="xbrl_enrichment").completed
    assert len(list((tmp_path / "_cache" / "xbrl_instances").iterdir())) == len(FILINGS)
    loads = _count_loads(monkeypatch)

    assert run_sec_filings_silver(tmp_path, from_stage="xbrl_linkbases").completed

    assert loads == []
    assert _rows(tmp_path, "sec_filings.xbrl_relationships")

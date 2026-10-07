from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from pyiceberg.expressions import EqualTo

from scripts.sec_filings.validate_silver import main
from src.data_pipeline.contracts.sec_filings.xbrl_core import context_key_for
from src.data_pipeline.contracts.sec_filings.xbrl_networks import network_id_for
from src.data_pipeline.pipelines.sec_filings.bronze_to_silver import (
    run_sec_filings_silver,
)
from src.data_pipeline.quality.sec_filings.silver_audit import (
    SILVER_TABLES,
    validate_silver,
)
from src.data_pipeline.storage.iceberg import (
    create_or_load_table,
    open_silver_catalog,
    silver_table_location,
)
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
TEN_K = "sec:0000320193-24-000123:document:0001"
TEN_Q = "sec:0000320193-25-000057:document:0001"
EIGHT_K = "sec:0000789019-25-000010:document:0001"
EIGHT_K_TABLE = f"{EIGHT_K}:block:000005:table"
FACTS = "sec_filings.xbrl_facts"
DIMENSIONS = "sec_filings.xbrl_dimensions"
RELATIONSHIPS = "sec_filings.xbrl_relationships"
TEN_K_INSTANCE = "sec:0000320193-24-000123:xbrl:inline:default"
PRESENTATION = network_id_for(TEN_K_INSTANCE, "presentation", STATEMENT_ROLE)
SCHEMAS = {table.identifier: table.spec.arrow_schema for table in SILVER_TABLES}


@pytest.fixture
def silver(tmp_path: Path, write_sec_filing) -> Path:
    """A complete Silver layer built by the pipeline from three fake filings."""

    for cik, form, accession, filed in [
        ("0000320193", "10-K", "0000320193-24-000123", "20241101"),
        ("0000320193", "10-Q", "0000320193-25-000057", "20250502"),
        ("0000789019", "8-K", "0000789019-25-000010", "20250129"),
    ]:
        write_sec_filing(tmp_path, cik=cik, form=form, accession=accession, filed=filed)
    assert run_sec_filings_silver(tmp_path).completed
    return tmp_path


def _snapshots(data_dir: Path) -> dict[str, int | None]:
    with open_silver_catalog(data_dir) as catalog:
        snapshots = {}
        for table in SILVER_TABLES:
            snapshot = catalog.load_table(table.identifier).current_snapshot()
            snapshots[table.identifier] = snapshot and snapshot.snapshot_id
        return snapshots


def _files(data_dir: Path) -> dict[str, bytes]:
    return {
        str(path): path.read_bytes()
        for path in sorted((data_dir / "silver").rglob("*"))
        if path.is_file()
    } | {"catalog": (data_dir / "_catalog" / "silver.db").read_bytes()}


def _rewrite(
    identifier: str, column: str, value: str, changes: dict[str, Any] | None = None
) -> Callable[[Path], None]:
    """Delete the matching rows, then append them back with `changes` applied.

    Rows are appended as raw Arrow, so they may break the record contract.
    """

    def mutate(data_dir: Path) -> None:
        with open_silver_catalog(data_dir) as catalog:
            table = catalog.load_table(identifier)
            rows = [
                row
                for row in table.scan().to_arrow().to_pylist()
                if row[column] == value
            ]
            assert rows
            table.delete(EqualTo(column, value))
            if changes is not None:
                table.append(
                    pa.Table.from_pylist(
                        [{**row, **changes} for row in rows],
                        schema=SCHEMAS[identifier],
                    )
                )

    return mutate


def _append_copy(identifier: str, column: str, value: str, changes: dict[str, Any]):
    """Append a changed copy of the matching row next to the original."""

    def mutate(data_dir: Path) -> None:
        with open_silver_catalog(data_dir) as catalog:
            table = catalog.load_table(identifier)
            row = next(
                row
                for row in table.scan().to_arrow().to_pylist()
                if row[column] == value
            )
            table.append(
                pa.Table.from_pylist([{**row, **changes}], schema=SCHEMAS[identifier])
            )

    return mutate


def test_complete_silver_passes_without_changing_anything(silver: Path) -> None:
    snapshots, files = _snapshots(silver), _files(silver)

    report = validate_silver(silver, verify_source_hashes=True)

    assert (report.status, report.errors) == ("passed", 0)
    assert report.tables_found == [table.identifier for table in SILVER_TABLES]
    assert report.tables_missing == []
    assert report.row_counts[DOCUMENTS] == 18
    assert set(report.duplicate_counts.values()) == {0}
    assert set(report.orphan_counts.values()) == {0}
    assert not report.group_violation_counts
    assert report.source_hash_failures == 0
    # The fixture's unresolved links are valid data: a warning, not an error.
    assert report.issue_counts[("warning", "unresolved_links")] == 1
    assert report.metrics["block_types"]["table"] == report.row_counts[TABLES]
    assert report.snapshot_ids_before == report.snapshot_ids_after == snapshots
    assert _files(silver) == files


def test_missing_table_is_reported_and_not_created(silver: Path) -> None:
    with open_silver_catalog(silver) as catalog:
        catalog.drop_table(LINKS)

    report = validate_silver(silver)

    assert report.status == "failed"
    assert report.tables_missing == [LINKS]
    assert report.issue_counts[("error", "missing_table")] == 1
    assert report.issue_counts[("info", "row_checks_skipped")] == 1
    with open_silver_catalog(silver) as catalog:
        assert not catalog.table_exists(LINKS)


def test_schema_mismatch_is_reported(silver: Path) -> None:
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

    report = validate_silver(silver)

    (issue,) = [i for i in report.issues if i.check_name == "table_definition"]
    assert issue.table == LINKS
    assert f"column {title + 1}" in issue.message
    assert "title" in issue.message
    assert report.status == "failed"


@pytest.mark.parametrize(
    ("mutate", "table", "check"),
    [
        (_append_copy(DOCUMENTS, "document_id", f"{TEN_Q[:-4]}0003", {}), DOCUMENTS, "duplicate_key"),
        (_rewrite(FILINGS, "filing_id", "sec:0000789019-25-000010"), DOCUMENTS, "orphan"),
        (_rewrite(DOCUMENTS, "document_id", f"{TEN_K[:-4]}0003"), DOCUMENTS, "group"),
        (_rewrite(SECTIONS, "section_id", f"{TEN_K}:section:0008"), SECTIONS, "group"),
        (_rewrite(SECTIONS, "section_id", f"{TEN_K}:section:0004", {"level": 1}), SECTIONS, "group"),
        (_rewrite(BLOCKS, "block_id", f"{TEN_Q}:block:000012"), BLOCKS, "group"),
        (_append_copy(BLOCKS, "block_id", f"{TEN_Q}:block:000012", {"block_id": f"{TEN_Q}:block:000013", "block_type": "table_cell"}), BLOCKS, "table_semantics"),
        (_rewrite(TABLES, "table_id", EIGHT_K_TABLE), TABLES, "group"),
        (_rewrite(TABLES, "table_id", EIGHT_K_TABLE, {"table_id": f"{EIGHT_K}:block:000002:table", "block_id": f"{EIGHT_K}:block:000002"}), TABLES, "group"),
        (_rewrite(CELLS, "cell_id", f"{EIGHT_K_TABLE}:cell:000002"), CELLS, "group"),
        (_rewrite(CELLS, "table_id", EIGHT_K_TABLE), CELLS, "group"),
        (_rewrite(LINKS, "link_id", f"{EIGHT_K}:link:000003"), LINKS, "group"),
        (_rewrite(LINKS, "link_id", f"{EIGHT_K}:link:000002", {"target_document_id": None}), LINKS, "row_contract"),
        (_rewrite(LINKS, "link_id", f"{EIGHT_K}:link:000002", {"target_document_id": f"{TEN_K[:-4]}0002"}), LINKS, "group"),
        (_rewrite(FACTS, "fact_id", f"{TEN_K_INSTANCE}:fact:00000001"), FACTS, "group"),
        (_rewrite(DIMENSIONS, "context_key", context_key_for(TEN_K_INSTANCE, "FY_Widgets")), DIMENSIONS, "group"),
        (_rewrite(RELATIONSHIPS, "network_id", PRESENTATION), RELATIONSHIPS, "group"),
    ],
    ids=[
        "duplicate-document-id",
        "orphan-documents-of-a-deleted-filing",
        "document-missing-from-its-filing",
        "truncated-sections",
        "section-level-not-below-parent",
        "truncated-blocks",
        "legacy-table-cell-block",
        "table-block-without-document-table",
        "document-table-on-a-non-table-block",
        "truncated-cells",
        "table-without-cells",
        "truncated-links",
        "resolved-link-without-target",
        "resolved-target-in-another-filing",
        "instance-missing-a-fact",
        "context-without-its-dimensions",
        "network-without-relationships",
    ],
)  # fmt: skip
def test_broken_silver_is_reported_without_changes(
    silver: Path, mutate: Callable[[Path], None], table: str, check: str
) -> None:
    mutate(silver)
    snapshots = _snapshots(silver)

    report = validate_silver(silver)

    assert report.status == "failed"
    assert report.issue_counts[("error", check)] >= 1
    assert any(i.check_name == check and i.table == table for i in report.issues)
    assert report.snapshot_ids_after == report.snapshot_ids_before == snapshots


def test_source_hash_mismatch_is_found_only_when_verified(silver: Path) -> None:
    submission = next(silver.glob("bronze/**/0000320193-25-000057/full-submission.txt"))
    submission.write_bytes(submission.read_bytes().replace(b"None.", b"Nope."))

    assert validate_silver(silver).status == "passed"
    report = validate_silver(silver, verify_source_hashes=True)

    assert report.status == "failed"
    assert report.source_hash_failures == 1
    (issue,) = [i for i in report.issues if i.check_name == "source_hash"]
    assert issue.identifier == "sec:0000320193-25-000057"


@pytest.mark.parametrize(
    ("damage", "exit_code"),
    [(None, 0), (LINKS, 1), ("catalog", 2)],
    ids=["passed", "validation-error", "missing-catalog"],
)
def test_cli_exit_codes(
    silver: Path, monkeypatch: pytest.MonkeyPatch, damage: str | None, exit_code: int
) -> None:
    data_dir = silver
    if damage == LINKS:
        with open_silver_catalog(silver) as catalog:
            catalog.drop_table(LINKS)
    elif damage == "catalog":
        data_dir = silver / "missing"
    monkeypatch.setattr(
        sys, "argv", ["validate_silver.py", "--data-dir", str(data_dir)]
    )

    assert main() == exit_code
    assert not (silver / "missing").exists()

from __future__ import annotations

import sys
from contextlib import closing
from dataclasses import fields, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from scripts.sec_filings import build_silver
from src.data_pipeline.pipelines.sec_filings.bronze_to_silver import SilverRunResult
from src.data_pipeline.pipelines.sec_filings.silver_stages import (
    SILVER_STAGES,
    SilverStage,
    check_stage_order,
    prerequisite_tables,
    select_stages,
)
from src.data_pipeline.quality.sec_filings.silver_audit_report import (
    SAMPLE_IDS,
    AuditCheck,
    SilverAuditReport,
)
from src.data_pipeline.quality.sec_filings.silver_audit_sql import (
    ScratchDatabase,
    run_sql_checks,
)
from src.data_pipeline.quality.sec_filings.silver_model import SILVER_TABLES
from src.data_pipeline.storage.iceberg import SilverTableSpec

pytestmark = pytest.mark.unit

FILINGS = "sec_filings.filings"
DOCUMENTS = "sec_filings.documents"
SECTIONS = "sec_filings.sections"
TABLES = "sec_filings.document_tables"
STAGES = {stage.name: stage for stage in SILVER_STAGES}


def _report() -> SilverAuditReport:
    return SilverAuditReport(data_dir="data", started_at=datetime.now(UTC))


def _audit(
    tmp_path: Path, rows: dict[str, list[dict[str, Any]]]
) -> dict[str, AuditCheck]:
    """Run the set-based checks on a scratch database holding only `rows`.

    Columns a row leaves out are stored as null.
    """

    report = _report()
    tmp_path.mkdir(exist_ok=True)
    with closing(ScratchDatabase(tmp_path / "audit.db")) as scratch:
        for identifier, table_rows in rows.items():
            columns = scratch.columns[identifier]
            scratch.insert(identifier, [dict.fromkeys(columns) | r for r in table_rows])
        run_sql_checks(scratch, report)
    return {f"{c.table}:{c.check_name}": c for c in report.checks}


def _names(stages: tuple[SilverStage, ...]) -> list[str]:
    return [stage.name for stage in stages]


def test_registry_is_one_valid_order_over_every_audited_table() -> None:
    check_stage_order(SILVER_STAGES)

    owned = [spec.identifier for s in SILVER_STAGES for spec in s.owned_tables]
    assert sorted(owned) == sorted(table.identifier for table in SILVER_TABLES)
    assert SILVER_STAGES[-1].name == "integrity"
    assert set(SILVER_STAGES[-1].dependencies) == set(STAGES) - {"integrity"}
    # The run result has exactly one summary field per stage.
    names = [f.name for f in fields(SilverRunResult)]
    assert names[-len(SILVER_STAGES) :] == list(STAGES)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s: (s[1], s[0], *s[2:]), "depends on \\['filings'\\], not run before"),
        (lambda s: (*s, replace(s[0], dependencies=("nowhere",))), "registered twice"),
        (
            lambda s: (s[0], replace(s[1], dependencies=("nowhere",)), *s[2:]),
            "depends on \\['nowhere'\\]",
        ),
        (
            lambda s: (s[0], replace(s[1], owned_tables=s[0].owned_tables), *s[2:]),
            "owned by 'filings' and 'documents'",
        ),
    ],
    ids=[
        "dependency-after-dependent",
        "duplicate-stage",
        "unknown-dependency",
        "two-owners",
    ],
)
def test_invalid_stage_orders_are_rejected(change, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_stage_order(change(SILVER_STAGES))


def test_stage_ranges_select_contiguous_stages_in_run_order() -> None:
    assert _names(select_stages(SILVER_STAGES)) == list(STAGES)
    assert _names(select_stages(SILVER_STAGES, "xbrl_core", "xbrl_facts")) == [
        "xbrl_core",
        "xbrl_facts",
    ]
    assert _names(select_stages(SILVER_STAGES, to_stage="documents")) == [
        "filings",
        "documents",
    ]
    with pytest.raises(ValueError, match="'xbrl_facts' runs after stage 'narrative'"):
        select_stages(SILVER_STAGES, "xbrl_facts", "narrative")
    with pytest.raises(ValueError, match="unknown stage 'gold'"):
        select_stages(SILVER_STAGES, to_stage="gold")


def test_prerequisites_are_the_tables_of_every_unselected_ancestor() -> None:
    facts = select_stages(SILVER_STAGES, "xbrl_facts", "xbrl_facts")
    audit = select_stages(SILVER_STAGES, "integrity")

    assert [spec.identifier for spec in prerequisite_tables(SILVER_STAGES, facts)] == [
        "sec_filings.filings",
        "sec_filings.documents",
        "sec_filings.sections",
        "sec_filings.content_blocks",
        "sec_filings.document_tables",
        "sec_filings.document_table_cells",
        "sec_filings.xbrl_instances",
        "sec_filings.xbrl_contexts",
        "sec_filings.xbrl_units",
    ]
    assert len(prerequisite_tables(SILVER_STAGES, audit)) == len(SILVER_TABLES)
    assert prerequisite_tables(SILVER_STAGES, SILVER_STAGES) == ()


def test_duplicate_primary_and_grain_keys_are_counted_with_samples(
    tmp_path: Path,
) -> None:
    blocks = [f"doc:block:{n}" for n in range(8)]
    # Two tables on one block break content_blocks 1 -- 0..1 tables.
    tables = [
        {
            "table_id": f"{block}:table",
            "block_id": block,
            "document_id": "doc",
            "table_ordinal": n,
        }
        for n, block in enumerate([*blocks, "doc:block:0"], start=1)
    ]
    tables[-1]["table_id"] = "doc:block:0:table"

    checks = _audit(tmp_path, {TABLES: tables})

    assert (
        checks[f"{TABLES}:unique:table_id"].failed_count,
        checks[f"{TABLES}:unique:table_id"].sample_ids,
    ) == (1, ("doc:block:0:table",))
    assert checks[f"{TABLES}:unique:block_id"].sample_ids == ("doc:block:0",)
    assert checks[f"{TABLES}:unique:document_id+table_ordinal"].status == "PASS"


def _filings_and_documents(
    document_dates: list[date],
) -> dict[str, list[dict[str, Any]]]:
    return {
        FILINGS: [
            {
                "filing_id": "sec:1",
                "filing_date": date(2025, 1, 2),
                "full_submission_path": "bronze/1/full-submission.txt",
                "full_submission_sha256": "a" * 64,
            }
        ],
        DOCUMENTS: [
            {
                "document_id": f"sec:{n}:document:0001",
                "filing_id": f"sec:{n}",
                "filing_date": filing_date,
                "source_path": "bronze/1/full-submission.txt",
                "source_submission_sha256": "a" * 64,
            }
            for n, filing_date in zip((1, 2), document_dates, strict=True)
        ],
    }


def test_orphans_and_copied_parent_fields_that_differ_fail(tmp_path: Path) -> None:
    checks = _audit(
        tmp_path, _filings_and_documents([date(2025, 1, 3), date(2025, 1, 2)])
    )

    orphan = checks[f"{DOCUMENTS}:orphan:documents.filing_id -> filings"]
    assert (orphan.status, orphan.sample_ids) == ("FAIL", ("sec:2:document:0001",))
    lineage = checks[f"{DOCUMENTS}:lineage:documents.filing_id -> filings"]
    assert (lineage.failed_count, lineage.sample_ids) == (1, ("sec:1:document:0001",))


def test_section_parents_must_exist_in_the_same_document(tmp_path: Path) -> None:
    sections = [
        {"section_id": "a:section:0001", "document_id": "a", "parent_section_id": None},
        {"section_id": "a:section:0002", "document_id": "a", "parent_section_id": "a:section:0001"},
        {"section_id": "b:section:0001", "document_id": "b", "parent_section_id": None},
        # b's second section points at a's root: a cross-document parent.
        {"section_id": "b:section:0002", "document_id": "b", "parent_section_id": "a:section:0001"},
    ]  # fmt: skip
    name = f"{SECTIONS}:lineage:sections.parent_section_id -> sections"

    checks = _audit(tmp_path / "cross", {SECTIONS: sections})
    sections[3]["parent_section_id"] = "a:section:0009"
    orphaned = _audit(tmp_path / "orphan", {SECTIONS: sections})

    assert checks[name].sample_ids == ("b:section:0002",)
    orphan = orphaned[f"{SECTIONS}:orphan:sections.parent_section_id -> sections"]
    assert (orphan.status, orphan.sample_ids) == ("FAIL", ("b:section:0002",))


def test_failures_keep_a_bounded_sample_of_identifiers() -> None:
    report = _report()
    with report.check("complete_group", TABLES, "groups are complete") as run:
        for n in range(SAMPLE_IDS + 3):
            run.found(f"id-{n}")

    (check,) = report.checks
    assert check.failed_count == SAMPLE_IDS + 3
    assert check.sample_ids == tuple(f"id-{n}" for n in range(SAMPLE_IDS))
    summary = report.summary()
    assert (summary["checks_failed"], summary["status"]) == (1, "FAIL")


def test_check_only_never_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write(*args: object, **kwargs: object) -> None:
        raise AssertionError("--check-only must not write")

    monkeypatch.setattr(build_silver, "run_sec_filings_silver", write)
    monkeypatch.setattr(SilverTableSpec, "open", write)
    monkeypatch.setattr(
        sys, "argv", ["build_silver.py", "--data-dir", str(tmp_path), "--check-only"]
    )

    # Without a Silver catalog there is nothing to audit: exit 2, nothing made.
    assert build_silver.main() == 2
    assert list(tmp_path.iterdir()) == []

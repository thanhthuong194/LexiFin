from __future__ import annotations

import sys
from dataclasses import fields, replace
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
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
from src.data_pipeline.quality.sec_filings.silver_audit import SILVER_TABLES
from src.data_pipeline.quality.sec_filings.silver_constraints import (
    DOCUMENT_RELATIONS,
    check_keys,
    check_references,
)
from src.data_pipeline.quality.sec_filings.silver_report import (
    SAMPLE_IDS,
    SilverAuditReport,
)
from src.data_pipeline.storage.iceberg import SilverTableSpec

pytestmark = pytest.mark.unit

FILINGS = "sec_filings.filings"
DOCUMENTS = "sec_filings.documents"
SECTIONS = "sec_filings.sections"
TABLES = "sec_filings.document_tables"
STAGES = {stage.name: stage for stage in SILVER_STAGES}


def _report() -> SilverAuditReport:
    return SilverAuditReport(started_at=datetime.now(UTC))


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


def test_duplicate_primary_and_grain_keys_are_counted_with_samples() -> None:
    blocks = [f"doc:block:{n}" for n in range(8)]
    tables = pa.table(
        {
            "table_id": [f"{block}:table" for block in blocks] + ["doc:block:0:table"],
            # Two tables on one block break content_blocks 1 -- 0..1 tables.
            "block_id": [*blocks, "doc:block:0"],
            "document_id": ["doc"] * 9,
            "table_ordinal": [*range(1, 9), 9],
        }
    )
    report = _report()

    check_keys(
        {TABLES: tables},
        {TABLES: (("table_id",), ("block_id",), ("document_id", "table_ordinal"))},
        report,
    )

    assert report.duplicate_counts == {
        "document_tables.table_id": 1,
        "document_tables.block_id": 1,
        "document_tables.document_id+table_ordinal": 0,
    }
    (failure,) = report.failures.values()
    assert (failure.check_name, failure.table, failure.affected_count) == (
        "duplicate_key",
        TABLES,
        2,
    )
    assert failure.sample_ids == ["doc:block:0:table", "doc:block:0"]


def _filings_and_documents(document_dates: list[date]) -> dict[str, pa.Table]:
    return {
        FILINGS: pa.table(
            {
                "filing_id": ["sec:1"],
                "filing_date": [date(2025, 1, 2)],
                "full_submission_path": ["bronze/1/full-submission.txt"],
                "full_submission_sha256": ["a" * 64],
            }
        ),
        DOCUMENTS: pa.table(
            {
                "document_id": [f"sec:{n}:document:0001" for n in (1, 2)],
                "filing_id": ["sec:1", "sec:2"],
                "filing_date": document_dates,
                "source_path": ["bronze/1/full-submission.txt"] * 2,
                "source_submission_sha256": ["a" * 64] * 2,
            }
        ),
    }


def test_orphans_and_copied_parent_fields_that_differ_are_errors() -> None:
    data = _filings_and_documents([date(2025, 1, 3), date(2025, 1, 2)])
    report = _report()

    check_references(data, DOCUMENT_RELATIONS[:1], report)

    failures = {(f.check_name, f.parent_table): f for f in report.failures.values()}
    assert failures[("orphan", FILINGS)].sample_ids == ["sec:2:document:0001"]
    lineage = failures[("lineage", FILINGS)]
    assert (lineage.affected_count, lineage.sample_ids) == (1, ["sec:1:document:0001"])
    assert report.orphan_counts == {"documents.filing_id -> filings": 1}


def test_section_parents_must_exist_in_the_same_document() -> None:
    sections = pa.table(
        {
            "section_id": ["a:section:0001", "a:section:0002", "b:section:0001", "b:section:0002"],
            "document_id": ["a", "a", "b", "b"],
            # b's second section points at a's root: a cross-document parent.
            "parent_section_id": [None, "a:section:0001", None, "a:section:0001"],
        }
    )  # fmt: skip
    relation = next(r for r in DOCUMENT_RELATIONS if r.child_key == "parent_section_id")
    report = _report()

    check_references({SECTIONS: sections}, [relation], report)

    (failure,) = report.failures.values()
    assert (failure.check_name, failure.sample_ids) == ("lineage", ["b:section:0002"])
    orphaned = sections.set_column(
        2, "parent_section_id", pa.array([None, "a:section:0009", None, None])
    )
    report = _report()
    check_references({SECTIONS: orphaned}, [relation], report)
    assert [f.check_name for f in report.failures.values()] == ["orphan"]


def test_failures_keep_a_bounded_sample_of_identifiers() -> None:
    report = _report()
    for n in range(SAMPLE_IDS + 3):
        report.add("error", "group", TABLES, "broken", f"id-{n}")

    (failure,) = report.failures.values()
    assert failure.affected_count == SAMPLE_IDS + 3
    assert failure.sample_ids == [f"id-{n}" for n in range(SAMPLE_IDS)]
    summary = report.summary()
    assert (summary["checks_failed"], summary["status"]) == (1, "failed")


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

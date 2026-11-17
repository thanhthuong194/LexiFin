from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.data_pipeline.pipelines.native_rag.chunks import (
    SILVER_RULES_PROPERTY,
    run_native_rag_chunks,
)
from src.data_pipeline.pipelines.sec_filings.bronze_to_silver import (
    run_sec_filings_silver,
)
from src.data_pipeline.quality.native_rag.audit import audit_chunks
from src.data_pipeline.storage.iceberg import (
    RULES_VERSION_PROPERTY,
    iter_rows,
    open_gold_catalog,
)
from src.data_pipeline.storage.native_rag.chunks import CHUNKS_SPEC, CHUNKS_TABLE
from src.data_pipeline.transforms.native_rag.document_chunks import ChunkParams
from tests.conftest import PRESS_RELEASE
from tests.native_rag_support import WordCounter

pytestmark = pytest.mark.integration

FILINGS = [
    ("0000320193", "10-K", "0000320193-24-000123", "20241101"),
    ("0000320193", "10-Q", "0000320193-25-000057", "20250502"),
    ("0000789019", "8-K", "0000789019-25-000010", "20250129"),
]
# Rows digest of `native_rag.chunks` built from FILINGS with WordCounter and
# default parameters, at CHUNKS_SPEC.rules_version. A different digest means
# the same Silver now gives different chunks: bump the rules version.
FIXTURE_CHUNKS = (
    3,
    "5a0bb996e6d94d9eb5d4715e87c8a2a1933773b76d9547a677f1331deb7a5eb0",
)
PROCESSED_AT = datetime(2026, 11, 16, tzinfo=UTC)
_ROWS = "".join(
    f"<tr><td>Segment {name}</td><td>$</td><td>{a:,}</td><td>$</td><td>{b:,}</td>"
    f"<td>{p}</td><td>%</td></tr>"
    for name, a, b, p in [
        ("Americas", 167045, 162560, 3),
        ("Europe", 101328, 94294, 7),
        ("Greater China", 66952, 72559, -8),
        ("Japan", 25052, 24257, 3),
        ("Rest of Asia Pacific", 30658, 29615, 4),
    ]
)
_SENTENCES = " ".join(
    f"Sentence {i} explains how demand for widgets changed in region {i}."
    for i in range(70)
)
# An EX-99 with what the primary fixtures lack: a data table over 100 chars
# (lead-in title, units row, `$`/`%` cells, a footnote) and a paragraph longer
# than the cap (split by sentences).
RESULTS = (
    "EX-99.3",
    "10",
    "ex99-3.htm",
    "QUARTERLY RESULTS",
    (
        "\n<html><body><p>Segment Results</p>"
        "<p>Net sales by reportable segment were as follows:</p>"
        "<table><tr><td></td><td colspan='2'>2025</td><td colspan='2'>2024</td>"
        "<td colspan='2'>Change</td></tr>"
        f"{_ROWS}</table>"
        "<p>(1) Segments are managed on a geographic basis.</p>"
        f"<p>{_SENTENCES}</p>"
        "</body></html>\n"
    ),
)


def _rows(data_dir: Path) -> list[dict[str, object]]:
    with open_gold_catalog(data_dir) as gold:
        return sorted(
            iter_rows(gold.load_table(CHUNKS_TABLE)), key=lambda r: str(r["chunk_id"])
        )


def _digest(rows: list[dict[str, object]]) -> str:
    content = "\n".join(
        repr(sorted((k, v) for k, v in row.items() if k != "processed_at"))
        for row in rows
    )
    return hashlib.sha256(content.encode()).hexdigest()


def test_chunks_from_fixture_silver(tmp_path: Path, write_sec_filing) -> None:
    for cik, form, accession, filed in FILINGS:
        extra = (PRESS_RELEASE, RESULTS) if form == "8-K" else ()
        write_sec_filing(
            tmp_path,
            cik=cik,
            form=form,
            accession=accession,
            filed=filed,
            extra_documents=extra,
        )
    assert run_sec_filings_silver(tmp_path).completed
    counter = WordCounter()

    summary = run_native_rag_chunks(tmp_path, counter, PROCESSED_AT)
    assert summary.status == "completed", summary.problems
    assert summary.documents_chunked == 5
    rows = _rows(tmp_path)
    assert len(rows) == summary.chunks_written
    by_form = {row["form"] for row in rows}
    assert by_form == {"10-K", "10-Q", "8-K"}
    topics = {(row["form"], row["section_topic"]) for row in rows}
    assert ("10-K", "mdna") in topics
    assert ("10-Q", "legal_proceedings") in topics
    assert ("8-K", "ex99") in topics
    tables = [r for r in rows if r["chunk_kind"] == "table"]
    assert len(tables) == 1
    table = tables[0]
    assert table["table_title"] == (
        "Segment Results Net sales by reportable segment were as follows:"
    )
    assert str(table["text"]).splitlines()[:2] == [
        " | 2025 | 2024 | Change",
        "Segment Americas | $167,045 | $162,560 | 3%",
    ]
    assert str(table["text"]).endswith(
        "(1) Segments are managed on a geographic basis."
    )
    parts = [r for r in rows if r["part"] is not None]
    assert len(parts) >= 2 and {len(r["block_ids"]) for r in parts} == {1}
    # Two-column fixture tables are read as text rows.
    assert any("Revenue $1,000" in str(r["text"]) for r in rows if r["form"] == "10-K")
    for row in rows:
        assert str(row["context"]).startswith(str(row["company_name"]))
        assert str(row["sec_url"]).startswith(
            "https://www.sec.gov/Archives/edgar/data/"
        )

    with open_gold_catalog(tmp_path) as gold:
        table = gold.load_table(CHUNKS_TABLE)
        assert table.properties[RULES_VERSION_PROPERTY] == str(
            CHUNKS_SPEC.rules_version
        )
    report = audit_chunks(tmp_path, counter)
    assert report.status != "FAIL", [c for c in report.checks if c.status == "FAIL"]
    assert (CHUNKS_SPEC.rules_version, _digest(rows)) == FIXTURE_CHUNKS

    rerun = run_native_rag_chunks(tmp_path, counter, PROCESSED_AT)
    assert (rerun.status, rerun.chunks_written, rerun.documents_skipped) == (
        "completed",
        0,
        5,
    )
    changed = run_native_rag_chunks(
        tmp_path, counter, PROCESSED_AT, params=ChunkParams(max_tokens=256)
    )
    assert changed.status == "rejected"
    assert "lexifin.chunk-params" in changed.problems[0]

    # Chunks built from other Silver rules versions are refused too.
    with open_gold_catalog(tmp_path) as gold:
        table = gold.load_table(CHUNKS_TABLE)
        with table.transaction() as transaction:
            transaction.set_properties(
                {SILVER_RULES_PROPERTY: "sec_filings.sections=2"}
            )
    stale = run_native_rag_chunks(tmp_path, counter, PROCESSED_AT)
    assert stale.status == "rejected"
    assert SILVER_RULES_PROPERTY in stale.problems[0]
    assert audit_chunks(tmp_path, counter).status == "FAIL"

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.data_pipeline.contracts.native_rag.chunks import (
    CHUNKS_ARROW_SCHEMA,
    ChunkRecord,
    chunk_id_for,
    chunk_sha256,
)
from src.data_pipeline.contracts.native_rag.embeddings import (
    EmbeddingRecord,
    embedding_id_for,
)
from src.data_pipeline.storage.iceberg import append_rows, iter_rows, open_gold_catalog
from src.data_pipeline.storage.native_rag.chunks import CHUNKS_SPEC, CHUNKS_TABLE
from src.data_pipeline.storage.native_rag.embeddings import EMBEDDINGS_TABLE
from src.data_pipeline.transforms.native_rag.context import embedded_text
from src.llm.types import LLMProviderError
from src.rag_engine.embedding import pipeline
from src.rag_engine.embedding.audit import audit_embeddings
from src.rag_engine.embedding.pipeline import normalized, run_native_rag_embeddings

pytestmark = pytest.mark.integration

NOW = datetime(2026, 11, 17, tzinfo=UTC)


class FakeEmbedder:
    """Deterministic 8-dim vectors from the text hash; counts texts embedded."""

    model = "test/fake-embedding"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [
            [b - 127.5 for b in hashlib.sha256(text.encode()).digest()[:8]]
            for text in texts
        ]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def _chunk(
    document: str, ordinal: int, count: int, filed: date, text: str
) -> ChunkRecord:
    context = f"Example Corp | 10-K | FY{filed.year}"
    return ChunkRecord(
        chunk_id=chunk_id_for(document, ordinal),
        filing_id="sec:1",
        document_id=document,
        section_id=f"{document}:section:0001",
        table_id=None,
        block_ids=[f"{document}:block:{ordinal:06d}"],
        filing_date=filed,
        cik="0000320193",
        tickers=["EX"],
        form="10-K",
        fiscal_year=filed.year,
        fiscal_period="FY",
        period_end_date=None,
        section_topic="mdna",
        document_type="10-K",
        chunk_kind="text",
        company_name="Example Corp",
        accession_number="0000320193-25-000001",
        sec_url="https://www.sec.gov/Archives/edgar/data/320193/000032019325000001/a.htm",
        section_title="Item 7",
        subheading=None,
        table_title=None,
        chunk_ordinal=ordinal,
        document_chunk_count=count,
        part=None,
        part_count=None,
        context=context,
        text=text,
        token_count=10,
        chunk_sha256=chunk_sha256(context, text),
        processed_at=NOW,
    )


def _write_chunks(gold_dir: Path, chunks: list[ChunkRecord]) -> None:
    with open_gold_catalog(gold_dir) as gold:
        if gold.table_exists(CHUNKS_TABLE):
            gold.purge_table(CHUNKS_TABLE)
        table, _ = CHUNKS_SPEC.open(gold, gold_dir)
        append_rows(table, chunks, CHUNKS_ARROW_SCHEMA)


def _chunks(texts_2024: int = 3, texts_2025: int = 4) -> list[ChunkRecord]:
    return [
        _chunk("d24", i, texts_2024, date(2024, 2, 1), f"Revenue note {i}")
        for i in range(1, texts_2024 + 1)
    ] + [
        _chunk("d25", i, texts_2025, date(2025, 2, 1), f"Margin note {i}")
        for i in range(1, texts_2025 + 1)
    ]


@pytest.fixture(autouse=True)
def _small_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    # Several request groups per call and several commits per run.
    monkeypatch.setattr(pipeline, "CHUNKS_PER_GROUP", 2)
    monkeypatch.setattr(pipeline, "ROWS_PER_COMMIT", 3)


def _stored(gold_dir: Path) -> list[dict[str, object]]:
    with open_gold_catalog(gold_dir) as gold:
        return list(iter_rows(gold.load_table(EMBEDDINGS_TABLE)))


def test_embed_resume_audit_and_stale(tmp_path: Path) -> None:
    chunks = _chunks()
    _write_chunks(tmp_path, chunks)
    embedder = FakeEmbedder()

    trial = run_native_rag_embeddings(tmp_path, embedder, NOW, limit=2)
    assert (trial.status, trial.chunks_embedded) == ("completed", 2)
    assert audit_embeddings(tmp_path, embedder.model).status == "WARN"

    rest = run_native_rag_embeddings(tmp_path, embedder, NOW, concurrency=2)
    assert (rest.chunks_total, rest.chunks_skipped, rest.chunks_embedded) == (7, 2, 5)
    assert sorted(embedder.texts) == sorted(
        embedded_text(c.context, c.text) for c in chunks
    )
    rows = _stored(tmp_path)
    assert len(rows) == 7
    by_chunk = {c.chunk_id: c for c in chunks}
    for row in rows:
        chunk = by_chunk[str(row["chunk_id"])]
        text = embedded_text(chunk.context, chunk.text)
        expected = normalized(FakeEmbedder().embed_documents([text]))[0]
        assert row["vector"] == pytest.approx(expected, abs=1e-6)
        assert row["chunk_sha256"] == chunk.chunk_sha256
    assert {r["embedding_id"] for r in rows} == {
        embedding_id_for(embedder.model, c.chunk_id) for c in chunks
    }
    for row in rows:
        assert row["dimensions"] == 8
        assert math.isclose(
            math.sqrt(sum(v * v for v in row["vector"])), 1, abs_tol=1e-6
        )

    report = audit_embeddings(tmp_path, embedder.model, embedder=embedder, sample=3)
    assert report.status == "PASS", [c for c in report.checks if c.status != "PASS"]

    again = run_native_rag_embeddings(tmp_path, embedder, NOW)
    assert (again.chunks_skipped, again.chunks_embedded) == (7, 0)

    # Chunks rebuilt with other text: stored vectors are stale.
    changed = _chunks()
    changed[0] = _chunk("d24", 1, 3, date(2024, 2, 1), "Rewritten revenue note")
    _write_chunks(tmp_path, changed)
    stale = run_native_rag_embeddings(tmp_path, embedder, NOW)
    assert stale.status == "rejected"
    assert "purge native_rag.embeddings" in stale.problems[0]
    report = audit_embeddings(tmp_path, embedder.model, embedder=embedder, sample=7)
    assert report.status == "FAIL"
    failed = {c.name for c in report.checks if c.status == "FAIL"}
    assert failed == {"current_chunks"}

    # A whole year dropped from the chunks: its vectors are still seen.
    _write_chunks(tmp_path, [c for c in chunks if c.filing_date.year == 2025])
    gone = run_native_rag_embeddings(tmp_path, embedder, NOW)
    assert gone.status == "rejected" and "of 2024" in gone.problems[0]


class FailingEmbedder(FakeEmbedder):
    """Fails from its `fail_at`-th request on, like a closed tunnel."""

    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self.calls = 0
        self.fail_at = fail_at

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls >= self.fail_at:
            raise LLMProviderError("tunnel closed")
        return super().embed_documents(texts)


def test_server_failure_keeps_vectors_and_resumes(tmp_path: Path) -> None:
    _write_chunks(tmp_path, _chunks())
    # concurrency 1: one group of up to 2 chunks per request, in order; 2024
    # has 3 chunks (requests 1-2, committed at 3 rows), 2025 fails at once.
    broken = run_native_rag_embeddings(
        tmp_path, FailingEmbedder(fail_at=3), NOW, concurrency=1
    )
    assert broken.status == "interrupted"
    assert "run again" in broken.problems[0]
    assert len(_stored(tmp_path)) == broken.chunks_embedded == 3
    resumed = run_native_rag_embeddings(tmp_path, FakeEmbedder(), NOW)
    assert (resumed.chunks_skipped, resumed.chunks_embedded) == (3, 4)
    ids = [r["embedding_id"] for r in _stored(tmp_path)]
    assert len(ids) == len(set(ids)) == 7


def test_failed_verification_never_appends_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_chunks(tmp_path, _chunks())

    def broken_verify(*args: object, **kwargs: object) -> None:
        raise RuntimeError("read failed")

    monkeypatch.setattr(pipeline, "verify_append", broken_verify)
    with pytest.raises(RuntimeError, match="read failed"):
        run_native_rag_embeddings(tmp_path, FakeEmbedder(), NOW, concurrency=1)
    ids = [r["embedding_id"] for r in _stored(tmp_path)]
    assert len(ids) == len(set(ids)) == 3  # the first commit, written once


def test_limit_crosses_years(tmp_path: Path) -> None:
    _write_chunks(tmp_path, _chunks())
    run = run_native_rag_embeddings(tmp_path, FakeEmbedder(), NOW, limit=4)
    years = sorted(str(r["chunk_id"])[:3] for r in _stored(tmp_path))
    assert run.chunks_embedded == 4 and years == ["d24", "d24", "d24", "d25"]


def test_normalized_rejects_bad_vectors() -> None:
    assert normalized([[3.0, 4.0]]) == [[0.6, 0.8]]
    for bad in ([[0.0, 0.0]], [[1.0, 2.0], [1.0]], [[math.inf, 1.0]]):
        with pytest.raises(ValueError):
            normalized(bad)


def test_embedding_contract() -> None:
    row = {
        "embedding_id": embedding_id_for("m", "c"),
        "chunk_id": "c",
        "document_id": "d",
        "filing_date": date(2025, 1, 1),
        "model_id": "m",
        "chunk_sha256": "0" * 64,
        "dimensions": 2,
        "vector": [0.6, 0.8],
        "processed_at": NOW,
    }
    EmbeddingRecord.model_validate(row)
    for changes in (
        {"embedding_id": "c"},
        {"dimensions": 3},
        {"vector": [1.0, 1.0]},
        {"vector": [math.nan, 1.0]},
    ):
        with pytest.raises(ValueError):
            EmbeddingRecord.model_validate({**row, **changes})


def test_audit_warns_when_server_is_down(tmp_path: Path) -> None:
    _write_chunks(tmp_path, _chunks())
    run_native_rag_embeddings(tmp_path, FakeEmbedder(), NOW)
    report = audit_embeddings(tmp_path, FakeEmbedder.model, embedder=FailingEmbedder(1))
    statuses = {c.name: c.status for c in report.checks}
    assert statuses["sample_reembed"] == "WARN"
    assert report.status == "WARN"

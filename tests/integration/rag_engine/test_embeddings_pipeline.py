from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest

from src.data_pipeline.contracts.native_rag.embeddings import (
    EmbeddingRecord,
    embedding_id_for,
)
from src.data_pipeline.storage.iceberg import iter_rows, open_gold_catalog
from src.data_pipeline.storage.native_rag.embeddings import EMBEDDINGS_TABLE
from src.data_pipeline.transforms.native_rag.context import embedded_text
from src.llm.types import LLMProviderError
from src.rag_engine.embedding import pipeline
from src.rag_engine.embedding.audit import audit_embeddings
from src.rag_engine.embedding.pipeline import (
    Throughput,
    normalized,
    run_native_rag_embeddings,
)
from tests.native_rag_support import (
    NOW,
    FakeEmbedder,
    gold_chunk,
    gold_chunks,
    write_chunks,
)

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _small_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    # Several request groups per call and several commits per run.
    monkeypatch.setattr(pipeline, "BATCH_SIZE", 2)
    monkeypatch.setattr(pipeline, "ROWS_PER_COMMIT", 3)


def _stored(gold_dir: Path) -> list[dict[str, object]]:
    with open_gold_catalog(gold_dir) as gold:
        return list(iter_rows(gold.load_table(EMBEDDINGS_TABLE)))


def test_embed_resume_audit_and_stale(tmp_path: Path) -> None:
    chunks = gold_chunks()
    write_chunks(tmp_path, chunks)
    embedder = FakeEmbedder()

    trial = run_native_rag_embeddings(tmp_path, embedder, NOW, limit=2)
    assert (trial.status, trial.chunks_embedded) == ("completed", 2)
    assert audit_embeddings(tmp_path, embedder.model).status == "WARN"

    rest = run_native_rag_embeddings(
        tmp_path, embedder, NOW, throughput=Throughput(concurrency=2)
    )
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
    changed = gold_chunks()
    changed[0] = gold_chunk("d24", 1, 3, date(2024, 2, 1), "Rewritten revenue note")
    write_chunks(tmp_path, changed)
    stale = run_native_rag_embeddings(tmp_path, embedder, NOW)
    assert stale.status == "rejected"
    assert "purge native_rag.embeddings" in stale.problems[0]
    report = audit_embeddings(tmp_path, embedder.model, embedder=embedder, sample=7)
    assert report.status == "FAIL"
    failed = {c.name for c in report.checks if c.status == "FAIL"}
    assert failed == {"current_chunks"}

    # A whole year dropped from the chunks: its vectors are still seen.
    write_chunks(tmp_path, [c for c in chunks if c.filing_date.year == 2025])
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
    write_chunks(tmp_path, gold_chunks())
    # concurrency 1: one group of up to 2 chunks per request, in order; 2024
    # has 3 chunks (requests 1-2, committed at 3 rows), 2025 fails at once.
    broken = run_native_rag_embeddings(
        tmp_path, FailingEmbedder(fail_at=3), NOW, throughput=Throughput(concurrency=1)
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
    write_chunks(tmp_path, gold_chunks())

    def broken_verify(*args: object, **kwargs: object) -> None:
        raise RuntimeError("read failed")

    monkeypatch.setattr(pipeline, "verify_append", broken_verify)
    with pytest.raises(RuntimeError, match="read failed"):
        run_native_rag_embeddings(
            tmp_path, FakeEmbedder(), NOW, throughput=Throughput(concurrency=1)
        )
    ids = [r["embedding_id"] for r in _stored(tmp_path)]
    assert len(ids) == len(set(ids)) == 3  # the first commit, written once


def test_limit_crosses_years(tmp_path: Path) -> None:
    write_chunks(tmp_path, gold_chunks())
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
    write_chunks(tmp_path, gold_chunks())
    run_native_rag_embeddings(tmp_path, FakeEmbedder(), NOW)
    report = audit_embeddings(tmp_path, FakeEmbedder.model, embedder=FailingEmbedder(1))
    statuses = {c.name: c.status for c in report.checks}
    assert statuses["sample_reembed"] == "WARN"
    assert report.status == "WARN"

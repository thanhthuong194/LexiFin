from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from pyiceberg.table import Table

from src.data_pipeline.contracts.native_rag.embeddings import NORM_TOLERANCE
from src.data_pipeline.quality.native_rag.audit import AuditReport
from src.data_pipeline.storage.iceberg import open_gold_catalog_read_only, scan_batches
from src.data_pipeline.storage.native_rag.chunks import CHUNKS_TABLE
from src.data_pipeline.storage.native_rag.embeddings import (
    EMBEDDINGS_SPEC,
    EMBEDDINGS_TABLE,
)
from src.data_pipeline.transforms.native_rag.context import embedded_text
from src.llm.base import EmbeddingModel
from src.llm.types import LLMError
from src.rag_engine.embedding.pipeline import (
    current_hashes,
    stale_vectors,
    table_years,
    year_filter,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# A re-embedded sample vector must match the stored one at least this well
# (GPU kernels are not bit-exact).
REEMBED_MIN_COSINE = 0.999


def _vector_problems(batch: pa.RecordBatch) -> tuple[list[str], list[str]]:
    """IDs of vectors whose length differs from `dimensions`, and of vectors
    not of unit length (a NaN counts as not unit length)."""

    vectors = batch["vector"]
    ids = batch["embedding_id"].to_pylist()
    lengths = pc.list_value_length(vectors).to_pylist()
    declared = batch["dimensions"].to_pylist()
    wrong_size = [i for i, n, d in zip(ids, lengths, declared, strict=True) if n != d]
    parents = pc.list_parent_indices(vectors)
    values = pc.list_flatten(vectors).cast(pa.float64())
    squares = pa.table({"row": parents, "square": pc.multiply(values, values)})
    sums = squares.group_by("row").aggregate([("square", "sum")])
    totals = dict(
        zip(sums["row"].to_pylist(), sums["square_sum"].to_pylist(), strict=True)
    )
    not_unit = [
        ids[row]
        for row in range(len(ids))
        if not abs(math.sqrt(totals.get(row, 0.0)) - 1) <= NORM_TOLERANCE
    ]
    return wrong_size, not_unit


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    norms = math.sqrt(math.fsum(a * a for a in left)) * math.sqrt(
        math.fsum(b * b for b in right)
    )
    return dot / norms if norms else 0.0


def _rank(chunk_id: str) -> str:
    return hashlib.sha256(chunk_id.encode()).hexdigest()


@dataclass
class _Findings:
    model_id: str
    sample_size: int
    rows: int = 0
    duplicates: list[str] = field(default_factory=list)
    sizes: Counter[int] = field(default_factory=Counter)
    wrong_size: list[str] = field(default_factory=list)
    not_unit: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    missing: int = 0
    # The `sample` current vectors that rank first by chunk-id hash:
    # (rank, chunk_id, year, vector).
    sample: list[tuple[str, str, int, list[float]]] = field(default_factory=list)


def _audit_year(table: Table, chunks: Table, year: int, found: _Findings) -> None:
    current = current_hashes(chunks, year)
    ids: Counter[str] = Counter()
    stored: dict[str, str] = {}
    columns = (
        "embedding_id",
        "chunk_id",
        "model_id",
        "chunk_sha256",
        "dimensions",
        "vector",
    )
    for batch in scan_batches(table, columns, files=year_filter(year)):
        mine = batch.filter(pc.equal(batch["model_id"], found.model_id))
        found.rows += mine.num_rows
        ids.update(mine["embedding_id"].to_pylist())
        found.sizes.update(mine["dimensions"].to_pylist())
        wrong_size, not_unit = _vector_problems(mine)
        found.wrong_size += wrong_size
        found.not_unit += not_unit
        for chunk_id, digest, vector in zip(
            mine["chunk_id"].to_pylist(),
            mine["chunk_sha256"].to_pylist(),
            mine["vector"].to_pylist(),
            strict=True,
        ):
            stored[chunk_id] = digest
            if current.get(chunk_id) == digest:
                found.sample.append((_rank(chunk_id), chunk_id, year, vector))
        found.sample = sorted(found.sample)[: found.sample_size]
    found.duplicates += [key for key, n in ids.items() if n > 1]
    found.stale += stale_vectors(stored, current)
    found.missing += sum(1 for chunk_id in current if chunk_id not in stored)


def audit_embeddings(
    gold_dir: Path,
    model_id: str,
    *,
    embedder: EmbeddingModel | None = None,
    sample: int = 5,
) -> AuditReport:
    """Read-only audit of one model's vectors in `native_rag.embeddings`.

    Checks: table definition; unique keys; one vector size, and every vector
    as long as its `dimensions`; every vector L2-normalized (NaN fails);
    every vector's chunk exists with the hash it was made from (else stale:
    FAIL); chunks without a vector (WARN, e.g. a trial run with `limit`);
    and, with an `embedder`, the `sample` current chunks that rank first by
    chunk-id hash re-embedded must match their stored vector (cosine >=
    0.999; WARN when the server cannot be reached). Reads one filing year at a time (years of either table), each
    vector once.
    """

    report = AuditReport()
    with open_gold_catalog_read_only(gold_dir) as gold:
        table = gold.load_table(EMBEDDINGS_TABLE)
        chunks = gold.load_table(CHUNKS_TABLE)
        problem = EMBEDDINGS_SPEC.definition_problem(table)
        report.add("table_definition", [problem] if problem else [], EMBEDDINGS_TABLE)
        found = _Findings(model_id, sample)
        for year in table_years(chunks, table):
            _audit_year(table, chunks, year, found)
        report.add(
            "unique_embedding_id", found.duplicates, f"{found.rows} {model_id} vectors"
        )
        sizes = sorted(found.sizes)
        report.add(
            "one_vector_size",
            [f"sizes {sizes}"] if len(sizes) > 1 else [],
            f"dimensions {sizes}",
        )
        report.add(
            "vector_length", found.wrong_size, "every vector has `dimensions` values"
        )
        report.add("normalized", found.not_unit, f"|norm - 1| <= {NORM_TOLERANCE}")
        report.add(
            "current_chunks", found.stale, "every vector's chunk exists with its hash"
        )
        report.add(
            "chunks_without_vector",
            [f"{found.missing} chunks"] if found.missing else [],
            "every chunk has a vector of this model",
            warn=True,
        )
        if embedder is not None:
            detail = f"{len(found.sample)} chunks re-embedded, cosine >= {REEMBED_MIN_COSINE}"
            try:
                report.add(
                    "sample_reembed",
                    _reembed_problems(embedder, chunks, found.sample),
                    detail,
                )
            except LLMError as error:
                report.add(
                    "sample_reembed",
                    [f"embedding server unavailable: {str(error)[:200]}"],
                    detail,
                    warn=True,
                )
    logger.info("native_rag_embeddings_audit_completed", status=report.status)
    return report


def _texts(
    chunks: Table, chosen: Sequence[tuple[str, str, int, list[float]]]
) -> dict[str, str]:
    """Embedded strings of the chosen chunks, reading only their years."""

    texts: dict[str, str] = {}
    for year in sorted({item[2] for item in chosen}):
        wanted = pa.array([c for _, c, y, _ in chosen if y == year], type=pa.string())
        for batch in scan_batches(
            chunks, ("chunk_id", "context", "text"), files=year_filter(year)
        ):
            matching = batch.filter(pc.is_in(batch["chunk_id"], value_set=wanted))
            for row in matching.to_pylist():
                texts[row["chunk_id"]] = embedded_text(row["context"], row["text"])
    return texts


def _reembed_problems(
    embedder: EmbeddingModel,
    chunks: Table,
    chosen: Sequence[tuple[str, str, int, list[float]]],
) -> list[str]:
    if not chosen:
        return []
    texts = _texts(chunks, chosen)
    vectors = embedder.embed_documents([texts[c] for _, c, _, _ in chosen])
    return [
        f"{chunk_id}: cosine {cosine:.5f}"
        for (_, chunk_id, _, stored), vector in zip(chosen, vectors, strict=True)
        if (cosine := _cosine(stored, vector)) < REEMBED_MIN_COSINE
    ]

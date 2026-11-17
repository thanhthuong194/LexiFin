from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyiceberg.table import Table
from qdrant_client import QdrantClient, models

from src.data_pipeline.quality.native_rag.audit import AuditReport
from src.data_pipeline.storage.iceberg import open_gold_catalog_read_only
from src.data_pipeline.storage.native_rag.chunks import CHUNKS_TABLE
from src.data_pipeline.storage.native_rag.embeddings import EMBEDDINGS_TABLE
from src.data_pipeline.transforms.native_rag.context import embedded_text
from src.rag_engine.embedding.pipeline import table_years
from src.rag_engine.indexing.pipeline import (
    chunks_in,
    data_files,
    file_rows,
    vector_hashes,
    vectors_of,
)
from src.rag_engine.indexing.qdrant_store import (
    DATETIME_FIELDS,
    DENSE,
    INTEGER_FIELDS,
    KEYWORD_FIELDS,
    SPARSE,
    TENANT_FIELD,
    CollectionSettings,
    Corpus,
    is_large,
    payload,
    point_id,
    search_params,
)
from src.rag_engine.indexing.sparse import document_vector, query_vector
from src.utils.logger import get_logger

logger = get_logger(__name__)

MIN_VECTOR_COSINE = 0.999
# HNSW vs exact top-10: the mean over sampled queries must reach this, and
# no single query may fall below the floor.
MIN_MEAN_ANN_RECALL = 0.95
MIN_QUERY_ANN_RECALL = 0.7
# The ANN check needs this share of points in HNSW (else it compares exact
# search with itself).
MIN_INDEXED_SHARE = 0.9
# Qdrant's `indexed_vectors_count` counts every named vector: dense and BM25.
VECTORS_PER_POINT = 2
SPARSE_TOLERANCE = 1e-5
TOP_K = 10


@dataclass
class _Sample:
    chunk: dict[str, Any]
    vector: list[float]


@dataclass
class _Expected:
    points: int = 0
    # (rank, year, chunk_id) of the `size` chunks ranking first by hash.
    ranked: list[tuple[str, int, str]] = field(default_factory=list)


def _rank(chunk_id: str) -> str:
    return hashlib.sha256(chunk_id.encode()).hexdigest()


def _expected(chunks: Table, embeddings: Table, model_id: str, size: int) -> _Expected:
    """Points Gold implies (chunks with a current vector), from hashes only,
    and which chunks to sample."""

    found = _Expected()
    for year in table_years(chunks, embeddings):
        hashes = vector_hashes(embeddings, model_id, year)
        for path in data_files(chunks, year):
            for row in file_rows(chunks, path, ("chunk_id", "chunk_sha256")):
                if hashes.get(row["chunk_id"]) != row["chunk_sha256"]:
                    continue
                found.points += 1
                entry = (_rank(row["chunk_id"]), year, row["chunk_id"])
                if len(found.ranked) < size or entry[0] < found.ranked[-1][0]:
                    found.ranked = sorted([*found.ranked, entry])[:size]
    return found


def _samples(
    chunks: Table, embeddings: Table, model_id: str, expected: _Expected
) -> list[_Sample]:
    """Full chunk rows and vectors of the sampled chunks, year by year."""

    by_year: dict[int, set[str]] = {}
    for _, year, chunk_id in expected.ranked:
        by_year.setdefault(year, set()).add(chunk_id)
    samples: list[_Sample] = []
    for year, ids in by_year.items():
        vectors = vectors_of(embeddings, model_id, year, ids)
        samples += [
            _Sample(chunk, vectors.get(chunk["chunk_id"]))
            for chunk in chunks_in(chunks, year, ids)
        ]
    return samples


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    norms = math.sqrt(math.fsum(a * a for a in left)) * math.sqrt(
        math.fsum(b * b for b in right)
    )
    return dot / norms if norms else 0.0


def _index_problems(client: QdrantClient, name: str) -> list[str]:
    schema = client.get_collection(name).payload_schema
    wanted = {
        TENANT_FIELD: "keyword",
        **dict.fromkeys(KEYWORD_FIELDS, "keyword"),
        **dict.fromkeys(INTEGER_FIELDS, "integer"),
        **dict.fromkeys(DATETIME_FIELDS, "datetime"),
    }
    problems = []
    for field_name, kind in wanted.items():
        info = schema.get(field_name)
        found = getattr(info.data_type, "value", info.data_type) if info else None
        if found != kind:
            problems.append(f"{field_name}: {found or 'no index'}, expected {kind}")
    return problems


def _point_problems(
    item: _Sample, point: models.Record | None, corpus: Corpus
) -> list[str]:
    chunk_id = item.chunk["chunk_id"]
    if point is None:
        return [f"{chunk_id}: missing"]
    problems = []
    if point.payload != payload(item.chunk, corpus):
        problems.append(f"{chunk_id}: payload differs from Gold")
    vectors = point.vector if isinstance(point.vector, dict) else {}
    dense = vectors.get(DENSE)
    values = dense if isinstance(dense, list) else []
    flat = [float(v) for v in values if isinstance(v, float | int)]
    if (
        not flat
        or len(flat) != len(values)
        or _cosine(flat, item.vector) < MIN_VECTOR_COSINE
    ):
        problems.append(f"{chunk_id}: dense vector differs from Gold")
    sparse = vectors.get(SPARSE)
    expected = document_vector(embedded_text(item.chunk["context"], item.chunk["text"]))
    stored = (
        dict(zip(sparse.indices, sparse.values, strict=True))
        if isinstance(sparse, models.SparseVector)
        else {}
    )
    if sorted(stored) != expected.indices or any(
        abs(stored[i] - v) > SPARSE_TOLERANCE
        for i, v in zip(expected.indices, expected.values, strict=True)
    ):
        problems.append(f"{chunk_id}: sparse vector differs from BM25 of the text")
    return problems


def _sample_problems(
    client: QdrantClient, name: str, sample: Sequence[_Sample], corpus: Corpus
) -> list[str]:
    """Each sampled point holds Gold's payload, dense and sparse vectors."""

    ids = [point_id(s.chunk["chunk_id"]) for s in sample]
    points = {
        str(p.id): p
        for p in client.retrieve(name, ids=ids, with_payload=True, with_vectors=True)
    }
    return [
        problem
        for item, pid in zip(sample, ids, strict=True)
        for problem in _point_problems(item, points.get(pid), corpus)
    ]


def _dense(
    client: QdrantClient,
    name: str,
    vector: list[float],
    settings: CollectionSettings,
    *,
    exact: bool,
) -> list[str]:
    points = client.query_points(
        name,
        query=vector,
        using=DENSE,
        limit=TOP_K,
        search_params=search_params(settings, exact=exact),
    ).points
    return [str(p.id) for p in points]


def _hybrid_hits(
    client: QdrantClient, name: str, item: _Sample
) -> list[tuple[str, Any]]:
    chunk = item.chunk
    sparse = query_vector(" ".join(chunk["text"].split()[:30]))
    points = client.query_points(
        name,
        prefetch=[
            models.Prefetch(query=item.vector, using=DENSE, limit=50),
            models.Prefetch(
                query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                using=SPARSE,
                limit=50,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key=TENANT_FIELD, match=models.MatchValue(value=chunk["cik"])
                )
            ]
        ),
        limit=TOP_K,
        with_payload=[TENANT_FIELD],
    ).points
    return [(str(p.id), (p.payload or {}).get(TENANT_FIELD)) for p in points]


def _indexed_problem(client: QdrantClient, name: str) -> str | None:
    info = client.get_collection(name)
    points = info.points_count or 0
    indexed = info.indexed_vectors_count or 0
    expected = VECTORS_PER_POINT * points
    if points and indexed < MIN_INDEXED_SHARE * expected:
        return f"only {indexed}/{expected} vectors indexed: HNSW not built yet"
    return None


@dataclass(frozen=True)
class AuditRequest:
    model_id: str
    collection: str
    corpus: Corpus
    # Embedded Qdrant has no payload indexes and no HNSW: those checks skip.
    embedded: bool = False
    sample: int = 20


def _search_checks(
    report: AuditReport,
    client: QdrantClient,
    request: AuditRequest,
    sample: Sequence[_Sample],
) -> None:
    name = request.collection
    settings = CollectionSettings(
        dimensions=len(sample[0].vector), large=is_large(client, name)
    )
    self_misses: list[str] = []
    recalls: dict[str, float] = {}
    hybrid: list[str] = []
    for item in sample:
        chunk_id = item.chunk["chunk_id"]
        own = point_id(chunk_id)
        approximate = _dense(client, name, item.vector, settings, exact=False)
        if not approximate or approximate[0] != own:
            self_misses.append(chunk_id)
        exact = set(_dense(client, name, item.vector, settings, exact=True))
        recalls[chunk_id] = len(exact & set(approximate)) / max(len(exact), 1)
        hits = _hybrid_hits(client, name, item)
        if own not in {pid for pid, _ in hits} or any(
            c != item.chunk["cik"] for _, c in hits
        ):
            hybrid.append(chunk_id)
    report.add("self_retrieval", self_misses, "a chunk's own vector finds it first")
    report.add(
        "hybrid_filtered",
        hybrid,
        "dense + BM25 (RRF) filtered by cik returns the chunk",
    )
    detail = (
        f"HNSW vs exact top-{TOP_K}: mean >= {MIN_MEAN_ANN_RECALL}, "
        f"each >= {MIN_QUERY_ANN_RECALL}"
    )
    not_indexed = _indexed_problem(client, name)
    if request.embedded:
        report.skip("ann_recall", detail, "embedded Qdrant searches exactly (no HNSW)")
        return
    if not_indexed:
        report.skip("ann_recall", detail, not_indexed)
        return
    mean = sum(recalls.values()) / len(recalls)
    problems = [f"mean recall {mean:.3f}"] if mean < MIN_MEAN_ANN_RECALL else []
    problems += [
        f"{c}: recall {r:.2f}" for c, r in recalls.items() if r < MIN_QUERY_ANN_RECALL
    ]
    report.add("ann_recall", problems, f"{detail}; mean {mean:.3f}")


def audit_qdrant(
    gold_dir: Path, client: QdrantClient, request: AuditRequest
) -> AuditReport:
    """Audit a loaded collection against Gold (read-only on both sides).

    Checks: the collection exists; point count equals Gold's chunks with a
    current vector; payload indexes (server); sampled points hold Gold's
    payload, dense vector (cosine >= 0.999) and BM25 sparse vector (indices
    and weights); each sampled vector finds its own point first; a hybrid
    dense + BM25 query filtered by the chunk's `cik` returns it and only that
    company; HNSW top-10 against exact top-10 (mean recall >= 0.95, each
    query >= 0.7), skipped while HNSW is not built or on embedded Qdrant. An
    empty sample fails. Search settings follow the collection (rescoring when
    it is quantized). Gold is read without vectors except for the sample.
    """

    report = AuditReport()
    name = request.collection
    if not client.collection_exists(name):
        report.add("collection", [f"{name} missing"], name)
        return report
    report.add("collection", [], name)
    with open_gold_catalog_read_only(gold_dir) as gold:
        chunks = gold.load_table(CHUNKS_TABLE)
        embeddings = gold.load_table(EMBEDDINGS_TABLE)
        expected = _expected(chunks, embeddings, request.model_id, request.sample)
        sample = _samples(chunks, embeddings, request.model_id, expected)
    count = client.count(name, exact=True).count
    report.add(
        "point_count",
        []
        if count == expected.points
        else [f"{count} points, Gold has {expected.points}"],
        f"{count} points",
    )
    if request.embedded:
        report.skip(
            "payload_indexes", "tenant cik + filter fields", "embedded Qdrant has none"
        )
    else:
        report.add(
            "payload_indexes",
            _index_problems(client, name),
            "tenant cik + filter fields",
        )
    if not sample:
        report.add("sample_points", ["no point to sample"], "sampled points match Gold")
        return report
    report.add(
        "sample_points",
        _sample_problems(client, name, sample, request.corpus),
        f"{len(sample)} points match Gold (payload, dense, BM25)",
    )
    _search_checks(report, client, request, sample)
    logger.info("qdrant_audit_completed", status=report.status, collection=name)
    return report

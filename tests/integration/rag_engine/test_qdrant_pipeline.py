from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient, models
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.data_pipeline.storage.iceberg import open_gold_catalog
from src.data_pipeline.storage.native_rag.embeddings import EMBEDDINGS_TABLE
from src.rag_engine.embedding.pipeline import run_native_rag_embeddings
from src.rag_engine.indexing import pipeline as pipeline_module
from src.rag_engine.indexing.audit import AuditRequest, audit_qdrant
from src.rag_engine.indexing.companies import (
    companies_collection,
    company_point_id,
    stored_companies,
)
from src.rag_engine.indexing.pipeline import (
    LoadRequest,
    run_qdrant_load,
    vector_index,
    vectors_in,
)
from src.rag_engine.indexing.qdrant_store import (
    DENSE,
    SEC_FILINGS,
    SPARSE,
    CollectionSettings,
    collection_name,
    collection_problems,
    point_id,
    switch_alias,
)
from src.rag_engine.indexing.retry import retried, transient
from src.rag_engine.indexing.sparse import document_vector, query_vector, terms
from tests.native_rag_support import (
    NOW,
    FakeEmbedder,
    gold_chunk,
    gold_chunks,
    write_chunks,
)

pytestmark = [
    pytest.mark.integration,
    # Embedded Qdrant ignores payload indexes; the server path is audited live.
    pytest.mark.filterwarnings("ignore:Payload indexes have no effect"),
    pytest.mark.filterwarnings("ignore:Local mode performs exact"),
]

MODEL = FakeEmbedder.model
COLLECTION = collection_name(SEC_FILINGS, 4, MODEL)


def _gold(tmp_path: Path, chunks: list | None = None) -> None:
    write_chunks(tmp_path, chunks or gold_chunks())
    with open_gold_catalog(tmp_path) as gold:
        if gold.table_exists(EMBEDDINGS_TABLE):
            gold.purge_table(EMBEDDINGS_TABLE)
    assert (
        run_native_rag_embeddings(tmp_path, FakeEmbedder(), NOW).status == "completed"
    )


def _audit(tmp_path: Path, client: QdrantClient) -> dict[str, str]:
    report = audit_qdrant(
        tmp_path,
        client,
        AuditRequest(
            model_id=MODEL,
            collection=COLLECTION,
            corpus=SEC_FILINGS,
            embedded=True,
            sample=7,
        ),
    )
    return {check.name: check.status for check in report.checks}


def test_load_resume_update_delete_and_audit(tmp_path: Path) -> None:
    _gold(tmp_path)
    client = QdrantClient(":memory:")
    request = LoadRequest(model_id=MODEL, collection=COLLECTION)

    first = run_qdrant_load(tmp_path, client, request)
    assert (first.status, first.created, first.points_upserted) == (
        "completed",
        True,
        7,
    )
    assert client.count(COLLECTION, exact=True).count == 7
    assert client.get_aliases().aliases == []  # switched by the caller after the audit
    statuses = _audit(tmp_path, client)
    assert set(statuses.values()) == {"PASS", "SKIP"}, statuses
    assert {k for k, v in statuses.items() if v == "SKIP"} == {
        "payload_indexes",
        "ann_recall",
    }
    switch_alias(client, COLLECTION, "sec_filings")
    switch_alias(client, COLLECTION, "sec_filings")  # already there: no-op
    assert {a.alias_name: a.collection_name for a in client.get_aliases().aliases} == {
        "sec_filings": COLLECTION
    }

    point = client.retrieve(
        COLLECTION,
        ids=[point_id("d24:chunk:000001")],
        with_payload=True,
        with_vectors=True,
    )[0]
    assert point.payload is not None
    assert point.payload["filing_date"] == "2024-02-01T00:00:00Z"
    assert point.payload["corpus"] == "sec_filings"
    assert isinstance(point.vector, dict) and set(point.vector) == {DENSE, SPARSE}

    again = run_qdrant_load(tmp_path, client, request)
    assert (again.created, again.points_skipped, again.points_upserted) == (False, 7, 0)

    # One chunk rewritten, one gone: re-embed, reload.
    changed = gold_chunks()
    changed[0] = gold_chunk("d24", 1, 3, date(2024, 2, 1), "Rewritten revenue note")
    del changed[-1]
    _gold(tmp_path, changed)
    update = run_qdrant_load(tmp_path, client, request)
    assert (update.points_upserted, update.points_deleted, update.points_skipped) == (
        1,
        1,
        5,
    )
    assert client.count(COLLECTION, exact=True).count == 6
    rewritten = client.retrieve(COLLECTION, ids=[point_id("d24:chunk:000001")])[0]
    assert rewritten.payload is not None
    assert rewritten.payload["text"] == "Rewritten revenue note"
    assert set(_audit(tmp_path, client).values()) == {"PASS", "SKIP"}


def test_hybrid_query_finds_keyword_chunk(tmp_path: Path) -> None:
    _gold(tmp_path)
    client = QdrantClient(":memory:")
    run_qdrant_load(
        tmp_path, client, LoadRequest(model_id=MODEL, collection=COLLECTION)
    )
    sparse = query_vector("margin note 3")
    hits = client.query_points(
        COLLECTION,
        query=models.SparseVector(indices=sparse.indices, values=sparse.values),
        using=SPARSE,
        limit=1,
        with_payload=["chunk_id"],
    ).points
    assert hits[0].payload == {"chunk_id": "d25:chunk:000003"}


def test_chunks_without_vectors_leave_alias_alone(tmp_path: Path) -> None:
    _gold(tmp_path)
    write_chunks(
        tmp_path, [*gold_chunks(), gold_chunk("d25", 5, 5, date(2025, 2, 1), "New")]
    )
    client = QdrantClient(":memory:")
    run = run_qdrant_load(
        tmp_path, client, LoadRequest(model_id=MODEL, collection=COLLECTION)
    )
    assert run.status == "incomplete" and run.chunks_without_vector == 1
    assert client.count(COLLECTION, exact=True).count == 7


def test_no_vectors_rejects(tmp_path: Path) -> None:
    write_chunks(tmp_path, gold_chunks())
    with open_gold_catalog(tmp_path) as gold:
        gold.create_namespace_if_not_exists("native_rag")
    run_native_rag_embeddings(tmp_path, FakeEmbedder(), NOW, limit=0)
    run = run_qdrant_load(
        tmp_path, QdrantClient(":memory:"), LoadRequest(model_id=MODEL)
    )
    assert (
        run.status == "rejected" and "no test/fake-embedding vectors" in run.problems[0]
    )


def test_bm25_terms_and_weights() -> None:
    assert terms("Net sales of $1,234.5 million in the 10-K (FY2025).") == [
        "net",
        "sales",
        "1234.5",
        "million",
        "10",
        "k",
        "fy2025",
    ]
    vector = document_vector("revenue revenue growth")
    assert vector.indices == sorted(set(vector.indices)) and len(vector.indices) == 2
    weights = dict(zip(vector.indices, vector.values, strict=True))
    revenue, growth = (
        document_vector(word).indices[0] for word in ("revenue", "growth")
    )
    assert weights[revenue] > weights[growth] > 0
    assert query_vector("revenue revenue").values == [1.0]
    assert document_vector("the of and").indices == []


def test_existing_collection_with_other_layout_is_refused(tmp_path: Path) -> None:
    _gold(tmp_path)
    client = QdrantClient(":memory:")
    request = LoadRequest(model_id=MODEL, collection=COLLECTION)
    run_qdrant_load(tmp_path, client, request)
    with pytest.raises(ValueError, match="differs from the requested layout"):
        run_qdrant_load(
            tmp_path,
            client,
            LoadRequest(model_id=MODEL, collection=COLLECTION, large=True),
        )
    assert (
        collection_problems(client, COLLECTION, CollectionSettings(dimensions=8)) == []
    )


@pytest.mark.parametrize("tamper", ["payload", "dense", "sparse"])
def test_audit_fails_on_tampered_point(tmp_path: Path, tamper: str) -> None:
    _gold(tmp_path)
    client = QdrantClient(":memory:")
    run_qdrant_load(
        tmp_path, client, LoadRequest(model_id=MODEL, collection=COLLECTION)
    )
    pid = point_id("d24:chunk:000002")
    if tamper == "payload":
        client.set_payload(COLLECTION, payload={"text": "tampered"}, points=[pid])
    else:
        update: dict[str, list[float] | models.SparseVector] = (
            {DENSE: [1.0] + [0.0] * 7}
            if tamper == "dense"
            else {SPARSE: models.SparseVector(indices=[1], values=[1.0])}
        )
        client.update_vectors(
            COLLECTION, points=[models.PointVectors(id=pid, vector=update)]
        )
    assert _audit(tmp_path, client)["sample_points"] == "FAIL"


def test_year_boundary_two_companies_and_period_end(tmp_path: Path) -> None:
    chunks = [
        gold_chunk("dDec", 1, 1, date(2024, 12, 31), "December filing about widgets",
                   cik="0000000001", period_end=date(2024, 9, 30)),
        gold_chunk("dJan", 1, 1, date(2025, 1, 1), "January filing about gadgets",
                   cik="0000000002", period_end=date(2024, 12, 31)),
    ]  # fmt: skip
    _gold(tmp_path, chunks)
    client = QdrantClient(":memory:")
    request = LoadRequest(model_id=MODEL, collection=COLLECTION)
    assert run_qdrant_load(tmp_path, client, request).points_upserted == 2
    point = client.retrieve(COLLECTION, ids=[point_id("dJan:chunk:000001")])[0]
    assert point.payload is not None
    assert point.payload["filing_date"] == "2025-01-01T00:00:00Z"
    assert point.payload["period_end_date"] == "2024-12-31T00:00:00Z"
    assert set(_audit(tmp_path, client).values()) == {"PASS", "SKIP"}
    again = run_qdrant_load(tmp_path, client, request)
    assert (again.points_skipped, again.points_upserted, again.points_deleted) == (
        2,
        0,
        0,
    )


def test_points_of_years_gold_dropped_are_deleted(tmp_path: Path) -> None:
    _gold(tmp_path)
    client = QdrantClient(":memory:")
    request = LoadRequest(model_id=MODEL, collection=COLLECTION)
    run_qdrant_load(tmp_path, client, request)
    _gold(tmp_path, [c for c in gold_chunks() if c.filing_date.year == 2025])
    run = run_qdrant_load(tmp_path, client, request)
    assert run.points_deleted == 3
    assert client.count(COLLECTION, exact=True).count == 4


def test_company_directory_is_loaded_and_audited(tmp_path: Path) -> None:
    chunks = [
        *gold_chunks(),
        gold_chunk("dB", 1, 1, date(2025, 3, 1), "Other company", cik="0000000002"),
    ]
    _gold(tmp_path, chunks)
    client = QdrantClient(":memory:")
    run = run_qdrant_load(
        tmp_path, client, LoadRequest(model_id=MODEL, collection=COLLECTION)
    )
    assert run.companies == 2
    rows = {row["cik"]: row for row in stored_companies(client, COLLECTION)}
    assert rows["0000320193"]["chunk_count"] == 7
    assert rows["0000000002"]["latest_filing_date"] == "2025-03-01"
    assert _audit(tmp_path, client)["companies"] == "PASS"
    client.delete(
        companies_collection(COLLECTION),
        points_selector=models.PointIdsList(points=[company_point_id("0000000002")]),
    )
    assert _audit(tmp_path, client)["companies"] == "FAIL"
    # The next load writes the directory again and drops companies Gold lost.
    _gold(tmp_path, gold_chunks())
    assert (
        run_qdrant_load(
            tmp_path, client, LoadRequest(model_id=MODEL, collection=COLLECTION)
        ).companies
        == 1
    )
    assert [r["cik"] for r in stored_companies(client, COLLECTION)] == ["0000320193"]
    assert _audit(tmp_path, client)["companies"] == "PASS"


def test_company_directory_lists_loaded_points_only(tmp_path: Path) -> None:
    _gold(tmp_path)
    # A new company's chunk without a vector yet: the load is incomplete.
    write_chunks(
        tmp_path,
        [
            *gold_chunks(),
            gold_chunk("dN", 1, 1, date(2025, 6, 1), "New", cik="0000000009"),
        ],
    )
    client = QdrantClient(":memory:")
    run = run_qdrant_load(
        tmp_path, client, LoadRequest(model_id=MODEL, collection=COLLECTION)
    )
    assert run.status == "incomplete" and run.companies == 1
    assert [r["cik"] for r in stored_companies(client, COLLECTION)] == ["0000320193"]


def test_vectors_are_read_only_from_the_files_holding_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [
        gold_chunk("dBig", i, 250, date(2024, 3, 1), f"Paragraph number {i}")
        for i in range(1, 251)
    ]
    write_chunks(tmp_path, chunks)
    with open_gold_catalog(tmp_path) as gold:
        gold.create_namespace_if_not_exists("native_rag")
    for _ in range(3):  # three commits: three embeddings files
        run_native_rag_embeddings(tmp_path, FakeEmbedder(), NOW, limit=100)
    with open_gold_catalog(tmp_path) as gold:
        embeddings = gold.load_table(EMBEDDINGS_TABLE)
        index = vector_index(embeddings, MODEL, 2024)
        assert len(index.hashes) == 250 and len(set(index.files.values())) == 3
        read: list[list[str]] = []
        original = pipeline_module.scan_batches

        def recording(table: Any, columns: Any = None, **kwargs: Any) -> Any:
            read.append(list(kwargs.get("files", [])))
            return original(table, columns, **kwargs)

        monkeypatch.setattr(pipeline_module, "scan_batches", recording)
        first_file = min(index.files.values())
        wanted = {c for c, f in index.files.items() if f == first_file}
        assert len(wanted) > 0
        vectors = vectors_in(embeddings, MODEL, index, wanted)
        assert read == [[first_file]] and set(vectors.rows) == wanted
        many = set(index.hashes)  # over PyIceberg's 200-value pruning limit
        assert len(vectors_in(embeddings, MODEL, index, many).rows) == 250
    client = QdrantClient(":memory:")
    monkeypatch.undo()
    run = run_qdrant_load(
        tmp_path, client, LoadRequest(model_id=MODEL, collection=COLLECTION)
    )
    assert (run.status, run.points_upserted) == ("completed", 250)


def test_payload_change_without_text_change_rewrites_the_point(tmp_path: Path) -> None:
    _gold(tmp_path)
    client = QdrantClient(":memory:")
    request = LoadRequest(model_id=MODEL, collection=COLLECTION)
    run_qdrant_load(tmp_path, client, request)
    changed = gold_chunks()
    changed[0] = changed[0].model_copy(update={"section_topic": "risk_factors"})
    write_chunks(tmp_path, changed)  # same text: the vectors stay current
    run = run_qdrant_load(tmp_path, client, request)
    assert (run.points_upserted, run.points_skipped) == (1, 6)
    point = client.retrieve(COLLECTION, ids=[point_id("d24:chunk:000001")])[0]
    assert point.payload is not None
    assert point.payload["section_topic"] == "risk_factors"
    assert set(_audit(tmp_path, client).values()) == {"PASS", "SKIP"}


def test_point_digests_catch_a_swap_the_count_misses(tmp_path: Path) -> None:
    _gold(tmp_path)
    client = QdrantClient(":memory:")
    run_qdrant_load(
        tmp_path, client, LoadRequest(model_id=MODEL, collection=COLLECTION)
    )
    stale = client.retrieve(
        COLLECTION, ids=[point_id("d25:chunk:000004")], with_vectors=True
    )[0]
    client.delete(COLLECTION, points_selector=models.PointIdsList(points=[stale.id]))
    assert isinstance(stale.vector, dict)
    client.upsert(
        COLLECTION,
        [models.PointStruct(id=point_id("gone:chunk:000001"), vector=stale.vector,
                            payload=stale.payload)],
    )  # fmt: skip
    statuses = _audit(tmp_path, client)
    assert statuses["point_count"] == "PASS" and statuses["point_digests"] == "FAIL"


def test_retry_only_transient_errors() -> None:
    delays: list[float] = []
    calls: list[int] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise UnexpectedResponse(503, "busy", b"", httpx.Headers())
        if len(calls) == 2:
            raise ResourceExhaustedResponse("slow down", 7)
        return "ok"

    assert retried(flaky, "flaky", sleep=delays.append) == "ok"
    assert delays == [2.0, 7.0]  # backoff, then the server's Retry-After

    def bad_request() -> None:
        raise UnexpectedResponse(400, "bad", b"", httpx.Headers())

    with pytest.raises(UnexpectedResponse):
        retried(bad_request, "bad", sleep=delays.append)
    assert transient(ResponseHandlingException(httpx.ConnectError("down")))
    try:
        models.CountResult.model_validate({"count": "many"})
    except ValidationError as error:
        assert not transient(ResponseHandlingException(error))


def test_bare_os_errors_are_retried() -> None:
    delays: list[float] = []
    calls: list[int] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionResetError("reset")
        return "ok"

    assert retried(flaky, "flaky", sleep=delays.append) == "ok" and delays == [2.0]


def test_points_survive_while_their_vectors_are_re_embedded(tmp_path: Path) -> None:
    _gold(tmp_path)
    client = QdrantClient(":memory:")
    request = LoadRequest(model_id=MODEL, collection=COLLECTION)
    run_qdrant_load(tmp_path, client, request)
    with open_gold_catalog(tmp_path) as gold:
        gold.purge_table(EMBEDDINGS_TABLE)  # vectors purged, chunks unchanged
    gold_dir = tmp_path
    with open_gold_catalog(gold_dir) as gold:
        gold.create_namespace_if_not_exists("native_rag")
    run_native_rag_embeddings(gold_dir, FakeEmbedder(), NOW, limit=2)
    run = run_qdrant_load(tmp_path, client, request)
    assert run.status == "incomplete" and run.points_deleted == 0
    assert client.count(COLLECTION, exact=True).count == 7

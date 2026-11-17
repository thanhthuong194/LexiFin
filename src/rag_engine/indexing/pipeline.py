from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from pyiceberg.expressions import And
from pyiceberg.table import Table
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.data_pipeline.storage.iceberg import (
    open_gold_catalog_read_only,
    scan_batches,
    snapshot_total,
    table_rules_version,
    values_filter,
)
from src.data_pipeline.storage.native_rag.chunks import CHUNKS_TABLE
from src.data_pipeline.storage.native_rag.embeddings import EMBEDDINGS_TABLE
from src.data_pipeline.transforms.native_rag.context import embedded_text
from src.rag_engine.embedding.pipeline import table_years, year_filter
from src.rag_engine.indexing.qdrant_store import (
    DENSE,
    SEC_FILINGS,
    SPARSE,
    CollectionSettings,
    Corpus,
    collection_name,
    enable_indexing,
    ensure_collection,
    payload,
    point_id,
)
from src.rag_engine.indexing.sparse import document_vector
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)

POINTS_PER_UPSERT = 512
# Above this many chunks the collection must use the large layout.
LARGE_MIN_CHUNKS = 500_000
SCROLL_PAGE = 10_000
# Transient server errors (5xx, 429, connection) retried with backoff.
WRITE_ATTEMPTS = 5
RETRY_BASE_SECONDS = 2.0
_TRANSIENT = (ResponseHandlingException, UnexpectedResponse, OSError)
CHUNK_COLUMNS = (
    "chunk_id",
    "filing_id",
    "document_id",
    "section_id",
    "filing_date",
    "cik",
    "tickers",
    "company_name",
    "form",
    "document_type",
    "fiscal_year",
    "fiscal_period",
    "period_end_date",
    "section_topic",
    "section_title",
    "subheading",
    "chunk_kind",
    "table_title",
    "chunk_ordinal",
    "part",
    "part_count",
    "accession_number",
    "sec_url",
    "context",
    "text",
    "token_count",
    "chunk_sha256",
)


@dataclass
class QdrantLoadSummary:
    """What one Gold -> Qdrant load did."""

    status: str = "running"
    collection: str = ""
    created: bool = False
    chunks_total: int = 0
    chunks_without_vector: int = 0
    points_skipped: int = 0
    points_upserted: int = 0
    points_deleted: int = 0
    problems: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


def year_points(year: int) -> models.Filter:
    """Points of one filing year (payload `filing_date`)."""

    return models.Filter(
        must=[
            models.FieldCondition(
                key="filing_date",
                range=models.DatetimeRange(
                    gte=datetime(year, 1, 1, tzinfo=UTC),
                    lt=datetime(year + 1, 1, 1, tzinfo=UTC),
                ),
            )
        ]
    )


def data_files(table: Table, year: int) -> list[str]:
    """The current data files of one `year(filing_date)` partition. Gold
    files are commit batches spanning many months, so they are read one at a
    time instead of by month (a month filter would reread a year 12 times)."""

    tasks = list(table.scan(row_filter=year_filter(year)).plan_files())
    if any(task.delete_files for task in tasks):
        raise ValueError(f"{table.name()} has delete files")
    return [task.file.file_path for task in tasks]


def vector_hashes(embeddings: Table, model_id: str, year: int) -> dict[str, str]:
    """`chunk_id` -> `chunk_sha256` of the model's vectors of one year (no
    vector data read)."""

    found: dict[str, str] = {}
    columns = ("chunk_id", "model_id", "chunk_sha256")
    for batch in scan_batches(embeddings, columns, files=year_filter(year)):
        mine = batch.filter(pc.equal(batch["model_id"], model_id))
        found.update(
            zip(
                mine["chunk_id"].to_pylist(),
                mine["chunk_sha256"].to_pylist(),
                strict=True,
            )
        )
    return found


@dataclass(frozen=True)
class Vectors:
    """Vectors of some chunks, kept as Arrow float32 (4 bytes a value)
    until a point is built, not as Python floats (~32 bytes each)."""

    rows: dict[str, int]
    vectors: pa.Array

    def get(self, chunk_id: str) -> list[float]:
        vector: list[float] = self.vectors[self.rows[chunk_id]].as_py()
        return vector


def vectors_of(
    embeddings: Table, model_id: str, year: int, chunk_ids: set[str]
) -> Vectors:
    """The model's vectors of the given chunks of one year; only data files
    whose `chunk_id` bounds can hold them are read."""

    empty = Vectors({}, pa.array([], pa.list_(pa.float32())))
    if not chunk_ids:
        return empty
    wanted = pa.array(sorted(chunk_ids), type=pa.string())
    files = And(year_filter(year), values_filter("chunk_id", chunk_ids))
    batches: list[pa.RecordBatch] = []
    for batch in scan_batches(
        embeddings, ("chunk_id", "model_id", "vector"), files=files
    ):
        mine = batch.filter(
            pc.and_(
                pc.equal(batch["model_id"], model_id),
                pc.is_in(batch["chunk_id"], value_set=wanted),
            )
        )
        if mine.num_rows:
            batches.append(mine.select(["chunk_id", "vector"]))
    if not batches:
        return empty
    table = pa.Table.from_batches(batches)
    del batches
    return Vectors(
        {chunk_id: i for i, chunk_id in enumerate(table["chunk_id"].to_pylist())},
        table["vector"].combine_chunks(),
    )


def file_rows(
    chunks: Table, path: str, columns: tuple[str, ...] = CHUNK_COLUMNS
) -> Iterator[dict[str, Any]]:
    for batch in scan_batches(chunks, columns, files=[path]):
        yield from batch.to_pylist()


def chunks_in(
    chunks: Table, year: int, chunk_ids: set[str]
) -> Iterator[dict[str, Any]]:
    """Full rows of the given chunks of one year, reading only data files
    whose `chunk_id` bounds can hold them."""

    if not chunk_ids:
        return
    files = And(year_filter(year), values_filter("chunk_id", chunk_ids))
    for batch in scan_batches(chunks, CHUNK_COLUMNS, files=files):
        for row in batch.to_pylist():
            if row["chunk_id"] in chunk_ids:
                yield row


def stored_points(
    client: QdrantClient, name: str, scope: models.Filter
) -> dict[str, str]:
    """Point ID -> `chunk_sha256` of the collection's points matching `scope`."""

    def page(offset: Any) -> tuple[list[models.Record], Any]:
        return retried(
            lambda: client.scroll(
                name,
                scroll_filter=scope,
                limit=SCROLL_PAGE,
                offset=offset,
                with_payload=["chunk_sha256"],
                with_vectors=False,
            ),
            "scroll",
        )

    found: dict[str, str] = {}
    offset: Any = None
    while True:
        points, offset = page(offset)
        for point in points:
            found[str(point.id)] = (point.payload or {}).get("chunk_sha256", "")
        if offset is None:
            return found


def _transient(error: Exception) -> bool:
    """Server-side or rate-limit failures; a 4xx (bad request) is not."""

    if isinstance(error, UnexpectedResponse):
        status = error.status_code or 0
        return status == 429 or status >= 500
    return True


def retried[T](action: Callable[[], T], what: str) -> T:
    """Run a server call, retrying transient errors with backoff."""

    for attempt in range(1, WRITE_ATTEMPTS + 1):
        try:
            return action()
        except _TRANSIENT as error:
            if attempt == WRITE_ATTEMPTS or not _transient(error):
                raise
            logger.warning(
                "qdrant_call_retry", what=what, attempt=attempt, error=str(error)[:200]
            )
            time.sleep(RETRY_BASE_SECONDS * 2 ** (attempt - 1))
    raise AssertionError("unreachable")


def _point(
    chunk: dict[str, Any], vector: list[float], corpus: Corpus
) -> models.PointStruct:
    sparse = document_vector(embedded_text(chunk["context"], chunk["text"]))
    return models.PointStruct(
        id=point_id(chunk["chunk_id"]),
        vector={
            DENSE: vector,
            SPARSE: models.SparseVector(indices=sparse.indices, values=sparse.values),
        },
        payload=payload(chunk, corpus),
    )


@dataclass
class _Load:
    client: QdrantClient
    name: str
    corpus: Corpus
    model_id: str
    summary: QdrantLoadSummary

    def upsert(self, points: list[models.PointStruct]) -> None:
        if not points:
            return
        retried(
            lambda: self.client.upsert(self.name, points=points, wait=True), "upsert"
        )
        self.summary.points_upserted += len(points)

    def delete(self, ids: list[str]) -> None:
        for start in range(0, len(ids), POINTS_PER_UPSERT):
            selector = models.PointIdsList(
                points=list(ids[start : start + POINTS_PER_UPSERT])
            )
            self._delete_points(selector)
        self.summary.points_deleted += len(ids)

    def _delete_points(self, selector: models.PointIdsList) -> None:
        retried(
            lambda: self.client.delete(self.name, points_selector=selector, wait=True),
            "delete",
        )

    def load_year(self, chunks: Table, embeddings: Table, year: int) -> None:
        """Per data file of the year's chunks: pass 1 on hashes only finds
        the chunks that need a point; pass 2 reads the vectors of just those
        chunks and upserts them. Then points no current Gold chunk backs are
        deleted."""

        hashes = vector_hashes(embeddings, self.model_id, year)
        stored = stored_points(self.client, self.name, year_points(year))
        current: set[str] = set()
        for path in data_files(chunks, year):
            pending: set[str] = set()
            for row in file_rows(chunks, path, ("chunk_id", "chunk_sha256")):
                self.summary.chunks_total += 1
                chunk_id, digest = row["chunk_id"], row["chunk_sha256"]
                if hashes.get(chunk_id) != digest:
                    self.summary.chunks_without_vector += 1
                    continue
                pid = point_id(chunk_id)
                current.add(pid)
                if stored.get(pid) == digest:
                    self.summary.points_skipped += 1
                else:
                    pending.add(chunk_id)
            if pending:
                self._upsert_file(chunks, embeddings, path, year, pending)
        self.delete(sorted(set(stored) - current))
        logger.info(
            "qdrant_load_year_done", year=year, upserted=self.summary.points_upserted
        )

    def _upsert_file(
        self, chunks: Table, embeddings: Table, path: str, year: int, pending: set[str]
    ) -> None:
        vectors = vectors_of(embeddings, self.model_id, year, pending)
        points: list[models.PointStruct] = []
        for chunk in file_rows(chunks, path):
            if chunk["chunk_id"] not in pending:
                continue
            points.append(_point(chunk, vectors.get(chunk["chunk_id"]), self.corpus))
            if len(points) >= POINTS_PER_UPSERT:
                self.upsert(points)
                points = []
        self.upsert(points)


def outside_years(years: list[int]) -> models.Filter:
    """Points whose filing year is none of `years`."""

    return models.Filter(
        must_not=[
            models.FieldCondition(
                key="filing_date",
                range=models.DatetimeRange(
                    gte=datetime(year, 1, 1, tzinfo=UTC),
                    lt=datetime(year + 1, 1, 1, tzinfo=UTC),
                ),
            )
            for year in years
        ]
    )


def model_dimensions(embeddings: Table, model_id: str) -> int | None:
    for batch in scan_batches(embeddings, ("model_id", "dimensions")):
        mine = batch.filter(pc.equal(batch["model_id"], model_id))
        if mine.num_rows:
            return int(mine["dimensions"][0].as_py())
    return None


@dataclass(frozen=True)
class LoadRequest:
    """Which vectors go where: model, collection (default derived from the
    corpus, the chunks' rules version and the model), layout."""

    model_id: str
    corpus: Corpus = SEC_FILINGS
    collection: str | None = None
    large: bool = False


def target_collection(chunks: Table, request: LoadRequest) -> str:
    return request.collection or collection_name(
        request.corpus, table_rules_version(chunks), request.model_id
    )


def _load(
    gold_dir: Path,
    client: QdrantClient,
    request: LoadRequest,
    summary: QdrantLoadSummary,
) -> None:
    with open_gold_catalog_read_only(gold_dir) as gold:
        chunks = gold.load_table(CHUNKS_TABLE)
        embeddings = gold.load_table(EMBEDDINGS_TABLE)
        dimensions = model_dimensions(embeddings, request.model_id)
        if dimensions is None:
            summary.problems.append(
                f"no {request.model_id} vectors in {EMBEDDINGS_TABLE}"
            )
            return
        name = target_collection(chunks, request)
        summary.collection = name
        total = snapshot_total(chunks, "total-records")
        if total > LARGE_MIN_CHUNKS and not request.large:
            summary.problems.append(
                f"{total} chunks: load with --large (vectors on disk, int8 in RAM); "
                f"in-RAM float vectors would need ~{total * dimensions * 4 // 2**30} GB"
            )
            return
        settings = CollectionSettings(dimensions=dimensions, large=request.large)
        summary.created = ensure_collection(client, name, settings)
        load = _Load(client, name, request.corpus, request.model_id, summary)
        years = table_years(chunks, embeddings)
        for year in years:
            load.load_year(chunks, embeddings, year)
        # Points of years Gold no longer has (lookback moved, rebuild).
        outside = outside_years(years)
        gone = retried(
            lambda: client.count(name, count_filter=outside, exact=True), "count"
        ).count
        if gone:
            retried(
                lambda: client.delete(
                    name,
                    points_selector=models.FilterSelector(filter=outside),
                    wait=True,
                ),
                "delete outside years",
            )
            summary.points_deleted += gone
        if summary.chunks_without_vector:
            summary.problems.append(
                f"{summary.chunks_without_vector} chunks have no current "
                f"{request.model_id} vector: embed them, then load again"
            )
        enable_indexing(client, name, settings)


def run_qdrant_load(
    gold_dir: Path, client: QdrantClient, request: LoadRequest
) -> QdrantLoadSummary:
    """Load Gold chunks with their current vectors into a Qdrant collection.

    One month of filing dates at a time: points are compared with Gold by
    `chunk_sha256`; vectors are read only for new or changed chunks, which
    are upserted (dense vector, BM25 sparse vector of the embedded text,
    payload) in acknowledged batches of POINTS_PER_UPSERT with retries;
    unchanged points are skipped; points no current Gold chunk backs (deleted,
    changed and not re-embedded, or of a year Gold no longer has) are
    deleted. Indexing is turned on at the end. The alias is not touched: the
    caller switches it once the collection is indexed and audited
    (`switch_alias`). Status `incomplete` when chunks lack a current vector
    (embed, then load again). Gold is read-only. A rerun resumes.
    """

    started = time.perf_counter()
    summary = QdrantLoadSummary()
    with log_context(pipeline="qdrant_load", model_id=request.model_id):
        logger.info("qdrant_load_started", gold_dir=str(gold_dir))
        with timed_operation("qdrant_load.total"):
            _load(gold_dir, client, request, summary)
        summary.status = (
            "incomplete"
            if summary.chunks_without_vector
            else "rejected"
            if summary.problems
            else "completed"
        )
        summary.duration_seconds = round(time.perf_counter() - started, 3)
        log = logger.info if summary.status == "completed" else logger.error
        log("qdrant_load_completed", **asdict(summary))
    return summary

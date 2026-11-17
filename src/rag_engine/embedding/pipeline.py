from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from pyiceberg.exceptions import CommitFailedException
from pyiceberg.expressions import BooleanExpression
from pyiceberg.expressions.parser import parse
from pyiceberg.table import Table

from src.data_pipeline.contracts.native_rag.embeddings import (
    EMBEDDINGS_ARROW_SCHEMA,
    EmbeddingRecord,
    embedding_id_for,
)
from src.data_pipeline.storage.iceberg import (
    append_rows,
    current_snapshot_id,
    open_gold_catalog,
    scan_batches,
    snapshot_total,
    verify_append,
)
from src.data_pipeline.storage.native_rag.chunks import CHUNKS_TABLE
from src.data_pipeline.storage.native_rag.embeddings import EMBEDDINGS_SPEC
from src.data_pipeline.transforms.native_rag.context import embedded_text
from src.llm.base import EmbeddingModel
from src.llm.types import LLMError
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)

# Chunks per request group; `concurrency` groups are sent at once (the server
# batches concurrent requests), then their vectors are written in order.
CHUNKS_PER_GROUP = 128
ROWS_PER_COMMIT = 4096


@dataclass
class EmbeddingsSummary:
    """What one embedding run did."""

    status: str = "running"
    model_id: str = ""
    chunks_total: int = 0
    chunks_skipped: int = 0
    chunks_embedded: int = 0
    commits: int = 0
    problems: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    chunks_per_second: float = 0.0


@dataclass(frozen=True)
class _Pending:
    chunk_id: str
    document_id: str
    filing_date: date
    chunk_sha256: str
    text: str


def year_filter(year: int) -> BooleanExpression:
    """Data files of one `year(filing_date)` partition."""

    return parse(f"filing_date >= '{year}-01-01' and filing_date < '{year + 1}-01-01'")


def table_years(*tables: Table) -> list[int]:
    """Filing years present in any of the tables (both are partitioned by
    year), so vectors of a year the chunks no longer have are still seen."""

    years: set[int] = set()
    for table in tables:
        for batch in scan_batches(table, ("filing_date",), batch_size=100_000):
            years.update(day.year for day in batch["filing_date"].to_pylist())
    return sorted(years)


def stored_hashes(
    embeddings: Table, model_id: str, year: int
) -> tuple[dict[str, str], set[int]]:
    """`chunk_id` -> `chunk_sha256` of the model's stored vectors of one year,
    and the vector sizes stored."""

    found: dict[str, str] = {}
    sizes: set[int] = set()
    columns = ("chunk_id", "model_id", "chunk_sha256", "dimensions")
    for batch in scan_batches(embeddings, columns, files=year_filter(year)):
        for chunk_id, model, digest, size in zip(
            *(batch[c].to_pylist() for c in columns), strict=True
        ):
            if model == model_id:
                found[chunk_id] = digest
                sizes.add(size)
    return found, sizes


def current_hashes(chunks: Table, year: int) -> dict[str, str]:
    columns = ("chunk_id", "chunk_sha256")
    found: dict[str, str] = {}
    for batch in scan_batches(chunks, columns, files=year_filter(year)):
        found.update(
            zip(
                batch["chunk_id"].to_pylist(),
                batch["chunk_sha256"].to_pylist(),
                strict=True,
            )
        )
    return found


def stale_vectors(stored: dict[str, str], current: dict[str, str]) -> list[str]:
    """Stored vectors whose chunk is gone or whose text changed."""

    return sorted(c for c, digest in stored.items() if current.get(c) != digest)


def _pending(chunks: Table, year: int, stored: dict[str, str]) -> Iterator[_Pending]:
    columns = (
        "chunk_id",
        "document_id",
        "filing_date",
        "chunk_sha256",
        "context",
        "text",
    )
    for batch in scan_batches(chunks, columns, files=year_filter(year)):
        for chunk_id, document_id, day, digest, context, text in zip(
            *(batch[c].to_pylist() for c in columns), strict=True
        ):
            if stored.get(chunk_id) != digest:
                yield _Pending(
                    chunk_id, document_id, day, digest, embedded_text(context, text)
                )


def normalized(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """Vectors scaled to unit length; raises on mixed sizes, a non-finite
    value or a zero vector."""

    if len({len(vector) for vector in vectors}) > 1:
        raise ValueError("embedder returned vectors of mixed sizes")
    result: list[list[float]] = []
    for vector in vectors:
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("embedder returned a zero or non-finite vector")
        result.append([value / norm for value in vector])
    return result


@dataclass
class _Run:
    table: Table
    embedder: EmbeddingModel
    processed_at: datetime
    concurrency: int
    summary: EmbeddingsSummary
    rows: list[EmbeddingRecord] = field(default_factory=list)
    dimensions: int | None = None

    def embed(self, pending: Sequence[_Pending]) -> None:
        groups = [
            pending[i : i + CHUNKS_PER_GROUP]
            for i in range(0, len(pending), CHUNKS_PER_GROUP)
        ]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            results = list(
                pool.map(
                    lambda group: self.embedder.embed_documents(
                        [p.text for p in group]
                    ),
                    groups,
                )
            )
        vectors = normalized([v for result in results for v in result])
        width = len(vectors[0])
        if self.dimensions not in (None, width):
            raise ValueError(f"vector size changed from {self.dimensions} to {width}")
        self.dimensions = width
        model_id = self.embedder.model
        for item, vector in zip(pending, vectors, strict=True):
            self.rows.append(
                EmbeddingRecord(
                    embedding_id=embedding_id_for(model_id, item.chunk_id),
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    filing_date=item.filing_date,
                    model_id=model_id,
                    chunk_sha256=item.chunk_sha256,
                    dimensions=width,
                    vector=vector,
                    processed_at=self.processed_at,
                )
            )
        self.summary.chunks_embedded += len(pending)
        if len(self.rows) >= ROWS_PER_COMMIT:
            self.commit()

    def commit(self) -> None:
        if not self.rows:
            return
        rows = sorted(self.rows, key=lambda r: r.embedding_id)
        # Taken out before the write: whatever happens next, these rows are
        # never appended twice (a failed write is re-embedded by the next run).
        self.rows = []
        before = current_snapshot_id(self.table)
        expected = snapshot_total(self.table, "total-records") + len(rows)
        append_rows(self.table, rows, EMBEDDINGS_ARROW_SCHEMA)
        verify_append(
            self.table,
            key_column="embedding_id",
            expected_rows=expected,
            inserted_keys=[r.embedding_id for r in rows],
            snapshot_before=before,
        )
        self.summary.commits += 1
        logger.info(
            "native_rag_embeddings_committed",
            rows=len(rows),
            chunks_embedded=self.summary.chunks_embedded,
        )


def _embed_all(
    gold_dir: Path,
    run_args: dict[str, Any],
    summary: EmbeddingsSummary,
    limit: int | None,
) -> None:
    with open_gold_catalog(gold_dir) as gold:
        chunks = gold.load_table(CHUNKS_TABLE)
        table, _ = EMBEDDINGS_SPEC.open(gold, gold_dir)
        model_id = summary.model_id
        years = table_years(chunks, table)
        sizes: set[int] = set()
        # Pre-pass, one year at a time and keeping only counts: stale vectors
        # reject the run before any request.
        for year in years:
            stored, year_sizes = stored_hashes(table, model_id, year)
            sizes |= year_sizes
            current = current_hashes(chunks, year)
            summary.chunks_total += len(current)
            summary.chunks_skipped += sum(
                current.get(c) == d for c, d in stored.items()
            )
            stale = stale_vectors(stored, current)
            if stale:
                summary.problems.append(
                    f"{len(stale)} stored {model_id} vectors of {year} have no current "
                    f"chunk with their hash (e.g. {stale[0]}): chunks were rebuilt; "
                    "purge native_rag.embeddings and embed again"
                )
        if len(sizes) > 1:
            summary.problems.append(
                f"stored {model_id} vectors have sizes {sorted(sizes)}"
            )
        if summary.problems:
            return
        run = _Run(
            table, summary=summary, dimensions=next(iter(sizes), None), **run_args
        )
        try:
            _embed_years(run, chunks, years, limit)
        except BaseException:
            # Keep the vectors already made when a request fails mid-run; a
            # failure of this commit is logged, the original error re-raised.
            try:
                run.commit()
            except (OSError, ValueError, RuntimeError, CommitFailedException) as error:
                logger.error(
                    "native_rag_embeddings_final_commit_failed", error=str(error)
                )
            raise
        run.commit()


def _embed_years(
    run: _Run, chunks: Table, years: Sequence[int], limit: int | None
) -> None:
    group = CHUNKS_PER_GROUP * run.concurrency
    for year in years:
        stored, _ = stored_hashes(run.table, run.embedder.model, year)
        buffer: list[_Pending] = []
        for item in _pending(chunks, year, stored):
            if limit is not None and run.summary.chunks_embedded + len(buffer) >= limit:
                break
            buffer.append(item)
            if len(buffer) >= group:
                run.embed(buffer)
                buffer = []
        if buffer:
            run.embed(buffer)


def run_native_rag_embeddings(
    gold_dir: Path,
    embedder: EmbeddingModel,
    processed_at: datetime,
    *,
    concurrency: int = 4,
    limit: int | None = None,
) -> EmbeddingsSummary:
    """Embed every chunk of `native_rag.chunks` that has no vector of this
    model yet (or whose text changed) into `native_rag.embeddings`.

    The embedded string is the chunk's `context + "\\n\\n" + text`, as counted
    by the chunker; documents get no instruction (queries do, at search time).
    Vectors are L2-normalized and written in verified commits of
    ROWS_PER_COMMIT rows, so an interrupted run (the Colab tunnel closing)
    resumes where it stopped: a server failure ends the run with status
    `interrupted` and the vectors made so far committed. Stored vectors whose
    chunk is gone or changed, or of another size, reject the run before any
    request: purge the table and embed again.
    `limit` caps the chunks embedded this run (trial runs).
    """

    started = perf_counter()
    summary = EmbeddingsSummary(model_id=embedder.model)
    with log_context(pipeline="native_rag_embeddings", model_id=embedder.model):
        logger.info("native_rag_embeddings_started", gold_dir=str(gold_dir))
        run_args = {
            "embedder": embedder,
            "processed_at": processed_at,
            "concurrency": concurrency,
        }
        interrupted = False
        try:
            with timed_operation("native_rag_embeddings.total"):
                _embed_all(gold_dir, run_args, summary, limit)
        except LLMError as error:
            # The server is gone (a closed tunnel) or answered wrongly: vectors
            # made so far are committed; the next run continues from there.
            interrupted = True
            summary.problems.append(
                f"embedding server failed ({type(error).__name__}: {str(error)[:300]}); "
                "vectors made so far are kept: check the server (a new Colab tunnel "
                "URL goes in .env) and run again to continue"
            )
        except Exception as error:
            logger.error(
                "native_rag_embeddings_failed",
                error_type=type(error).__name__,
                error=str(error)[:500],
                chunks_embedded=summary.chunks_embedded,
            )
            raise
        summary.status = (
            "interrupted"
            if interrupted
            else "rejected"
            if summary.problems
            else "completed"
        )
        summary.duration_seconds = round(perf_counter() - started, 3)
        if summary.duration_seconds:
            summary.chunks_per_second = round(
                summary.chunks_embedded / summary.duration_seconds, 1
            )
        log = logger.info if summary.status == "completed" else logger.error
        log("native_rag_embeddings_completed", **asdict(summary))
    return summary

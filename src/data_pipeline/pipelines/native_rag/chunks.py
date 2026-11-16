from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from pyiceberg.catalog import Catalog
from pyiceberg.expressions import BooleanExpression
from pyiceberg.table import Table

from src.data_pipeline.contracts.native_rag.chunks import (
    CHUNKS_ARROW_SCHEMA,
    ChunkRecord,
    chunk_id_for,
    chunk_sha256,
)
from src.data_pipeline.pipelines.native_rag.filing_metadata import (
    DocumentMeta,
    FilingMeta,
    read_document_meta,
    read_filing_meta,
    sec_url,
)
from src.data_pipeline.storage.iceberg import (
    append_rows,
    current_snapshot_id,
    filing_dates_filter,
    open_gold_catalog,
    open_silver_catalog_read_only,
    read_rows_in,
    scan_batches,
    snapshot_total,
    summarize_groups,
    table_rules_version,
    verify_append,
)
from src.data_pipeline.storage.native_rag.chunks import CHUNKS_SPEC
from src.data_pipeline.storage.sec_filings.content_blocks import (
    CONTENT_BLOCKS_SPEC,
    CONTENT_BLOCKS_TABLE,
)
from src.data_pipeline.storage.sec_filings.document_table_cells import (
    DOCUMENT_TABLE_CELLS_SPEC,
    DOCUMENT_TABLE_CELLS_TABLE,
)
from src.data_pipeline.storage.sec_filings.document_tables import (
    DOCUMENT_TABLES_SPEC,
    DOCUMENT_TABLES_TABLE,
)
from src.data_pipeline.storage.sec_filings.documents import DOCUMENTS_SPEC
from src.data_pipeline.storage.sec_filings.exhibits import EXHIBITS_SPEC
from src.data_pipeline.storage.sec_filings.filing_entities import FILING_ENTITIES_SPEC
from src.data_pipeline.storage.sec_filings.filings import FILINGS_SPEC
from src.data_pipeline.storage.sec_filings.sections import (
    SECTIONS_SPEC,
    SECTIONS_TABLE,
)
from src.data_pipeline.storage.sec_filings.xbrl_core import (
    XBRL_CONTEXTS_SPEC,
    XBRL_INSTANCES_SPEC,
)
from src.data_pipeline.storage.sec_filings.xbrl_facts import (
    XBRL_CONCEPTS_SPEC,
    XBRL_FACTS_SPEC,
)
from src.data_pipeline.transforms.native_rag.blocks import SourceBlock, SourceSection
from src.data_pipeline.transforms.native_rag.context import (
    FilingContext,
    chunk_context,
    short_title,
)
from src.data_pipeline.transforms.native_rag.document_chunks import (
    ChunkParams,
    DocumentInput,
    chunk_document,
)
from src.data_pipeline.transforms.native_rag.section_topics import (
    SectionHead,
    section_topic,
)
from src.data_pipeline.transforms.native_rag.table_chunks import Cell
from src.data_pipeline.transforms.native_rag.tokenizer import TokenCounter
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)

DOCUMENTS_PER_COMMIT = 200
CELLS_PER_BATCH = 300_000
# processed_at of rebuilt rows (compared without it).
REBUILD_TIME = datetime(2000, 1, 1, tzinfo=UTC)
COVER_PAGE = "Cover page"
# Path of the prose kept from a table-of-contents section (it precedes PART I).
FRONT_MATTER = "Front matter"
# Table properties pinning what the stored chunks were built with.
TOKENIZER_PROPERTY = "lexifin.tokenizer"
PARAMS_PROPERTY = "lexifin.chunk-params"
SILVER_RULES_PROPERTY = "lexifin.silver-rules"
# Every Silver table chunks or their metadata are built from.
_SILVER_INPUTS = (
    FILINGS_SPEC,
    DOCUMENTS_SPEC,
    FILING_ENTITIES_SPEC,
    SECTIONS_SPEC,
    CONTENT_BLOCKS_SPEC,
    DOCUMENT_TABLES_SPEC,
    DOCUMENT_TABLE_CELLS_SPEC,
    EXHIBITS_SPEC,
    XBRL_INSTANCES_SPEC,
    XBRL_CONTEXTS_SPEC,
    XBRL_CONCEPTS_SPEC,
    XBRL_FACTS_SPEC,
)


@dataclass
class ChunksSummary:
    """What one chunking run did."""

    status: str = "running"
    documents_total: int = 0
    documents_skipped: int = 0
    documents_chunked: int = 0
    documents_without_chunks: int = 0
    documents_rejected: int = 0
    chunks_written: int = 0
    batches_committed: int = 0
    tokens_max: int = 0
    tokens_mean: float = 0.0
    dropped_blocks: dict[str, int] = field(default_factory=dict)
    chunk_kinds: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


def silver_rules(catalog: Catalog) -> str:
    """Rules versions of the Silver tables chunks are built from."""

    return ",".join(
        f"{spec.identifier}={table_rules_version(catalog.load_table(spec.identifier))}"
        for spec in _SILVER_INPUTS
    )


def _expected_properties(
    catalog: Catalog, counter: TokenCounter, params: ChunkParams
) -> dict[str, str]:
    return {
        TOKENIZER_PROPERTY: counter.identity,
        PARAMS_PROPERTY: params.as_property(),
        SILVER_RULES_PROPERTY: silver_rules(catalog),
    }


def _open_chunks(
    gold: Catalog, data_dir: Path, expected: Mapping[str, str]
) -> tuple[Table, list[str]]:
    """Open or create the chunks table; list how stored chunks were built
    differently from this run (tokenizer, parameters, Silver rules)."""

    table, created = CHUNKS_SPEC.open(gold, data_dir)
    if created:
        with table.transaction() as transaction:
            transaction.set_properties(dict(expected))
        table.refresh()
        return table, []
    problems = [
        f"{key} is {table.properties.get(key)!r}, this run uses {value!r}"
        for key, value in expected.items()
        if table.properties.get(key) != value
    ]
    return table, problems


def _section_paths(
    rows: Sequence[dict[str, Any]], document: DocumentMeta, form: str
) -> dict[str, SourceSection]:
    by_id = {row["section_id"]: row for row in rows}
    exhibit = document.document_type.upper().startswith("EX-99")
    sections: dict[str, SourceSection] = {}
    for row in sorted(rows, key=lambda r: r["section_ordinal"]):
        parent = sections.get(row["parent_section_id"] or "")
        if row["parent_section_id"] is None:
            path: tuple[str, ...] = (document.label if exhibit else COVER_PAGE,)
        elif row["section_kind"] == "table_of_contents":
            path = (FRONT_MATTER,)
        elif parent is not None and parent.parent_section_id is None and not exhibit:
            path = (short_title(row["title"]),)
        else:
            path = (*(parent.path if parent else ()), short_title(row["title"]))
        part_code = _part_code(row, by_id)
        head = SectionHead(
            row["section_kind"], row["section_code"], row["title"], part_code
        )
        sections[row["section_id"]] = SourceSection(
            section_id=row["section_id"],
            section_ordinal=row["section_ordinal"],
            parent_section_id=row["parent_section_id"],
            section_kind=row["section_kind"],
            section_code=row["section_code"],
            title=row["title"],
            start_block_ordinal=row["start_block_ordinal"],
            path=path,
            topic=section_topic(form, document.document_type, head),
        )
    return sections


def _part_code(
    row: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]
) -> str | None:
    parent = by_id.get(row["parent_section_id"] or "")
    while parent is not None:
        if parent["section_kind"] == "part":
            return str(parent["section_code"])
        parent = by_id.get(parent["parent_section_id"] or "")
    return None


@dataclass(frozen=True)
class _BatchRows:
    sections: dict[str, list[dict[str, Any]]]
    blocks: dict[str, list[dict[str, Any]]]
    cells: dict[str, dict[str, list[Cell]]]


def _read_batch(silver: Catalog, documents: Sequence[DocumentMeta]) -> _BatchRows:
    ids = [d.document_id for d in documents]
    files = filing_dates_filter({d.filing_date for d in documents})
    sections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_rows_in(
        silver.load_table(SECTIONS_TABLE),
        "document_id",
        ids,
        columns=(
            "section_id",
            "section_ordinal",
            "parent_section_id",
            "section_kind",
            "section_code",
            "title",
            "start_block_ordinal",
        ),
        files=files,
    ):
        sections[row["document_id"]].append(row)
    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_rows_in(
        silver.load_table(CONTENT_BLOCKS_TABLE),
        "document_id",
        ids,
        columns=("block_id", "block_ordinal", "section_id", "block_type", "text"),
        files=files,
    ):
        blocks[row["document_id"]].append(row)
    return _BatchRows(dict(sections), dict(blocks), _read_cells(silver, ids, files))


def _read_cells(
    silver: Catalog, ids: Sequence[str], files: BooleanExpression
) -> dict[str, dict[str, list[Cell]]]:
    """Cells of the documents as `Cell`s by document and table, in cell order,
    built batch by batch (no intermediate row dicts)."""

    wanted = pa.array(list(ids), type=pa.string())
    found: dict[str, dict[str, list[tuple[int, Cell]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    columns = (
        "document_id",
        "table_id",
        "cell_ordinal",
        "row_ordinal",
        "column_ordinal",
        "colspan",
        "text",
    )
    table = silver.load_table(DOCUMENT_TABLE_CELLS_TABLE)
    for batch in scan_batches(table, columns, files=files):
        matching = batch.filter(pc.is_in(batch["document_id"], value_set=wanted))
        values = [matching[column].to_pylist() for column in columns]
        for document_id, table_id, ordinal, row, column, span, text in zip(
            *values, strict=True
        ):
            found[document_id][table_id].append(
                (ordinal, Cell(row, column, span, text))
            )
    return {
        document_id: {
            table_id: [cell for _, cell in sorted(cells, key=lambda pair: pair[0])]
            for table_id, cells in tables.items()
        }
        for document_id, tables in found.items()
    }


@dataclass(frozen=True)
class ChunkingSettings:
    """How text is measured and cut: tokenizer and chunk parameters."""

    counter: TokenCounter
    params: ChunkParams


@dataclass
class _Run:
    silver: Catalog
    table: Table
    filings: Mapping[str, FilingMeta]
    settings: ChunkingSettings
    processed_at: datetime
    summary: ChunksSummary
    tokens_total: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    kinds: Counter[str] = field(default_factory=Counter)


def build_document_records(
    filing: FilingMeta,
    document: DocumentMeta,
    rows: _BatchRows,
    settings: ChunkingSettings,
    processed_at: datetime,
) -> tuple[list[ChunkRecord], dict[str, str]]:
    """Chunk rows of one document of a batch; also return its dropped blocks."""

    sections = _section_paths(
        rows.sections.get(document.document_id, []), document, filing.form
    )
    blocks = [
        SourceBlock(
            r["block_id"],
            r["block_ordinal"],
            r["section_id"],
            r["block_type"],
            r["text"],
        )
        for r in sorted(
            rows.blocks.get(document.document_id, []), key=lambda r: r["block_ordinal"]
        )
    ]
    context = FilingContext(
        filing.company_name,
        filing.form,
        filing.filing_date,
        filing.period_end_date,
        filing.fiscal_year,
        filing.fiscal_period,
    )

    def build_context(
        path: tuple[str, ...], subheading: str | None, table: str | None
    ) -> str:
        return chunk_context(context, path, subheading=subheading, table=table)

    result = chunk_document(
        DocumentInput(
            blocks, sections, rows.cells.get(document.document_id, {}), build_context
        ),
        settings.counter,
        settings.params,
    )
    count = len(result.chunks)
    url = sec_url(filing.cik, filing.accession_number, document.filename)
    records: list[ChunkRecord] = []
    for ordinal, draft in enumerate(result.chunks, start=1):
        section = sections[draft.section_id]
        if draft.token_count > settings.params.max_tokens:
            raise ValueError(
                f"{document.document_id} chunk {ordinal} has {draft.token_count} tokens"
            )
        records.append(
            ChunkRecord(
                chunk_id=chunk_id_for(document.document_id, ordinal),
                filing_id=filing.filing_id,
                document_id=document.document_id,
                section_id=draft.section_id,
                table_id=draft.table_id,
                block_ids=list(draft.block_ids),
                filing_date=filing.filing_date,
                cik=filing.cik,
                tickers=list(filing.tickers),
                form=filing.form,
                fiscal_year=filing.fiscal_year,
                fiscal_period=filing.fiscal_period,  # type: ignore[arg-type]
                period_end_date=filing.period_end_date,
                section_topic=section.topic,  # type: ignore[arg-type]
                document_type=document.document_type,
                chunk_kind=draft.chunk_kind,  # type: ignore[arg-type]
                company_name=filing.company_name,
                accession_number=filing.accession_number,
                sec_url=url,
                section_title=section.path[-1],
                subheading=draft.subheading,
                table_title=draft.table_title,
                chunk_ordinal=ordinal,
                document_chunk_count=count,
                part=draft.part,
                part_count=draft.part_count,
                context=draft.context,
                text=draft.text,
                token_count=draft.token_count,
                chunk_sha256=chunk_sha256(draft.context, draft.text),
                processed_at=processed_at,
            )
        )
    return records, result.dropped


def rebuild_document(
    silver: Catalog,
    filings: Mapping[str, FilingMeta],
    document: DocumentMeta,
    settings: ChunkingSettings,
) -> tuple[list[ChunkRecord], dict[str, str], set[str]]:
    """Chunk one stored document again from Silver (for the audit): its
    chunks, dropped blocks with reasons, and every block of the document."""

    rows = _read_batch(silver, [document])
    records, dropped = build_document_records(
        filings[document.filing_id], document, rows, settings, REBUILD_TIME
    )
    blocks = {row["block_id"] for row in rows.blocks.get(document.document_id, [])}
    return records, dropped, blocks


def _run_batch(run: _Run, batch: Sequence[DocumentMeta]) -> None:
    rows = _read_batch(run.silver, batch)
    records: list[ChunkRecord] = []
    for document in batch:
        try:
            chunks, dropped = build_document_records(
                run.filings[document.filing_id],
                document,
                rows,
                run.settings,
                run.processed_at,
            )
        except ValueError as error:
            # Its chunks are not written; the next run tries it again.
            run.summary.documents_rejected += 1
            run.summary.problems.append(f"{document.document_id}: {error}"[:300])
            logger.error(
                "native_rag_chunks_document_rejected",
                document_id=document.document_id,
                error=str(error)[:500],
            )
            continue
        run.dropped.update(dropped.values())
        if chunks:
            run.summary.documents_chunked += 1
        else:
            run.summary.documents_without_chunks += 1
        records.extend(chunks)
    records.sort(key=lambda r: r.chunk_id)
    if not records:
        return
    before = current_snapshot_id(run.table)
    expected = snapshot_total(run.table, "total-records") + len(records)
    append_rows(run.table, records, CHUNKS_ARROW_SCHEMA)
    verify_append(
        run.table,
        key_column="chunk_id",
        expected_rows=expected,
        inserted_keys=[r.chunk_id for r in records],
        snapshot_before=before,
    )
    run.summary.chunks_written += len(records)
    run.summary.batches_committed += 1
    run.tokens_total += sum(r.token_count for r in records)
    run.summary.tokens_max = max(
        run.summary.tokens_max, *(r.token_count for r in records)
    )
    run.kinds.update(r.chunk_kind for r in records)
    logger.info(
        "native_rag_chunks_batch_committed",
        documents=len(batch),
        chunks=len(records),
        first_filing_date=str(batch[0].filing_date),
    )


def _cell_counts(silver: Catalog) -> Counter[str]:
    """Visible cells per document, to size batches."""

    counts: Counter[str] = Counter()
    for batch in scan_batches(
        silver.load_table(DOCUMENT_TABLES_TABLE), ("document_id", "cell_count")
    ):
        for document_id, cells in zip(
            batch["document_id"].to_pylist(),
            batch["cell_count"].to_pylist(),
            strict=True,
        ):
            counts[document_id] += cells
    return counts


def _batches(
    documents: Sequence[DocumentMeta], cells: Mapping[str, int]
) -> list[list[DocumentMeta]]:
    """Consecutive batches of at most DOCUMENTS_PER_COMMIT documents and, past
    the first document, CELLS_PER_BATCH cells: memory follows cells, and a
    10-K season packs many cell-heavy documents together."""

    batches: list[list[DocumentMeta]] = []
    current: list[DocumentMeta] = []
    total = 0
    for document in documents:
        size = cells.get(document.document_id, 0)
        if current and (
            len(current) >= DOCUMENTS_PER_COMMIT or total + size > CELLS_PER_BATCH
        ):
            batches.append(current)
            current, total = [], 0
        current.append(document)
        total += size
    if current:
        batches.append(current)
    return batches


def _documents_to_chunk(
    table: Table, documents: Mapping[str, DocumentMeta], summary: ChunksSummary
) -> list[DocumentMeta] | None:
    stored = summarize_groups(
        table, "document_id", "chunk_ordinal", declared="document_chunk_count"
    )
    broken = sorted(doc for doc, group in stored.items() if not group.complete)
    unknown = sorted(set(stored) - set(documents))
    if broken or unknown:
        summary.problems.append(
            f"stored chunk groups incomplete {broken[:5]} or without a Silver "
            f"narrative document {unknown[:5]}: purge and rebuild the table"
        )
        return None
    summary.documents_skipped = len(stored)
    return sorted(
        (d for d in documents.values() if d.document_id not in stored),
        key=lambda d: (d.filing_date, d.document_id),
    )


@dataclass(frozen=True)
class _Settings:
    data_dir: Path
    gold_dir: Path
    counter: TokenCounter
    params: ChunkParams
    processed_at: datetime


def _chunk_all(settings: _Settings, summary: ChunksSummary) -> None:
    with (
        open_silver_catalog_read_only(settings.data_dir) as silver,
        open_gold_catalog(settings.gold_dir) as gold,
    ):
        expected = _expected_properties(silver, settings.counter, settings.params)
        table, problems = _open_chunks(gold, settings.gold_dir, expected)
        if problems:
            summary.problems.extend(problems)
            return
        documents = read_document_meta(silver)
        summary.documents_total = len(documents)
        pending = _documents_to_chunk(table, documents, summary)
        if not pending:
            return
        with timed_operation("native_rag_chunks.metadata"):
            filings = read_filing_meta(silver)
        run = _Run(
            silver,
            table,
            filings,
            ChunkingSettings(settings.counter, settings.params),
            settings.processed_at,
            summary,
        )
        with timed_operation("native_rag_chunks.build"):
            for batch in _batches(pending, _cell_counts(silver)):
                _run_batch(run, batch)
        summary.dropped_blocks = dict(sorted(run.dropped.items()))
        summary.chunk_kinds = dict(sorted(run.kinds.items()))
        if summary.chunks_written:
            summary.tokens_mean = round(run.tokens_total / summary.chunks_written, 1)


def run_native_rag_chunks(
    data_dir: Path,
    counter: TokenCounter,
    processed_at: datetime,
    *,
    gold_dir: Path | None = None,
    params: ChunkParams | None = None,
) -> ChunksSummary:
    """Build `native_rag.chunks` from the Silver narrative under `data_dir`.

    Gold lives under `gold_dir` (default `data_dir`): catalog
    `<gold_dir>/_catalog/gold.db`, files `<gold_dir>/gold/native_rag/chunks`.

    Incremental by document: documents whose chunks are all stored (ordinals
    1..N with N declared) are skipped; the rest are chunked in batches of
    DOCUMENTS_PER_COMMIT sorted by filing date, each one commit verified after
    the write. A run refuses to write (status `rejected`) when stored chunks
    were built with another tokenizer, other parameters or other Silver rules
    versions, or when a stored document is incomplete: purge and rebuild.
    """

    started = perf_counter()
    summary = ChunksSummary()
    params = params or ChunkParams()
    with log_context(
        pipeline="native_rag_chunks",
        data_dir=str(data_dir),
        gold_dir=str(gold_dir or data_dir),
    ):
        logger.info("native_rag_chunks_started", params=params.as_property())
        try:
            with timed_operation("native_rag_chunks.total"):
                settings = _Settings(
                    data_dir, gold_dir or data_dir, counter, params, processed_at
                )
                _chunk_all(settings, summary)
        except Exception as error:
            logger.error(
                "native_rag_chunks_failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        summary.status = (
            "partial"
            if summary.documents_rejected
            else "rejected"
            if summary.problems
            else "completed"
        )
        summary.duration_seconds = round(perf_counter() - started, 3)
        log = logger.info if summary.status == "completed" else logger.error
        log("native_rag_chunks_completed", **asdict(summary))
    return summary

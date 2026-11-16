from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pyiceberg.catalog import Catalog
from pyiceberg.expressions import And, BooleanExpression, GreaterThanOrEqual, LessThan
from pyiceberg.table import Table

from src.data_pipeline.contracts.native_rag.chunks import ChunkRecord, chunk_sha256
from src.data_pipeline.pipelines.native_rag.chunks import (
    PARAMS_PROPERTY,
    SILVER_RULES_PROPERTY,
    TOKENIZER_PROPERTY,
    ChunkingSettings,
    rebuild_document,
    silver_rules,
)
from src.data_pipeline.pipelines.native_rag.filing_metadata import (
    DocumentMeta,
    FilingMeta,
    read_document_meta,
    read_filing_meta,
)
from src.data_pipeline.storage.iceberg import (
    open_gold_catalog_read_only,
    open_silver_catalog_read_only,
    scan_batches,
)
from src.data_pipeline.storage.native_rag.chunks import CHUNKS_SPEC, CHUNKS_TABLE
from src.data_pipeline.storage.sec_filings.content_blocks import CONTENT_BLOCKS_TABLE
from src.data_pipeline.storage.sec_filings.sections import SECTIONS_TABLE
from src.data_pipeline.transforms.native_rag.document_chunks import ChunkParams
from src.data_pipeline.transforms.native_rag.tokenizer import TokenCounter
from src.utils.logger import get_logger

logger = get_logger(__name__)

Status = Literal["PASS", "WARN", "FAIL"]
# Documents whose share of characters outside every chunk exceeds this get a WARN.
UNCOVERED_WARN_SHARE = 0.30
SMALL_CHUNK_TOKENS = 16
SAMPLE_ROWS = 5


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str
    samples: tuple[str, ...] = ()


@dataclass
class AuditReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def status(self) -> Status:
        statuses = {check.status for check in self.checks}
        return (
            "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"
        )

    def add(
        self, name: str, problems: Sequence[str], detail: str, *, warn: bool = False
    ) -> None:
        status: Status = "PASS" if not problems else "WARN" if warn else "FAIL"
        self.checks.append(Check(name, status, detail, tuple(problems[:SAMPLE_ROWS])))


def _rows(
    table: Table, columns: Sequence[str] | None, files: BooleanExpression
) -> Iterator[dict[str, Any]]:
    for batch in scan_batches(table, columns, files=files):
        yield from batch.to_pylist()


def _year_filter(year: int) -> BooleanExpression:
    """Data files of one `year(filing_date)` partition."""

    return And(
        GreaterThanOrEqual("filing_date", f"{year}-01-01"),  # type: ignore[call-arg, arg-type]
        LessThan("filing_date", f"{year + 1}-01-01"),  # type: ignore[call-arg, arg-type]
    )


@dataclass
class _Findings:
    """Problems found so far, accumulated one filing year at a time."""

    rows: int = 0
    invalid: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    incomplete: list[str] = field(default_factory=list)
    over_cap: list[str] = field(default_factory=list)
    bad_hash: list[str] = field(default_factory=list)
    small: list[str] = field(default_factory=list)
    unknown_documents: list[str] = field(default_factory=list)
    missing_sections: list[str] = field(default_factory=list)
    wrong_blocks: list[str] = field(default_factory=list)
    without_chunks: list[str] = field(default_factory=list)
    repeated_blocks: list[str] = field(default_factory=list)
    uncovered: list[str] = field(default_factory=list)
    documents: int = 0
    stored: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )


@dataclass
class _YearChunks:
    ids: Counter[str] = field(default_factory=Counter)
    ordinals: dict[str, list[tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    covered: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # Occurrences of each block in chunks that are not part of a split chain.
    unsplit: Counter[str] = field(default_factory=Counter)
    block_sections: list[tuple[str, str, str]] = field(default_factory=list)


def _check_row(row: dict[str, Any], max_tokens: int, found: _Findings) -> None:
    chunk_id = row["chunk_id"]
    found.rows += 1
    if row["token_count"] > max_tokens:
        found.over_cap.append(chunk_id)
    if row["token_count"] < SMALL_CHUNK_TOKENS:
        found.small.append(chunk_id)
    if row["chunk_sha256"] != chunk_sha256(row["context"], row["text"]):
        found.bad_hash.append(chunk_id)
    try:
        ChunkRecord.model_validate(row)
    except ValueError as error:
        found.invalid.append(f"{chunk_id}: {str(error).splitlines()[0]}")


def _scan_year_chunks(
    table: Table, year: int, max_tokens: int, sample: set[str], found: _Findings
) -> _YearChunks:
    chunks = _YearChunks()
    for row in _rows(table, None, _year_filter(year)):
        document_id = row["document_id"]
        chunks.ids[row["chunk_id"]] += 1
        chunks.ordinals[document_id].append(
            (row["chunk_ordinal"], row["document_chunk_count"])
        )
        for block_id in row["block_ids"]:
            chunks.covered[document_id].add(block_id)
            if row["part"] is None:
                chunks.unsplit[block_id] += 1
            chunks.block_sections.append((row["chunk_id"], block_id, row["section_id"]))
        _check_row(row, max_tokens, found)
        if document_id in sample:
            found.stored[document_id].append(row)
    return chunks


def _audit_year(
    silver: Catalog,
    chunks: _YearChunks,
    year: int,
    documents: Mapping[str, DocumentMeta],
    found: _Findings,
) -> None:
    """Keys, ordinals, FKs and coverage of one filing year's chunks."""

    files = _year_filter(year)
    found.documents += len(chunks.ordinals)
    found.duplicates += [key for key, n in chunks.ids.items() if n > 1]
    found.incomplete += [
        document_id
        for document_id, pairs in chunks.ordinals.items()
        if sorted(o for o, _ in pairs) != list(range(1, len(pairs) + 1))
        or {n for _, n in pairs} != {len(pairs)}
    ]
    found.unknown_documents += sorted(set(chunks.ordinals) - set(documents))
    sections = {
        row["section_id"]
        for row in _rows(silver.load_table(SECTIONS_TABLE), ("section_id",), files)
    }
    block_section: dict[str, str] = {}
    tables: set[str] = set()
    length: dict[str, int] = {}
    characters: Counter[str] = Counter()
    for row in _rows(
        silver.load_table(CONTENT_BLOCKS_TABLE),
        ("block_id", "document_id", "section_id", "text_length", "block_type"),
        files,
    ):
        block_section[row["block_id"]] = row["section_id"]
        if row["block_type"] == "table":
            tables.add(row["block_id"])
        length[row["block_id"]] = row["text_length"]
        characters[row["document_id"]] += row["text_length"]
    found.missing_sections += sorted(
        {section for _, _, section in chunks.block_sections if section not in sections}
    )
    found.wrong_blocks += [
        f"{chunk}: {block}"
        for chunk, block, section in chunks.block_sections
        if block_section.get(block) != section
    ]
    # A layout table's rows may continue in the next chunk; any other block
    # outside a split chain is in exactly one chunk.
    found.repeated_blocks += sorted(
        block for block, n in chunks.unsplit.items() if n > 1 and block not in tables
    )
    found.without_chunks += sorted(
        d
        for d, n in characters.items()
        if n and d in documents and d not in chunks.ordinals
    )
    for document_id, total in sorted(characters.items()):
        if document_id not in chunks.covered or not total:
            continue
        kept = sum(length.get(b, 0) for b in chunks.covered[document_id])
        if 1 - kept / total > UNCOVERED_WARN_SHARE:
            found.uncovered.append(
                f"{document_id}: {kept}/{total} characters in chunks"
            )


def _sample_documents(
    documents: Mapping[str, DocumentMeta], filings: Mapping[str, FilingMeta], extra: int
) -> set[str]:
    """The newest primary document of each form plus `extra` chosen by hash."""

    newest: dict[str, tuple[date, str]] = {}
    for document_id, meta in documents.items():
        filing = filings[meta.filing_id]
        if meta.document_type != filing.form:
            continue
        key = (filing.filing_date, document_id)
        if filing.form not in newest or key > newest[filing.form]:
            newest[filing.form] = key
    chosen = {document_id for _, document_id in newest.values()}
    ranked = sorted(documents, key=lambda d: hashlib.sha256(d.encode()).hexdigest())
    chosen.update(ranked[:extra])
    return chosen


def audit_chunks(
    data_dir: Path,
    counter: TokenCounter,
    *,
    gold_dir: Path | None = None,
    params: ChunkParams | None = None,
    sample_documents: int = 5,
) -> AuditReport:
    """Read-only audit of `native_rag.chunks` against Silver.

    Checks: table definition and build properties; every row valid against
    the contract (parts, kinds, keys, hash); unique `chunk_id`; ordinals 1..N
    per document with N declared; token cap; `chunk_sha256` recomputed; FKs
    (document, section, every block in Silver and in the chunk's section);
    narrative documents with text but no chunks (WARN); documents with over
    30% of their characters outside chunks (WARN: noise); chunks under 16
    tokens (WARN); an exact rebuild of sampled documents (newest
    10-K/10-Q/8-K plus `sample_documents` by hash), which also pins coverage
    and token counts. Gold and Silver are read one filing year (partition) at
    a time, so memory follows the largest year.
    """

    params = params or ChunkParams()
    report = AuditReport()
    gold_root = gold_dir or data_dir
    with (
        open_silver_catalog_read_only(data_dir) as silver,
        open_gold_catalog_read_only(gold_root) as gold,
    ):
        table = gold.load_table(CHUNKS_TABLE)
        problem = CHUNKS_SPEC.definition_problem(table)
        report.add("table_definition", [problem] if problem else [], CHUNKS_TABLE)
        expected = {
            TOKENIZER_PROPERTY: counter.identity,
            PARAMS_PROPERTY: params.as_property(),
            SILVER_RULES_PROPERTY: silver_rules(silver),
        }
        mismatched = [
            f"{k}={table.properties.get(k)!r}"
            for k, v in expected.items()
            if table.properties.get(k) != v
        ]
        report.add(
            "build_properties",
            mismatched,
            "tokenizer, chunk parameters and Silver rules versions",
        )

        filings = read_filing_meta(silver)
        documents = read_document_meta(silver)
        sample = _sample_documents(documents, filings, sample_documents)
        found = _Findings()
        for year in sorted({f.filing_date.year for f in filings.values()}):
            chunks = _scan_year_chunks(table, year, params.max_tokens, sample, found)
            _audit_year(silver, chunks, year, documents, found)
        _report_findings(report, found, params)

        settings = ChunkingSettings(counter, params)
        differences: list[str] = []
        coverage: list[str] = []
        for document_id in sorted(sample):
            rebuilt, dropped, blocks = rebuild_document(
                silver, filings, documents[document_id], settings
            )
            if _comparable(rebuilt) != _comparable(found.stored.get(document_id, [])):
                differences.append(document_id)
            coverage += _coverage_problems(document_id, rebuilt, dropped, blocks)
        report.add(
            "sample_rebuild", differences, f"{len(sample)} documents rebuilt exactly"
        )
        report.add(
            "sample_coverage",
            coverage,
            "every block of the sampled documents is in a chunk or dropped, not both",
        )
    logger.info(
        "native_rag_audit_completed", status=report.status, checks=len(report.checks)
    )
    return report


def _report_findings(
    report: AuditReport, found: _Findings, params: ChunkParams
) -> None:
    report.add("contract", found.invalid, f"{found.rows} rows validated")
    report.add("unique_chunk_id", found.duplicates, f"{found.rows} keys")
    report.add("document_ordinals", found.incomplete, f"{found.documents} documents")
    report.add("token_cap", found.over_cap, f"max {params.max_tokens} tokens")
    report.add("chunk_sha256", found.bad_hash, "recomputed from context + text")
    report.add(
        "small_chunks",
        found.small,
        f"chunks under {SMALL_CHUNK_TOKENS} tokens",
        warn=True,
    )
    report.add(
        "fk_document",
        found.unknown_documents,
        "chunk documents are Silver narrative documents",
    )
    report.add("fk_section", found.missing_sections, "chunk sections exist in Silver")
    report.add(
        "blocks_in_section",
        found.wrong_blocks,
        "every block exists and lies in the chunk's section",
    )
    report.add(
        "block_in_one_chunk",
        found.repeated_blocks,
        "a non-table block outside a split chain is in exactly one chunk",
    )
    report.add(
        "documents_without_chunks",
        found.without_chunks,
        "narrative documents with text",
        warn=True,
    )
    report.add(
        "uncovered_text",
        found.uncovered,
        f"documents with over {UNCOVERED_WARN_SHARE:.0%} of characters outside chunks",
        warn=True,
    )


def _coverage_problems(
    document_id: str,
    chunks: Sequence[ChunkRecord],
    dropped: Mapping[str, str],
    blocks: set[str],
) -> list[str]:
    chunked = {block for chunk in chunks for block in chunk.block_ids}
    missing = blocks - chunked - set(dropped)
    both = chunked & set(dropped)
    unknown = (chunked | set(dropped)) - blocks
    return [
        f"{document_id}: {kind} {sorted(found)[:3]}"
        for kind, found in (("missing", missing), ("both", both), ("unknown", unknown))
        if found
    ]


def _comparable(rows: Sequence[dict[str, Any] | ChunkRecord]) -> list[str]:
    result = []
    for row in rows:
        values = row if isinstance(row, dict) else row.model_dump()
        result.append(
            repr([(k, values[k]) for k in sorted(values) if k != "processed_at"])
        )
    return sorted(result)

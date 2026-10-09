from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from itertools import batched
from pathlib import Path
from time import perf_counter
from typing import Literal

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.document_links import (
    DOCUMENT_LINKS_ARROW_SCHEMA,
    DocumentLinkRecord,
)
from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.pipelines.sec_filings.document_link_extraction import (
    extract_document_links,
    source_context,
)
from src.data_pipeline.quality.sec_filings.document_links import links_problem
from src.data_pipeline.quality.sec_filings.narrative import stored_group_problem
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.storage.iceberg import (
    GroupSummary,
    append_rows,
    current_snapshot_id,
    filing_dates_filter,
    iter_rows,
    open_silver_catalog,
    read_groups_in,
    snapshot_total,
    summarize_groups,
    verify_append,
)
from src.data_pipeline.storage.sec_filings.document_links import (
    DOCUMENT_LINKS_TABLE,
    open_document_links_table,
)
from src.data_pipeline.storage.sec_filings.documents import open_documents_table
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)

# Filings whose HTML documents are parsed, validated and committed together.
FILINGS_PER_COMMIT = 50
LinkGroups = Mapping[str, Sequence[DocumentLinkRecord]]


@dataclass
class DocumentLinksStageSummary:
    """Outcome of one `sec_filings.document_links` run.

    Each HTML source document is counted once: inserted, skipped (complete
    stored group, not re-read), without links (parsed, nothing stored, so
    parsed again next run), conflicted, or failed (unreadable source or
    invalid links). A rejected run inserts nothing. `links_*` count the
    inserted links by resolution status; `links_external` those of kind
    external_url.
    """

    status: Literal["completed", "rejected"] = "rejected"
    documents_considered: int = 0
    documents_inserted: int = 0
    documents_skipped: int = 0
    documents_without_links: int = 0
    documents_conflicted: int = 0
    documents_failed: int = 0
    links_inserted: int = 0
    links_resolved: int = 0
    links_unresolved: int = 0
    links_ambiguous: int = 0
    links_not_applicable: int = 0
    links_external: int = 0
    duration_seconds: float = 0.0
    snapshot_before: int | None = None
    snapshot_after: int | None = None


@dataclass(frozen=True)
class _Source:
    """What classifying an HTML source document needs from its row."""

    document_id: str
    filing_id: str
    filing_date: date
    source_path: str
    content_sha256: str


def _plan(
    sources: Sequence[_Source],
    stored: Mapping[str, GroupSummary],
    summary: DocumentLinksStageSummary,
) -> list[_Source]:
    """Skip complete stored groups and flag the others; return sources to parse.

    Stored groups are judged by their summaries (complete ordinals, the
    source's lineage); the integrity audit checks their rows and targets.
    """

    for document_id in sorted(set(stored) - {source.document_id for source in sources}):
        summary.documents_conflicted += 1
        logger.error(
            "document_links_conflict",
            source_document_id=document_id,
            reason="links have no HTML source document in sec_filings.documents",
        )
    candidates = []
    skipped_links = 0
    for source in sources:
        group = stored.get(source.document_id)
        if group is None:
            candidates.append(source)
            continue
        lineage = {
            "filing_id": source.filing_id,
            "filing_date": source.filing_date,
            "source_document_sha256": source.content_sha256,
        }
        problem = stored_group_problem(group, lineage, "links")
        if problem is None:
            summary.documents_skipped += 1
            skipped_links += group.rows
        else:
            summary.documents_conflicted += 1
            logger.error(
                "document_links_conflict",
                reason=problem,
                filing_id=source.filing_id,
                source_document_id=source.document_id,
                source_path=source.source_path,
            )
    logger.info(
        "document_links_skipped",
        documents=summary.documents_skipped,
        links=skipped_links,
    )
    return candidates


def _validate_built(
    candidates: Sequence[DocumentRecord],
    built: LinkGroups,
    documents: Mapping[str, DocumentRecord],
    summary: DocumentLinksStageSummary,
) -> None:
    """Check every built group against its source and the document inventory."""

    for source in candidates:
        records = built.get(source.document_id)
        problem = None if records is None else links_problem(source, records, documents)
        if problem is not None:
            summary.documents_failed += 1
            logger.error(
                "document_links_invalid", error=problem, **source_context(source)
            )


def _verify_groups(
    table: Table, sources: Sequence[DocumentRecord], built: LinkGroups
) -> None:
    """Re-read the batch's groups: each built group is stored exactly."""

    stored = read_groups_in(
        table,
        "source_document_id",
        built,
        files=filing_dates_filter({source.filing_date for source in sources}),
    )
    for document_id, records in built.items():
        rows = sorted(stored.get(document_id, []), key=lambda row: row["link_ordinal"])
        for row in rows:
            del row["processed_at"]
        if rows != [record.model_dump(exclude={"processed_at"}) for record in records]:
            raise ValueError(f"Links of {document_id} differ after commit")


def _write(
    table: Table,
    sources: Sequence[DocumentRecord],
    built: LinkGroups,
    summary: DocumentLinksStageSummary,
) -> None:
    """Append the batch's new groups in one commit, then verify them."""

    new = [record for records in built.values() for record in records]
    stored_rows = snapshot_total(table, "total-records")
    if new:
        with timed_operation("document_links.write", records=len(new)):
            append_rows(table, new, DOCUMENT_LINKS_ARROW_SCHEMA)
    statuses = Counter(record.resolution_status for record in new)
    summary.documents_inserted += len(built)
    summary.links_inserted += len(new)
    summary.links_resolved += statuses["resolved"]
    summary.links_unresolved += statuses["unresolved"]
    summary.links_ambiguous += statuses["ambiguous"]
    summary.links_not_applicable += statuses["not_applicable"]
    summary.links_external += sum(r.link_kind == "external_url" for r in new)
    logger.info("document_links_inserted", documents=len(built), links=len(new))
    with timed_operation("document_links.verify"):
        summary.snapshot_after = verify_append(
            table,
            key_column="link_id",
            expected_rows=stored_rows + len(new),
            inserted_keys=[record.link_id for record in new],
            snapshot_before=summary.snapshot_after,
        )
        _verify_groups(table, sources, built)


def _run_batch(
    tables: tuple[Table, Table],
    batch: Sequence[_Source],
    run: tuple[SecBronze, datetime],
    summary: DocumentLinksStageSummary,
) -> bool:
    """Parse, resolve, validate and commit the sources of a few filings.

    Links resolve within their filing, so the batch reads every document of
    its filings. Returns whether the batch was written.
    """

    documents_table, links_table = tables
    documents = {
        row["document_id"]: DocumentRecord.model_validate(row)
        for rows in read_groups_in(
            documents_table,
            "filing_id",
            {source.filing_id for source in batch},
            files=filing_dates_filter({s.filing_date for s in batch}),
        ).values()
        for row in rows
    }
    candidates = [documents[source.document_id] for source in batch]
    extraction = extract_document_links(candidates, documents, *run)
    summary.documents_failed += extraction.failures
    summary.documents_without_links += extraction.without_links
    with timed_operation("document_links.validate", phase="built_records"):
        _validate_built(candidates, extraction.built, documents, summary)
    if summary.documents_failed:
        return False
    _write(links_table, candidates, extraction.built, summary)
    return True


def _read_sources(documents_table: Table) -> list[_Source]:
    """HTML documents with content, in (filing_date, document_id) order."""

    columns = (
        "document_id",
        "filing_id",
        "filing_date",
        "source_path",
        "content_sha256",
    )
    # DocumentRecord sets content offsets and checksum together.
    return sorted(
        (
            _Source(*(row[column] for column in columns))
            for row in iter_rows(documents_table, (*columns, "content_kind"))
            if row["content_kind"] == "html" and row["content_sha256"] is not None
        ),
        key=lambda source: (source.filing_date, source.document_id),
    )


def _run_links(
    catalog: Catalog,
    data_dir: Path,
    run: tuple[SecBronze, datetime],
    summary: DocumentLinksStageSummary,
) -> None:
    table, created = open_document_links_table(catalog, data_dir)
    logger.info(
        "document_links_table_created" if created else "document_links_table_loaded",
        location=table.location(),
    )
    summary.snapshot_before = summary.snapshot_after = current_snapshot_id(table)
    with timed_operation("document_links.read"):
        documents_table, _ = open_documents_table(catalog, data_dir)
        sources = _read_sources(documents_table)
        stored = summarize_groups(
            table,
            "source_document_id",
            "link_ordinal",
            declared="document_link_count",
            uniform=("filing_id", "filing_date", "source_document_sha256"),
        )
    summary.documents_considered = len(sources)
    logger.info("html_link_documents_loaded", documents=len(sources))

    with timed_operation("document_links.validate", phase="existing_state"):
        candidates = _plan(sources, stored, summary)
    if summary.documents_conflicted:
        return
    by_filing: dict[str, list[_Source]] = defaultdict(list)
    for source in candidates:
        by_filing[source.filing_id].append(source)
    for filings in batched(by_filing.values(), FILINGS_PER_COMMIT, strict=False):
        batch = [source for group in filings for source in group]
        if not _run_batch((documents_table, table), batch, run, summary):
            return
    summary.status = "completed"


def run_document_links_stage(
    data_dir: Path, processed_at: datetime, bronze: SecBronze | None = None
) -> DocumentLinksStageSummary:
    """Build `sec_filings.document_links` from every HTML document.

    Complete stored groups, judged by their summaries, are skipped without
    reading Bronze; any conflict rejects the run before the first write.
    Every other HTML document is read by offsets, verified, parsed once and
    its `<a href>` elements resolved against the documents of its filing, in
    batches of FILINGS_PER_COMMIT filings, one commit each. An invalid source
    rejects its batch before that batch's write; earlier batches stay
    committed. Unexpected errors are logged and re-raised.
    """

    started = perf_counter()
    summary = DocumentLinksStageSummary()
    with log_context(source="sec_filings", table=DOCUMENT_LINKS_TABLE):
        logger.info("document_links_stage_started")
        try:
            with (
                timed_operation("document_links.total"),
                open_silver_catalog(data_dir) as catalog,
            ):
                _run_links(
                    catalog,
                    data_dir,
                    (bronze or SecBronze.under(data_dir), processed_at),
                    summary,
                )
        except Exception as error:
            logger.error(
                "document_links_stage_failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise

        summary.duration_seconds = round(perf_counter() - started, 3)
        if summary.status == "completed":
            logger.info("document_links_stage_completed", **asdict(summary))
        else:
            logger.error("document_links_stage_failed", **asdict(summary))
    return summary

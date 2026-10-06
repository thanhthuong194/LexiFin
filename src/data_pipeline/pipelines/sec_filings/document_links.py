from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import ValidationError
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
from src.data_pipeline.storage.iceberg import (
    append_rows,
    current_snapshot_id,
    group_rows_by,
    open_silver_catalog,
    verify_append,
)
from src.data_pipeline.storage.sec_filings.document_links import (
    DOCUMENT_LINKS_TABLE,
    open_document_links_table,
)
from src.data_pipeline.storage.sec_filings.documents import (
    open_documents_table,
    read_documents,
)
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)

StoredRows = Mapping[str, Sequence[dict[str, Any]]]
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


def _stored_problem(
    source: DocumentRecord,
    rows: Sequence[dict[str, Any]],
    documents: Mapping[str, DocumentRecord],
) -> str | None:
    try:
        links = [DocumentLinkRecord.model_validate(row) for row in rows]
    except ValidationError as error:
        return f"stored row is invalid: {error.errors()[0]['msg']}"
    return links_problem(source, links, documents)


def _plan(
    sources: Sequence[DocumentRecord],
    stored: StoredRows,
    documents: Mapping[str, DocumentRecord],
    summary: DocumentLinksStageSummary,
) -> list[DocumentRecord]:
    """Skip complete stored groups and flag the others; return sources to parse."""

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
        rows = stored.get(source.document_id)
        if rows is None:
            candidates.append(source)
            continue
        problem = _stored_problem(source, rows, documents)
        if problem is None:
            summary.documents_skipped += 1
            skipped_links += len(rows)
        else:
            summary.documents_conflicted += 1
            logger.error(
                "document_links_conflict", reason=problem, **source_context(source)
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
    """Re-read the table: no orphan group, and each built group stored exactly."""

    stored = group_rows_by(table, "source_document_id")
    orphans = set(stored) - {source.document_id for source in sources}
    if orphans:
        raise ValueError(f"Links without an HTML source: {sorted(orphans)[:5]}")
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
    stored_rows: int,
    summary: DocumentLinksStageSummary,
) -> None:
    """Append every new group in one commit, then verify the table."""

    new = sorted(
        (record for records in built.values() for record in records),
        key=lambda r: (r.filing_date, r.source_document_id, r.link_ordinal),
    )
    if new:
        with timed_operation("document_links.write", records=len(new)):
            append_rows(table, new, DOCUMENT_LINKS_ARROW_SCHEMA)
    statuses = Counter(record.resolution_status for record in new)
    summary.documents_inserted, summary.links_inserted = len(built), len(new)
    summary.links_resolved = statuses["resolved"]
    summary.links_unresolved = statuses["unresolved"]
    summary.links_ambiguous = statuses["ambiguous"]
    summary.links_not_applicable = statuses["not_applicable"]
    summary.links_external = sum(r.link_kind == "external_url" for r in new)
    logger.info("document_links_inserted", documents=len(built), links=len(new))
    with timed_operation("document_links.verify"):
        summary.snapshot_after = verify_append(
            table,
            key_column="link_id",
            expected_rows=stored_rows + len(new),
            inserted_keys=[record.link_id for record in new],
            snapshot_before=summary.snapshot_before,
        )
        _verify_groups(table, sources, built)


def _run_links(
    catalog: Catalog,
    data_dir: Path,
    processed_at: datetime,
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
        documents = {d.document_id: d for d in read_documents(documents_table)}
        stored = group_rows_by(table, "source_document_id")
    # DocumentRecord sets content offsets and checksum together.
    sources = sorted(
        (
            d
            for d in documents.values()
            if d.content_kind == "html" and d.content_sha256 is not None
        ),
        key=lambda d: (d.filing_date, d.document_id),
    )
    summary.documents_considered = len(sources)
    logger.info("html_link_documents_loaded", documents=len(sources))

    with timed_operation("document_links.validate", phase="existing_state"):
        candidates = _plan(sources, stored, documents, summary)
    if summary.documents_conflicted:
        return
    extraction = extract_document_links(candidates, documents, data_dir, processed_at)
    summary.documents_failed = extraction.failures
    summary.documents_without_links = extraction.without_links
    with timed_operation("document_links.validate", phase="built_records"):
        _validate_built(candidates, extraction.built, documents, summary)
    if summary.documents_failed:
        return
    _write(table, sources, extraction.built, sum(map(len, stored.values())), summary)
    summary.status = "completed"


def run_document_links_stage(
    data_dir: Path, processed_at: datetime
) -> DocumentLinksStageSummary:
    """Build `sec_filings.document_links` from every HTML document.

    Complete stored groups are skipped without reading Bronze; every other
    HTML document is read by offsets, verified, parsed once and its `<a href>`
    elements resolved against the documents of its filing. Any conflict or
    invalid source rejects the batch before the single append; unexpected
    errors are logged and re-raised.
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
                _run_links(catalog, data_dir, processed_at, summary)
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

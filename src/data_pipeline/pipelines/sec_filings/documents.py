from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import batched
from pathlib import Path
from time import perf_counter
from typing import Literal

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.documents import (
    DOCUMENTS_ARROW_SCHEMA,
    DocumentRecord,
)
from src.data_pipeline.contracts.sec_filings.filings import FilingRecord
from src.data_pipeline.quality.sec_filings.documents import documents_problem
from src.data_pipeline.storage.bronze import (
    SecBronze,
    sha256_bytes,
)
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
from src.data_pipeline.storage.sec_filings.documents import (
    DOCUMENTS_TABLE,
    open_documents_table,
)
from src.data_pipeline.storage.sec_filings.filings import (
    open_filings_table,
    read_filings,
)
from src.data_pipeline.transforms.sec_filings.documents import (
    build_document_records,
    check_document_offsets,
)
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)

# Filings whose submissions are parsed, validated and committed together.
FILINGS_PER_COMMIT = 500


@dataclass
class DocumentsStageSummary:
    """Outcome of one `sec_filings.documents` stage run.

    `filings_*` count parent filings and `documents_*` count document rows,
    except `documents_failed`: the parent filings whose documents could not be
    built (unreadable or changed submission, invalid block). A filing is
    reported at its first invalid block. `rejected` means nothing was written.
    `document_count_mismatches` counts parsed filings whose block count differs
    from the header's PUBLIC DOCUMENT COUNT; EDGAR overstates that count for
    some filings, so a mismatch is logged but does not reject the batch.
    """

    status: Literal["completed", "rejected"] = "rejected"
    filings_considered: int = 0
    filings_inserted: int = 0
    filings_skipped: int = 0
    filings_conflicted: int = 0
    documents_inserted: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    document_count_mismatches: int = 0
    duration_seconds: float = 0.0
    snapshot_before: int | None = None
    snapshot_after: int | None = None


def _stored_problem(
    parent: FilingRecord, group: GroupSummary, primaries: int
) -> str | None:
    """Why a stored group, known by its summary, is not one complete inventory."""

    if not group.complete:
        return "document ordinals are not exactly 1..N"
    if group.value("source_submission_sha256") != parent.full_submission_sha256:
        return "source submission SHA-256 differs from sec_filings.filings"
    if primaries != 1:
        return f"expected one primary_document, found {primaries}"
    return None


def _pending_parents(
    parents: Sequence[FilingRecord],
    stored: Mapping[str, GroupSummary],
    primaries: Mapping[str, int],
    summary: DocumentsStageSummary,
) -> list[FilingRecord]:
    """Skip filings whose stored group is complete, flag broken groups as
    conflicts, and return the filings that have no documents yet.

    Stored groups are judged by their summaries and primary counts; the
    integrity audit checks their rows.
    """

    for filing_id in sorted(set(stored) - {parent.filing_id for parent in parents}):
        summary.filings_conflicted += 1
        logger.error(
            "filing_documents_conflict",
            filing_id=filing_id,
            reason="documents have no parent in sec_filings.filings",
        )
    pending: list[FilingRecord] = []
    for parent in parents:
        group = stored.get(parent.filing_id)
        if group is None:
            pending.append(parent)
            continue
        problem = _stored_problem(parent, group, primaries.get(parent.filing_id, 0))
        if problem is None:
            summary.filings_skipped += 1
            summary.documents_skipped += group.rows
        else:
            summary.filings_conflicted += 1
            logger.error(
                "filing_documents_conflict",
                filing_id=parent.filing_id,
                source_path=parent.full_submission_path,
                reason=problem,
            )
    logger.info(
        "filing_documents_skipped",
        filings=summary.filings_skipped,
        documents=summary.documents_skipped,
    )
    return pending


def _read_submission(parent: FilingRecord, bronze: SecBronze) -> bytes:
    """Read a parent's full submission and check it against the parent row."""

    raw = bronze.path(parent.full_submission_path).read_bytes()
    if not raw:
        raise ValueError("Full submission is empty")
    if sha256_bytes(raw) != parent.full_submission_sha256:
        raise ValueError("Full submission SHA-256 differs from sec_filings.filings")
    return raw


def _parse_pending(
    parents: Sequence[FilingRecord],
    bronze: SecBronze,
    processed_at: datetime,
    summary: DocumentsStageSummary,
) -> list[DocumentRecord]:
    """Build and offset-check each filing's documents, one submission at a time,
    so only one raw submission is held in memory."""

    records: list[DocumentRecord] = []
    for parent in parents:
        with log_context(filing_id=parent.filing_id):
            try:
                raw = _read_submission(parent, bronze)
                documents = build_document_records(parent, raw, processed_at)
                check_document_offsets(documents, raw)
            except (OSError, ValueError) as error:
                summary.documents_failed += 1
                logger.error(
                    "filing_documents_invalid",
                    source_path=parent.full_submission_path,
                    error=str(error),
                )
                continue
            logger.info(
                "submission_documents_parsed",
                documents=len(documents),
                public_document_count=parent.public_document_count,
            )
            if parent.public_document_count not in (None, len(documents)):
                summary.document_count_mismatches += 1
                logger.warning(
                    "submission_document_count_mismatch",
                    public_document_count=parent.public_document_count,
                    parsed_documents=len(documents),
                )
            records.extend(documents)
    return records


def _validate_new_groups(
    records: Sequence[DocumentRecord],
    parents: Mapping[str, FilingRecord],
    summary: DocumentsStageSummary,
) -> None:
    """Check that every new filing group is one complete inventory."""

    groups: dict[str, list[DocumentRecord]] = defaultdict(list)
    for record in records:
        groups[record.filing_id].append(record)
    for filing_id, group in groups.items():
        problem = documents_problem(parents[filing_id], group)
        if problem is not None:
            summary.documents_failed += 1
            logger.error(
                "filing_documents_invalid",
                filing_id=filing_id,
                source_path=parents[filing_id].full_submission_path,
                error=problem,
            )


def _verify_written_groups(table: Table, parents: Sequence[FilingRecord]) -> None:
    """Re-read the batch's groups: each must be a complete inventory."""

    stored = read_groups_in(
        table,
        "filing_id",
        {parent.filing_id for parent in parents},
        files=filing_dates_filter({parent.filing_date for parent in parents}),
    )
    for parent in parents:
        group = [
            DocumentRecord.model_validate(row)
            for row in stored.get(parent.filing_id, [])
        ]
        problem = documents_problem(parent, group) if group else "no documents stored"
        if problem is not None:
            raise ValueError(f"Documents of {parent.filing_id} after commit: {problem}")


def _write_batch(
    table: Table,
    batch: Sequence[FilingRecord],
    run: tuple[SecBronze, datetime],
    summary: DocumentsStageSummary,
) -> bool:
    """Parse, validate and commit one batch of filings; return whether it was written."""

    with timed_operation("documents.parse", filings=len(batch)):
        records = _parse_pending(batch, *run, summary)
    with timed_operation("documents.validate", records=len(records)):
        _validate_new_groups(
            records, {parent.filing_id: parent for parent in batch}, summary
        )
    if summary.documents_failed:
        return False
    rows = snapshot_total(table, "total-records")
    with timed_operation("documents.write", records=len(records)):
        append_rows(table, records, DOCUMENTS_ARROW_SCHEMA)
    summary.filings_inserted += len(batch)
    summary.documents_inserted += len(records)
    logger.info("filing_documents_inserted", filings=len(batch), documents=len(records))
    with timed_operation("documents.verify"):
        summary.snapshot_after = verify_append(
            table,
            key_column="document_id",
            expected_rows=rows + len(records),
            inserted_keys=[record.document_id for record in records],
            snapshot_before=summary.snapshot_after,
        )
        _verify_written_groups(table, batch)
    return True


def _run_documents(
    catalog: Catalog,
    data_dir: Path,
    run: tuple[SecBronze, datetime],
    summary: DocumentsStageSummary,
) -> None:
    filings_table, _ = open_filings_table(catalog, data_dir)
    parents = sorted(
        read_filings(filings_table), key=lambda p: (p.filing_date, p.filing_id)
    )
    summary.filings_considered = len(parents)

    table, created = open_documents_table(catalog, data_dir)
    logger.info(
        "documents_table_created" if created else "documents_table_loaded",
        location=table.location(),
    )
    summary.snapshot_before = summary.snapshot_after = current_snapshot_id(table)
    stored = summarize_groups(
        table, "filing_id", "document_ordinal", uniform=("source_submission_sha256",)
    )
    primaries = Counter(
        row["filing_id"]
        for row in iter_rows(table, ("filing_id", "role"))
        if row["role"] == "primary_document"
    )
    pending = _pending_parents(parents, stored, primaries, summary)
    if summary.filings_conflicted:
        return
    for batch in batched(pending, FILINGS_PER_COMMIT, strict=False):
        if not _write_batch(table, batch, run, summary):
            return
    summary.status = "completed"


def run_documents_stage(
    data_dir: Path,
    processed_at: datetime,
    bronze: SecBronze | None = None,
) -> DocumentsStageSummary:
    """Inventory the `<DOCUMENT>` blocks of every filing in `sec_filings.filings`.

    Filings whose documents are already stored completely (judged by group
    summaries) are skipped without re-reading Bronze; a broken stored group
    rejects the run before the first write. The others are parsed one
    submission at a time and committed in batches of FILINGS_PER_COMMIT. An
    invalid filing rejects its batch before that batch's write; earlier
    batches stay committed. Unexpected errors are logged and re-raised.
    """

    started = perf_counter()
    summary = DocumentsStageSummary()
    with log_context(source="sec_filings", table=DOCUMENTS_TABLE):
        logger.info("documents_stage_started")
        try:
            with (
                timed_operation("documents.total"),
                open_silver_catalog(data_dir) as catalog,
            ):
                _run_documents(
                    catalog,
                    data_dir,
                    (bronze or SecBronze.under(data_dir), processed_at),
                    summary,
                )
        except Exception as error:
            logger.error(
                "documents_stage_failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise

        summary.duration_seconds = round(perf_counter() - started, 3)
        if summary.status == "completed":
            logger.info("documents_stage_completed", **asdict(summary))
        else:
            logger.error("documents_stage_failed", **asdict(summary))
    return summary

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

from pyiceberg.table import Table

from src.data_pipeline.connectors.sec_filings.bronze import (
    BronzeFiling,
    discover_filings,
    load_filing_manifest,
)
from src.data_pipeline.contracts.sec_filings.filings import (
    FILINGS_ARROW_SCHEMA,
    FilingRecord,
)
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.storage.iceberg import (
    append_rows,
    current_snapshot_id,
    open_silver_catalog,
    verify_append,
)
from src.data_pipeline.storage.sec_filings.filings import (
    FILINGS_TABLE,
    StoredChecksums,
    open_filings_table,
    read_stored_checksums,
)
from src.data_pipeline.transforms.sec_filings.filings import (
    build_filing_record,
    index_manifest,
)
from src.utils.logger import get_logger
from src.utils.timer import timed_operation

logger = get_logger(__name__)


@dataclass
class FilingsStageSummary:
    """Outcome of the `sec_filings.filings` stage of one Silver run.

    Filled in as the stage progresses. `rejected` means the batch had invalid
    filings or checksum conflicts and nothing was written. Snapshot IDs are
    None when the table has no snapshot yet, or was not opened because Bronze
    validation already failed.
    """

    status: Literal["completed", "rejected"] = "rejected"
    discovered: int = 0
    parsed: int = 0
    inserted: int = 0
    skipped: int = 0
    conflicts: int = 0
    failed: int = 0
    duration_seconds: float = 0.0
    snapshot_before: int | None = None
    snapshot_after: int | None = None


@dataclass(frozen=True)
class _WritePlan:
    """Validated records split by how they relate to rows already stored."""

    new: tuple[FilingRecord, ...]
    skipped: int
    conflicts: int


def _parse_batch(
    filings: Sequence[BronzeFiling],
    bronze: SecBronze,
    processed_at: datetime,
) -> tuple[list[FilingRecord], int]:
    """Build a record for every filing; return the records and the failure count."""

    manifest = index_manifest(load_filing_manifest(bronze))
    accession_counts = Counter(filing.accession_number for filing in filings)
    records: list[FilingRecord] = []
    failed = 0
    for filing in filings:
        try:
            if accession_counts[filing.accession_number] > 1:
                raise ValueError(
                    "Accession number appears in more than one Bronze folder"
                )
            records.append(
                build_filing_record(
                    filing,
                    bronze=bronze,
                    manifest=manifest,
                    processed_at=processed_at,
                )
            )
        except (OSError, ValueError) as error:
            failed += 1
            logger.error(
                "sec_filing_invalid", path=str(filing.directory), error=str(error)
            )
    return records, failed


def _plan_writes(
    records: Sequence[FilingRecord],
    stored: dict[str, StoredChecksums],
) -> _WritePlan:
    """Classify records as new, unchanged (skipped) or conflicting."""

    new: list[FilingRecord] = []
    skipped = conflicts = 0
    for record in records:
        current = StoredChecksums(
            full_submission_sha256=record.full_submission_sha256,
            primary_document_sha256=record.primary_document_sha256,
        )
        existing = stored.get(record.filing_id)
        if existing is None:
            new.append(record)
        elif existing == current:
            skipped += 1
        else:
            conflicts += 1
            logger.error(
                "sec_filing_conflict",
                filing_id=record.filing_id,
                path=record.full_submission_path,
                stored=asdict(existing),
                current=asdict(current),
            )
    # Ordering rows before the single append keeps each data file clustered.
    new.sort(key=lambda r: (r.filing_date, r.cik, r.form, r.accession_number))
    return _WritePlan(new=tuple(new), skipped=skipped, conflicts=conflicts)


def _write_batch(
    table: Table, records: Sequence[FilingRecord], summary: FilingsStageSummary
) -> None:
    """Append the new records unless any conflicts, then verify the commit."""

    summary.snapshot_before = summary.snapshot_after = current_snapshot_id(table)
    stored = read_stored_checksums(table)
    plan = _plan_writes(records, stored)
    summary.skipped, summary.conflicts = plan.skipped, plan.conflicts
    logger.info("sec_filings_skipped", skipped=plan.skipped)
    if plan.conflicts:
        return

    if plan.new:
        with timed_operation("iceberg.write", records=len(plan.new)):
            append_rows(table, plan.new, FILINGS_ARROW_SCHEMA)
    summary.inserted = len(plan.new)
    logger.info("sec_filings_inserted", inserted=summary.inserted)

    with timed_operation("iceberg.verify"):
        summary.snapshot_after = verify_append(
            table,
            key_column="filing_id",
            expected_rows=len(stored) + len(plan.new),
            inserted_keys=[record.filing_id for record in plan.new],
            snapshot_before=summary.snapshot_before,
        )
    summary.status = "completed"


def _run_filings(
    data_dir: Path,
    bronze: SecBronze,
    processed_at: datetime,
    summary: FilingsStageSummary,
) -> None:
    with timed_operation("bronze.discovery"):
        filings = discover_filings(bronze)
    summary.discovered = len(filings)
    logger.info("bronze_filings_discovered", discovered=summary.discovered)

    with timed_operation("bronze.transform"):
        records, summary.failed = _parse_batch(filings, bronze, processed_at)
    summary.parsed = len(records)
    logger.info("bronze_filings_parsed", parsed=summary.parsed, failed=summary.failed)
    if summary.failed:
        return

    with open_silver_catalog(data_dir) as catalog:
        table, created = open_filings_table(catalog, data_dir)
        logger.info(
            "iceberg_table_created" if created else "iceberg_table_loaded",
            table=FILINGS_TABLE,
            location=table.location(),
        )
        _write_batch(table, records, summary)


def run_filings_stage(
    data_dir: Path, processed_at: datetime, bronze: SecBronze | None = None
) -> FilingsStageSummary:
    """Build or incrementally update `sec_filings.filings` from all SEC Bronze filings.

    Bronze is read from `bronze` (default `<data_dir>/bronze/sec_filings`);
    the Silver catalog lives under `data_dir`. The whole batch is validated
    before any write and new filings are appended in one commit. Invalid
    filings or checksum conflicts reject the batch without writing.
    """

    started = perf_counter()
    summary = FilingsStageSummary()
    _run_filings(data_dir, bronze or SecBronze.under(data_dir), processed_at, summary)
    summary.duration_seconds = round(perf_counter() - started, 3)
    if summary.status == "completed":
        logger.info("filings_stage_completed", **asdict(summary))
    else:
        logger.error("filings_stage_failed", **asdict(summary))
    return summary

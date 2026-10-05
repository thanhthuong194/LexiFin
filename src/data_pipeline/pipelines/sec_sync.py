from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter

from src.data_pipeline.connectors.sec_filings.sec_edgar import (
    FilingDownloader,
    FormDownloadRequest,
    download_form,
    filing_form_root,
    inspect_filing_directory,
    iter_accession_directories,
)
from src.data_pipeline.contracts.ingestion import (
    CompanySyncResult,
    FilingManifestEntry,
    FormType,
    IngestionRequest,
)
from src.data_pipeline.pipelines.sec_state import (
    CompanySyncPlan,
    complete_success_accessions,
)
from src.data_pipeline.quality.sec_bronze import manifest_entry_is_complete
from src.data_pipeline.storage.bronze import append_jsonl, sha256_file
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)


def _folder_state(path: Path) -> dict[str, tuple[int, str]]:
    if not path.exists():
        return {}
    return {
        file_path.relative_to(path).as_posix(): (
            file_path.stat().st_size,
            sha256_file(file_path),
        )
        for file_path in sorted(path.rglob("*"))
        if file_path.is_file()
    }


def _downloaded_bytes(
    before: dict[str, tuple[int, str]],
    after: dict[str, tuple[int, str]],
) -> int:
    return sum(
        size
        for path, (size, digest) in after.items()
        if before.get(path) != (size, digest)
    )


def _append_entries(
    manifest_path: Path,
    entries: list[FilingManifestEntry],
) -> None:
    if not entries:
        return
    with timed_operation("manifest.write", entries=len(entries)):
        for entry in entries:
            append_jsonl(manifest_path, entry)


@dataclass(frozen=True)
class SecSyncContext:
    """Run-wide inputs shared by every company sync."""

    request: IngestionRequest
    downloader: FilingDownloader
    source_root: Path
    data_dir: Path
    manifest_path: Path
    run_id: str


@dataclass
class _CompanyProgress:
    """Totals and manifest view accumulated while one company syncs.

    Updated in place, so a failure part-way through still reports everything
    counted before it.
    """

    manifest: list[FilingManifestEntry]
    effective_sync_from: date
    discovered: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    downloaded_bytes: int = 0


@dataclass(frozen=True)
class _FormScan:
    """Accession sets that decide how each downloaded directory is recorded."""

    cik: str
    form: FormType
    sync_from: date
    complete_before: set[str]
    eligible: set[str]
    known_success: set[str]
    changed: set[str]


def _latest_success(
    manifest: list[FilingManifestEntry],
    scan: _FormScan,
    accession: str,
) -> FilingManifestEntry | None:
    return next(
        (
            entry
            for entry in reversed(manifest)
            if entry.cik == scan.cik
            and entry.form == scan.form
            and entry.accession_number == accession
            and entry.status == "success"
        ),
        None,
    )


def _failed_entry(
    accession: str,
    error: Exception,
    scan: _FormScan,
    context: SecSyncContext,
    progress: _CompanyProgress,
) -> FilingManifestEntry:
    entry = FilingManifestEntry(
        run_id=context.run_id,
        filing_id=f"{scan.cik}:{scan.form}:{accession}",
        cik=scan.cik,
        accession_number=accession,
        form=scan.form,
        status="failed",
        error=str(error),
        recorded_at=datetime.now(UTC),
    )
    progress.failed += 1
    logger.error("filing_failed", accession_number=accession, error=str(error))
    return entry


def _accession_entry(
    directory: Path,
    scan: _FormScan,
    context: SecSyncContext,
    progress: _CompanyProgress,
) -> FilingManifestEntry | None:
    """Return the manifest entry to append for one accession directory, if any."""

    accession = directory.name
    if accession in scan.complete_before:
        if accession in scan.eligible:
            logger.info(
                "filing_skipped",
                accession_number=accession,
                reason="complete_manifest_entry",
            )
        return None
    try:
        filing = inspect_filing_directory(directory, context.data_dir)
    except (OSError, ValueError) as error:
        return (
            _failed_entry(accession, error, scan, context, progress)
            if accession in scan.changed
            else None
        )

    if (
        filing.filing_date is not None
        and not scan.sync_from <= filing.filing_date <= context.request.as_of_date
    ):
        return None
    existing_success = _latest_success(progress.manifest, scan, accession)
    if existing_success is not None and manifest_entry_is_complete(
        existing_success, context.data_dir
    ):
        if accession in scan.changed:
            progress.downloaded += 1
        logger.info("filing_downloaded", accession_number=accession, recovered=True)
        return None
    if accession in scan.known_success and accession not in scan.changed:
        return None

    entry = FilingManifestEntry(
        run_id=context.run_id,
        filing_id=f"{scan.cik}:{scan.form}:{accession}",
        cik=scan.cik,
        accession_number=accession,
        form=scan.form,
        filing_date=filing.filing_date,
        status="success",
        artifacts=filing.artifacts,
        recorded_at=datetime.now(UTC),
    )
    if accession in scan.changed:
        progress.downloaded += 1
    logger.info("filing_downloaded", accession_number=accession)
    return entry


def _sync_form(
    form: FormType,
    plan: CompanySyncPlan,
    sync_from: date,
    context: SecSyncContext,
    progress: _CompanyProgress,
) -> None:
    as_of_date = context.request.as_of_date
    recovery_dates = [
        entry.filing_date
        for entry in progress.manifest
        if entry.cik == plan.cik
        and entry.form == form
        and entry.status == "success"
        and entry.filing_date is not None
        and not manifest_entry_is_complete(entry, context.data_dir)
    ]
    form_sync_from = min([sync_from, *recovery_dates])
    progress.effective_sync_from = min(progress.effective_sync_from, form_sync_from)
    complete_before = complete_success_accessions(
        tuple(progress.manifest),
        data_dir=context.data_dir,
        cik=plan.cik,
        form=form,
    )
    eligible_accessions = {
        entry.accession_number
        for entry in progress.manifest
        if entry.cik == plan.cik
        and entry.form == form
        and entry.accession_number in complete_before
        and entry.filing_date is not None
        and form_sync_from <= entry.filing_date <= as_of_date
    }
    form_root = filing_form_root(context.source_root, plan.cik, form)
    before_state = _folder_state(form_root)
    with timed_operation("sec.download"):
        reported_downloaded = download_form(
            context.downloader,
            FormDownloadRequest(
                cik=plan.cik, form=form, after=form_sync_from, before=as_of_date
            ),
            accession_numbers_to_skip=complete_before,
        )
    after_state = _folder_state(form_root)
    progress.downloaded_bytes += _downloaded_bytes(before_state, after_state)
    progress.skipped += len(eligible_accessions)
    progress.discovered += reported_downloaded + len(eligible_accessions)

    scan = _FormScan(
        cik=plan.cik,
        form=form,
        sync_from=form_sync_from,
        complete_before=complete_before,
        eligible=eligible_accessions,
        known_success={
            entry.accession_number
            for entry in progress.manifest
            if entry.cik == plan.cik
            and entry.form == form
            and entry.status == "success"
        },
        changed={
            relative_path.split("/", maxsplit=1)[0]
            for relative_path, signature in after_state.items()
            if before_state.get(relative_path) != signature
        },
    )
    new_entries: list[FilingManifestEntry] = []
    for directory in iter_accession_directories(context.source_root, plan.cik, form):
        entry = _accession_entry(directory, scan, context, progress)
        if entry is not None:
            new_entries.append(entry)

    _append_entries(context.manifest_path, new_entries)
    progress.manifest.extend(new_entries)


def sync_company(
    *,
    plan: CompanySyncPlan,
    sync_from: date,
    context: SecSyncContext,
    manifest: tuple[FilingManifestEntry, ...],
) -> tuple[CompanySyncResult, int, tuple[FilingManifestEntry, ...]]:
    """Synchronize all requested forms for one universe company."""

    started = perf_counter()
    progress = _CompanyProgress(manifest=list(manifest), effective_sync_from=sync_from)

    logger.info("company_sync_started", membership=plan.membership, sync_from=sync_from)
    try:
        for form in context.request.forms:
            with log_context(form=form):
                _sync_form(form, plan, sync_from, context, progress)
    except Exception as error:
        progress.failed += 1
        logger.exception("company_sync_failed", error=str(error))

    result = CompanySyncResult(
        cik=plan.cik,
        membership=plan.membership,
        sync_from=progress.effective_sync_from,
        sync_to=context.request.as_of_date,
        discovered=progress.discovered,
        downloaded=progress.downloaded,
        skipped=progress.skipped,
        failed=progress.failed,
        duration_seconds=perf_counter() - started,
    )
    logger.info(
        "company_sync_completed",
        downloaded=progress.downloaded,
        skipped=progress.skipped,
        failed=progress.failed,
    )
    return result, progress.downloaded_bytes, tuple(progress.manifest)

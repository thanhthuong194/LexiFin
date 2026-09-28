from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter

from src.data_pipeline.connectors.sec_filings.sec_edgar import (
    FilingDownloader,
    download_form,
    filing_form_root,
    inspect_filing_directory,
    iter_accession_directories,
)
from src.data_pipeline.contracts.ingestion import (
    CompanySyncResult,
    FilingManifestEntry,
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


def sync_company(
    *,
    plan: CompanySyncPlan,
    request: IngestionRequest,
    sync_from: date,
    downloader: FilingDownloader,
    source_root: Path,
    data_dir: Path,
    manifest_path: Path,
    manifest: tuple[FilingManifestEntry, ...],
    run_id: str,
) -> tuple[CompanySyncResult, int, tuple[FilingManifestEntry, ...]]:
    """Synchronize all requested forms for one universe company."""

    started = perf_counter()
    discovered = downloaded = skipped = failed = downloaded_bytes = 0
    current_manifest = list(manifest)
    effective_sync_from = sync_from

    logger.info("company_sync_started", membership=plan.membership, sync_from=sync_from)
    try:
        for form in request.forms:
            with log_context(form=form):
                recovery_dates = [
                    entry.filing_date
                    for entry in current_manifest
                    if entry.cik == plan.cik
                    and entry.form == form
                    and entry.status == "success"
                    and entry.filing_date is not None
                    and not manifest_entry_is_complete(entry, data_dir)
                ]
                form_sync_from = min([sync_from, *recovery_dates])
                effective_sync_from = min(effective_sync_from, form_sync_from)
                complete_before = complete_success_accessions(
                    tuple(current_manifest),
                    data_dir=data_dir,
                    cik=plan.cik,
                    form=form,
                )
                eligible_accessions = {
                    entry.accession_number
                    for entry in current_manifest
                    if entry.cik == plan.cik
                    and entry.form == form
                    and entry.accession_number in complete_before
                    and entry.filing_date is not None
                    and form_sync_from <= entry.filing_date <= request.as_of_date
                }
                form_root = filing_form_root(source_root, plan.cik, form)
                before_state = _folder_state(form_root)
                with timed_operation("sec.download"):
                    reported_downloaded = download_form(
                        downloader,
                        cik=plan.cik,
                        form=form,
                        after=form_sync_from,
                        before=request.as_of_date,
                        accession_numbers_to_skip=complete_before,
                    )
                after_state = _folder_state(form_root)
                downloaded_bytes += _downloaded_bytes(before_state, after_state)
                skipped += len(eligible_accessions)
                discovered += reported_downloaded + len(eligible_accessions)

                new_entries: list[FilingManifestEntry] = []
                known_success = {
                    entry.accession_number
                    for entry in current_manifest
                    if entry.cik == plan.cik
                    and entry.form == form
                    and entry.status == "success"
                }
                changed_accessions = {
                    relative_path.split("/", maxsplit=1)[0]
                    for relative_path, signature in after_state.items()
                    if before_state.get(relative_path) != signature
                }
                for directory in iter_accession_directories(
                    source_root, plan.cik, form
                ):
                    accession = directory.name
                    if accession in complete_before:
                        if accession in eligible_accessions:
                            logger.info(
                                "filing_skipped",
                                accession_number=accession,
                                reason="complete_manifest_entry",
                            )
                        continue
                    try:
                        filing = inspect_filing_directory(directory, data_dir)
                    except (OSError, ValueError) as error:
                        if accession not in changed_accessions:
                            continue
                        new_entries.append(
                            FilingManifestEntry(
                                run_id=run_id,
                                filing_id=f"{plan.cik}:{form}:{accession}",
                                cik=plan.cik,
                                accession_number=accession,
                                form=form,
                                status="failed",
                                error=str(error),
                                recorded_at=datetime.now(UTC),
                            )
                        )
                        failed += 1
                        logger.error(
                            "filing_failed",
                            accession_number=accession,
                            error=str(error),
                        )
                        continue

                    if (
                        filing.filing_date is not None
                        and not form_sync_from
                        <= filing.filing_date
                        <= request.as_of_date
                    ):
                        continue
                    existing_success = next(
                        (
                            entry
                            for entry in reversed(current_manifest)
                            if entry.cik == plan.cik
                            and entry.form == form
                            and entry.accession_number == accession
                            and entry.status == "success"
                        ),
                        None,
                    )
                    if existing_success is not None and manifest_entry_is_complete(
                        existing_success, data_dir
                    ):
                        if accession in changed_accessions:
                            downloaded += 1
                        logger.info(
                            "filing_downloaded",
                            accession_number=accession,
                            recovered=True,
                        )
                        continue
                    if (
                        accession in known_success
                        and accession not in changed_accessions
                    ):
                        continue

                    new_entries.append(
                        FilingManifestEntry(
                            run_id=run_id,
                            filing_id=f"{plan.cik}:{form}:{accession}",
                            cik=plan.cik,
                            accession_number=accession,
                            form=form,
                            filing_date=filing.filing_date,
                            status="success",
                            artifacts=filing.artifacts,
                            recorded_at=datetime.now(UTC),
                        )
                    )
                    if accession in changed_accessions:
                        downloaded += 1
                    logger.info("filing_downloaded", accession_number=accession)

                _append_entries(manifest_path, new_entries)
                current_manifest.extend(new_entries)
    except Exception as error:
        failed += 1
        logger.exception("company_sync_failed", error=str(error))

    result = CompanySyncResult(
        cik=plan.cik,
        membership=plan.membership,
        sync_from=effective_sync_from,
        sync_to=request.as_of_date,
        discovered=discovered,
        downloaded=downloaded,
        skipped=skipped,
        failed=failed,
        duration_seconds=perf_counter() - started,
    )
    logger.info(
        "company_sync_completed",
        downloaded=downloaded,
        skipped=skipped,
        failed=failed,
    )
    return result, downloaded_bytes, tuple(current_manifest)

from __future__ import annotations

from pathlib import Path

from src.data_pipeline.contracts.ingestion import (
    FilingManifestEntry,
    IngestionRunRecord,
    UniverseMember,
)
from src.data_pipeline.storage.bronze import resolve_relative_path, sha256_file


def validate_universe(
    members: tuple[UniverseMember, ...],
    company_limit: int,
) -> None:
    """Validate SEC universe cardinality and ranking invariants."""

    if len(members) != company_limit:
        raise ValueError(
            f"Universe has {len(members)} companies; expected {company_limit}"
        )
    ciks = [member.cik for member in members]
    if len(ciks) != len(set(ciks)):
        raise ValueError("Universe contains duplicate CIK values")
    if [member.rank for member in members] != list(range(1, company_limit + 1)):
        raise ValueError("Universe ranks are not contiguous")
    if any(not member.ticker_aliases for member in members):
        raise ValueError("Universe contains empty ticker aliases")


def validate_nonempty_files(paths: tuple[Path, ...]) -> None:
    """Validate that controlled source snapshot files exist and are non-empty."""

    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Required output is missing or empty: {path}")


def manifest_entry_is_complete(
    entry: FilingManifestEntry,
    data_dir: Path,
) -> bool:
    """Return whether every success artifact still matches disk."""

    if entry.status != "success" or not entry.artifacts:
        return False
    try:
        for artifact in entry.artifacts:
            path = resolve_relative_path(artifact.relative_path, data_dir)
            if not path.is_file() or path.stat().st_size != artifact.size_bytes:
                return False
            if artifact.size_bytes == 0 or sha256_file(path) != artifact.sha256:
                return False
    except (OSError, ValueError):
        return False
    return True


def validate_manifest(
    entries: tuple[FilingManifestEntry, ...],
    data_dir: Path,
) -> None:
    """Validate successful artifacts and reject duplicate entries within a run."""

    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (entry.run_id, entry.filing_id)
        if key in seen:
            raise ValueError(f"Duplicate filing manifest entry: {key}")
        seen.add(key)
        if entry.status == "success" and not manifest_entry_is_complete(
            entry, data_dir
        ):
            raise ValueError(f"Invalid success artifacts for {entry.filing_id}")


def validate_run_totals(run: IngestionRunRecord) -> None:
    """Validate aggregate run metrics and final status."""

    downloaded = sum(company.downloaded for company in run.companies)
    skipped = sum(company.skipped for company in run.companies)
    failed = sum(company.failed for company in run.companies)
    if (run.downloaded, run.skipped, run.failed) != (downloaded, skipped, failed):
        raise ValueError("Run totals do not match company results")
    expected_status = "partial" if failed else "completed"
    if run.status != expected_status:
        raise ValueError(
            f"Run status {run.status!r} does not match metrics; "
            f"expected {expected_status!r}"
        )

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from src.data_pipeline.contracts.ingestion import (
    FilingManifestEntry,
    IngestionRunRecord,
    MembershipStatus,
    UniverseMember,
)
from src.data_pipeline.quality.sec_bronze import manifest_entry_is_complete
from src.data_pipeline.storage.bronze import read_jsonl


@dataclass(frozen=True)
class CompanySyncPlan:
    """Incremental membership and date range for one company."""

    cik: str
    membership: MembershipStatus
    sync_from: date | None


def load_manifest(path: Path) -> tuple[FilingManifestEntry, ...]:
    """Load and validate the append-only SEC filing manifest."""

    return tuple(FilingManifestEntry.model_validate(row) for row in read_jsonl(path))


def load_usable_runs(
    runs_dir: Path, company_limit: int
) -> tuple[IngestionRunRecord, ...]:
    """Load completed or partial SEC runs for one universe size."""

    runs: list[IngestionRunRecord] = []
    if not runs_dir.exists():
        return ()
    for path in sorted(runs_dir.glob("run_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            run = IngestionRunRecord.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            raise ValueError(f"Invalid SEC run metadata: {path}") from error
        if (
            run.request.company_limit == company_limit
            and run.status in {"completed", "partial"}
            and run.finished_at is not None
        ):
            runs.append(run)
    return tuple(sorted(runs, key=lambda run: (run.finished_at, run.run_id)))


def _active_ciks(run: IngestionRunRecord) -> set[str]:
    return {company.cik for company in run.companies if company.membership != "exited"}


def _last_successful_sync(
    cik: str,
    runs: tuple[IngestionRunRecord, ...],
) -> date | None:
    for run in reversed(runs):
        for company in run.companies:
            if (
                company.cik == cik
                and company.membership != "exited"
                and company.failed == 0
            ):
                return company.sync_to
    return None


def classify_membership(
    members: tuple[UniverseMember, ...],
    prior_runs: tuple[IngestionRunRecord, ...],
) -> tuple[tuple[CompanySyncPlan, ...], tuple[str, ...]]:
    """Classify current members and companies that just exited the universe."""

    current_ciks = {member.cik for member in members}
    previous_active = _active_ciks(prior_runs[-1]) if prior_runs else set()
    ever_active = (
        set().union(*(_active_ciks(run) for run in prior_runs)) if prior_runs else set()
    )

    plans: list[CompanySyncPlan] = []
    for member in members:
        if member.cik in previous_active:
            membership: MembershipStatus = "unchanged"
        elif member.cik in ever_active:
            membership = "reentered"
        else:
            membership = "new"
        plans.append(
            CompanySyncPlan(
                cik=member.cik,
                membership=membership,
                sync_from=_last_successful_sync(member.cik, prior_runs),
            )
        )

    exited = tuple(sorted(previous_active - current_ciks))
    return tuple(plans), exited


def complete_success_accessions(
    entries: tuple[FilingManifestEntry, ...],
    *,
    data_dir: Path,
    cik: str,
    form: str,
) -> set[str]:
    """Return globally reusable accessions with intact manifest artifacts."""

    return {
        entry.accession_number
        for entry in entries
        if entry.cik == cik
        and entry.form == form
        and manifest_entry_is_complete(entry, data_dir)
    }

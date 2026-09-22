from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


FormType = Literal["10-K", "10-Q", "8-K"]

ArtifactRole = Literal[
    "full_submission",
    "primary_document",
]

FilingStatus = Literal["success", "failed"]

MembershipStatus = Literal[
    "new",
    "unchanged",
    "exited",
    "reentered",
]

RunStatus = Literal[
    "running",
    "completed",
    "partial",
    "failed",
]


class ContractModel(BaseModel):
    """Base model that rejects fields outside the declared contract."""

    model_config = ConfigDict(extra="forbid")


class IngestionRequest(ContractModel):
    """Parameters defining one SEC ingestion run."""

    company_limit: int = Field(ge=1, le=500)
    as_of_date: date
    lookback_years: int = Field(default=5, ge=1)
    forms: tuple[FormType, ...] = ("10-K", "10-Q", "8-K")


class UniverseMember(ContractModel):
    """One ranked company selected from an IVV holdings snapshot."""

    snapshot_id: str
    holdings_as_of_date: date
    rank: int = Field(ge=1)

    cik: str
    company_name: str
    primary_ticker: str
    ticker_aliases: tuple[str, ...]

    weight_pct: float = Field(ge=0)


class RawArtifact(ContractModel):
    """Metadata describing one raw filing file stored in Bronze."""

    role: ArtifactRole
    relative_path: str
    sha256: str
    size_bytes: int = Field(ge=0)


class FilingManifestEntry(ContractModel):
    """Manifest record for one downloaded or failed SEC filing."""

    run_id: str

    filing_id: str
    cik: str
    accession_number: str
    form: FormType
    filing_date: date | None = None

    status: FilingStatus
    artifacts: tuple[RawArtifact, ...] = ()
    error: str | None = None
    recorded_at: datetime


class CompanySyncResult(ContractModel):
    """Incremental ingestion result for one company in a run."""

    cik: str
    membership: MembershipStatus

    sync_from: date | None = None
    sync_to: date

    discovered: int = Field(default=0, ge=0)
    downloaded: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)

    duration_seconds: float = Field(default=0, ge=0)


class IngestionRunRecord(ContractModel):
    """Provenance, status, and metrics for one ingestion run."""

    run_id: str
    request: IngestionRequest

    universe_snapshot_id: str
    input_hashes: dict[str, str]
    code_version: str | None = None

    status: RunStatus
    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float | None = None

    companies: tuple[CompanySyncResult, ...] = ()

    downloaded: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    downloaded_bytes: int = Field(default=0, ge=0)
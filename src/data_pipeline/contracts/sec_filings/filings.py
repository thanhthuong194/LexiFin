from __future__ import annotations

from datetime import date

import pyarrow as pa
from pydantic import AwareDatetime, Field, NaiveDatetime, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel, FormType

SHA256_PATTERN = r"^[0-9a-f]{64}$"
INT32_MAX = 2**31 - 1

FILINGS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("filing_id", pa.string(), nullable=False),
        pa.field("accession_number", pa.string(), nullable=False),
        pa.field("cik", pa.string(), nullable=False),
        pa.field("form", pa.string(), nullable=False),
        pa.field("filing_date", pa.date32(), nullable=False),
        pa.field("report_date", pa.date32(), nullable=True),
        pa.field("accepted_at", pa.timestamp("us"), nullable=True),
        pa.field("public_document_count", pa.int32(), nullable=True),
        pa.field("full_submission_path", pa.string(), nullable=False),
        pa.field("primary_document_path", pa.string(), nullable=True),
        pa.field("full_submission_sha256", pa.string(), nullable=False),
        pa.field("primary_document_sha256", pa.string(), nullable=True),
        pa.field("ingestion_run_id", pa.string(), nullable=True),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def filing_id_for(accession_number: str) -> str:
    """Return the internal Silver key of one SEC submission."""

    return f"sec:{accession_number}"


class FilingRecord(ContractModel):
    """One SEC submission (one accession number) in Silver `sec_filings.filings`.

    Field order matches FILINGS_ARROW_SCHEMA. Paths are POSIX paths relative
    to the data directory. `accepted_at` is the naive EDGAR acceptance time
    as written in the header; `processed_at` is the Silver batch time.
    """

    filing_id: str
    accession_number: str = Field(pattern=r"^\d{10}-\d{2}-\d{6}$")
    cik: str = Field(pattern=r"^\d{10}$")
    form: FormType
    filing_date: date
    report_date: date | None
    accepted_at: NaiveDatetime | None
    public_document_count: int | None = Field(ge=0, le=INT32_MAX)
    full_submission_path: str
    primary_document_path: str | None
    full_submission_sha256: str = Field(pattern=SHA256_PATTERN)
    primary_document_sha256: str | None = Field(pattern=SHA256_PATTERN)
    ingestion_run_id: str | None
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_derived_fields(self) -> FilingRecord:
        if self.filing_id != filing_id_for(self.accession_number):
            raise ValueError("filing_id must be derived from accession_number")
        if (self.primary_document_path is None) != (
            self.primary_document_sha256 is None
        ):
            raise ValueError("primary document path and checksum must both be set")
        return self

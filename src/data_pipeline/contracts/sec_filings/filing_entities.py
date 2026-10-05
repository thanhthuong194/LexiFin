from __future__ import annotations

from datetime import date
from typing import Literal

import pyarrow as pa
from pydantic import AwareDatetime, Field, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel
from src.data_pipeline.contracts.sec_filings.filings import INT32_MAX, SHA256_PATTERN

EntityRole = Literal[
    "filer",
    "filed_by",
    "subject_company",
    "reporting_owner",
    "issuer",
    "serial_company",
    "filed_for",
]

FILING_ENTITIES_ARROW_SCHEMA = pa.schema(
    [
        pa.field("filing_entity_id", pa.string(), nullable=False),
        pa.field("filing_id", pa.string(), nullable=False),
        pa.field("filing_date", pa.date32(), nullable=False),
        pa.field("entity_ordinal", pa.int32(), nullable=False),
        pa.field("entity_role", pa.string(), nullable=False),
        pa.field("source_role", pa.string(), nullable=False),
        pa.field("cik", pa.string(), nullable=False),
        pa.field("company_name", pa.string(), nullable=False),
        pa.field("sic", pa.string(), nullable=True),
        pa.field("sic_description", pa.string(), nullable=True),
        pa.field("organization_name", pa.string(), nullable=True),
        pa.field("ein", pa.string(), nullable=True),
        pa.field("state_of_incorporation", pa.string(), nullable=True),
        pa.field("fiscal_year_end", pa.string(), nullable=True),
        pa.field("form_type", pa.string(), nullable=True),
        pa.field("sec_act", pa.string(), nullable=True),
        pa.field("file_number", pa.string(), nullable=True),
        pa.field("film_number", pa.string(), nullable=True),
        pa.field("business_street_1", pa.string(), nullable=True),
        pa.field("business_street_2", pa.string(), nullable=True),
        pa.field("business_city", pa.string(), nullable=True),
        pa.field("business_state_or_country", pa.string(), nullable=True),
        pa.field("business_zip", pa.string(), nullable=True),
        pa.field("business_phone", pa.string(), nullable=True),
        pa.field("mailing_street_1", pa.string(), nullable=True),
        pa.field("mailing_street_2", pa.string(), nullable=True),
        pa.field("mailing_city", pa.string(), nullable=True),
        pa.field("mailing_state_or_country", pa.string(), nullable=True),
        pa.field("mailing_zip", pa.string(), nullable=True),
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("source_submission_sha256", pa.string(), nullable=False),
        pa.field("entity_start_byte", pa.int64(), nullable=False),
        pa.field("entity_end_byte", pa.int64(), nullable=False),
        pa.field("entity_block_sha256", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def filing_entity_id_for(filing_id: str, entity_ordinal: int) -> str:
    """Return the internal key of the N-th entity-role block of a filing."""

    return f"{filing_id}:entity:{entity_ordinal:04d}"


class FilingEntityRecord(ContractModel):
    """One entity-role block (FILER, SUBJECT COMPANY, ...) of a filing's SEC header.

    Field order matches FILING_ENTITIES_ARROW_SCHEMA. Values are the header's
    text, trimmed; codes such as SIC, EIN, ZIP and fiscal year end (MMDD) stay
    strings. The entity span is a zero-based byte range of the raw
    `source_path` file from the role label up to the next entity block or the
    end of the SEC header (start inclusive, end exclusive). Optional metadata
    defaults to None because headers omit fields they do not have.
    """

    filing_entity_id: str
    filing_id: str
    filing_date: date
    entity_ordinal: int = Field(ge=1, le=INT32_MAX)
    entity_role: EntityRole
    source_role: str = Field(min_length=1)
    cik: str = Field(pattern=r"^\d{10}$")
    company_name: str = Field(min_length=1)
    sic: str | None = Field(default=None, pattern=r"^\d+$")
    sic_description: str | None = None
    organization_name: str | None = None
    ein: str | None = None
    state_of_incorporation: str | None = None
    fiscal_year_end: str | None = None
    form_type: str | None = None
    sec_act: str | None = None
    file_number: str | None = None
    film_number: str | None = None
    business_street_1: str | None = None
    business_street_2: str | None = None
    business_city: str | None = None
    business_state_or_country: str | None = None
    business_zip: str | None = None
    business_phone: str | None = None
    mailing_street_1: str | None = None
    mailing_street_2: str | None = None
    mailing_city: str | None = None
    mailing_state_or_country: str | None = None
    mailing_zip: str | None = None
    source_path: str
    source_submission_sha256: str = Field(pattern=SHA256_PATTERN)
    entity_start_byte: int = Field(ge=0)
    entity_end_byte: int = Field(ge=0)
    entity_block_sha256: str = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_identity_and_span(self) -> FilingEntityRecord:
        if self.filing_entity_id != filing_entity_id_for(
            self.filing_id, self.entity_ordinal
        ):
            raise ValueError(
                "filing_entity_id must be derived from filing_id and ordinal"
            )
        if self.entity_start_byte >= self.entity_end_byte:
            raise ValueError("entity_start_byte must be before entity_end_byte")
        return self

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import httpx
import polars as pl
from sec_edgar_downloader import Downloader  # type: ignore[attr-defined]  # no __all__

from src.data_pipeline.contracts.ingestion import FormType, RawArtifact
from src.data_pipeline.storage.bronze import artifact_for_path

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FILING_DATE_PATTERNS = (
    re.compile(rb"<FILING-DATE>\s*(\d{8})"),
    re.compile(rb"FILED AS OF DATE:\s*(\d{8})"),
)


class FilingDownloader(Protocol):
    """Small protocol implemented by sec-edgar-downloader and test fakes."""

    # The signature mirrors sec_edgar_downloader.Downloader.get.
    def get(  # noqa: PLR0913
        self,
        form: str,
        ticker_or_cik: str,
        *,
        after: date,
        before: date,
        include_amends: bool,
        download_details: bool,
        accession_numbers_to_skip: set[str],
    ) -> int: ...


@dataclass(frozen=True)
class DownloadedFiling:
    """A filing discovered in the downloader's on-disk output."""

    accession_number: str
    filing_date: date | None
    artifacts: tuple[RawArtifact, ...]


def sec_user_agent(company_name: str, email: str) -> str:
    """Build the SEC-compliant request identity."""

    company_name = company_name.strip()
    email = email.strip()
    if not company_name or not email:
        raise ValueError("SEC_COMPANY_NAME and SEC_EMAIL are both required")
    return f"{company_name} {email}"


def download_company_tickers(
    company_name: str,
    email: str,
    client: httpx.Client | None = None,
) -> bytes:
    """Download the unmodified SEC company ticker mapping bytes."""

    headers = {"User-Agent": sec_user_agent(company_name, email)}
    if client is not None:
        response = client.get(SEC_COMPANY_TICKERS_URL, headers=headers)
        response.raise_for_status()
        return response.content

    with httpx.Client(follow_redirects=True, timeout=30.0) as owned_client:
        response = owned_client.get(SEC_COMPANY_TICKERS_URL, headers=headers)
        response.raise_for_status()
        return response.content


def parse_company_tickers(raw_json: bytes) -> pl.DataFrame:
    """Parse SEC ticker, title, and zero-padded CIK values."""

    try:
        payload = json.loads(raw_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("SEC company ticker mapping is not valid JSON") from error

    if not isinstance(payload, dict):
        # Malformed SEC response, not a caller type error; callers handle
        # ValueError for every kind of invalid mapping.
        raise ValueError("SEC company ticker mapping must be a JSON object")  # noqa: TRY004

    records: list[dict[str, str]] = []
    for item in payload.values():
        if not isinstance(item, dict):
            continue
        try:
            cik = str(int(item["cik_str"])).zfill(10)
            ticker = str(item["ticker"]).strip()
            title = str(item["title"]).strip()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "SEC company ticker mapping contains an invalid row"
            ) from error
        records.append({"ticker": ticker, "company_name": title, "cik": cik})

    if not records:
        raise ValueError("SEC company ticker mapping contains no rows")
    return pl.DataFrame(records)


def create_downloader(
    company_name: str,
    email: str,
    download_root: Path,
) -> FilingDownloader:
    """Create the pinned sec-edgar-downloader adapter."""

    sec_user_agent(company_name, email)
    return Downloader(company_name, email, download_root)


@dataclass(frozen=True)
class FormDownloadRequest:
    """One company's filings of one form within an inclusive filing-date range."""

    cik: str
    form: FormType
    after: date
    before: date


def download_form(
    downloader: FilingDownloader,
    request: FormDownloadRequest,
    *,
    accession_numbers_to_skip: set[str],
) -> int:
    """Download one SEC form with the verified 5.1.x API semantics."""

    return downloader.get(
        request.form,
        request.cik,
        after=request.after,
        before=request.before,
        include_amends=False,
        download_details=True,
        accession_numbers_to_skip=accession_numbers_to_skip,
    )


def filing_form_root(download_root: Path, cik: str, form: FormType) -> Path:
    """Return the actual output folder used by sec-edgar-downloader 5.1.x."""

    return download_root / "sec-edgar-filings" / cik / form


def _filing_date(full_submission: Path) -> date | None:
    contents = full_submission.read_bytes()[:100_000]
    for pattern in FILING_DATE_PATTERNS:
        match = pattern.search(contents)
        if match:
            return date.fromisoformat(match.group(1).decode())
    return None


def inspect_filing_directory(directory: Path, data_dir: Path) -> DownloadedFiling:
    """Validate and describe artifacts in one accession directory."""

    full_submission = directory / "full-submission.txt"
    if not full_submission.is_file() or full_submission.stat().st_size == 0:
        raise ValueError(f"Missing or empty filing submission: {full_submission}")

    artifacts = [artifact_for_path("full_submission", full_submission, data_dir)]
    primary_documents = sorted(directory.glob("primary-document.*"))
    artifacts.extend(
        artifact_for_path("primary_document", path, data_dir)
        for path in primary_documents
        if path.is_file() and path.stat().st_size > 0
    )
    return DownloadedFiling(
        accession_number=directory.name,
        filing_date=_filing_date(full_submission),
        artifacts=tuple(artifacts),
    )


def iter_accession_directories(
    download_root: Path,
    cik: str,
    form: FormType,
) -> Iterable[Path]:
    """Yield deterministic accession directories from downloader output."""

    form_root = filing_form_root(download_root, cik, form)
    if not form_root.exists():
        return ()
    return tuple(sorted(path for path in form_root.iterdir() if path.is_dir()))

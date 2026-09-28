from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

from scripts.ingest_sec_edgar import REPOSITORY_ROOT, parse_args
from src.data_pipeline.contracts.ingestion import IngestionRequest
from src.data_pipeline.pipelines.sec_edgar import run_sec_edgar
from src.data_pipeline.pipelines.sec_state import load_manifest
from src.data_pipeline.quality.sec_bronze import manifest_entry_is_complete
from src.data_pipeline.storage.bronze import sha256_file


def _holdings(weight_by_ticker: dict[str, float]) -> bytes:
    rows = [
        '"Fund Holdings as of","Jan 15, 2026"',
        '"Ticker","Name","Asset Class","Weight (%)"',
    ]
    rows.extend(
        f'"{ticker}","{ticker} Corp","Equity","{weight}"'
        for ticker, weight in weight_by_ticker.items()
    )
    return ("\n".join(rows) + "\n").encode()


TICKERS = json.dumps(
    {
        str(index): {"cik_str": index, "ticker": ticker, "title": f"{ticker} Corp"}
        for index, ticker in enumerate(("AAA", "BBB", "CCC"), start=1)
    }
).encode()


class FakeDownloader:
    def __init__(self, root: Path) -> None:
        self.root = root

    def get(
        self,
        form: str,
        ticker_or_cik: str,
        *,
        after: date,
        before: date,
        include_amends: bool,
        download_details: bool,
        accession_numbers_to_skip: set[str],
    ) -> int:
        assert include_amends is False
        assert download_details is True
        filing_date = date(2026, 1, 15)
        if not after <= filing_date <= before:
            return 0
        accession = f"{ticker_or_cik[-4:]}-{form.replace('-', '')}-000001"
        if accession in accession_numbers_to_skip:
            return 0
        folder = self.root / "sec-edgar-filings" / ticker_or_cik / form / accession
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "full-submission.txt").write_bytes(
            b"<SEC-HEADER>\n<FILING-DATE>20260115\n"
        )
        (folder / "primary-document.html").write_bytes(b"<html>filing</html>")
        return 1


def _run(
    data_dir: Path,
    holdings: bytes,
    as_of: date,
):
    return run_sec_edgar(
        IngestionRequest(company_limit=2, as_of_date=as_of, lookback_years=5),
        data_dir,
        "LexiFin",
        "test@example.com",
        ivv_fetcher=lambda: holdings,
        ticker_fetcher=lambda _company, _email: TICKERS,
        downloader_factory=lambda _company, _email, root: FakeDownloader(root),
    )


@pytest.mark.integration
def test_bronze_layout_manifest_checksums_and_idempotent_recovery(
    tmp_path: Path,
) -> None:
    holdings = _holdings({"AAA": 5.0, "BBB": 4.0, "CCC": 3.0})
    first = _run(tmp_path, holdings, date(2026, 1, 31))
    source_root = tmp_path / "bronze" / "sec_filings"
    manifest_path = source_root / "_metadata" / "filing_manifest.jsonl"
    first_manifest = load_manifest(manifest_path)

    assert first.status == "completed"
    assert first.downloaded == 6
    assert len(first_manifest) == 6
    assert len(list(source_root.glob("universe_snapshots/*/universe.jsonl"))) == 1
    for entry in first_manifest:
        assert entry.status == "success"
        for artifact in entry.artifacts:
            assert not Path(artifact.relative_path).is_absolute()
            artifact_path = tmp_path / artifact.relative_path
            assert artifact_path.stat().st_size == artifact.size_bytes
            assert sha256_file(artifact_path) == artifact.sha256

    rerun = _run(tmp_path, holdings, date(2026, 1, 31))
    assert rerun.downloaded == 0
    assert rerun.skipped == 0
    assert len(load_manifest(manifest_path)) == 6

    damaged_entry = load_manifest(manifest_path)[0]
    damaged_path = tmp_path / damaged_entry.artifacts[0].relative_path
    damaged_path.unlink()
    assert not manifest_entry_is_complete(damaged_entry, tmp_path)

    recovered = _run(tmp_path, holdings, date(2026, 1, 31))
    assert recovered.downloaded == 1
    assert manifest_entry_is_complete(damaged_entry, tmp_path)
    assert len(load_manifest(manifest_path)) == 6


@pytest.mark.integration
def test_membership_new_unchanged_exited_and_reentered(tmp_path: Path) -> None:
    first = _run(
        tmp_path,
        _holdings({"AAA": 5.0, "BBB": 4.0, "CCC": 3.0}),
        date(2026, 1, 31),
    )
    second = _run(
        tmp_path,
        _holdings({"AAA": 5.0, "CCC": 4.0, "BBB": 3.0}),
        date(2026, 2, 28),
    )
    third = _run(
        tmp_path,
        _holdings({"AAA": 5.0, "BBB": 4.0, "CCC": 3.0}),
        date(2026, 3, 31),
    )

    assert {company.membership for company in first.companies} == {"new"}
    second_membership = {
        company.cik: company.membership for company in second.companies
    }
    assert second_membership == {
        "0000000001": "unchanged",
        "0000000002": "exited",
        "0000000003": "new",
    }
    third_membership = {company.cik: company.membership for company in third.companies}
    assert third_membership == {
        "0000000001": "unchanged",
        "0000000002": "reentered",
        "0000000003": "exited",
    }


@pytest.mark.integration
def test_cli_default_data_dir_is_repository_relative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["ingest_sec_edgar.py"])
    args = parse_args()
    assert args.data_dir == REPOSITORY_ROOT / "data"

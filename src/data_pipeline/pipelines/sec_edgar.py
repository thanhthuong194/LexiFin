from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.data_pipeline.connectors.sec_filings.ishares import (
    download_ivv_holdings,
    parse_ivv_holdings,
)
from src.data_pipeline.connectors.sec_filings.sec_edgar import (
    FilingDownloader,
    create_downloader,
    download_company_tickers,
    parse_company_tickers,
    sec_user_agent,
)
from src.data_pipeline.contracts.ingestion import (
    CompanySyncResult,
    IngestionRequest,
    IngestionRunRecord,
)
from src.data_pipeline.pipelines.sec_state import (
    classify_membership,
    load_manifest,
    load_usable_runs,
)
from src.data_pipeline.pipelines.sec_sync import sync_company
from src.data_pipeline.quality.sec_bronze import (
    validate_manifest,
    validate_nonempty_files,
    validate_run_totals,
    validate_universe,
)
from src.data_pipeline.storage.bronze import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    sha256_bytes,
)
from src.data_pipeline.transforms.sec_universe import build_sec_universe
from src.utils.logger import (
    bind_log_context,
    clear_log_context,
    get_logger,
    log_context,
)
from src.utils.timer import timed_operation

logger = get_logger(__name__)
IVVFetcher = Callable[[], bytes]
TickerFetcher = Callable[[str, str], bytes]
DownloaderFactory = Callable[[str, str, Path], FilingDownloader]


def _years_before(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def run_sec_edgar(
    request: IngestionRequest,
    data_dir: Path,
    sec_company_name: str,
    sec_email: str,
    *,
    ivv_fetcher: IVVFetcher = download_ivv_holdings,
    ticker_fetcher: TickerFetcher = download_company_tickers,
    downloader_factory: DownloaderFactory = create_downloader,
) -> IngestionRunRecord:
    """Ingest the ranked IVV company universe and SEC filings into Bronze."""

    sec_user_agent(sec_company_name, sec_email)
    if not 1 <= request.company_limit <= 500:
        raise ValueError("company_limit must be between 1 and 500")

    data_dir = data_dir.expanduser().resolve()
    source_root = data_dir / "bronze" / "sec_filings"
    runs_dir = source_root / "_metadata" / "runs"
    manifest_path = source_root / "_metadata" / "filing_manifest.jsonl"
    run_id = uuid4().hex
    run_path = runs_dir / f"run_{run_id}.json"
    started_at = datetime.now(UTC)
    started_clock = perf_counter()
    snapshot_id = ""
    input_hashes: dict[str, str] = {}
    company_results: list[CompanySyncResult] = []
    total_bytes = 0

    bind_log_context(
        run_id=run_id,
        source="sec_filings",
        company_limit=request.company_limit,
        as_of_date=request.as_of_date.isoformat(),
    )
    logger.info("run_started")
    try:
        with timed_operation("pipeline.total"):
            with timed_operation("universe.fetch"):
                ivv_bytes = ivv_fetcher()
                input_hashes["ivv_holdings.csv"] = sha256_bytes(ivv_bytes)
                ticker_bytes = ticker_fetcher(sec_company_name, sec_email)
                input_hashes["company_tickers.json"] = sha256_bytes(ticker_bytes)
            snapshot_id = sha256_bytes(
                f"{input_hashes['ivv_holdings.csv']}:{input_hashes['company_tickers.json']}".encode()
            )
            bind_log_context(snapshot_id=snapshot_id)
            snapshot_folder = (
                source_root
                / "universe_snapshots"
                / (f"snapshot_date_{started_at.strftime('%Y%m%dT%H%M%S%fZ')}")
            )
            ivv_path = snapshot_folder / "ivv_holdings.csv"
            ticker_path = snapshot_folder / "company_tickers.json"
            universe_path = snapshot_folder / "universe.jsonl"
            atomic_write_bytes(ivv_path, ivv_bytes)
            atomic_write_bytes(ticker_path, ticker_bytes)
            logger.info("sources_downloaded", snapshot_folder=str(snapshot_folder))

            with timed_operation("universe.build"):
                holdings = parse_ivv_holdings(ivv_bytes)
                ticker_mapping = parse_company_tickers(ticker_bytes)
                universe = build_sec_universe(
                    holdings.rows,
                    ticker_mapping,
                    snapshot_id=snapshot_id,
                    holdings_as_of_date=holdings.holdings_as_of_date,
                    company_limit=request.company_limit,
                )
                validate_universe(universe.members, request.company_limit)
                atomic_write_jsonl(universe_path, universe.members)
            for ticker in universe.unmapped_tickers:
                logger.warning("ticker_mapping_missing", ticker=ticker)
            logger.info("universe_built", companies=len(universe.members))

            prior_runs = load_usable_runs(runs_dir, request.company_limit)
            plans, exited_ciks = classify_membership(universe.members, prior_runs)
            manifest = load_manifest(manifest_path)
            downloader = downloader_factory(
                sec_company_name,
                sec_email,
                source_root,
            )
            for plan in plans:
                sync_from = plan.sync_from or _years_before(
                    request.as_of_date, request.lookback_years
                )
                with log_context(cik=plan.cik):
                    with timed_operation("company.sync"):
                        result, company_bytes, manifest = sync_company(
                            plan=plan,
                            request=request,
                            sync_from=sync_from,
                            downloader=downloader,
                            source_root=source_root,
                            data_dir=data_dir,
                            manifest_path=manifest_path,
                            manifest=manifest,
                            run_id=run_id,
                        )
                company_results.append(result)
                total_bytes += company_bytes

            for cik in exited_ciks:
                company_results.append(
                    CompanySyncResult(
                        cik=cik,
                        membership="exited",
                        sync_to=request.as_of_date,
                    )
                )

            total_failed = sum(result.failed for result in company_results)
            run = IngestionRunRecord(
                run_id=run_id,
                request=request,
                universe_snapshot_id=snapshot_id,
                input_hashes=input_hashes,
                status="partial" if total_failed else "completed",
                started_at=started_at,
                companies=tuple(company_results),
                downloaded=sum(result.downloaded for result in company_results),
                skipped=sum(result.skipped for result in company_results),
                failed=total_failed,
                downloaded_bytes=total_bytes,
            )
            with timed_operation("quality.check"):
                validate_nonempty_files((ivv_path, ticker_path, universe_path))
                validate_manifest(manifest, data_dir)
                run = run.model_copy(
                    update={
                        "finished_at": datetime.now(UTC),
                        "duration_seconds": perf_counter() - started_clock,
                    }
                )
                validate_run_totals(run)
            atomic_write_json(run_path, run)

        logger.info(
            "run_completed",
            status=run.status,
            downloaded=run.downloaded,
            skipped=run.skipped,
            failed=run.failed,
            downloaded_bytes=run.downloaded_bytes,
            duration_seconds=run.duration_seconds,
        )
        return run
    except Exception:
        failed_run = IngestionRunRecord(
            run_id=run_id,
            request=request,
            universe_snapshot_id=snapshot_id,
            input_hashes=input_hashes,
            status="failed",
            started_at=started_at,
            finished_at=datetime.now(UTC),
            duration_seconds=perf_counter() - started_clock,
            companies=tuple(company_results),
            downloaded=sum(result.downloaded for result in company_results),
            skipped=sum(result.skipped for result in company_results),
            failed=sum(result.failed for result in company_results),
            downloaded_bytes=total_bytes,
        )
        atomic_write_json(run_path, failed_run, overwrite=True)
        logger.exception("run_failed", run_record=str(run_path))
        raise
    finally:
        clear_log_context()

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
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
    FilingManifestEntry,
    IngestionRequest,
    IngestionRunRecord,
    RunStatus,
    UniverseMember,
)
from src.data_pipeline.pipelines.sec_state import (
    classify_membership,
    load_manifest,
    load_usable_runs,
)
from src.data_pipeline.pipelines.sec_sync import SecSyncContext, sync_company
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


@dataclass(frozen=True)
class SourceAdapters:
    """External sources a run reads from; tests replace them with fakes."""

    ivv_fetcher: IVVFetcher = download_ivv_holdings
    ticker_fetcher: TickerFetcher = download_company_tickers
    downloader_factory: DownloaderFactory = create_downloader


LIVE_SOURCES = SourceAdapters()


@dataclass(frozen=True)
class _RunContext:
    """Fixed inputs and Bronze locations of one ingestion run."""

    request: IngestionRequest
    data_dir: Path
    sec_company_name: str
    sec_email: str
    sources: SourceAdapters
    run_id: str
    started_at: datetime
    started_clock: float

    @property
    def source_root(self) -> Path:
        return self.data_dir / "bronze" / "sec_filings"

    @property
    def runs_dir(self) -> Path:
        return self.source_root / "_metadata" / "runs"

    @property
    def manifest_path(self) -> Path:
        return self.source_root / "_metadata" / "filing_manifest.jsonl"

    @property
    def run_path(self) -> Path:
        return self.runs_dir / f"run_{self.run_id}.json"


@dataclass
class _RunProgress:
    """Results accumulated during a run; also recorded when the run fails."""

    snapshot_id: str = ""
    input_hashes: dict[str, str] = field(default_factory=dict)
    company_results: list[CompanySyncResult] = field(default_factory=list)
    total_bytes: int = 0


@dataclass(frozen=True)
class _UniverseSnapshot:
    """Universe members and the snapshot files they were built from."""

    members: tuple[UniverseMember, ...]
    files: tuple[Path, ...]


def _run_record(
    context: _RunContext,
    progress: _RunProgress,
    status: RunStatus,
    finished_at: datetime | None = None,
    duration_seconds: float | None = None,
) -> IngestionRunRecord:
    results = progress.company_results
    return IngestionRunRecord(
        run_id=context.run_id,
        request=context.request,
        universe_snapshot_id=progress.snapshot_id,
        input_hashes=progress.input_hashes,
        status=status,
        started_at=context.started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        companies=tuple(results),
        downloaded=sum(result.downloaded for result in results),
        skipped=sum(result.skipped for result in results),
        failed=sum(result.failed for result in results),
        downloaded_bytes=progress.total_bytes,
    )


def _fetch_sources(context: _RunContext, progress: _RunProgress) -> tuple[bytes, bytes]:
    with timed_operation("universe.fetch"):
        ivv_bytes = context.sources.ivv_fetcher()
        progress.input_hashes["ivv_holdings.csv"] = sha256_bytes(ivv_bytes)
        ticker_bytes = context.sources.ticker_fetcher(
            context.sec_company_name, context.sec_email
        )
        progress.input_hashes["company_tickers.json"] = sha256_bytes(ticker_bytes)
    progress.snapshot_id = sha256_bytes(
        f"{progress.input_hashes['ivv_holdings.csv']}:{progress.input_hashes['company_tickers.json']}".encode()
    )
    bind_log_context(snapshot_id=progress.snapshot_id)
    return ivv_bytes, ticker_bytes


def _write_universe_snapshot(
    context: _RunContext,
    progress: _RunProgress,
    ivv_bytes: bytes,
    ticker_bytes: bytes,
) -> _UniverseSnapshot:
    snapshot_folder = (
        context.source_root
        / "universe_snapshots"
        / (f"snapshot_date_{context.started_at.strftime('%Y%m%dT%H%M%S%fZ')}")
    )
    ivv_path = snapshot_folder / "ivv_holdings.csv"
    ticker_path = snapshot_folder / "company_tickers.json"
    universe_path = snapshot_folder / "universe.jsonl"
    atomic_write_bytes(ivv_path, ivv_bytes)
    atomic_write_bytes(ticker_path, ticker_bytes)
    logger.info("sources_downloaded", snapshot_folder=str(snapshot_folder))

    company_limit = context.request.company_limit
    with timed_operation("universe.build"):
        holdings = parse_ivv_holdings(ivv_bytes)
        ticker_mapping = parse_company_tickers(ticker_bytes)
        universe = build_sec_universe(
            holdings.rows,
            ticker_mapping,
            snapshot_id=progress.snapshot_id,
            holdings_as_of_date=holdings.holdings_as_of_date,
            company_limit=company_limit,
        )
        validate_universe(universe.members, company_limit)
        atomic_write_jsonl(universe_path, universe.members)
    for ticker in universe.unmapped_tickers:
        logger.warning("ticker_mapping_missing", ticker=ticker)
    logger.info("universe_built", companies=len(universe.members))
    return _UniverseSnapshot(
        members=universe.members, files=(ivv_path, ticker_path, universe_path)
    )


def _sync_companies(
    context: _RunContext,
    progress: _RunProgress,
    members: tuple[UniverseMember, ...],
) -> tuple[FilingManifestEntry, ...]:
    """Sync current members, record exited ones, and return the updated manifest."""

    request = context.request
    prior_runs = load_usable_runs(context.runs_dir, request.company_limit)
    plans, exited_ciks = classify_membership(members, prior_runs)
    manifest = load_manifest(context.manifest_path)
    sync_context = SecSyncContext(
        request=request,
        downloader=context.sources.downloader_factory(
            context.sec_company_name,
            context.sec_email,
            context.source_root,
        ),
        source_root=context.source_root,
        data_dir=context.data_dir,
        manifest_path=context.manifest_path,
        run_id=context.run_id,
    )
    for plan in plans:
        sync_from = plan.sync_from or _years_before(
            request.as_of_date, request.lookback_years
        )
        with log_context(cik=plan.cik), timed_operation("company.sync"):
            result, company_bytes, manifest = sync_company(
                plan=plan,
                sync_from=sync_from,
                context=sync_context,
                manifest=manifest,
            )
        progress.company_results.append(result)
        progress.total_bytes += company_bytes

    for cik in exited_ciks:
        progress.company_results.append(
            CompanySyncResult(
                cik=cik,
                membership="exited",
                sync_to=request.as_of_date,
            )
        )
    return manifest


def _checked_run_record(
    context: _RunContext,
    progress: _RunProgress,
    snapshot: _UniverseSnapshot,
    manifest: tuple[FilingManifestEntry, ...],
) -> IngestionRunRecord:
    total_failed = sum(result.failed for result in progress.company_results)
    run = _run_record(context, progress, "partial" if total_failed else "completed")
    with timed_operation("quality.check"):
        validate_nonempty_files(snapshot.files)
        validate_manifest(manifest, context.data_dir)
        run = run.model_copy(
            update={
                "finished_at": datetime.now(UTC),
                "duration_seconds": perf_counter() - context.started_clock,
            }
        )
        validate_run_totals(run)
    return run


def run_sec_edgar(
    request: IngestionRequest,
    data_dir: Path,
    sec_company_name: str,
    sec_email: str,
    *,
    sources: SourceAdapters = LIVE_SOURCES,
) -> IngestionRunRecord:
    """Ingest the ranked IVV company universe and SEC filings into Bronze."""

    sec_user_agent(sec_company_name, sec_email)
    if not 1 <= request.company_limit <= 500:
        raise ValueError("company_limit must be between 1 and 500")

    context = _RunContext(
        request=request,
        data_dir=data_dir.expanduser().resolve(),
        sec_company_name=sec_company_name,
        sec_email=sec_email,
        sources=sources,
        run_id=uuid4().hex,
        started_at=datetime.now(UTC),
        started_clock=perf_counter(),
    )
    progress = _RunProgress()

    bind_log_context(
        run_id=context.run_id,
        source="sec_filings",
        company_limit=request.company_limit,
        as_of_date=request.as_of_date.isoformat(),
    )
    logger.info("run_started")
    try:
        with timed_operation("pipeline.total"):
            ivv_bytes, ticker_bytes = _fetch_sources(context, progress)
            snapshot = _write_universe_snapshot(
                context, progress, ivv_bytes, ticker_bytes
            )
            manifest = _sync_companies(context, progress, snapshot.members)
            run = _checked_run_record(context, progress, snapshot, manifest)
            atomic_write_json(context.run_path, run)

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
        failed_run = _run_record(
            context,
            progress,
            "failed",
            finished_at=datetime.now(UTC),
            duration_seconds=perf_counter() - context.started_clock,
        )
        atomic_write_json(context.run_path, failed_run, overwrite=True)
        logger.exception("run_failed", run_record=str(context.run_path))
        raise
    finally:
        clear_log_context()

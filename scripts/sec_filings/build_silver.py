from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from dotenv import load_dotenv

from src.data_pipeline.connectors.sec_filings.sec_edgar import sec_user_agent
from src.data_pipeline.pipelines.sec_filings.bronze_to_silver import (
    run_sec_filings_silver,
)
from src.data_pipeline.pipelines.sec_filings.silver_stages import (
    SILVER_STAGES,
    select_stages,
)
from src.data_pipeline.quality.sec_filings.silver_audit import (
    AuditOptions,
    audit_silver,
)
from src.data_pipeline.transforms.sec_filings.xbrl_processor import ProcessorSettings
from src.utils.logger import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse the SEC Silver build CLI; invalid combinations exit with code 2."""

    names = [stage.name for stage in SILVER_STAGES]
    parser = argparse.ArgumentParser(
        description=(
            "Build the sec_filings Iceberg tables from SEC Bronze, then audit "
            "them read-only"
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data",
        help="Silver root: catalog and tables (default: <repository_root>/data)",
    )
    parser.add_argument(
        "--bronze-dir",
        type=Path,
        default=None,
        help="SEC Bronze directory (sec-edgar-filings/, _metadata/) "
        "(default: <data-dir>/bronze/sec_filings)",
    )
    parser.add_argument(
        "--taxonomy-cache-dir",
        type=Path,
        default=None,
        help="XBRL taxonomy cache (default: <data-dir>/_cache/arelle)",
    )
    parser.add_argument(
        "--allow-taxonomy-network",
        action="store_true",
        help=(
            "Download taxonomy files missing from the cache, identifying with "
            "SEC_COMPANY_NAME and SEC_EMAIL from .env (default: offline)"
        ),
    )
    parser.add_argument(
        "--from-stage",
        choices=names,
        help="Start at this stage; the tables of the stages it needs must exist",
    )
    parser.add_argument(
        "--to-stage", choices=names, help="Stop after this stage (default: integrity)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only audit the existing Silver tables; write nothing",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Print the stages in run order with their dependencies and tables",
    )
    args = parser.parse_args()
    if args.check_only and (
        args.from_stage or args.to_stage or args.allow_taxonomy_network
    ):
        parser.error("--check-only takes no stage range or taxonomy download")
    try:
        select_stages(SILVER_STAGES, args.from_stage, args.to_stage)
    except ValueError as error:
        parser.error(str(error))
    return args


def _taxonomy(args: argparse.Namespace) -> ProcessorSettings | None:
    """The XBRL taxonomy settings the options ask for; None for the default."""

    if args.taxonomy_cache_dir is None and not args.allow_taxonomy_network:
        return None
    user_agent = None
    if args.allow_taxonomy_network:
        # Taxonomy hosts include sec.gov, which requires a declared identity.
        load_dotenv(REPOSITORY_ROOT / ".env")
        try:
            user_agent = sec_user_agent(
                os.getenv("SEC_COMPANY_NAME", ""), os.getenv("SEC_EMAIL", "")
            )
        except ValueError as error:
            raise SystemExit(
                f"{error}; add them to .env or the environment to download taxonomies"
            ) from error
    data_dir = args.data_dir.expanduser().resolve()
    return ProcessorSettings(
        cache_dir=(args.taxonomy_cache_dir or data_dir / "_cache" / "arelle")
        .expanduser()
        .resolve(),
        allow_network=args.allow_taxonomy_network,
        http_user_agent=user_agent,
    )


def _print_stages() -> None:
    print(f"{'#':>2}  {'stage':<16} {'dependencies':<52} owned tables")
    for order, stage in enumerate(SILVER_STAGES, start=1):
        dependencies = ", ".join(stage.dependencies) or "-"
        tables = ", ".join(spec.identifier for spec in stage.owned_tables) or "-"
        if len(dependencies) > 52:
            dependencies = "every table stage"
        print(f"{order:>2}  {stage.name:<16} {dependencies:<52} {tables}")


def _print_summary(summary: Mapping[str, Any]) -> None:
    """Print the summary fields worth reading at a terminal."""

    for name in (
        "status",
        "run_id",
        "duration_seconds",
        "stages_completed",
        "stages_skipped",
        "failed_stage",
        "prerequisite_problems",
        "tables_created",
        "rows_inserted",
        "rows_skipped",
        "snapshot_changes",
        "tables_discovered",
        "total_rows_scanned",
        "integrity_checks_run",
        "integrity_checks_passed",
        "integrity_checks_failed",
        "checks_passed",
        "checks_warned",
        "checks_failed",
        "checks_skipped",
    ):
        if name in summary:
            print(f"{name}: {summary[name]}")
    for label, key in (("failure", "failures"), ("warning", "warnings")):
        for check in summary.get(key) or summary.get(f"integrity_{key}") or []:
            print(f"{label}: {check}")


def main() -> int:
    """Run the pipeline, or only its audit.

    Exit 0 on success, 1 when a stage is rejected or the audit finds errors,
    2 on invalid arguments or, with --check-only, a missing Silver catalog.
    """

    args = parse_args()
    if args.list_stages:
        _print_stages()
        return 0
    configure_logging()
    if args.check_only:
        try:
            report = audit_silver(
                args.data_dir, AuditOptions(bronze_dir=args.bronze_dir)
            )
        except FileNotFoundError:  # no catalog: logged as silver_integrity_failed
            return 2
        _print_summary(report.summary())
        return 1 if report.status == "FAIL" else 0
    result = run_sec_filings_silver(
        args.data_dir,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        taxonomy=_taxonomy(args),
        bronze_dir=args.bronze_dir,
    )
    _print_summary(result.summary())
    return 0 if result.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())

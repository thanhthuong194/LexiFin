from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_pipeline.quality.sec_filings.silver_audit import (
    AuditOptions,
    audit_silver,
)
from src.data_pipeline.quality.sec_filings.silver_audit_report import (
    SilverAuditReport,
)
from src.utils.logger import configure_logging


def _count(minimum: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}")
        return value

    return parse


def parse_args() -> argparse.Namespace:
    """Parse the SEC Silver audit CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit the sec_filings Iceberg tables read-only: inventory, contracts, "
            "keys, references, groups, hierarchy and sampled Bronze lineage"
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
        help="SEC Bronze directory (default: <data-dir>/bronze/sec_filings)",
    )
    parser.add_argument(
        "--sample-per-form",
        type=_count(0),
        default=1,
        help="Latest filings of each form (10-K, 10-Q, 8-K) traced to Bronze (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=_count(1),
        default=10_000,
        help="Rows read and validated at a time (default: 10000)",
    )
    parser.add_argument(
        "--verify-source-hashes",
        action="store_true",
        help="Also re-hash every filing's Bronze files and stored byte spans",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Keep the scratch database here, so an interrupted audit resumes "
        "when run again (default: a temporary directory)",
    )
    parser.add_argument(
        "--workers",
        type=_count(1),
        default=1,
        help="Processes validating rows against their contracts (default: 1)",
    )
    return parser.parse_args()


def _print_report(report: SilverAuditReport) -> None:
    print(f"\nSEC Silver audit: {report.status} ({report.data_dir})")
    print("\nTables:")
    for table in report.tables.values():
        print(
            f"  {report.table_status(table.identifier):<4} {table.identifier:<36} "
            f"{table.row_count:>9,} rows {table.data_file_count:>3} files "
            f"({table.data_file_bytes / 2**20:.1f} MiB), "
            f"snapshot {table.snapshot_id} ({table.snapshot_count} in history), "
            f"schema {table.schema_id}, {', '.join(table.partition_spec)}, "
            f"{table.sort_order}"
        )
    print("\nChecks (status, table, check: failed of checked):")
    for check in report.checks:
        print(
            f"  {check.status:<4} {check.table or '-'} {check.check_name}: "
            f"{check.failed_count:,} of {check.rows_checked:,}"
        )
        if check.status in ("FAIL", "WARN", "SKIP"):
            print(f"       {check.message}")
        if check.sample_ids:
            print(f"       samples: {', '.join(check.sample_ids)}")
    print("\nSampled filings:")
    for sample in report.samples:
        print(f"  [{sample.form}] {sample.filing_id}")
        for line in sample.lines:
            print(f"    {line}")
    print("\nSummary:")
    for key, value in report.summary().items():
        print(f"  {key}: {value}")


def main() -> int:
    """Exit 0 without FAIL checks, 1 with one, 2 when the audit cannot run."""

    args = parse_args()
    configure_logging()
    try:
        report = audit_silver(
            args.data_dir,
            AuditOptions(
                samples_per_form=args.sample_per_form,
                batch_size=args.batch_size,
                verify_source_hashes=args.verify_source_hashes,
                bronze_dir=args.bronze_dir,
                work_dir=args.work_dir,
                workers=args.workers,
            ),
        )
    except Exception:  # noqa: BLE001 - logged as silver_integrity_failed; exit code 2 reports it
        return 2
    _print_report(report)
    return 1 if report.status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

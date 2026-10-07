from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_pipeline.quality.sec_filings.silver_audit import validate_silver
from src.utils.logger import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse the SEC Silver audit CLI."""

    parser = argparse.ArgumentParser(
        description="Audit the sec_filings Iceberg tables without writing to them"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data",
        help="Data root (default: <repository_root>/data)",
    )
    parser.add_argument(
        "--verify-source-hashes",
        action="store_true",
        help="Also re-hash the Bronze submissions and every stored byte span",
    )
    return parser.parse_args()


def main() -> int:
    """Exit 0 when the audit passes, 1 on validation errors, 2 when it cannot run."""

    args = parse_args()
    configure_logging()
    try:
        report = validate_silver(
            args.data_dir, verify_source_hashes=args.verify_source_hashes
        )
    except Exception:  # noqa: BLE001 - logged as silver_integrity_failed; exit code 2 reports it
        return 2
    return 0 if report.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

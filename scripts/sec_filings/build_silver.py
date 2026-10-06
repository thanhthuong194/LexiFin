from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_pipeline.pipelines.sec_filings.bronze_to_silver import (
    run_sec_filings_silver,
)
from src.utils.logger import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse the SEC Silver build CLI."""

    parser = argparse.ArgumentParser(
        description="Build the sec_filings Iceberg tables from SEC Bronze"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data",
        help="Data root (default: <repository_root>/data)",
    )
    return parser.parse_args()


def main() -> int:
    """Run one Bronze -> Silver batch; exit 1 when any stage is rejected."""

    args = parse_args()
    configure_logging()
    return 0 if run_sec_filings_silver(args.data_dir).completed else 1


if __name__ == "__main__":
    raise SystemExit(main())

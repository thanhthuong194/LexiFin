from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from dotenv import load_dotenv

from src.data_pipeline.connectors.sec_filings.sec_edgar import sec_user_agent
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
    return parser.parse_args()


def main() -> int:
    """Run one Bronze -> Silver batch; exit 1 when any stage is rejected."""

    args = parse_args()
    configure_logging()
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
    result = run_sec_filings_silver(
        args.data_dir,
        taxonomy_cache_dir=args.taxonomy_cache_dir,
        allow_taxonomy_network=args.allow_taxonomy_network,
        http_user_agent=user_agent,
    )
    return 0 if result.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())

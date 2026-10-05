from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from dotenv import load_dotenv

from src.data_pipeline.contracts.ingestion import IngestionRequest
from src.data_pipeline.pipelines.sec_edgar import run_sec_edgar
from src.utils.logger import configure_logging, get_logger


def _company_limit(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 500:
        raise argparse.ArgumentTypeError("company-limit must be between 1 and 500")
    return parsed


def _positive_years(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("lookback-years must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse the minimal SEC Bronze ingestion CLI."""

    parser = argparse.ArgumentParser(description="Ingest SEC filings into Bronze")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data",
        help="Data root (default: <repository_root>/data)",
    )
    parser.add_argument("--company-limit", type=_company_limit, default=5)
    parser.add_argument("--lookback-years", type=_positive_years, default=5)
    return parser.parse_args()


def main() -> int:
    """Load runtime configuration and execute one SEC ingestion run."""

    args = parse_args()
    configure_logging()
    logger = get_logger(__name__)
    load_dotenv(REPOSITORY_ROOT / ".env")

    company_name = os.getenv("SEC_COMPANY_NAME", "").strip()
    email = os.getenv("SEC_EMAIL", "").strip()
    if not company_name or not email:
        raise SystemExit(
            "SEC_COMPANY_NAME and SEC_EMAIL are required; add them to .env or the environment"
        )

    as_of_date = datetime.now(UTC).date()
    request = IngestionRequest(
        company_limit=args.company_limit,
        as_of_date=as_of_date,
        lookback_years=args.lookback_years,
    )
    try:
        run = run_sec_edgar(
            request=request,
            data_dir=args.data_dir,
            sec_company_name=company_name,
            sec_email=email,
        )
    except Exception as error:
        logger.exception("sec_ingestion_failed", error=str(error))
        return 1
    return 0 if run.status in {"completed", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

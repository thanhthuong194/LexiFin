from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.llm import get_embedder
from src.rag_engine.embedding.audit import audit_embeddings
from src.rag_engine.embedding.pipeline import run_native_rag_embeddings
from src.utils.logger import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed native_rag.chunks into native_rag.embeddings (Gold)"
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data",
        help="Gold root: _catalog/gold.db, gold/ (default: <repo>/data)",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Embedding profile in config/llm_profiles.toml (default: default)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Request groups sent at once to the embedding server (default 4)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Embed at most this many chunks"
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Only audit (read-only)"
    )
    parser.add_argument(
        "--skip-audit", action="store_true", help="Embed without the audit after it"
    )
    parser.add_argument(
        "--reembed-sample",
        type=int,
        default=5,
        help="Chunks the audit re-embeds to compare with stored vectors (default 5)",
    )
    return parser.parse_args()


def main() -> int:
    """Exit 0 when the run completed and the audit has no FAIL, else 1."""

    args = parse_args()
    configure_logging()
    # The OpenAI client logs one INFO line per HTTP request: it would bury the
    # run's own logs.
    for name in ("openai", "httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
    embedder = get_embedder(args.profile)
    if not args.check_only:
        summary = run_native_rag_embeddings(
            args.gold_dir,
            embedder,
            datetime.now(UTC),
            concurrency=args.concurrency,
            limit=args.limit,
        )
        print(json.dumps(asdict(summary), indent=2))
        if summary.status != "completed":
            return 1
        if args.skip_audit:
            return 0
    report = audit_embeddings(
        args.gold_dir, embedder.model, embedder=embedder, sample=args.reembed_sample
    )
    for check in report.checks:
        print(f"{check.status:4} {check.name}: {check.detail}")
        for sample in check.samples:
            print(f"       {sample}")
    print(f"audit: {report.status}")
    return 1 if report.status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

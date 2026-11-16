from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_pipeline.pipelines.native_rag.chunks import run_native_rag_chunks
from src.data_pipeline.quality.native_rag.audit import AuditReport, audit_chunks
from src.data_pipeline.transforms.native_rag.tokenizer import (
    QwenTokenCounter,
    tokenizer_path,
)
from src.utils.logger import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build native_rag.chunks (Gold) from the Silver narrative"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data",
        help="Silver root: _catalog/silver.db, silver/ (default: <repo>/data)",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=None,
        help="Gold root: _catalog/gold.db, gold/ (default: --data-dir)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Pinned Qwen3-Embedding tokenizer.json "
        "(default: <data-dir>/_cache/models/qwen3-embedding-0.6b/tokenizer.json)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only audit the stored chunks (read-only)",
    )
    parser.add_argument(
        "--skip-audit", action="store_true", help="Build without the audit after it"
    )
    parser.add_argument(
        "--sample-documents",
        type=int,
        default=5,
        help="Documents rebuilt exactly by the audit besides the newest of each form",
    )
    return parser.parse_args()


def print_report(report: AuditReport) -> None:
    for check in report.checks:
        print(f"{check.status:4} {check.name}: {check.detail}")
        for sample in check.samples:
            print(f"       {sample}")
    print(f"audit: {report.status}")


def main() -> int:
    """Exit 0 when the build completed and the audit has no FAIL, else 1."""

    args = parse_args()
    configure_logging()
    counter = QwenTokenCounter(args.tokenizer or tokenizer_path(args.data_dir))
    if not args.check_only:
        summary = run_native_rag_chunks(
            args.data_dir, counter, datetime.now(UTC), gold_dir=args.gold_dir
        )
        print(json.dumps(asdict(summary), indent=2))
        if summary.status != "completed":
            return 1
        if args.skip_audit:
            return 0
    report = audit_chunks(
        args.data_dir,
        counter,
        gold_dir=args.gold_dir,
        sample_documents=args.sample_documents,
    )
    print_report(report)
    return 1 if report.status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

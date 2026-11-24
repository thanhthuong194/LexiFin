from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evals.sampling import (
    SampleSettings,
    coverage_summary,
    sample_tasks,
    split_batches,
)
from src.evals.testset import SetManifest, dump_json, write_jsonl
from src.utils.logger import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample eval tasks (passages + XBRL facts) from Gold and Silver"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data",
        help="Silver root: _catalog/silver.db, silver/ (default: <repo>/data)",
    )
    parser.add_argument("--gold-dir", type=Path, help="Gold root (default: --data-dir)")
    parser.add_argument("--set-id", default="v1")
    parser.add_argument(
        "--out-dir", type=Path, help="Default: <repo>/evals/sets/<set-id>"
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        help="Date relative questions are read against (default: today, UTC)",
    )
    parser.add_argument("--total", type=int, default=SampleSettings.total)
    parser.add_argument("--seed", type=int, default=SampleSettings.seed)
    parser.add_argument("--batches", type=int, default=6, help="Writer batches")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=SampleSettings.max_candidates,
        help="Chunks whose metadata the first pass keeps (hash sample)",
    )
    parser.add_argument(
        "--max-fact-instances",
        type=int,
        default=SampleSettings.max_fact_instances,
        help="XBRL instances whose facts are read for numeric tasks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.total < 1 or args.batches < 1:
        print("--total and --batches must be positive", file=sys.stderr)
        return 2
    configure_logging()
    settings = SampleSettings(
        total=args.total,
        seed=args.seed,
        max_candidates=args.max_candidates,
        max_fact_instances=args.max_fact_instances,
    )
    tasks, rules_version = sample_tasks(
        args.data_dir, args.gold_dir or args.data_dir, settings
    )
    out: Path = args.out_dir or REPOSITORY_ROOT / "evals" / "sets" / args.set_id
    manifest = SetManifest(
        set_id=args.set_id,
        as_of=args.as_of or datetime.now(UTC).date(),
        seed=args.seed,
        total=len(tasks),
        chunks_rules_version=rules_version,
    )
    dump_json(out / "set.json", manifest.model_dump(mode="json"))
    write_jsonl(out / "tasks.jsonl", tasks)
    for number, batch in enumerate(split_batches(tasks, args.batches), start=1):
        write_jsonl(out / "batches" / f"batch_{number}.jsonl", batch)
    summary = {"tasks": len(tasks), "seed": args.seed, **coverage_summary(tasks)}
    dump_json(out / "coverage.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evals.sampling import (
    PLANS,
    SampleSettings,
    coverage_summary,
    read_exclusion,
    sample_tasks,
    split_batches,
    write_tasks,
)
from src.evals.testset import SetManifest, dump_json
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
    parser.add_argument(
        "--bronze-dir",
        type=Path,
        help="SEC Bronze (universe snapshots, filings) read by --plan v2 "
        "(default: <data-dir>/bronze/sec_filings)",
    )
    parser.add_argument(
        "--plan",
        choices=sorted(PLANS),
        default="v1",
        help="v1: 9 types, 100 tasks; v2: 16 types with minimums, sector and "
        "size strata, tags and hard negatives (use --total 150); v2dev: v2 "
        "inputs, exact quotas of weak types for dev-only tasks (--total 120)",
    )
    parser.add_argument(
        "--exclude-set",
        type=Path,
        help="Earlier set dir: skip companies of its test questions, every "
        "chunk its tasks show or list as hard negatives, its outside companies",
    )
    parser.add_argument(
        "--task-prefix", default="t", help="Task IDs <prefix>0001... (default: t)"
    )
    parser.add_argument(
        "--split", choices=("dev", "test"), help="Fixed split of every task"
    )
    parser.add_argument(
        "--min-relative",
        type=int,
        help="relative_time tasks to aim for (default: the plan's, 20 of 120 "
        "for v2dev, else 0)",
    )
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
    out: Path = args.out_dir or REPOSITORY_ROOT / "evals" / "sets" / args.set_id
    exclude_dir: Path | None = args.exclude_set
    if exclude_dir is not None and out.resolve() == exclude_dir.resolve():
        print("--out-dir must differ from --exclude-set", file=sys.stderr)
        return 2
    configure_logging()
    as_of: date = args.as_of or datetime.now(UTC).date()
    settings = SampleSettings(
        total=args.total,
        seed=args.seed,
        max_candidates=args.max_candidates,
        max_fact_instances=args.max_fact_instances,
        plan_id=args.plan,
        bronze_dir=args.bronze_dir,
        as_of=as_of,
        exclude=read_exclusion(exclude_dir) if exclude_dir is not None else None,
        task_prefix=args.task_prefix,
        split=args.split,
        min_relative=args.min_relative,
    )
    tasks, rules_version = sample_tasks(
        args.data_dir, args.gold_dir or args.data_dir, settings
    )
    manifest = SetManifest(
        set_id=args.set_id,
        as_of=as_of,
        seed=args.seed,
        total=len(tasks),
        chunks_rules_version=rules_version,
    )
    dump_json(out / "set.json", manifest.model_dump(mode="json"))
    write_tasks(out / "tasks.jsonl", tasks)
    for number, batch in enumerate(split_batches(tasks, args.batches), start=1):
        write_tasks(out / "batches" / f"batch_{number}.jsonl", batch)
    summary = {"tasks": len(tasks), "seed": args.seed, **coverage_summary(tasks)}
    dump_json(out / "coverage.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

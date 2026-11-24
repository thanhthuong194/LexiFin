"""Merge writer drafts with sampled tasks, check them, write the question set.

Reads `<set-dir>/tasks.jsonl` and `<set-dir>/drafts/*.jsonl`, maps evidence
quotes to Silver blocks and checks them against current Gold chunks, writes
`questions.jsonl` (only without errors) and `check_report.md`.
Exit 0 = written (warnings allowed), 1 = errors, 2 = bad arguments.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pydantic import ValidationError

from src.evals.checks import (
    check_testset,
    load_blocks,
    load_chunks,
    load_company_chunks,
    load_drafts,
    read_blocks,
    render_report,
    widen_evidence,
)
from src.evals.testset import EvalTask, SetManifest, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check drafts and build questions")
    parser.add_argument(
        "--set-dir", type=Path, default=REPOSITORY_ROOT / "evals" / "sets" / "v1.1"
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Silver root")
    parser.add_argument("--gold-dir", type=Path, help="Gold root (default: --data-dir)")
    parser.add_argument("--set-id", help="Default: set.json, else the set dir name")
    parser.add_argument("--tasks", type=Path, help="Default: <set-dir>/tasks.jsonl")
    parser.add_argument("--drafts", type=Path, nargs="+", help="Files or dirs")
    parser.add_argument("--out", type=Path, help="Default: <set-dir>/questions.jsonl")
    parser.add_argument("--report", type=Path, help="Default: check_report.md by --out")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_dir: Path = args.set_dir
    tasks_path: Path = args.tasks or set_dir / "tasks.jsonl"
    draft_paths: list[Path] = args.drafts or [set_dir / "drafts"]
    out: Path = args.out or set_dir / "questions.jsonl"
    report: Path = args.report or out.parent / "check_report.md"
    gold_dir: Path = args.gold_dir or args.data_dir
    missing = [
        p for p in (tasks_path, *draft_paths, args.data_dir, gold_dir) if not p.exists()
    ]
    if missing:
        print(f"Not found: {', '.join(map(str, missing))}", file=sys.stderr)
        return 2
    manifest = SetManifest.read(set_dir)
    set_id = args.set_id or (manifest.set_id if manifest else set_dir.name)
    try:
        tasks = read_jsonl(tasks_path, EvalTask)
    except ValidationError as error:
        print(f"Invalid tasks file {tasks_path}: {error}", file=sys.stderr)
        return 1
    drafts, draft_errors = load_drafts(draft_paths)
    blocks = load_blocks(args.data_dir, tasks)
    chunks = load_chunks(gold_dir, tasks)
    result = check_testset(set_id, tasks, drafts, blocks, chunks)
    widen_evidence(
        result,
        tasks,
        partial(load_company_chunks, gold_dir),
        partial(read_blocks, args.data_dir),
    )
    result.errors[:0] = draft_errors

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(result), encoding="utf-8")
    print(
        f"{len(tasks)} tasks, {len(drafts)} drafts: {len(result.questions)} questions, "
        f"{len(result.errors)} errors, {len(result.warnings)} warnings, "
        f"{sum(result.skipped.values())} skipped; report {report}"
    )
    for issue in result.errors[:20]:
        print(f"ERROR {issue.task_id} {issue.check}: {issue.message}")
    if result.errors:
        print("Questions not written.")
        return 1
    count = write_jsonl(out, result.questions)
    print(f"Wrote {count} questions to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

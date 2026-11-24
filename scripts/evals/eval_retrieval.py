"""Score retrieval against a frozen question set, or compare scored runs.

Run:     eval_retrieval.py [--questions Q] [--resume RUN_DIR]
Compare: eval_retrieval.py --compare RUN_DIR RUN_DIR ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_pipeline.storage.bronze import sha256_file
from src.evals.retrieval_eval import (
    HEADLINE,
    Probe,
    collect,
    compare_runs,
    evidence_chunks,
    failures,
    filter_admits,
    load_rows,
    means,
    pending,
    resume_rows,
    retried,
    write_outputs,
)
from src.evals.testset import (
    LANGUAGES,
    EvalQuestion,
    SetManifest,
    dump_json,
    read_jsonl,
)
from src.rag_engine.retrieval.factory import (
    add_retriever_arguments,
    retriever_from_args,
)
from src.rag_engine.retrieval.retriever import RetrieverSettings
from src.utils.logger import configure_logging

EVALS = REPOSITORY_ROOT / "evals"
# Config entries a resumed run must keep.
RESUME_KEYS = (
    "languages",
    "limit",
    "today",
    "chunks_rules_version",
    "profiles",
    "models",
    "settings",
    "collections",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score retrieval against a frozen question set, or compare runs"
    )
    parser.add_argument(
        "--questions", type=Path, default=EVALS / "sets" / "v1.1" / "questions.jsonl"
    )
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--gold-dir", type=Path, help="Default: --data-dir")
    add_retriever_arguments(parser)
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=LANGUAGES)
    parser.add_argument("--limit", type=int, help="First N questions only")
    parser.add_argument(
        "--today", type=date.fromisoformat, help="Default: as_of of set.json"
    )
    parser.add_argument("--label", default="run")
    parser.add_argument("--out-dir", type=Path, default=EVALS / "runs")
    parser.add_argument("--resume", type=Path, help="Run directory to continue")
    parser.add_argument("--compare", type=Path, nargs="+", help="Run directories")
    parser.add_argument("--stat-test", choices=("student", "fisher"), default="student")
    parser.add_argument("--max-p", type=float, default=0.05)
    parser.add_argument("--out", type=Path, help="compare.md path")
    args = parser.parse_args()
    args.gold_dir = args.gold_dir or args.data_dir
    if not args.no_llm and args.llm_cache_dir is None:
        args.llm_cache_dir = EVALS / "cache" / "llm"
    return args


def conflicts(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    return [
        k
        for k in RESUME_KEYS
        if k in new and json.dumps(old.get(k)) != json.dumps(new[k])
    ]


def run(args: argparse.Namespace) -> int:
    questions: list[EvalQuestion] = read_jsonl(args.questions, EvalQuestion)
    questions = questions[: args.limit] if args.limit else questions
    manifest = SetManifest.read(args.questions.parent)
    today: date | None = args.today or (manifest.as_of if manifest else None)
    if today is None:
        print("no set.json next to the questions: pass --today", file=sys.stderr)
        return 2
    relevance, version = evidence_chunks(args.gold_dir, questions)
    if manifest and manifest.chunks_rules_version not in (None, version):
        print(
            f"set sampled from chunks v{manifest.chunks_rules_version}, Gold is "
            f"v{version}: unmatched evidence is listed in the report",
            file=sys.stderr,
        )
    config: dict[str, Any] = {
        "label": args.label,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "questions": {
            "path": str(args.questions),
            "sha256": sha256_file(args.questions),
            "count": len(questions),
        },
        "languages": list(args.languages),
        "limit": args.limit,
        "today": today.isoformat(),
        "chunks_rules_version": version,
        "qdrant": "path" if args.qdrant_path else "url",
        "profiles": {
            "query": None if args.no_llm else args.query_profile,
            "embedding": args.embedding_profile,
            "rerank": None if args.no_rerank else args.rerank_profile,
        },
        "settings": asdict(RetrieverSettings()),
    }
    if args.resume:
        run_dir = args.resume
        previous = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        differ = conflicts(previous, config) + (
            ["questions"]
            if previous["questions"]["sha256"] != config["questions"]["sha256"]
            else []
        )
        if differ:
            print(f"resumed run differs in {', '.join(differ)}", file=sys.stderr)
            return 2
        config = previous | {k: v for k, v in config.items() if k in RESUME_KEYS}
        done = resume_rows(run_dir / "results.jsonl")
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_dir = args.out_dir / f"{stamp}_{args.label}"
        run_dir.mkdir(parents=True)
        done = []
    todo = pending(questions, args.languages, done)
    print(f"{run_dir}: {len(todo)} to retrieve, {len(done)} done", file=sys.stderr)
    if todo:  # rescoring alone needs no server
        retriever = retriever_from_args(args, today=today)
        found = {
            "collections": {n: t.collection for n, t in retriever.targets.items()},
            "models": {
                "query": retriever.chat.model if retriever.chat else None,
                "embedding": retriever.embedder.model,
                "rerank": retriever.reranker.model if retriever.reranker else None,
            },
        }
        if args.resume and (differ := conflicts(config, found)):
            print(f"resumed run differs in {', '.join(differ)}", file=sys.stderr)
            return 2
        config |= found | {"retriever_notes": retriever.notes}
        dump_json(run_dir / "config.json", config)
        probe = Probe(
            retriever.retrieve,
            partial(filter_admits, retriever.client),
            query_model=retriever.chat is not None,
        )
        collect(probe, todo, relevance, run_dir / "results.jsonl")
    scoring = write_outputs(run_dir, config, questions, relevance)
    for language in args.languages:
        scores = means(s for s in scoring.scored if s.row["language"] == language)
        print(
            f"{language}: { {k: round(scores[k][0], 3) for k in HEADLINE if k in scores} }"
        )
    left = sum(retried(r) for r in load_rows(run_dir / "results.jsonl"))
    print(
        f"failures {len(failures(scoring))}, errors or degraded {left}; {run_dir / 'report.md'}"
    )
    return 1 if left else 0


def main() -> int:
    """Exit 0 when every pair was retrieved without degrading, 1 when a
    server error or a degraded row remains (resume to retry), 2 on bad input."""

    args = parse_args()
    configure_logging()
    for noisy in ("httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if not args.compare:
        return run(args)
    if len(args.compare) < 2:
        print("--compare needs two run directories or more", file=sys.stderr)
        return 2
    try:
        text = compare_runs(args.compare, args.stat_test, args.max_p)
    except ValueError as error:
        print(f"cannot compare: {error}", file=sys.stderr)
        return 2
    out = args.out or args.compare[0] / "compare.md"
    out.write_text(text, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

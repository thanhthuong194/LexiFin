"""Generate answers from retrieved passages for a frozen question set and score them.

Run:     eval_answers.py [--questions Q] [--judge] [--resume RUN_DIR]
Rejudge: eval_answers.py --rejudge RUN_DIR  (stored answers, new run directory)
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
from src.evals.answer_eval import (
    Judge,
    Steps,
    collect,
    rejudge,
    to_retry,
    write_outputs,
)
from src.evals.answer_judge import judge_answer
from src.evals.retrieval_eval import (
    evidence_chunks,
    load_rows,
    means,
    pending,
    resume_rows,
)
from src.evals.testset import (
    LANGUAGES,
    EvalQuestion,
    SetManifest,
    dump_json,
    read_jsonl,
)
from src.llm import get_chat
from src.rag_engine.generation.answer import generate_answer
from src.rag_engine.retrieval.factory import (
    add_retriever_arguments,
    retriever_from_args,
)
from src.rag_engine.retrieval.retriever import RetrieverSettings
from src.utils.logger import configure_logging

EVALS = REPOSITORY_ROOT / "evals"
ANSWERS = "answers.jsonl"
# Config entries a resumed run must keep.
RESUME_KEYS = (
    "languages",
    "limit",
    "today",
    "judge",
    "profiles",
    "models",
    "settings",
    "chunks_rules_version",
)
SHOWN = ("answer.value", "answer.false_refusal", "answer.citation_evidence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and score answers for a frozen question set"
    )
    parser.add_argument(
        "--questions", type=Path, default=EVALS / "sets" / "v1.2" / "questions.jsonl"
    )
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--gold-dir", type=Path, help="Default: --data-dir")
    add_retriever_arguments(parser)
    parser.add_argument("--generator-profile", default="generator")
    parser.add_argument("--judge-profile", default="judge")
    parser.add_argument("--judge", action="store_true", help="Also judge answers")
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=LANGUAGES)
    parser.add_argument("--limit", type=int, help="First N questions only")
    parser.add_argument("--label", default="run")
    parser.add_argument("--out-dir", type=Path, default=EVALS / "runs")
    runs = parser.add_mutually_exclusive_group()
    runs.add_argument("--resume", type=Path, help="Run directory to continue")
    runs.add_argument("--rejudge", type=Path, help="Run directory to judge again")
    args = parser.parse_args()
    args.gold_dir = args.gold_dir or args.data_dir
    if args.llm_cache_dir is None:
        args.llm_cache_dir = EVALS / "cache" / "llm"
    return args


def read_config(run_dir: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    return loaded


def questions_entry(path: Path, count: int) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "count": count}


def new_run_dir(args: argparse.Namespace) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir: Path = args.out_dir / f"{stamp}_answers_{args.label}"
    run_dir.mkdir(parents=True)
    return run_dir


def differences(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Config entries (and the questions' sha256) a resumed run changes."""

    changed = [
        k
        for k in RESUME_KEYS
        if k in new and json.dumps(old.get(k)) != json.dumps(new[k])
    ]
    if "questions" in new and old["questions"]["sha256"] != new["questions"]["sha256"]:
        changed.insert(0, "questions")
    return changed


def judge_step(args: argparse.Namespace) -> tuple[Judge, str]:
    chat = get_chat(args.judge_profile, cache_dir=args.llm_cache_dir)
    return partial(judge_answer, chat), chat.model


def finish(run_dir: Path, config: dict[str, Any], questions: list[EvalQuestion]) -> int:
    scored = write_outputs(run_dir, config, questions)
    for language in config["languages"]:
        found = means(s for s in scored if s.row["language"] == language)
        print(
            f"{language}: { {k: round(found[k][0], 3) for k in SHOWN if k in found} }"
        )
    rejudged = bool(config.get("rejudged_from"))
    rows = load_rows(run_dir / ANSWERS)
    left = sum(to_retry(r, rejudged=rejudged) for r in rows)
    print(f"to retry {left}; {run_dir / 'report.md'}")
    return 1 if left else 0


def run_rejudge(args: argparse.Namespace) -> int:
    source = read_config(args.rejudge)
    entry = source["questions"]
    path = Path(entry["path"])
    if not path.exists() or sha256_file(path) != entry["sha256"]:
        print(f"{path} changed since {args.rejudge}: cannot rejudge", file=sys.stderr)
        return 2
    questions: list[EvalQuestion] = read_jsonl(path, EvalQuestion)
    rows = load_rows(args.rejudge / ANSWERS)
    if any("text" not in p for r in rows for p in r.get("passages", [])):
        print(f"{args.rejudge} has no passage texts: cannot rejudge", file=sys.stderr)
        return 2
    judge, model = judge_step(args)
    run_dir = new_run_dir(args)
    config = source | {
        "label": args.label,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "judge": True,
        "rejudged_from": str(args.rejudge),
        "profiles": source["profiles"] | {"judge": args.judge_profile},
        "models": source["models"] | {"judge": model},
    }
    dump_json(run_dir / "config.json", config)
    print(f"{run_dir}: judging {len(rows)} stored rows", file=sys.stderr)
    rejudge(judge, questions, rows, run_dir / ANSWERS)
    return finish(run_dir, config, questions)


def run(args: argparse.Namespace) -> int:
    questions: list[EvalQuestion] = read_jsonl(args.questions, EvalQuestion)
    questions = questions[: args.limit] if args.limit else questions
    manifest = SetManifest.read(args.questions.parent)
    if manifest is None:
        print("no set.json next to the questions", file=sys.stderr)
        return 2
    relevance, _, version = evidence_chunks(args.gold_dir, questions)
    config: dict[str, Any] = {
        "label": args.label,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "questions": questions_entry(args.questions, len(questions)),
        "languages": list(args.languages),
        "limit": args.limit,
        "today": manifest.as_of.isoformat(),
        "judge": args.judge,
        "chunks_rules_version": version,
        "profiles": {
            "query": None if args.no_llm else args.query_profile,
            "embedding": args.embedding_profile,
            "rerank": None if args.no_rerank else args.rerank_profile,
            "generator": args.generator_profile,
            "judge": args.judge_profile if args.judge else None,
        },
        "settings": asdict(RetrieverSettings()),
    }
    if args.resume:
        run_dir = args.resume
        previous = read_config(run_dir)
        if differ := differences(previous, config):
            print(f"resumed run differs in {', '.join(differ)}", file=sys.stderr)
            return 2
        config = previous
        done = resume_rows(run_dir / ANSWERS)
    else:
        run_dir = new_run_dir(args)
        done = []
    todo = pending(questions, args.languages, done)
    print(f"{run_dir}: {len(todo)} to answer, {len(done)} done", file=sys.stderr)
    if todo:  # rescoring alone needs no server
        today = date.fromisoformat(config["today"])
        retriever = retriever_from_args(args, today=today)
        generator = get_chat(args.generator_profile, cache_dir=args.llm_cache_dir)
        judge, judge_model = judge_step(args) if args.judge else (None, None)
        models = {
            "query": retriever.chat.model if retriever.chat else None,
            "embedding": retriever.embedder.model,
            "rerank": retriever.reranker.model if retriever.reranker else None,
            "generator": generator.model,
            "judge": judge_model,
        }
        if args.resume and (differ := differences(config, {"models": models})):
            print(f"resumed run differs in {', '.join(differ)}", file=sys.stderr)
            return 2
        config |= {"models": models, "retriever_notes": retriever.notes}
        dump_json(run_dir / "config.json", config)
        steps = Steps(
            retriever.retrieve,
            partial(generate_answer, generator),
            judge,
            query_model=retriever.chat is not None,
        )
        collect(steps, todo, relevance, run_dir / ANSWERS)
    return finish(run_dir, config, questions)


def main() -> int:
    """Exit 0 when every pair was answered without degrading, 1 when an
    error or a degraded row remains (resume to retry), 2 on bad input."""

    args = parse_args()
    configure_logging()
    for noisy in ("httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return run_rejudge(args) if args.rejudge else run(args)


if __name__ == "__main__":
    raise SystemExit(main())

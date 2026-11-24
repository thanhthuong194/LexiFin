from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.rag_engine.retrieval.factory import (
    SERVER_ERRORS,
    add_retriever_arguments,
    retriever_from_args,
)
from src.rag_engine.retrieval.retriever import RetrievalResult, RetrieverSettings
from src.utils.logger import configure_logging

PREVIEW_CHARS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve cited passages for questions (Qdrant + model servers)"
    )
    parser.add_argument("questions", nargs="*", help="Questions to search")
    parser.add_argument(
        "--questions-file",
        type=Path,
        help="One question per line, or a JSON list of strings / {question: ...}",
    )
    add_retriever_arguments(parser)
    parser.add_argument("--top-k", type=int, default=RetrieverSettings.top_k)
    parser.add_argument(
        "--context-tokens", type=int, default=RetrieverSettings.context_tokens
    )
    parser.add_argument("--json", action="store_true", help="Print JSON results")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON results to this file (logs stay on stdout)",
    )
    return parser.parse_args()


def load_questions(args: argparse.Namespace) -> list[str]:
    questions = list(args.questions)
    if args.questions_file:
        raw = args.questions_file.read_text(encoding="utf-8")
        if raw.lstrip().startswith("["):
            questions += [
                item if isinstance(item, str) else item["question"]
                for item in json.loads(raw)
            ]
        else:
            questions += [line.strip() for line in raw.splitlines() if line.strip()]
    return questions


def as_json(result: RetrievalResult) -> dict[str, Any]:
    return {
        "question": result.question,
        "plan": asdict(result.plan),
        "companies": result.companies,
        "unresolved_companies": result.unresolved_companies,
        "filters": result.filters,
        "notes": result.notes,
        "timings": result.timings,
        "context_tokens": result.context_tokens,
        "passages": [p.as_dict() for p in result.passages],
        "ranking": result.ranking,
    }


def show(result: RetrievalResult) -> None:
    plan = result.plan
    print(f"\n=== {result.question}")
    print(
        f"plan ({plan.source}): language {plan.language}; "
        f"search: {dict(plan.texts) or plan.question!r}"
    )
    for corpus, found in result.filters.items():
        print(f"filters {corpus}: {'; '.join(found) or 'none'}")
    for note in result.notes:
        print(f"note: {note}")
    print(f"timings: {result.timings}; context {result.context_tokens} tokens")
    for passage in result.passages:
        period = (
            f"FY{passage.fiscal_year} {passage.fiscal_period or ''}".strip()
            if passage.fiscal_year
            else f"filed {passage.filing_date}"
        )
        print(
            f"\n[{passage.rank}] {passage.score:.4f} {passage.company_name} | "
            f"{passage.document_type} | {period} | {passage.section_title}"
        )
        print(f"    {passage.sec_url}")
        print(f"    chunks {', '.join(passage.chunk_ids)} ({passage.tokens} tokens)")
        print(f"    {passage.context}")
        preview = " ".join(passage.text.split())
        print(
            f"    {preview[:PREVIEW_CHARS]}{'…' if len(preview) > PREVIEW_CHARS else ''}"
        )


def main() -> int:
    """Exit 0 when every question was searched (even with no passage, e.g. a
    company outside the corpus), 1 when one failed (server error)."""

    args = parse_args()
    configure_logging()
    for noisy in ("httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    questions = load_questions(args)
    if not questions:
        print("no question given", file=sys.stderr)
        return 2
    retriever = retriever_from_args(
        args,
        RetrieverSettings(top_k=args.top_k, context_tokens=args.context_tokens),
    )
    results: list[RetrievalResult] = []
    failed: list[dict[str, str]] = []
    for question in questions:
        try:
            results.append(retriever.retrieve(question))
        except SERVER_ERRORS as error:
            # A server down for one question must not lose the others.
            failed.append(
                {"question": question, "error": f"{type(error).__name__}: {error}"}
            )
            print(f"FAILED {question!r}: {failed[-1]['error']}", file=sys.stderr)
        if args.output:  # after each question: an interrupted run keeps its results
            rows = [as_json(r) for r in results] + failed
            args.output.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    rows = [as_json(r) for r in results] + failed
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for result in results:
            show(result)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

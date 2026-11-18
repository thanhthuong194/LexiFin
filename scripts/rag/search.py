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

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.llm import (
    ChatModel,
    LLMConfigError,
    LLMError,
    Reranker,
    get_chat,
    get_embedder,
    get_reranker,
)
from src.rag_engine.retrieval.retriever import (
    Models,
    RetrievalResult,
    Retriever,
    RetrieverSettings,
)
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
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--qdrant-url", default="http://localhost:6333")
    target.add_argument("--qdrant-path", type=Path, help="Embedded Qdrant directory")
    parser.add_argument("--query-profile", default="query", help="[chat.<name>]")
    parser.add_argument("--embedding-profile", default="default")
    parser.add_argument("--rerank-profile", default="default")
    parser.add_argument(
        "--no-llm", action="store_true", help="Read questions with rules only"
    )
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument(
        "--llm-cache-dir", type=Path, help="Cache query-model replies here"
    )
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


def optional_chat(args: argparse.Namespace) -> tuple[ChatModel | None, str | None]:
    if args.no_llm:
        return None, None
    try:
        return get_chat(args.query_profile, cache_dir=args.llm_cache_dir), None
    except LLMConfigError as error:
        return None, f"query model unavailable ({error}): rules"


def optional_reranker(args: argparse.Namespace) -> tuple[Reranker | None, str | None]:
    if args.no_rerank:
        return None, None
    try:
        return get_reranker(args.rerank_profile), None
    except LLMConfigError as error:
        return None, f"reranker unavailable ({error})"


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
    client = (
        QdrantClient(path=str(args.qdrant_path))
        if args.qdrant_path
        else QdrantClient(url=args.qdrant_url)
    )
    chat, chat_note = optional_chat(args)
    reranker, rerank_note = optional_reranker(args)
    retriever = Retriever(
        client,
        Models(get_embedder(args.embedding_profile), chat, reranker),
        settings=RetrieverSettings(
            top_k=args.top_k, context_tokens=args.context_tokens
        ),
    )
    retriever.notes += [n for n in (chat_note, rerank_note) if n]
    results: list[RetrievalResult] = []
    failed: list[dict[str, str]] = []
    for question in questions:
        try:
            results.append(retriever.retrieve(question))
        except (
            LLMError,
            ResponseHandlingException,
            UnexpectedResponse,
            OSError,
        ) as error:
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

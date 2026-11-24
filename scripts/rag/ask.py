from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.llm import get_chat
from src.rag_engine.generation.answer import generate_answer
from src.rag_engine.retrieval.factory import (
    SERVER_ERRORS,
    add_retriever_arguments,
    retriever_from_args,
)
from src.utils.logger import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer a question from SEC filings with cited sources"
    )
    parser.add_argument("question")
    add_retriever_arguments(parser)  # --llm-cache-dir also caches the generator
    parser.add_argument(
        "--generator-profile", default="generator", help="[chat.<name>]"
    )
    parser.add_argument("--output", type=Path, help="Write the answer as JSON here")
    return parser.parse_args()


def main() -> int:
    """Exit 0 when answered (a refusal included), 1 on a server error."""

    args = parse_args()
    configure_logging()
    for noisy in ("httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    retriever = retriever_from_args(args)
    generator = get_chat(args.generator_profile, cache_dir=args.llm_cache_dir)
    try:
        result = retriever.retrieve(args.question)
        answer = generate_answer(
            generator, args.question, result.passages, result.plan.language
        )
    except SERVER_ERRORS as error:
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"\n{answer.text}\n")
    for note in result.notes + answer.notes:
        print(f"note: {note}")
    for number in answer.citations:
        passage = result.passages[number - 1]
        period = (
            f"FY{passage.fiscal_year} {passage.fiscal_period or ''}".strip()
            if passage.fiscal_year
            else f"filed {passage.filing_date}"
        )
        print(
            f"[{number}] {passage.company_name} | {passage.document_type} | "
            f"{period} | {passage.sec_url}"
        )
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "question": args.question,
                    "language": result.plan.language,
                    "answer": answer.model_dump(),
                    "notes": result.notes,
                    "passages": [p.as_dict() for p in result.passages],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

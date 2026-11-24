"""Build a `Retriever` from command-line arguments (search and eval scripts).

The query model and the reranker are optional: when their profile is not
configured, retrieval degrades (rules, fused order) and says so in its notes.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

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
from src.rag_engine.retrieval.retriever import Models, Retriever, RetrieverSettings

# Errors of one question's server calls: record them and go on.
SERVER_ERRORS = (LLMError, ResponseHandlingException, UnexpectedResponse, OSError)


def add_retriever_arguments(parser: argparse.ArgumentParser) -> None:
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


def _optional_chat(args: argparse.Namespace) -> tuple[ChatModel | None, str | None]:
    if args.no_llm:
        return None, None
    try:
        return get_chat(args.query_profile, cache_dir=args.llm_cache_dir), None
    except LLMConfigError as error:
        return None, f"query model unavailable ({error}): rules"


def _optional_reranker(args: argparse.Namespace) -> tuple[Reranker | None, str | None]:
    if args.no_rerank:
        return None, None
    try:
        return get_reranker(args.rerank_profile), None
    except LLMConfigError as error:
        return None, f"reranker unavailable ({error})"


def retriever_from_args(
    args: argparse.Namespace,
    settings: RetrieverSettings | None = None,
    today: date | None = None,
) -> Retriever:
    """A retriever for the arguments of `add_retriever_arguments`; a missing
    query model or reranker is noted in `retriever.notes`. `today` pins the
    date relative questions are read against (default: the real date)."""

    client = (
        QdrantClient(path=str(args.qdrant_path))
        if args.qdrant_path
        else QdrantClient(url=args.qdrant_url)
    )
    chat, chat_note = _optional_chat(args)
    reranker, rerank_note = _optional_reranker(args)
    retriever = Retriever(
        client,
        Models(get_embedder(args.embedding_profile), chat, reranker),
        settings=settings,
        today=today,
    )
    retriever.notes += [n for n in (chat_note, rerank_note) if n]
    return retriever

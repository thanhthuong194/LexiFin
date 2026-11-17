"""Send one small real request through an LLM profile and print the result.

    uv run python scripts/llm/smoke.py chat.default
    uv run python scripts/llm/smoke.py embedding.default
    uv run python scripts/llm/smoke.py rerank.default [--config path.toml]

Calls the real provider (costs a few tokens); keys and URLs come from .env.
Exit 0 = the profile works, 1 = config or provider error.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.llm import (
    LLMError,
    Message,
    get_chat,
    get_embedder,
    get_reranker,
    load_profiles,
)

DOCUMENTS = [
    "Total net sales were $391.0 billion in fiscal 2024.",
    "The company leases office space in Cupertino, California.",
    "Risk factors include supply chain disruption.",
]
QUERY = "What were total net sales in fiscal 2024?"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def smoke(kind: str, name: str, config: Path | None) -> None:
    profiles = load_profiles(config) if config else None
    if kind == "chat":
        chat = get_chat(name, profiles=profiles)
        result = chat.generate([Message.user("Reply with the single word: OK")])
        print(f"{result.provider} {result.model}: {result.text!r}")
        print(
            f"stop={result.stop_reason} latency={result.latency_s:.2f}s "
            f"tokens in/out={result.usage.input_tokens}/{result.usage.output_tokens}"
        )
    elif kind == "embedding":
        embedder = get_embedder(name, profiles=profiles)
        documents = embedder.embed_documents(DOCUMENTS)
        (query,) = embedder.embed_queries([QUERY])
        print(f"{embedder.model}: {len(documents)} vectors of size {len(query)}")
        for text, vector in zip(DOCUMENTS, documents, strict=True):
            print(f"  cosine={cosine(query, vector):.3f}  {text}")
    elif kind == "rerank":
        reranker = get_reranker(name, profiles=profiles)
        print(f"{reranker.model}:")
        for result in reranker.rerank(QUERY, DOCUMENTS):
            print(f"  score={result.score:.3f}  {DOCUMENTS[result.index]}")
    else:
        raise LLMError(f"Unknown kind {kind!r}: use chat, embedding or rerank")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("profile", help="<kind>.<name>, e.g. chat.default")
    parser.add_argument("--config", type=Path, help="profiles TOML (default: config)")
    args = parser.parse_args()
    kind, _, name = args.profile.partition(".")
    try:
        smoke(kind, name or "default", args.config)
    except LLMError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

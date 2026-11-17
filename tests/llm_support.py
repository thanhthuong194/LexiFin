"""Test doubles for src/llm: scripted models and a recording HTTP transport."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.llm.types import (
    ChatOptions,
    ChatResult,
    ContentBlock,
    Message,
    RerankResult,
    StopReason,
    TextBlock,
    Usage,
)


def reply(
    text: str = "",
    *,
    content: tuple[ContentBlock, ...] | None = None,
    stop_reason: StopReason = "end",
) -> ChatResult:
    return ChatResult(
        content=content if content is not None else (TextBlock(text=text),),
        stop_reason=stop_reason,
        raw_stop_reason=stop_reason,
        usage=Usage(input_tokens=1, output_tokens=1),
        provider="fake",
        model="fake-model",
        latency_s=0.0,
    )


@dataclass
class FakeChat:
    """Returns `replies` in order and records every request."""

    replies: list[ChatResult]
    calls: list[tuple[list[Message], ChatOptions | None]] = field(default_factory=list)
    provider: str = "fake"
    model: str = "fake-model"

    def generate(
        self, messages: Sequence[Message], options: ChatOptions | None = None
    ) -> ChatResult:
        self.calls.append((list(messages), options))
        return self.replies.pop(0)


@dataclass
class FakeEmbedding:
    """Deterministic vectors: [len(text), index, 1 for queries else 0]."""

    model: str = "fake-embedding"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(t)), float(i), 0.0] for i, t in enumerate(texts)]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(t)), float(i), 1.0] for i, t in enumerate(texts)]


@dataclass
class FakeReranker:
    """Scores a document by how many query words it contains."""

    model: str = "fake-reranker"

    def rerank(
        self, query: str, documents: Sequence[str], top_n: int | None = None
    ) -> list[RerankResult]:
        words = set(query.lower().split())
        results = [
            RerankResult(index=i, score=float(len(words & set(d.lower().split()))))
            for i, d in enumerate(documents)
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_n] if top_n is not None else results


@dataclass
class RecordingServer:
    """A MockTransport handler: answers with queued (status, JSON) pairs and
    keeps every request's method, URL, headers and JSON body."""

    responses: list[tuple[int, Any]]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def handler(self, response_type: Callable[..., Any]) -> Callable[[Any], Any]:
        def handle(request: Any) -> Any:
            body = request.content
            self.requests.append(
                {
                    "method": request.method,
                    "url": str(request.url),
                    "headers": dict(request.headers),
                    "json": json.loads(body) if body else None,
                }
            )
            status, payload = self.responses.pop(0)
            return response_type(status, json=payload)

        return handle

    @property
    def last_json(self) -> Any:
        return self.requests[-1]["json"]

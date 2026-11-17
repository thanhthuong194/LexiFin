"""The three interfaces callers use; adapters in `providers/` implement them."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.llm.types import ChatOptions, ChatResult, Message, RerankResult


class ChatModel(Protocol):
    """Text/image in, text/tool calls out (RAG answers, judges, agents)."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def generate(
        self, messages: Sequence[Message], options: ChatOptions | None = None
    ) -> ChatResult: ...


class EmbeddingModel(Protocol):
    """Documents and queries are embedded differently by retrieval models
    (Qwen3-Embedding instruction, Gemini task type), so they are two calls."""

    @property
    def model(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    @property
    def model(self) -> str: ...

    def rerank(
        self, query: str, documents: Sequence[str], top_n: int | None = None
    ) -> list[RerankResult]:
        """Results ordered by descending score, at most `top_n` of them."""
        ...

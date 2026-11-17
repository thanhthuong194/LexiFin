"""Shared embedding logic: batching, query instructions, shape checks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

from src.llm.config import EmbeddingProfile
from src.llm.types import LLMProviderError, LLMResponseError

EmbeddingTask = Literal["document", "query"]


class BatchedEmbedding(ABC):
    """Splits input into `batch_size` requests and checks every vector."""

    # SDK exceptions turned into LLMProviderError.
    sdk_errors: tuple[type[Exception], ...] = ()

    def __init__(self, profile: EmbeddingProfile) -> None:
        self.profile = profile

    @property
    def model(self) -> str:
        return self.profile.model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(list(texts), "document")

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        instruction = self.profile.query_instruction
        queries = [
            f"Instruct: {instruction}\nQuery:{text}" if instruction else text
            for text in texts
        ]
        return self._embed(queries, "query")

    def _embed(self, texts: list[str], task: EmbeddingTask) -> list[list[float]]:
        size = self.profile.batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), size):
            batch = texts[start : start + size]
            try:
                embedded = self._embed_batch(batch, task)
            except self.sdk_errors as error:
                raise LLMProviderError(f"{self.model}: {error}") from error
            if len(embedded) != len(batch):
                raise LLMResponseError(
                    f"{self.model}: {len(embedded)} vectors for {len(batch)} texts"
                )
            vectors.extend(embedded)
        widths = {len(vector) for vector in vectors}
        expected = self.profile.dimensions
        if len(widths) > 1 or (expected and widths and widths != {expected}):
            raise LLMResponseError(
                f"{self.model}: vector sizes {sorted(widths)}, expected {expected}"
            )
        return vectors

    @abstractmethod
    def _embed_batch(
        self, texts: list[str], task: EmbeddingTask
    ) -> list[list[float]]: ...

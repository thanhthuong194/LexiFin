"""Rerank over HTTP: `POST {base_url}/rerank` with the schema vLLM, TEI,
Cohere (v2), Jina and Voyage share: {model, query, documents, top_n} ->
{results: [{index, relevance_score}]}."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import structlog

from src.llm.config import RerankProfile, resolve_endpoint
from src.llm.types import LLMResponseError, RerankResult

logger = structlog.get_logger(__name__)

_RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 30.0


def create_client(profile: RerankProfile) -> httpx.Client:
    endpoint = resolve_endpoint(profile)
    headers = {}
    if endpoint.api_key and endpoint.api_key != "unused":
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    return httpx.Client(
        base_url=endpoint.base_url or "", headers=headers, timeout=profile.timeout_s
    )


class HttpReranker:
    def __init__(
        self,
        profile: RerankProfile,
        client: httpx.Client,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.profile = profile
        self._client = client
        self._sleep = sleep

    @property
    def model(self) -> str:
        return self.profile.model

    def rerank(
        self, query: str, documents: Sequence[str], top_n: int | None = None
    ) -> list[RerankResult]:
        if not documents:
            return []
        body: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": list(documents),
        }
        if top_n is not None:
            body["top_n"] = top_n
        payload = self._post(body | self.profile.options)
        try:
            results = [
                RerankResult(index=item["index"], score=item["relevance_score"])
                for item in payload["results"]
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise LLMResponseError(f"{self.model}: unexpected rerank body") from error
        if any(result.index >= len(documents) for result in results):
            raise LLMResponseError(f"{self.model}: rerank index out of range")
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_n] if top_n is not None else results

    def _post(self, body: dict[str, Any]) -> Any:
        for attempt in range(self.profile.max_retries + 1):
            if attempt:
                delay = min(_BACKOFF_S * 2 ** (attempt - 1), _MAX_BACKOFF_S)
                logger.warning(
                    "llm_rerank_retry", model=self.model, attempt=attempt, delay_s=delay
                )
                self._sleep(delay)
            try:
                response = self._client.post("/rerank", json=body)
            except httpx.TransportError as error:
                failure: Exception = error
                continue
            if response.status_code in _RETRY_STATUSES:
                failure = _http_error(self.model, response)
                continue
            if response.is_error:
                raise _http_error(self.model, response)
            return response.json()
        raise LLMResponseError(f"{self.model}: rerank request failed") from failure


def _http_error(model: str, response: httpx.Response) -> LLMResponseError:
    return LLMResponseError(
        f"{model}: rerank HTTP {response.status_code}: {response.text[:300]}"
    )

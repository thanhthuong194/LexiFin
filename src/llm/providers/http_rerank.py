"""Rerank over HTTP: `POST {base_url}/rerank` with the schema vLLM, TEI,
Cohere (v2), Jina and Voyage share: {model, query, documents, top_n} ->
{results: [{index, relevance_score}]}.

With `prompt_format = "qwen3"` the query and documents carry Qwen3-Reranker's
template (vLLM `--hf_overrides '{"architectures":
["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"],
"is_original_qwen3_reranker": true}'`). Measured on that server: bare text
scored a relevant passage 0.32 and an unrelated one 0.26; templated, 0.995
and 0.0006."""

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
QWEN3_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
_QWEN3_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based "
    'on the Query and the Instruct provided. Note that the answer can only be "yes" '
    'or "no".<|im_end|>\n<|im_start|>user\n'
)
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def formatted(
    profile: RerankProfile, query: str, documents: Sequence[str]
) -> tuple[str, list[str]]:
    """Query and documents in the layout of the profile's `prompt_format`."""

    if profile.prompt_format == "qwen3":
        instruction = profile.instruction or QWEN3_DEFAULT_INSTRUCTION
        return (
            f"{_QWEN3_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n",
            [f"<Document>: {document}{_QWEN3_SUFFIX}" for document in documents],
        )
    return query, list(documents)


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
        sent_query, sent_documents = formatted(self.profile, query, documents)
        body: dict[str, Any] = {
            "model": self.profile.request_model,
            "query": sent_query,
            "documents": sent_documents,
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
            try:
                return response.json()
            except ValueError as error:  # a tunnel's HTML page, a cut body
                raise LLMResponseError(
                    f"{self.model}: rerank reply is not JSON: {response.text[:200]}"
                ) from error
        raise LLMResponseError(f"{self.model}: rerank request failed") from failure


def _http_error(model: str, response: httpx.Response) -> LLMResponseError:
    return LLMResponseError(
        f"{model}: rerank HTTP {response.status_code}: {response.text[:300]}"
    )

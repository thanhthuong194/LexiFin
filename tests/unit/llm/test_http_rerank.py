from __future__ import annotations

import httpx
import pytest

from src.llm.config import RerankProfile
from src.llm.providers.http_rerank import HttpReranker, create_client
from src.llm.types import LLMResponseError, RerankResult
from tests.llm_support import RecordingServer

pytestmark = pytest.mark.unit

PROFILE = RerankProfile(
    kind="rerank",
    name="default",
    provider="http_rerank",
    model="Qwen/Qwen3-Reranker-0.6B",
    max_retries=2,
)


def reranker(server: RecordingServer, delays: list[float]) -> HttpReranker:
    transport = httpx.MockTransport(server.handler(httpx.Response))
    client = httpx.Client(base_url="http://vllm.test/v1", transport=transport)
    return HttpReranker(PROFILE, client, sleep=delays.append)


def results(*pairs: tuple[int, float]) -> dict[str, object]:
    return {"results": [{"index": i, "relevance_score": s} for i, s in pairs]}


def test_rerank_sorts_and_truncates_after_retrying_busy_server() -> None:
    server = RecordingServer(
        [(503, {"error": "busy"}), (200, results((0, 0.1), (2, 0.9), (1, 0.5)))]
    )
    delays: list[float] = []

    ranked = reranker(server, delays).rerank("revenue", ["a", "b", "c"], top_n=2)

    assert ranked == [
        RerankResult(index=2, score=0.9),
        RerankResult(index=1, score=0.5),
    ]
    assert delays == [1.0]
    assert server.requests[-1]["url"] == "http://vllm.test/v1/rerank"
    assert server.last_json == {
        "model": "Qwen/Qwen3-Reranker-0.6B",
        "query": "revenue",
        "documents": ["a", "b", "c"],
        "top_n": 2,
    }


def test_rerank_gives_up_after_max_retries() -> None:
    server = RecordingServer([(503, {}), (503, {}), (503, {})])
    delays: list[float] = []

    with pytest.raises(LLMResponseError, match="request failed"):
        reranker(server, delays).rerank("q", ["a"])
    assert delays == [1.0, 2.0]


def test_rerank_retries_transport_errors() -> None:
    calls: list[int] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json=results((0, 1.0)))

    client = httpx.Client(
        base_url="http://vllm.test/v1", transport=httpx.MockTransport(handle)
    )

    assert HttpReranker(PROFILE, client, sleep=lambda _: None).rerank("q", ["a"]) == [
        RerankResult(index=0, score=1.0)
    ]


@pytest.mark.parametrize(
    ("status", "payload", "message"),
    [
        (400, {"error": "bad"}, "HTTP 400"),
        (200, {"data": []}, "unexpected rerank body"),
        (200, results((5, 1.0)), "out of range"),
    ],
)
def test_rerank_errors(status: int, payload: dict[str, object], message: str) -> None:
    server = RecordingServer([(status, payload)])

    with pytest.raises(LLMResponseError, match=message):
        reranker(server, []).rerank("q", ["a"])


def test_no_documents_sends_nothing() -> None:
    server = RecordingServer([])

    assert reranker(server, []).rerank("q", []) == []
    assert server.requests == []


def test_client_sends_the_profile_key_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANK_DEFAULT_BASE_URL", "http://host/v1")
    monkeypatch.setenv("RERANK_DEFAULT_API_KEY", "secret")
    assert create_client(PROFILE).headers["Authorization"] == "Bearer secret"

    monkeypatch.delenv("RERANK_DEFAULT_API_KEY")
    assert "Authorization" not in create_client(PROFILE).headers

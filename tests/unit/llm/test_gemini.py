from __future__ import annotations

import httpx2
import pytest
from google import genai
from google.genai import types as gtypes

from src.llm.config import ChatProfile, EmbeddingProfile
from src.llm.providers.gemini import GeminiChat, GeminiEmbedding
from src.llm.types import (
    ChatOptions,
    ImageBlock,
    LLMProviderError,
    LLMResponseError,
    Message,
    OutputSchema,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolSpec,
)
from tests.llm_support import RecordingServer

pytestmark = pytest.mark.unit

CHAT = ChatProfile(
    kind="chat", name="gemini", provider="gemini", model="gemini-2.5-pro", effort="low"
)


def client(server: RecordingServer) -> genai.Client:
    transport = httpx2.MockTransport(server.handler(httpx2.Response))
    options = gtypes.HttpOptions(
        base_url="http://gemini.test/", httpx_client=httpx2.Client(transport=transport)
    )
    return genai.Client(api_key="key", http_options=options)


def response(parts: list[dict[str, object]], finish: str = "STOP") -> dict[str, object]:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": parts}, "finishReason": finish}
        ],
        "usageMetadata": {
            "promptTokenCount": 7,
            "candidatesTokenCount": 3,
            "thoughtsTokenCount": 2,
            "cachedContentTokenCount": 1,
        },
    }


def test_function_call_round_trip_keeps_thought_signature() -> None:
    server = RecordingServer(
        [
            (
                200,
                response(
                    [
                        {"text": "planning", "thought": True},
                        {
                            "functionCall": {
                                "name": "lookup_fact",
                                "args": {"concept": "Revenues"},
                            },
                            "thoughtSignature": "c2ln",
                        },
                    ]
                ),
            ),
            (200, response([{"text": "Revenue was 10."}])),
        ]
    )
    chat = GeminiChat(CHAT, client(server))
    tool = ToolSpec(name="lookup_fact", description="d", parameters={"type": "object"})
    options = ChatOptions(system="Use filings.", tools=(tool,))
    question = Message.user("Revenue?")

    first = chat.generate([question], options)

    assert first.stop_reason == "tool_call"
    assert first.content == (
        ThinkingBlock(text="planning", provider="gemini"),
        ToolCallBlock(
            id="call_1",
            name="lookup_fact",
            arguments={"concept": "Revenues"},
            signature="c2ln",
        ),
    )
    assert (first.usage.input_tokens, first.usage.output_tokens) == (7, 5)
    assert first.usage.cached_input_tokens == 1
    body = server.last_json
    assert server.requests[0]["url"].endswith("models/gemini-2.5-pro:generateContent")
    assert body["systemInstruction"]["parts"] == [{"text": "Use filings."}]
    assert body["generationConfig"]["thinkingConfig"] == {"thinking_level": "LOW"}
    assert body["tools"][0]["functionDeclarations"][0]["name"] == "lookup_fact"
    assert body["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}

    result = Message(
        role="user",
        content=(
            ToolResultBlock(tool_call_id="call_1", name="lookup_fact", content="10"),
        ),
    )
    second = chat.generate([question, first.as_message(), result], options)

    assert second.text == "Revenue was 10."
    contents = server.last_json["contents"]
    assert contents[1] == {
        "role": "model",
        "parts": [
            {
                "functionCall": {
                    "id": "call_1",
                    "name": "lookup_fact",
                    "args": {"concept": "Revenues"},
                },
                "thoughtSignature": "c2ln",
            }
        ],
    }
    assert contents[2]["parts"][0]["functionResponse"] == {
        "id": "call_1",
        "name": "lookup_fact",
        "response": {"result": "10"},
    }


def test_structured_output_and_image() -> None:
    server = RecordingServer([(200, response([{"text": "{}"}]))])
    message = Message(
        role="user",
        content=(
            TextBlock(text="Read"),
            ImageBlock(media_type="image/png", data="AAA="),
        ),
    )
    schema = OutputSchema(name="A", json_schema={"type": "object"})

    GeminiChat(CHAT, client(server)).generate(
        [message], ChatOptions(output_schema=schema, tool_choice="none")
    )

    config = server.last_json["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == {"type": "object"}
    assert server.last_json["contents"][0]["parts"][1] == {
        "inlineData": {"data": "AAA=", "mime_type": "image/png"}
    }


@pytest.mark.parametrize(
    ("payload", "stop"),
    [
        (response([{"text": "cut"}], "MAX_TOKENS"), "max_tokens"),
        (response([{"text": "no"}], "SAFETY"), "refusal"),
        (response([{"text": "?"}], "OTHER"), "other"),
        ({"promptFeedback": {"blockReason": "SAFETY"}}, "refusal"),
    ],
)
def test_stop_reasons(payload: dict[str, object], stop: str) -> None:
    server = RecordingServer([(200, payload)])

    assert (
        GeminiChat(CHAT, client(server)).generate([Message.user("x")]).stop_reason
        == stop
    )


def test_no_candidates_raises() -> None:
    server = RecordingServer([(200, {"candidates": []})])

    with pytest.raises(LLMResponseError, match="no candidates"):
        GeminiChat(CHAT, client(server)).generate([Message.user("x")])


def test_embeddings_use_task_types() -> None:
    server = RecordingServer(
        [
            (200, {"embeddings": [{"values": [1.0, 0.0]}, {"values": [0.0, 1.0]}]}),
            (200, {"embeddings": [{"values": [0.5, 0.5]}]}),
        ]
    )
    profile = EmbeddingProfile(
        kind="embedding",
        name="gemini",
        provider="gemini",
        model="gemini-embedding-001",
        dimensions=2,
    )
    embedder = GeminiEmbedding(profile, client(server))

    assert embedder.embed_documents(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert embedder.embed_queries(["q"]) == [[0.5, 0.5]]
    first, second = (request["json"] for request in server.requests)
    assert {r["taskType"] for r in first["requests"]} == {"RETRIEVAL_DOCUMENT"}
    assert first["requests"][0]["outputDimensionality"] == 2
    assert second["requests"][0]["taskType"] == "RETRIEVAL_QUERY"


def test_empty_embedding_raises() -> None:
    server = RecordingServer([(200, {"embeddings": [{"values": []}]})])
    profile = EmbeddingProfile(kind="embedding", name="g", provider="gemini", model="m")

    with pytest.raises(LLMResponseError, match="empty embedding"):
        GeminiEmbedding(profile, client(server)).embed_documents(["a"])


def test_http_errors_become_provider_errors() -> None:
    error = {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}}
    server = RecordingServer([(400, error)])

    with pytest.raises(LLMProviderError, match="gemini gemini-2.5-pro"):
        GeminiChat(CHAT, client(server)).generate([Message.user("x")])

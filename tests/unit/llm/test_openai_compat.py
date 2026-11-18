from __future__ import annotations

import httpx2
import openai
import pytest

from src.llm.config import ChatProfile, EmbeddingProfile
from src.llm.providers.openai_compat import OpenAIChat, OpenAIEmbedding
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

TOOL = ToolSpec(
    name="lookup_fact",
    description="Find an XBRL fact",
    parameters={"type": "object", "properties": {"concept": {"type": "string"}}},
)


def client(server: RecordingServer) -> openai.OpenAI:
    transport = httpx2.MockTransport(server.handler(httpx2.Response))
    return openai.OpenAI(
        api_key="key",
        base_url="http://vllm.test/v1",
        http_client=httpx2.Client(transport=transport),
        max_retries=0,
    )


def completion(
    message: dict[str, object], finish_reason: str = "stop"
) -> dict[str, object]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "Qwen/Qwen3-32B",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "total_tokens": 16,
            "prompt_tokens_details": {"cached_tokens": 3},
        },
    }


def chat(server: RecordingServer, **settings: object) -> OpenAIChat:
    profile = ChatProfile.model_validate(
        {
            "kind": "chat",
            "name": "local",
            "provider": "openai_compat",
            "model": "Qwen/Qwen3-32B",
        }
        | settings
    )
    return OpenAIChat(profile, client(server))


def test_text_reply_with_reasoning_and_usage() -> None:
    server = RecordingServer(
        [
            (
                200,
                completion(
                    {"role": "assistant", "content": "42", "reasoning_content": "sum"}
                ),
            )
        ]
    )
    model = chat(server, temperature=0.0, max_tokens=99, options={"top_k": 5})

    result = model.generate([Message.user("6*7?")], ChatOptions(system="Be terse."))

    assert result.text == "42"
    assert result.content[0] == ThinkingBlock(text="sum", provider="openai")
    assert result.stop_reason == "end"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (12, 4)
    assert result.usage.cached_input_tokens == 3
    assert (result.provider, result.model) == ("openai_compat", "Qwen/Qwen3-32B")
    body = server.last_json
    assert server.requests[0]["url"] == "http://vllm.test/v1/chat/completions"
    assert body["messages"] == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "6*7?"},
    ]
    assert (body["max_completion_tokens"], body["temperature"], body["top_k"]) == (
        99,
        0.0,
        5,
    )
    assert "tools" not in body
    assert "reasoning_effort" not in body


def test_tool_call_round_trip() -> None:
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup_fact", "arguments": '{"concept": "Revenues"}'},
    }
    server = RecordingServer(
        [
            (
                200,
                completion(
                    {"role": "assistant", "content": None, "tool_calls": [call]},
                    "tool_calls",
                ),
            ),
            (200, completion({"role": "assistant", "content": "Revenue was 10."})),
        ]
    )
    model = chat(server)
    question = Message.user("Revenue?")

    first = model.generate([question], ChatOptions(tools=(TOOL,)))
    assert first.stop_reason == "tool_call"
    assert first.tool_calls == (
        ToolCallBlock(
            id="call_1", name="lookup_fact", arguments={"concept": "Revenues"}
        ),
    )
    assert server.last_json["tools"][0]["function"]["name"] == "lookup_fact"
    assert server.last_json["tool_choice"] == "auto"

    answer = Message(
        role="user",
        content=(
            ToolResultBlock(tool_call_id="call_1", name="lookup_fact", content="10"),
            TextBlock(text="Answer now."),
        ),
    )
    second = model.generate(
        [question, first.as_message(), answer], ChatOptions(tools=(TOOL,))
    )

    assert second.text == "Revenue was 10."
    assert server.last_json["messages"][1:] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_fact",
                        "arguments": '{"concept": "Revenues"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "10"},
        {"role": "user", "content": "Answer now."},
    ]


def test_images_and_structured_output_and_effort() -> None:
    server = RecordingServer(
        [(200, completion({"role": "assistant", "content": "{}"}))]
    )
    model = chat(server, effort="low")
    message = Message(
        role="user",
        content=(
            TextBlock(text="Read"),
            ImageBlock(media_type="image/png", data="AAA="),
        ),
    )
    schema = OutputSchema(name="Answer", json_schema={"type": "object"})

    model.generate([message], ChatOptions(output_schema=schema))

    body = server.last_json
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": "Read"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}},
    ]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Answer", "schema": {"type": "object"}, "strict": True},
    }
    assert body["reasoning_effort"] == "low"


@pytest.mark.parametrize(
    ("message", "finish_reason", "stop"),
    [
        ({"role": "assistant", "content": "cut"}, "length", "max_tokens"),
        ({"role": "assistant", "content": None, "refusal": "no"}, "stop", "refusal"),
        ({"role": "assistant", "content": "?"}, "weird", "other"),
    ],
)
def test_stop_reasons(
    message: dict[str, object], finish_reason: str, stop: str
) -> None:
    server = RecordingServer([(200, completion(message, finish_reason))])

    result = chat(server).generate([Message.user("x")])

    assert result.stop_reason == stop
    assert result.raw_stop_reason == finish_reason


def test_bad_tool_arguments_and_empty_choices_raise() -> None:
    bad_call = {
        "id": "c",
        "type": "function",
        "function": {"name": "f", "arguments": "{not json"},
    }
    empty = completion({"role": "assistant", "content": "x"})
    empty["choices"] = []
    server = RecordingServer(
        [
            (
                200,
                completion(
                    {"role": "assistant", "tool_calls": [bad_call]}, "tool_calls"
                ),
            ),
            (200, empty),
        ]
    )
    model = chat(server)

    with pytest.raises(LLMResponseError, match="not JSON"):
        model.generate([Message.user("x")])
    with pytest.raises(LLMResponseError, match="no choices"):
        model.generate([Message.user("x")])


def embedding_response(vectors: list[list[float]]) -> dict[str, object]:
    data = [
        {"object": "embedding", "index": i, "embedding": v}
        for i, v in enumerate(vectors)
    ]
    return {
        "object": "list",
        "model": "m",
        "data": list(reversed(data)),
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


def test_embeddings_batch_order_instruction_and_dimensions() -> None:
    server = RecordingServer(
        [
            (200, embedding_response([[1.0, 0.0], [0.0, 1.0]])),
            (200, embedding_response([[1.0, 1.0]])),
            (200, embedding_response([[0.5, 0.5]])),
        ]
    )
    profile = EmbeddingProfile(
        kind="embedding",
        name="default",
        provider="openai_compat",
        model="Qwen/Qwen3-Embedding-0.6B",
        served_model="@cf/qwen/qwen3-embedding-0.6b",
        batch_size=2,
        dimensions=2,
        query_instruction="Find filings",
    )
    embedder = OpenAIEmbedding(profile, client(server))

    documents = embedder.embed_documents(["a", "b", "c"])
    queries = embedder.embed_queries(["revenue?"])

    # The server gets its own name; results keep the canonical one.
    assert embedder.model == "Qwen/Qwen3-Embedding-0.6B"
    assert {r["json"]["model"] for r in server.requests} == {
        "@cf/qwen/qwen3-embedding-0.6b"
    }
    assert documents == [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    assert queries == [[0.5, 0.5]]
    assert [r["json"]["input"] for r in server.requests] == [
        ["a", "b"],
        ["c"],
        ["Instruct: Find filings\nQuery:revenue?"],
    ]
    assert server.requests[0]["json"]["dimensions"] == 2


def test_embedding_shape_errors() -> None:
    server = RecordingServer(
        [
            (200, embedding_response([[1.0]])),
            (200, embedding_response([[1.0, 2.0, 3.0]])),
        ]
    )
    profile = EmbeddingProfile(
        kind="embedding", name="d", provider="openai", model="m", dimensions=2
    )
    embedder = OpenAIEmbedding(profile, client(server))

    with pytest.raises(LLMResponseError, match="1 vectors for 2 texts"):
        embedder.embed_documents(["a", "b"])
    with pytest.raises(LLMResponseError, match="expected 2"):
        embedder.embed_documents(["a"])


def test_http_errors_become_provider_errors() -> None:
    server = RecordingServer([(500, {"error": {"message": "boom"}})] * 2)
    profile = EmbeddingProfile(kind="embedding", name="d", provider="openai", model="m")

    with pytest.raises(LLMProviderError, match="openai_compat Qwen/Qwen3-32B"):
        chat(server).generate([Message.user("x")])
    with pytest.raises(LLMProviderError):
        OpenAIEmbedding(profile, client(server)).embed_documents(["a"])

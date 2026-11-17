from __future__ import annotations

import anthropic
import httpx2
import pytest

from src.llm.config import ChatProfile
from src.llm.providers.anthropic import AnthropicChat
from src.llm.types import (
    ChatOptions,
    ImageBlock,
    LLMProviderError,
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

PROFILE = ChatProfile(
    kind="chat",
    name="default",
    provider="anthropic",
    model="claude-opus-5-5",
    effort="high",
)


def model(server: RecordingServer) -> AnthropicChat:
    transport = httpx2.MockTransport(server.handler(httpx2.Response))
    client = anthropic.Anthropic(
        api_key="key",
        base_url="http://anthropic.test",
        http_client=httpx2.Client(transport=transport),
        max_retries=0,
    )
    return AnthropicChat(PROFILE, client)


def message(content: list[dict[str, object]], stop_reason: str) -> dict[str, object]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 6, "cache_read_input_tokens": 4},
    }


def test_tool_loop_replays_thinking_and_orders_tool_results_first() -> None:
    server = RecordingServer(
        [
            (
                200,
                message(
                    [
                        {
                            "type": "thinking",
                            "thinking": "need data",
                            "signature": "sig",
                        },
                        {"type": "redacted_thinking", "data": "opaque"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup_fact",
                            "input": {"concept": "Revenues"},
                        },
                    ],
                    "tool_use",
                ),
            ),
            (200, message([{"type": "text", "text": "Revenue was 10."}], "end_turn")),
        ]
    )
    chat = model(server)
    tool = ToolSpec(
        name="lookup_fact", description="d", parameters={"type": "object"}, strict=True
    )
    options = ChatOptions(system="You answer from filings.", tools=(tool,))
    question = Message.user("Revenue?")

    first = chat.generate([question], options)

    assert first.stop_reason == "tool_call"
    assert first.content == (
        ThinkingBlock(text="need data", signature="sig", provider="anthropic"),
        ThinkingBlock(text="", signature="opaque", redacted=True, provider="anthropic"),
        ToolCallBlock(
            id="toolu_1", name="lookup_fact", arguments={"concept": "Revenues"}
        ),
    )
    assert first.usage.cached_input_tokens == 4
    body = server.last_json
    assert server.requests[0]["url"] == "http://anthropic.test/v1/messages"
    assert body["system"] == "You answer from filings."
    assert body["output_config"] == {"effort": "high"}
    assert body["tools"] == [
        {
            "name": "lookup_fact",
            "description": "d",
            "input_schema": {"type": "object"},
            "strict": True,
        }
    ]
    assert body["tool_choice"] == {"type": "auto"}
    assert "temperature" not in body

    answer = Message(
        role="user",
        content=(
            TextBlock(text="Here:"),
            ToolResultBlock(tool_call_id="toolu_1", name="lookup_fact", content="10"),
        ),
    )
    second = chat.generate([question, first.as_message(), answer], options)

    assert second.text == "Revenue was 10."
    assert server.last_json["messages"][1:] == [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "need data", "signature": "sig"},
                {"type": "redacted_thinking", "data": "opaque"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup_fact",
                    "input": {"concept": "Revenues"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "10",
                    "is_error": False,
                },
                {"type": "text", "text": "Here:"},
            ],
        },
    ]


def test_structured_output_image_and_foreign_thinking() -> None:
    server = RecordingServer(
        [(200, message([{"type": "text", "text": "{}"}], "end_turn"))]
    )
    history = [
        Message(
            role="user",
            content=(
                ImageBlock(media_type="image/png", data="AAA="),
                TextBlock(text="Read"),
            ),
        ),
        Message(
            role="assistant",
            content=(
                ThinkingBlock(text="from vllm", provider="openai"),
                TextBlock(text="ok"),
            ),
        ),
        Message.user("Now JSON"),
    ]
    schema = OutputSchema(name="A", json_schema={"type": "object"})

    model(server).generate(history, ChatOptions(output_schema=schema, max_tokens=50))

    body = server.last_json
    assert body["max_tokens"] == 50
    assert body["output_config"]["format"] == {
        "type": "json_schema",
        "schema": {"type": "object"},
    }
    assert body["messages"][0]["content"][0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAA="},
    }
    assert body["messages"][1]["content"] == [{"type": "text", "text": "ok"}]


@pytest.mark.parametrize(
    ("raw", "stop"),
    [("max_tokens", "max_tokens"), ("refusal", "refusal"), ("pause_turn", "other")],
)
def test_stop_reasons(raw: str, stop: str) -> None:
    server = RecordingServer([(200, message([{"type": "text", "text": "x"}], raw))])

    assert model(server).generate([Message.user("x")]).stop_reason == stop


def test_http_errors_become_provider_errors() -> None:
    error = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "bad"},
    }
    server = RecordingServer([(400, error)])

    with pytest.raises(LLMProviderError, match="anthropic claude-opus-5-5"):
        model(server).generate([Message.user("x")])

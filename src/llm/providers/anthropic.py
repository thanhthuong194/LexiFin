"""Claude through the Anthropic Messages API (official `anthropic` SDK)."""

from __future__ import annotations

from typing import Any, override

import anthropic
from anthropic.types import Message as AnthropicMessage

from src.llm.chat import ChatAdapter, Completion
from src.llm.config import ChatProfile
from src.llm.types import (
    ChatOptions,
    ContentBlock,
    ImageBlock,
    Message,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolSpec,
    Usage,
)

PROVIDER = "anthropic"

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end",
    "stop_sequence": "end",
    "max_tokens": "max_tokens",
    "model_context_window_exceeded": "max_tokens",
    "tool_use": "tool_call",
    "refusal": "refusal",
}


def create_client(profile: ChatProfile) -> anthropic.Anthropic:
    """ANTHROPIC_API_KEY (or an `ant auth login` profile) from the environment."""

    return anthropic.Anthropic(
        timeout=profile.timeout_s, max_retries=profile.max_retries
    )


class AnthropicChat(ChatAdapter):
    provider_name = PROVIDER
    sdk_errors = (anthropic.APIError,)

    def __init__(self, profile: ChatProfile, client: anthropic.Anthropic) -> None:
        super().__init__(profile)
        self._client = client

    @override
    def _complete(self, messages: list[Message], options: ChatOptions) -> Completion:
        request: dict[str, Any] = {
            "model": self.profile.request_model,
            "max_tokens": self.max_tokens(options),
            "messages": [_message(message) for message in messages],
        }
        if options.system:
            request["system"] = options.system
        output_config: dict[str, Any] = {}
        if self.profile.effort:
            output_config["effort"] = self.profile.effort
        if options.output_schema:
            output_config["format"] = {
                "type": "json_schema",
                "schema": options.output_schema.json_schema,
            }
        if output_config:
            request["output_config"] = output_config
        if options.tools:
            request["tools"] = [_tool(tool) for tool in options.tools]
            request["tool_choice"] = {"type": options.tool_choice}
        extra = self.extra_body(options)
        response = self._client.messages.create(**request, extra_body=extra or None)
        return _completion(response)


def _tool(tool: ToolSpec) -> dict[str, Any]:
    converted: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }
    if tool.strict:
        converted["strict"] = True
    return converted


def _message(message: Message) -> dict[str, Any]:
    # Tool results must open the user turn that answers a tool call.
    results = [_block(b) for b in message.content if isinstance(b, ToolResultBlock)]
    others = [
        converted
        for b in message.content
        if not isinstance(b, ToolResultBlock) and (converted := _block(b)) is not None
    ]
    return {"role": message.role, "content": [*results, *others]}


def _block(block: ContentBlock) -> dict[str, Any] | None:
    match block:
        case TextBlock(text=text):
            return {"type": "text", "text": text}
        case ImageBlock(media_type=media_type, data=data):
            source = {"type": "base64", "media_type": media_type, "data": data}
            return {"type": "image", "source": source}
        case ToolCallBlock(id=call_id, name=name, arguments=arguments):
            return {"type": "tool_use", "id": call_id, "name": name, "input": arguments}
        case ToolResultBlock():
            return {
                "type": "tool_result",
                "tool_use_id": block.tool_call_id,
                "content": block.content,
                "is_error": block.is_error,
            }
        case ThinkingBlock():
            return _thinking(block)
    return None


def _thinking(block: ThinkingBlock) -> dict[str, Any] | None:
    if block.provider != PROVIDER:
        # Reasoning written by another provider cannot be replayed to Claude.
        return None
    if block.redacted:
        return {"type": "redacted_thinking", "data": block.signature}
    return {"type": "thinking", "thinking": block.text, "signature": block.signature}


def _completion(response: AnthropicMessage) -> Completion:
    blocks: list[ContentBlock] = []
    for item in response.content:
        if item.type == "text":
            blocks.append(TextBlock(text=item.text))
        elif item.type == "thinking":
            blocks.append(
                ThinkingBlock(
                    text=item.thinking, signature=item.signature, provider=PROVIDER
                )
            )
        elif item.type == "redacted_thinking":
            blocks.append(
                ThinkingBlock(
                    text="", signature=item.data, redacted=True, provider=PROVIDER
                )
            )
        elif item.type == "tool_use":
            blocks.append(
                ToolCallBlock(id=item.id, name=item.name, arguments=dict(item.input))
            )
    raw = response.stop_reason
    usage = response.usage
    return Completion(
        content=tuple(blocks),
        stop_reason=_STOP_REASONS.get(raw or "", "other"),
        raw_stop_reason=raw,
        usage=Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cache_read_input_tokens or 0,
        ),
    )

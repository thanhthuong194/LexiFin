"""OpenAI Chat Completions and Embeddings API: OpenAI itself (`openai`) and
any server that speaks it (`openai_compat`: vLLM on Colab/Kaggle...)."""

from __future__ import annotations

import json
from typing import Any, override

import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from src.llm.chat import ChatAdapter, Completion
from src.llm.config import ChatProfile, EmbeddingProfile, resolve_endpoint
from src.llm.embedding import BatchedEmbedding, EmbeddingTask
from src.llm.types import (
    ChatOptions,
    ContentBlock,
    ImageBlock,
    LLMResponseError,
    Message,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    Usage,
)

_STOP_REASONS: dict[str, StopReason] = {
    "stop": "end",
    "length": "max_tokens",
    "tool_calls": "tool_call",
    "function_call": "tool_call",
    "content_filter": "refusal",
}


def create_client(profile: ChatProfile | EmbeddingProfile) -> openai.OpenAI:
    endpoint = resolve_endpoint(profile)
    return openai.OpenAI(
        api_key=endpoint.api_key,
        base_url=endpoint.base_url,
        timeout=profile.timeout_s,
        max_retries=profile.max_retries,
    )


class OpenAIChat(ChatAdapter):
    provider_name = "openai"
    sdk_errors = (openai.APIError,)

    def __init__(self, profile: ChatProfile, client: openai.OpenAI) -> None:
        super().__init__(profile)
        self.provider_name = profile.provider
        self._client = client

    @override
    def _complete(self, messages: list[Message], options: ChatOptions) -> Completion:
        request: dict[str, Any] = {
            "model": self.profile.request_model,
            "messages": _messages(options.system, messages),
            "max_completion_tokens": self.max_tokens(options),
        }
        if self.profile.temperature is not None:
            request["temperature"] = self.profile.temperature
        if self.profile.effort:
            request["reasoning_effort"] = self.profile.effort
        if options.tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "strict": tool.strict,
                    },
                }
                for tool in options.tools
            ]
            request["tool_choice"] = options.tool_choice
        if options.output_schema:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": options.output_schema.name,
                    "schema": options.output_schema.json_schema,
                    "strict": True,
                },
            }
        extra = self.extra_body(options)
        response = self._client.chat.completions.create(
            **request, extra_body=extra or None
        )
        return _completion(response)


def _messages(system: str | None, messages: list[Message]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    if system:
        converted.append({"role": "system", "content": system})
    for message in messages:
        if message.role == "assistant":
            converted.append(_assistant_message(message))
        else:
            converted.extend(_user_messages(message))
    return converted


def _assistant_message(message: Message) -> dict[str, Any]:
    text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
    calls = [
        {
            "id": block.id,
            "type": "function",
            "function": {"name": block.name, "arguments": json.dumps(block.arguments)},
        }
        for block in message.content
        if isinstance(block, ToolCallBlock)
    ]
    converted: dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        converted["tool_calls"] = calls
    return converted


def _user_messages(message: Message) -> list[dict[str, Any]]:
    """Tool results become `tool` messages, placed first so they directly
    follow the assistant turn that called them; the rest is one user turn."""

    converted: list[dict[str, Any]] = [
        {"role": "tool", "tool_call_id": block.tool_call_id, "content": block.content}
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    parts: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            url = f"data:{block.media_type};base64,{block.data}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
    if parts and all(part["type"] == "text" for part in parts):
        converted.append(
            {"role": "user", "content": "".join(part["text"] for part in parts)}
        )
    elif parts:
        converted.append({"role": "user", "content": parts})
    return converted


def _completion(response: ChatCompletion) -> Completion:
    if not response.choices:
        raise LLMResponseError(f"{response.model}: response has no choices")
    choice = response.choices[0]
    content = _content(choice.message)
    raw = choice.finish_reason
    stop = _STOP_REASONS.get(raw, "other")
    if choice.message.refusal:
        stop = "refusal"
    usage = response.usage
    cached = 0
    if usage and usage.prompt_tokens_details:
        cached = usage.prompt_tokens_details.cached_tokens or 0
    return Completion(
        content=content,
        stop_reason=stop,
        raw_stop_reason=raw,
        usage=Usage(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cached_input_tokens=cached,
        ),
    )


def _content(message: ChatCompletionMessage) -> tuple[ContentBlock, ...]:
    blocks: list[ContentBlock] = []
    # vLLM reasoning parsers add `reasoning_content` (older) or `reasoning`.
    extra = message.model_extra or {}
    reasoning = extra.get("reasoning_content") or extra.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        blocks.append(ThinkingBlock(text=reasoning, provider="openai"))
    if message.content:
        blocks.append(TextBlock(text=message.content))
    for call in message.tool_calls or []:
        if call.type != "function":
            continue
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError as error:
            raise LLMResponseError(
                f"Tool call {call.function.name}: arguments are not JSON"
            ) from error
        blocks.append(
            ToolCallBlock(id=call.id, name=call.function.name, arguments=arguments)
        )
    return tuple(blocks)


class OpenAIEmbedding(BatchedEmbedding):
    sdk_errors = (openai.APIError,)

    def __init__(self, profile: EmbeddingProfile, client: openai.OpenAI) -> None:
        super().__init__(profile)
        self._client = client

    @override
    def _embed_batch(self, texts: list[str], task: EmbeddingTask) -> list[list[float]]:
        request: dict[str, Any] = {"model": self.profile.request_model, "input": texts}
        if self.profile.dimensions:
            request["dimensions"] = self.profile.dimensions
        extra = self.profile.options
        response = self._client.embeddings.create(**request, extra_body=extra or None)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

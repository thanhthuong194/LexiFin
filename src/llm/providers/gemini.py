"""Gemini through Google AI Studio (official `google-genai` SDK)."""

from __future__ import annotations

import base64
from typing import Any, override

import httpx2
from google import genai
from google.genai import errors as gerrors
from google.genai import types as gtypes

from src.llm.chat import ChatAdapter, Completion
from src.llm.config import ChatProfile, EmbeddingProfile
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

PROVIDER = "gemini"

_REFUSALS = frozenset(
    {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION"}
)
_SDK_ERRORS = (gerrors.APIError, httpx2.TransportError)
_TASK_TYPES: dict[EmbeddingTask, str] = {
    "document": "RETRIEVAL_DOCUMENT",
    "query": "RETRIEVAL_QUERY",
}


def create_client(profile: ChatProfile | EmbeddingProfile) -> genai.Client:
    """GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment."""

    options = gtypes.HttpOptions(
        timeout=int(profile.timeout_s * 1000),
        retry_options=gtypes.HttpRetryOptions(attempts=profile.max_retries + 1),
    )
    return genai.Client(http_options=options)


class GeminiChat(ChatAdapter):
    provider_name = PROVIDER
    sdk_errors = _SDK_ERRORS

    def __init__(self, profile: ChatProfile, client: genai.Client) -> None:
        super().__init__(profile)
        self._client = client

    @override
    def _complete(self, messages: list[Message], options: ChatOptions) -> Completion:
        config: dict[str, Any] = {
            "max_output_tokens": self.max_tokens(options),
            "automatic_function_calling": {"disable": True},
        }
        if options.system:
            config["system_instruction"] = options.system
        if self.profile.temperature is not None:
            config["temperature"] = self.profile.temperature
        if self.profile.effort:
            config["thinking_config"] = {"thinking_level": self.profile.effort}
        if options.tools:
            declarations = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters_json_schema": tool.parameters,
                }
                for tool in options.tools
            ]
            config["tools"] = [{"function_declarations": declarations}]
            mode = "AUTO" if options.tool_choice == "auto" else "NONE"
            config["tool_config"] = {"function_calling_config": {"mode": mode}}
        if options.output_schema:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = options.output_schema.json_schema
        config |= self.extra_body(options)
        response = self._client.models.generate_content(
            model=self.model,
            contents=[_content(message) for message in messages],
            config=gtypes.GenerateContentConfig.model_validate(config),
        )
        return _completion(response)


def _content(message: Message) -> gtypes.Content:
    parts = [part for block in message.content if (part := _part(block)) is not None]
    return gtypes.Content(
        role="model" if message.role == "assistant" else "user", parts=parts
    )


def _part(block: ContentBlock) -> gtypes.Part | None:
    match block:
        case TextBlock(text=text):
            return gtypes.Part(text=text)
        case ImageBlock(media_type=media_type, data=data):
            return gtypes.Part.from_bytes(
                data=base64.b64decode(data), mime_type=media_type
            )
        case ToolCallBlock(id=call_id, name=name, arguments=arguments):
            # Gemini 3 rejects a function call replayed without its signature.
            signature = base64.b64decode(block.signature) if block.signature else None
            return gtypes.Part(
                function_call=gtypes.FunctionCall(
                    id=call_id, name=name, args=arguments
                ),
                thought_signature=signature,
            )
        case ToolResultBlock():
            key = "error" if block.is_error else "result"
            return gtypes.Part(
                function_response=gtypes.FunctionResponse(
                    id=block.tool_call_id,
                    name=block.name,
                    response={key: block.content},
                )
            )
    # Thought summaries are not sent back; signatures ride on their parts.
    return None


def _completion(response: gtypes.GenerateContentResponse) -> Completion:
    usage = _usage(response.usage_metadata)
    feedback = response.prompt_feedback
    if feedback and feedback.block_reason:
        reason = str(feedback.block_reason.value)
        return Completion((), "refusal", reason, usage)
    if not response.candidates:
        raise LLMResponseError("Gemini response has no candidates")
    candidate = response.candidates[0]
    parts = candidate.content.parts if candidate.content else None
    blocks = [_block(part, index) for index, part in enumerate(parts or [])]
    content = tuple(block for block in blocks if block is not None)
    raw = candidate.finish_reason.value if candidate.finish_reason else None
    return Completion(content, _stop_reason(raw, content), raw, usage)


def _block(part: gtypes.Part, index: int) -> ContentBlock | None:
    signature = (
        base64.b64encode(part.thought_signature).decode("ascii")
        if part.thought_signature
        else None
    )
    if part.function_call:
        call = part.function_call
        return ToolCallBlock(
            id=call.id or f"call_{index}",
            name=call.name or "",
            arguments=dict(call.args or {}),
            signature=signature,
        )
    if part.text is None:
        return None
    if part.thought:
        return ThinkingBlock(text=part.text, signature=signature, provider=PROVIDER)
    return TextBlock(text=part.text)


def _stop_reason(raw: str | None, content: tuple[ContentBlock, ...]) -> StopReason:
    if any(isinstance(block, ToolCallBlock) for block in content):
        return "tool_call"
    if raw == "STOP":
        return "end"
    if raw == "MAX_TOKENS":
        return "max_tokens"
    if raw in _REFUSALS:
        return "refusal"
    return "other"


def _usage(metadata: gtypes.GenerateContentResponseUsageMetadata | None) -> Usage:
    if metadata is None:
        return Usage()
    return Usage(
        input_tokens=metadata.prompt_token_count or 0,
        output_tokens=(metadata.candidates_token_count or 0)
        + (metadata.thoughts_token_count or 0),
        cached_input_tokens=metadata.cached_content_token_count or 0,
    )


class GeminiEmbedding(BatchedEmbedding):
    sdk_errors = _SDK_ERRORS

    def __init__(self, profile: EmbeddingProfile, client: genai.Client) -> None:
        super().__init__(profile)
        self._client = client

    @override
    def _embed_batch(self, texts: list[str], task: EmbeddingTask) -> list[list[float]]:
        config: dict[str, Any] = {"task_type": _TASK_TYPES[task]}
        if self.profile.dimensions:
            config["output_dimensionality"] = self.profile.dimensions
        config |= self.profile.options
        response = self._client.models.embed_content(
            model=self.model,
            contents=[gtypes.Content(parts=[gtypes.Part(text=text)]) for text in texts],
            config=gtypes.EmbedContentConfig.model_validate(config),
        )
        vectors = [list(e.values or []) for e in response.embeddings or []]
        if any(not vector for vector in vectors):
            raise LLMResponseError(f"{self.model}: empty embedding in response")
        return vectors

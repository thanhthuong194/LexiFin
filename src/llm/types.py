"""Provider-neutral request and response types shared by every adapter."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TextBlock(_Frozen):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(_Frozen):
    """An inline image: base64 `data` of MIME type `media_type` (image/png...)."""

    type: Literal["image"] = "image"
    media_type: str
    data: str


class ToolCallBlock(_Frozen):
    """A call the model asks the caller to run.

    `signature` is an opaque provider token that must travel back with the
    call in the next request (Gemini thought signatures); None elsewhere.
    """

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any]
    signature: str | None = None


class ToolResultBlock(_Frozen):
    """The caller's answer to one ToolCallBlock, sent in a user message."""

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


class ThinkingBlock(_Frozen):
    """Model reasoning as the provider returns it.

    Anthropic needs these blocks sent back unchanged in tool loops:
    `signature` holds the block signature (or the payload of a redacted
    block when `redacted`), and `provider` limits replay to the provider that
    wrote it. Other providers' reasoning is informational only.
    """

    type: Literal["thinking"] = "thinking"
    text: str
    provider: str
    signature: str | None = None
    redacted: bool = False


ContentBlock = Annotated[
    TextBlock | ImageBlock | ToolCallBlock | ToolResultBlock | ThinkingBlock,
    Field(discriminator="type"),
]


class Message(_Frozen):
    """One conversation turn. The system prompt goes in ChatOptions."""

    role: Literal["user", "assistant"]
    content: tuple[ContentBlock, ...]

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=(TextBlock(text=text),))

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role="assistant", content=(TextBlock(text=text),))


class ToolSpec(_Frozen):
    """A function the model may call; `parameters` is a JSON schema object.

    `strict` asks providers that support it (OpenAI, Anthropic) to guarantee
    arguments valid against the schema; the schema must then follow the
    strict subset (see `strict_json_schema`).
    """

    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool = False


class OutputSchema(_Frozen):
    """Constrain the reply to JSON valid against `json_schema`."""

    name: str
    json_schema: dict[str, Any]


class ChatOptions(_Frozen):
    """Per-request settings; the profile supplies everything left None.

    `provider_options` is merged into the request body as is, for features
    only one provider has (vLLM `chat_template_kwargs`, Anthropic betas...).
    """

    system: str | None = None
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: Literal["auto", "none"] = "auto"
    output_schema: OutputSchema | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    provider_options: dict[str, Any] = Field(default_factory=dict)


StopReason = Literal["end", "max_tokens", "tool_call", "refusal", "other"]


class Usage(_Frozen):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


class ChatResult(_Frozen):
    """A model reply. `raw_stop_reason` is the provider's own value."""

    content: tuple[ContentBlock, ...]
    stop_reason: StopReason
    raw_stop_reason: str | None
    usage: Usage
    provider: str
    model: str
    latency_s: float

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> tuple[ToolCallBlock, ...]:
        return tuple(b for b in self.content if isinstance(b, ToolCallBlock))

    def as_message(self) -> Message:
        """The reply as an assistant turn, to continue the conversation."""

        return Message(role="assistant", content=self.content)


class RerankResult(_Frozen):
    """Relevance of `documents[index]`; higher is more relevant."""

    index: int = Field(ge=0)
    score: float


class LLMError(Exception):
    """Base class of every error this package raises."""


class LLMConfigError(LLMError):
    """A profile is missing, malformed, or lacks its environment variables."""


class LLMProviderError(LLMError):
    """The provider call failed (HTTP error, timeout) after the SDK retries."""


class LLMResponseError(LLMError):
    """The provider answered, but not with something usable."""


class LLMRefusalError(LLMResponseError):
    """The model declined the request (stop reason `refusal`)."""

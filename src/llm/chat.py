"""Shared chat logic: option resolution, timing, logging."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from src.llm.config import ChatProfile
from src.llm.types import (
    ChatOptions,
    ChatResult,
    ContentBlock,
    LLMProviderError,
    Message,
    StopReason,
    Usage,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Completion:
    """What an adapter extracts from one provider response."""

    content: tuple[ContentBlock, ...]
    stop_reason: StopReason
    raw_stop_reason: str | None
    usage: Usage


class ChatAdapter(ABC):
    """Base of every ChatModel adapter: subclasses translate one request."""

    provider_name: str
    # SDK exceptions turned into LLMProviderError.
    sdk_errors: tuple[type[Exception], ...] = ()

    def __init__(self, profile: ChatProfile) -> None:
        self.profile = profile

    @property
    def provider(self) -> str:
        return self.provider_name

    @property
    def model(self) -> str:
        return self.profile.model

    def generate(
        self, messages: Sequence[Message], options: ChatOptions | None = None
    ) -> ChatResult:
        resolved = options or ChatOptions()
        started = time.perf_counter()
        try:
            completion = self._complete(list(messages), resolved)
        except self.sdk_errors as error:
            raise LLMProviderError(f"{self.provider} {self.model}: {error}") from error
        latency = time.perf_counter() - started
        logger.debug(
            "llm_chat_call",
            profile=self.profile.name,
            provider=self.provider,
            model=self.model,
            latency_s=round(latency, 3),
            stop_reason=completion.stop_reason,
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
        )
        return ChatResult(
            content=completion.content,
            stop_reason=completion.stop_reason,
            raw_stop_reason=completion.raw_stop_reason,
            usage=completion.usage,
            provider=self.provider,
            model=self.model,
            latency_s=latency,
        )

    def max_tokens(self, options: ChatOptions) -> int:
        return options.max_tokens or self.profile.max_tokens

    def extra_body(self, options: ChatOptions) -> dict[str, Any]:
        """Profile options overridden by request options."""

        return self.profile.options | options.provider_options

    @abstractmethod
    def _complete(self, messages: list[Message], options: ChatOptions) -> Completion:
        """Send one request and translate the reply."""

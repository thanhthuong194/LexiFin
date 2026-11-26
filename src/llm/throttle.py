"""Space chat calls to stay under a provider's requests-per-minute limit,
instead of hitting 429 and waiting out its retry delay."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence

from src.llm.base import ChatModel
from src.llm.types import ChatOptions, ChatResult, Message


class ThrottledChatModel:
    """Wraps a ChatModel; each call starts at least 60 / rpm s after the last."""

    def __init__(
        self,
        inner: ChatModel,
        requests_per_minute: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._inner = inner
        self._interval = 60.0 / requests_per_minute
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    @property
    def provider(self) -> str:
        return self._inner.provider

    @property
    def model(self) -> str:
        return self._inner.model

    def generate(
        self, messages: Sequence[Message], options: ChatOptions | None = None
    ) -> ChatResult:
        with self._lock:
            wait = self._next - self._clock()
            if wait > 0:
                self._sleep(wait)
            self._next = self._clock() + self._interval
        return self._inner.generate(messages, options)

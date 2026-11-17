"""Disk cache of chat replies, so reruns (evals) cost nothing and repeat
exactly. One JSON file per request under `<cache_dir>/<key[:2]>/<key>.json`;
the key hashes provider, model, profile settings, messages and options."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from src.llm.base import ChatModel
from src.llm.types import ChatOptions, ChatResult, Message

CACHE_FORMAT = 1


class CachedChatModel:
    """Wraps a ChatModel; only replies that ended normally are stored."""

    def __init__(self, inner: ChatModel, cache_dir: Path, settings: str) -> None:
        self._inner = inner
        self._cache_dir = cache_dir
        self._settings = settings

    @property
    def provider(self) -> str:
        return self._inner.provider

    @property
    def model(self) -> str:
        return self._inner.model

    def generate(
        self, messages: Sequence[Message], options: ChatOptions | None = None
    ) -> ChatResult:
        resolved = options or ChatOptions()
        path = self._path(self._key(messages, resolved))
        try:
            return ChatResult.model_validate_json(path.read_bytes())
        except (FileNotFoundError, ValidationError):
            pass
        result = self._inner.generate(messages, resolved)
        if result.stop_reason in ("end", "tool_call"):
            _write_atomic(path, result.model_dump_json().encode())
        return result

    def _key(self, messages: Sequence[Message], options: ChatOptions) -> str:
        request = {
            "format": CACHE_FORMAT,
            "provider": self.provider,
            "model": self.model,
            "settings": self._settings,
            "messages": [message.model_dump(mode="json") for message in messages],
            "options": options.model_dump(mode="json"),
        }
        encoded = json.dumps(request, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _path(self, key: str) -> Path:
        return self._cache_dir / key[:2] / f"{key}.json"


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)

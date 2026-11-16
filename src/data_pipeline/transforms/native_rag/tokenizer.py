from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer

# The embedding model whose tokenizer sizes the chunks. Only `tokenizer.json`
# is needed (no weights); revision and hash pin the exact file.
TOKENIZER_NAME = "Qwen/Qwen3-Embedding-0.6B"
TOKENIZER_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
TOKENIZER_SHA256 = "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"
TOKENIZER_URL = (
    f"https://huggingface.co/{TOKENIZER_NAME}/resolve/{TOKENIZER_REVISION}/"
    "tokenizer.json"
)


def tokenizer_path(data_dir: Path) -> Path:
    """Where the pinned tokenizer file lives under a data directory."""

    return data_dir / "_cache" / "models" / "qwen3-embedding-0.6b" / "tokenizer.json"


class TokenCounter(Protocol):
    """Counts and cuts text in the embedding model's tokens."""

    @property
    def identity(self) -> str:
        """Names the tokenizer exactly (stored with the chunks)."""
        ...

    def count(self, text: str) -> int:
        """Tokens the embedder sees for `text`, special tokens included."""
        ...

    def split(self, text: str, max_tokens: int) -> list[str]:
        """Cut `text` into consecutive pieces of at most `max_tokens` tokens
        (special tokens excluded); joining the pieces gives `text` back."""
        ...


class QwenTokenCounter:
    """`TokenCounter` over the pinned Qwen3-Embedding `tokenizer.json`.

    `count` includes the `<|endoftext|>` token the tokenizer's template
    appends, as the embedder does.
    """

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(
                f"Tokenizer file is missing: {path}; download it with: "
                f"curl -L -o {path} {TOKENIZER_URL}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != TOKENIZER_SHA256:
            raise ValueError(
                f"Tokenizer file {path} has SHA-256 {digest}, expected "
                f"{TOKENIZER_SHA256} ({TOKENIZER_NAME}@{TOKENIZER_REVISION})"
            )
        self._tokenizer = Tokenizer.from_file(str(path))

    @property
    def identity(self) -> str:
        return f"{TOKENIZER_NAME}@{TOKENIZER_REVISION}"

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text).ids)

    def split(self, text: str, max_tokens: int) -> list[str]:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        offsets = self._tokenizer.encode(text, add_special_tokens=False).offsets
        pieces: list[str] = []
        start = 0
        for index in range(max_tokens, len(offsets), max_tokens):
            end = offsets[index][0]
            if end > start:
                pieces.append(text[start:end])
                start = end
        pieces.append(text[start:])
        return [piece for piece in pieces if piece]

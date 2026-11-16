from __future__ import annotations

import re

_WORD = re.compile(r"\S+\s*")


class WordCounter:
    """A `TokenCounter` for tests: one token per whitespace-separated word,
    plus one end-of-text token, like the real tokenizer's template."""

    identity = "test-words"

    def count(self, text: str) -> int:
        return len(text.split()) + 1

    def split(self, text: str, max_tokens: int) -> list[str]:
        words = _WORD.findall(text)
        head = text[: len(text) - len(text.lstrip())]
        pieces = [
            "".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)
        ]
        if pieces and head:
            pieces[0] = head + pieces[0]
        return pieces or [text]


class DigitCounter(WordCounter):
    """A `TokenCounter` for tests where digits are expensive (5 tokens each on
    top of one per word), so `(part 12/13)` costs well more than `(part 9/9)`:
    the real tokenizer also spends one token per digit."""

    identity = "test-digits"

    def count(self, text: str) -> int:
        return super().count(text) + 5 * sum(ch.isdigit() for ch in text)

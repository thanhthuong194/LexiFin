from __future__ import annotations

import re

# Words after which a period does not end a sentence (compared lowercase,
# without the period).
_ABBREVIATIONS = frozenset(
    {
        "inc", "corp", "co", "ltd", "llc", "plc", "no", "nos", "u.s", "u.k",
        "e.g", "i.e", "mr", "mrs", "ms", "dr", "vs", "st", "jr", "sr", "etc",
        "approx", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
        "sept", "oct", "nov", "dec", "fig", "sec", "art", "para", "dept",
    }
)  # fmt: skip
# A candidate boundary: terminal punctuation, optional closing quotes or
# brackets, whitespace, then something that can open a sentence.
_BOUNDARY = re.compile(r"[.!?][\"'”’)\]]*(\s+)(?=[\"“'‘(\[]?[A-Z0-9$])")
_WORD_BEFORE = re.compile(r"([A-Za-z.]+)\.$")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences; `"".join(result) == text`.

    Rule based: a sentence ends at `.`, `!` or `?` (plus closing quotes) followed
    by whitespace and an uppercase letter, digit, `$` or opening quote. A
    period after a known abbreviation (Inc., U.S., e.g.) or a single-letter
    initial does not end a sentence; decimals never match (no whitespace).
    The whitespace after a sentence stays with it.
    """

    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if text[match.start()] == "." and _is_abbreviation(text[: match.start() + 1]):
            continue
        end = match.end(1)
        sentences.append(text[start:end])
        start = end
    if start < len(text):
        sentences.append(text[start:])
    return sentences


def _is_abbreviation(head: str) -> bool:
    word = _WORD_BEFORE.search(head)
    if word is None:
        return False
    token = word.group(1).lower()
    return token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha())

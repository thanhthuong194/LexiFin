"""BM25 sparse vectors for hybrid search (Qdrant applies the IDF part).

Documents get the BM25 term-frequency part, `tf * (k1 + 1) / (tf + k1 * (1 - b
+ b * len / AVG_DOC_WORDS))`; queries get 1 per distinct term; Qdrant multiplies
by IDF over the collection (`Modifier.IDF`). The average length is a fixed
constant, so a chunk's vector never depends on the rest of the corpus and
incremental loads stay consistent. Terms are lowercased words and numbers
(`10-k`, `1,234.5` -> `10`, `k`, `1234.5`), English stopwords dropped, and
mapped to a stable 32-bit index (CRC32; collisions are rare and harmless).
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass

K1 = 1.2
B = 0.75
# Mean terms of an embedded chunk (`terms(context + text)`), measured on the
# local Gold (26,750 chunks, rules v4, 2026-11-17). Changing it changes every
# document vector: reload the collection.
AVG_DOC_WORDS = 116

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOPWORDS = frozenset(
    (
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
        "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
        "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
        "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
        "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
        "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
        "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
        "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
        "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
        "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
        "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you",
        "your", "yours", "yourself", "yourselves",
    )
)  # fmt: skip


@dataclass(frozen=True)
class SparseTerms:
    """A sparse vector as parallel index/value lists, sorted by index."""

    indices: list[int]
    values: list[float]


def terms(text: str) -> list[str]:
    """Lowercase word and number terms of `text`, stopwords dropped; digit
    group separators removed (`1,234.5` -> `1234.5`)."""

    found = []
    for token in _TOKEN.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        found.append(token.replace(",", "") if token[0].isdigit() else token)
    return found


def term_index(term: str) -> int:
    return zlib.crc32(term.encode("utf-8"))


def _vector(weights: dict[int, float]) -> SparseTerms:
    ordered = sorted(weights.items())
    return SparseTerms([i for i, _ in ordered], [w for _, w in ordered])


def document_vector(text: str) -> SparseTerms:
    """BM25 term-frequency weights of a document (IDF left to Qdrant)."""

    words = terms(text)
    counts = Counter(term_index(word) for word in words)
    norm = K1 * (1 - B + B * len(words) / AVG_DOC_WORDS)
    return _vector({i: tf * (K1 + 1) / (tf + norm) for i, tf in counts.items()})


def query_vector(text: str) -> SparseTerms:
    """Weight 1 for each distinct query term."""

    return _vector(dict.fromkeys((term_index(word) for word in terms(text)), 1.0))

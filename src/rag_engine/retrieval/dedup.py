"""Collapse candidates with the same text (boilerplate repeated across
filings: forward-looking statements, 8-K cover items, legal notes).

The key is the company and the chunk text without its prefix, lowercased
with whitespace collapsed, so the same paragraph in two filings of a company
is one candidate (two companies' "None." stay apart in a comparison). The
collapsed chunk IDs ride along, so Recall@k still sees them. The kept
copy takes the best score of the group; it is the newest filing's when the
question names no period (the current wording), else the best-scored one.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

from src.rag_engine.retrieval.search import Hit


def text_key(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def dedupe(hits: list[Hit], *, prefer_newest: bool) -> tuple[list[Hit], int]:
    """Hits in their original order with one per text; also the count
    removed. `hits` are ordered best first."""

    groups: dict[tuple[str, str], list[Hit]] = {}
    for hit in hits:
        key = (
            str(hit.payload.get("cik", "")),
            text_key(str(hit.payload.get("text", ""))),
        )
        groups.setdefault(key, []).append(hit)
    kept: list[Hit] = []
    for group in groups.values():
        best = group[0]
        chosen = (
            max(group, key=lambda h: str(h.payload.get("filing_date") or ""))
            if prefer_newest
            else best
        )
        others = tuple(h.chunk_id for h in group if h is not chosen)
        kept.append(replace(chosen, score=best.score, duplicates=others))
    kept.sort(key=lambda h: -h.score)
    return kept, len(hits) - len(kept)

"""Grow the selected chunks into passages the answer model can read.

Each selected chunk may bring in, from Qdrant only:
- the other parts of the paragraph or table it was split from (`part` of
  `part_count`: consecutive `chunk_ordinal`s of the same document);
- otherwise the chunks right before and after it in the same section;
- a whole 8-K item when that section is short (`small_section_chunks`).
Selected chunks go in first (best first) while they fit the token budget,
then the extra chunks round by round, nearest first, so every selected chunk
gets its closest context before any gets far context. Chunks touching in a
document merge into one passage; passages keep the rank of their best chunk.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient, models

from src.rag_engine.retrieval.search import Hit

_PART_LABEL = re.compile(r"\s*\(part \d+/\d+\)")


@dataclass(frozen=True)
class Passage:
    """Consecutive chunks of one document, ready to cite."""

    rank: int
    score: float
    chunk_ids: tuple[str, ...]
    # Chunks the search selected (the others are context).
    hit_chunk_ids: tuple[str, ...]
    context: str
    text: str
    tokens: int
    corpus: str
    company_name: str
    cik: str
    form: str
    document_type: str
    fiscal_year: int | None
    fiscal_period: str | None
    filing_date: str
    section_title: str
    sec_url: str

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass
class _Plan:
    rank: int
    hit: Hit
    document_id: str
    ordinal: int
    # Extra ordinals this hit may bring in, nearest first.
    extra: list[int] = field(default_factory=list)


def _ordinal(payload: dict[str, Any]) -> int:
    return int(payload["chunk_ordinal"])


def _extra_ordinals(hit: Hit, window: int) -> list[int]:
    ordinal, part, count = (
        _ordinal(hit.payload),
        hit.payload.get("part"),
        hit.payload.get("part_count"),
    )
    if part and count:
        first, last = ordinal - int(part) + 1, ordinal + int(count) - int(part)
    else:
        first, last = ordinal - window, ordinal + window
    others = [o for o in range(max(first, 1), last + 1) if o != ordinal]
    return sorted(others, key=lambda o: (abs(o - ordinal), o))


def _section_chunks(
    client: QdrantClient, collection: str, section_id: str, limit: int
) -> list[dict[str, Any]] | None:
    """Every chunk of a section when it has at most `limit`, else None."""

    points, _ = client.scroll(
        collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="section_id", match=models.MatchValue(value=section_id)
                )
            ]
        ),
        limit=limit + 1,
        with_payload=True,
    )
    if len(points) > limit:
        return None
    return [p.payload for p in points if p.payload]


def _fetch(
    client: QdrantClient, collection: str, document_id: str, ordinals: set[int]
) -> list[dict[str, Any]]:
    if not ordinals:
        return []
    points, _ = client.scroll(
        collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=document_id)
                ),
                models.FieldCondition(
                    key="chunk_ordinal", match=models.MatchAny(any=sorted(ordinals))
                ),
            ]
        ),
        limit=len(ordinals),
        with_payload=True,
    )
    return [p.payload for p in points if p.payload]


@dataclass(frozen=True)
class ExpandSettings:
    # Sum of chunk `token_count`s (prefix included: an upper bound).
    budget_tokens: int = 6000
    window: int = 1
    small_section_chunks: int = 4


def expand(
    client: QdrantClient,
    collection: str,
    hits: Sequence[Hit],
    settings: ExpandSettings,
) -> list[Passage]:
    """Passages for `hits` (best first) within the settings' token budget."""

    window, small_section_chunks = settings.window, settings.small_section_chunks

    plans: list[_Plan] = []
    chunks: dict[tuple[str, int], dict[str, Any]] = {}
    wanted: dict[str, set[int]] = {}
    for rank, hit in enumerate(hits, start=1):
        document_id = str(hit.payload["document_id"])
        plan = _Plan(rank, hit, document_id, _ordinal(hit.payload))
        chunks[(document_id, plan.ordinal)] = hit.payload
        section = (
            _section_chunks(
                client, collection, str(hit.payload["section_id"]), small_section_chunks
            )
            if hit.payload.get("document_type") == "8-K" and not hit.payload.get("part")
            else None
        )
        if section is not None:
            for payload in section:
                chunks[(document_id, _ordinal(payload))] = payload
            plan.extra = sorted(
                (_ordinal(p) for p in section if _ordinal(p) != plan.ordinal),
                key=lambda o: (abs(o - plan.ordinal), o),
            )
        else:
            plan.extra = _extra_ordinals(hit, window)
            wanted.setdefault(document_id, set()).update(plan.extra)
        plans.append(plan)
    for document_id, ordinals in wanted.items():
        missing = {o for o in ordinals if (document_id, o) not in chunks}
        for payload in _fetch(client, collection, document_id, missing):
            chunks[(document_id, _ordinal(payload))] = payload
    for plan in plans:
        # Neighbours stay in the hit's section; part groups always do.
        section_id = plan.hit.payload["section_id"]
        plan.extra = [
            o
            for o in plan.extra
            if (plan.document_id, o) in chunks
            and chunks[(plan.document_id, o)]["section_id"] == section_id
        ]
    chosen = _within_budget(plans, chunks, settings.budget_tokens)
    return _passages(plans, chunks, chosen)


def _within_budget(
    plans: list[_Plan],
    chunks: dict[tuple[str, int], dict[str, Any]],
    budget: int,
) -> set[tuple[str, int]]:
    chosen: set[tuple[str, int]] = set()
    used = 0

    def take(key: tuple[str, int], *, always: bool = False) -> bool:
        nonlocal used
        if key in chosen:
            return True
        tokens = int(chunks[key].get("token_count") or 0)
        if always or used + tokens <= budget:
            chosen.add(key)
            used += tokens
            return True
        return False

    for plan in plans:
        take((plan.document_id, plan.ordinal), always=not chosen)
    selected = {(p.document_id, p.ordinal) for p in plans} & chosen
    # A side (before / after the hit) stops at its first chunk that does not
    # fit: a farther chunk would not touch the passage and never be shown.
    blocked: set[tuple[int, int]] = set()
    depth = max((len(p.extra) for p in plans), default=0)
    for step in range(depth):
        for index, plan in enumerate(plans):
            if (plan.document_id, plan.ordinal) not in selected or step >= len(
                plan.extra
            ):
                continue
            ordinal = plan.extra[step]
            side = (index, 1 if ordinal > plan.ordinal else -1)
            if side not in blocked and not take((plan.document_id, ordinal)):
                blocked.add(side)
    return chosen


def _run_text(payloads: list[dict[str, Any]]) -> str:
    """Chunk texts joined; a chunk whose prefix differs from the previous
    one's (another subheading or table) starts with its prefix tail."""

    parts: list[str] = []
    previous = ""
    for payload in payloads:
        context = _PART_LABEL.sub("", str(payload["context"]))
        text = str(payload["text"])
        if parts and context != previous:
            tail = context.split(" | ", 3)[-1]
            text = f"[{tail}]\n{text}"
        parts.append(text)
        previous = context
    return "\n\n".join(parts)


def _passages(
    plans: list[_Plan],
    chunks: dict[tuple[str, int], dict[str, Any]],
    chosen: set[tuple[str, int]],
) -> list[Passage]:
    best: dict[tuple[str, int], _Plan] = {}
    for plan in plans:
        best.setdefault((plan.document_id, plan.ordinal), plan)
    runs: list[list[tuple[str, int]]] = []
    for key in sorted(chosen):
        last = runs[-1][-1] if runs else None
        if (
            last is not None
            and last[0] == key[0]
            and last[1] + 1 == key[1]
            and chunks[last]["section_id"] == chunks[key]["section_id"]
        ):
            runs[-1].append(key)
        else:
            runs.append([key])
    passages = []
    for run in runs:
        hits = [best[key] for key in run if key in best]
        if not hits:
            continue  # context of a hit that did not fit
        top = min(hits, key=lambda p: p.rank)
        first = chunks[run[0]]
        payloads = [chunks[key] for key in run]
        passages.append(
            Passage(
                rank=top.rank,
                score=top.hit.score,
                chunk_ids=tuple(str(p["chunk_id"]) for p in payloads),
                hit_chunk_ids=tuple(
                    p.hit.chunk_id for p in sorted(hits, key=lambda h: h.rank)
                ),
                context=_PART_LABEL.sub("", str(first["context"])),
                text=_run_text(payloads),
                tokens=sum(int(p.get("token_count") or 0) for p in payloads),
                corpus=str(first.get("corpus", "")),
                company_name=str(first["company_name"]),
                cik=str(first["cik"]),
                form=str(first["form"]),
                document_type=str(first["document_type"]),
                fiscal_year=first.get("fiscal_year"),
                fiscal_period=first.get("fiscal_period"),
                filing_date=str(first["filing_date"])[:10],
                section_title=str(first["section_title"]),
                sec_url=str(first["sec_url"]),
            )
        )
    passages.sort(key=lambda p: p.rank)
    return passages

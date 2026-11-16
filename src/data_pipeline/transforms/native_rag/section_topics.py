from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# One label per kind of content, the same across forms, so a query filters on
# "MD&A" without knowing it is 10-K Item 7 or 10-Q Part I Item 2.
SectionTopic = Literal[
    "cover",
    "business",
    "risk_factors",
    "cybersecurity",
    "properties",
    "legal_proceedings",
    "market_equity",
    "mdna",
    "market_risk",
    "financial_statements",
    "controls",
    "governance",
    "exhibits",
    "signatures",
    "8k_event",
    "ex99",
    "other",
]

_TEN_K_ITEMS: dict[str, SectionTopic] = {
    "ITEM 1": "business",
    "ITEM 1A": "risk_factors",
    "ITEM 1C": "cybersecurity",
    "ITEM 2": "properties",
    "ITEM 3": "legal_proceedings",
    "ITEM 5": "market_equity",
    "ITEM 7": "mdna",
    "ITEM 7A": "market_risk",
    "ITEM 8": "financial_statements",
    "ITEM 9A": "controls",
    "ITEM 10": "governance",
    "ITEM 11": "governance",
    "ITEM 12": "governance",
    "ITEM 13": "governance",
    "ITEM 14": "governance",
    "ITEM 15": "exhibits",
}
# 10-Q item codes repeat across parts (Part I Item 1 vs Part II Item 1).
_TEN_Q_ITEMS: dict[tuple[str, str], SectionTopic] = {
    ("PART I", "ITEM 1"): "financial_statements",
    ("PART I", "ITEM 2"): "mdna",
    ("PART I", "ITEM 3"): "market_risk",
    ("PART I", "ITEM 4"): "controls",
    ("PART II", "ITEM 1"): "legal_proceedings",
    ("PART II", "ITEM 1A"): "risk_factors",
    ("PART II", "ITEM 2"): "market_equity",
    ("PART II", "ITEM 6"): "exhibits",
}
# Title words that name a topic, for 10-Q items without a PART.
_TITLE_TOPICS: tuple[tuple[str, SectionTopic], ...] = (
    ("risk factors", "risk_factors"),
    ("legal proceedings", "legal_proceedings"),
    ("financial statements", "financial_statements"),
    ("discussion and analysis", "mdna"),
    ("market risk", "market_risk"),
    ("controls and procedures", "controls"),
    ("unregistered sales", "market_equity"),
    ("exhibits", "exhibits"),
)


_KIND_TOPICS: dict[str, SectionTopic] = {
    "document": "cover",
    "table_of_contents": "cover",
    "signature": "signatures",
    "exhibit_index": "exhibits",
}


@dataclass(frozen=True)
class SectionHead:
    """What names a section: kind, code, title, and the enclosing PART's code."""

    kind: str
    code: str | None
    title: str
    part_code: str | None = None


def section_topic(form: str, document_type: str, head: SectionHead) -> SectionTopic:
    """The topic of a section of a narrative document.

    Every EX-99 exhibit (press releases, CFO commentary, decks) is `ex99`;
    10-Q items are keyed by their PART, or named by title words without one.
    """

    if document_type.upper().startswith("EX-99"):
        return "ex99"
    if head.kind != "item" or head.code is None:
        return _KIND_TOPICS.get(head.kind, "other")
    if form == "8-K":
        return "exhibits" if head.code == "ITEM 9.01" else "8k_event"
    if form == "10-K":
        return _TEN_K_ITEMS.get(head.code, "other")
    if form == "10-Q":
        return _ten_q_topic(head)
    return "other"


def _ten_q_topic(head: SectionHead) -> SectionTopic:
    if head.part_code is not None:
        return _TEN_Q_ITEMS.get((head.part_code, head.code or ""), "other")
    lowered = head.title.lower()
    return next((topic for words, topic in _TITLE_TOPICS if words in lowered), "other")

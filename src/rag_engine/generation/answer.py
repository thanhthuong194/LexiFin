"""Answer a question from retrieved passages, citing them as [1], [2]...

One structured call to `[chat.generator]`; citations are read from the
answer text and checked against the passages (invalid markers are removed
and noted). Errors of `src/llm` propagate.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.llm import ChatModel, ChatOptions, Message, generate_structured
from src.rag_engine.retrieval.expand import Passage

_LANGUAGE_NAMES = {"en": "English", "vi": "Vietnamese"}

# A citation: numbers of at most 2 digits (`[2023]` is a year, not one),
# listed and/or as ranges: `[1]`, `[1, 3]`, `[1-3]`, `[1–3]`; `[1][3]` is two.
_ITEM = r"\d{1,2}(?:\s*[-–]\s*\d{1,2})?"
_CITATION = re.compile(rf"[ \t]*\[({_ITEM}(?:\s*,\s*{_ITEM})*)\]")
# gpt-oss (harmony) style: `【2】`, `【2†source】`, `【2†L3-L5】`.
_HARMONY = re.compile(r"【(\d{1,2})(?:†[^】]*)?】")

_NO_PASSAGES = {
    "en": "The filings found do not contain information to answer this question.",
    "vi": "Các hồ sơ tìm được không có thông tin để trả lời câu hỏi này.",
}


class Answer(BaseModel):
    model_config = ConfigDict(frozen=True)

    # The answer, citing passages as [n] (1-based passage order).
    text: str
    # Valid [n] numbers found in text, sorted, unique.
    citations: list[int]
    # The passages do not answer the question (a refusal).
    insufficient: bool
    notes: list[str] = Field(default_factory=list)


class _Reply(BaseModel):
    """What the generator returns."""

    answer: str = Field(description="The answer, citing passages as [n]")
    insufficient: bool = Field(
        description="True only when nothing in the passages answers the question"
    )


def language_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code)


def passage_header(number: int, passage: Passage) -> str:
    """`[n] company | form | period | filed date | section`."""

    period = (
        f"FY{passage.fiscal_year} {passage.fiscal_period or ''}".strip()
        if passage.fiscal_year
        else ""
    )
    fields = (
        passage.company_name,
        passage.document_type,
        period,
        f"filed {passage.filing_date}",
        passage.section_title,
    )
    return f"[{number}] " + " | ".join(f for f in fields if f)


def format_passages(passages: Sequence[Passage], numbers: Sequence[int]) -> str:
    """Passages under their header, each numbered as given."""

    return "\n\n".join(
        f"{passage_header(n, p)}\n{p.context}\n{p.text}"
        for n, p in zip(numbers, passages, strict=True)
    )


def system_prompt(language: str) -> str:
    name = language_name(language)
    return (
        "You answer questions about SEC filings (10-K, 10-Q, 8-K and their "
        "exhibits) using only the numbered passages the user gives.\n"
        "Rules:\n"
        "- Use only the passages: no outside knowledge, no guessing.\n"
        "- Cite every claim with the number of the passage it comes from, right "
        "after the claim. Cite only in the form [1] or [2, 3].\n"
        "- Keep numbers, units, currencies and periods exactly as the passages "
        "state them (e.g. '$1,234 million', 'three months ended June 30, 2024').\n"
        "- Amounts in tables are in the unit the table heading or context states "
        "(e.g. 'in millions'): always state the unit and currency with a number.\n"
        f"- Write the answer in {name}, whatever language the question is in.\n"
        "- Be concise: a few sentences, or a short list when the question asks "
        "for one.\n"
        "- Answer what the passages support and say briefly what is missing, if "
        "anything; then set insufficient to false.\n"
        "- Set insufficient to true only when nothing in the passages answers "
        f"the question; then say briefly, in {name}, what information is missing."
    )


def user_prompt(question: str, passages: Sequence[Passage]) -> str:
    numbers = range(1, len(passages) + 1)
    return f"Passages:\n\n{format_passages(passages, numbers)}\n\nQuestion: {question}"


def _numbers(listed: str) -> list[int] | None:
    """`1, 3-5` -> [1, 3, 4, 5]; None for a backward range (not a citation)."""

    found: list[int] = []
    for item in listed.split(","):
        first, _, last = item.replace("–", "-").partition("-")
        start, end = int(first), int(last or first)
        if end < start:
            return None
        found += range(start, end + 1)
    return found


def parse_citations(text: str, count: int) -> tuple[str, list[int], list[str]]:
    """Text with citations in canonical form (`[1]`, `[1, 2, 3]`; harmony
    `【n†...】` and ranges rewritten) and markers outside 1..count removed,
    the valid numbers (sorted, unique) and a note per dropped number."""

    valid: set[int] = set()
    notes: list[str] = []

    def canonical(match: re.Match[str]) -> str:
        numbers = _numbers(match.group(1))
        if numbers is None:
            return match.group(0)
        kept = list(dict.fromkeys(n for n in numbers if 1 <= n <= count))
        notes.extend(
            f"dropped citation [{n}]: no passage {n}"
            for n in dict.fromkeys(numbers)
            if n not in kept
        )
        valid.update(kept)
        if not kept:  # removed with its leading space
            return ""
        space = match.group(0)[: match.group(0).index("[")]
        return f"{space}[{', '.join(map(str, kept))}]"

    text = _HARMONY.sub(r"[\1]", text)
    return _CITATION.sub(canonical, text).strip(), sorted(valid), notes


def generate_answer(
    chat: ChatModel, question: str, passages: Sequence[Passage], language: str
) -> Answer:
    """Answer `question` in `language` (ISO 639-1) from `passages` only."""

    if not passages:
        return Answer(
            text=_NO_PASSAGES.get(language, _NO_PASSAGES["en"]),
            citations=[],
            insufficient=True,
            notes=["no passages: model not called"],
        )
    reply, _ = generate_structured(
        chat,
        [Message.user(user_prompt(question, passages))],
        _Reply,
        ChatOptions(system=system_prompt(language)),
    )
    text, citations, notes = parse_citations(reply.answer.strip(), len(passages))
    if not reply.insufficient and not citations:
        notes.append("answer cites no passage")
    return Answer(
        text=text, citations=citations, insufficient=reply.insufficient, notes=notes
    )

from __future__ import annotations

import json
from typing import Any

import pytest

from src.llm import LLMResponseError
from src.rag_engine.generation.answer import (
    Answer,
    generate_answer,
    parse_citations,
)
from src.rag_engine.retrieval.expand import Passage
from tests.llm_support import FakeChat, reply

pytestmark = pytest.mark.unit


def passage(number: int, **overrides: Any) -> Passage:
    values: dict[str, Any] = {
        "rank": number,
        "score": 1.0 / number,
        "chunk_ids": (f"c{number}",),
        "hit_chunk_ids": (f"c{number}",),
        "context": f"Apple 10-K 2024 > Item 7 > part {number}",
        "text": f"Net sales were ${number},000 million.",
        "tokens": 20,
        "corpus": "sec_filings",
        "company_name": "Apple Inc.",
        "cik": "0000320193",
        "form": "10-K",
        "document_type": "10-K",
        "fiscal_year": 2024,
        "fiscal_period": "FY",
        "filing_date": "2024-11-01",
        "section_title": "ITEM 7. MD&A",
        "sec_url": f"https://www.sec.gov/{number}",
    }
    values.update(overrides)
    return Passage(**values)


def model_reply(answer: str, *, insufficient: bool = False) -> Any:
    return reply(json.dumps({"answer": answer, "insufficient": insufficient}))


@pytest.mark.parametrize(
    ("text", "cleaned", "citations", "dropped"),
    [
        ("Sales rose [1].", "Sales rose [1].", [1], []),
        ("A [2][1] and B [1, 3].", "A [2][1] and B [1, 3].", [1, 2, 3], []),
        ("A [1,3].", "A [1, 3].", [1, 3], []),
        ("A [9].", "A.", [], [9]),
        ("A [1, 9] and [0] B.", "A [1] and B.", [1], [9, 0]),
        ("Item [a] stays.", "Item [a] stays.", [], []),
        # Years and longer numbers are not citations: kept, no note.
        ("In [2023] sales [1].", "In [2023] sales [1].", [1], []),
        ("Note [123] and [2023, 2024].", "Note [123] and [2023, 2024].", [], []),
        # Ranges and gpt-oss (harmony) markers become canonical.
        ("A [1-3].", "A [1, 2, 3].", [1, 2, 3], []),
        ("A [2–4].", "A [2, 3].", [2, 3], [4]),
        ("A [3-1].", "A [3-1].", [], []),
        ("A 【2】 B 【1†source】【3†L3-L5】.", "A [2] B [1][3].", [1, 2, 3], []),
        ("A 【7†source】.", "A.", [], [7]),
    ],
)
def test_parse_citations(
    text: str, cleaned: str, citations: list[int], dropped: list[int]
) -> None:
    result, found, notes = parse_citations(text, 3)
    assert result == cleaned
    assert found == citations
    assert notes == [f"dropped citation [{n}]: no passage {n}" for n in dropped]


def test_answer_cites_valid_passages_and_drops_others() -> None:
    chat = FakeChat([model_reply("Net sales were $1,000 million [1][4]. [2]")])
    answer = generate_answer(
        chat, "Apple net sales 2024?", [passage(1), passage(2)], "en"
    )
    assert answer == Answer(
        text="Net sales were $1,000 million [1]. [2]",
        citations=[1, 2],
        insufficient=False,
        notes=["dropped citation [4]: no passage 4"],
    )


def test_prompts_carry_language_rules_and_numbered_passages() -> None:
    chat = FakeChat([model_reply("Doanh thu là $1,000 triệu [1].")])
    eight_k = passage(2, fiscal_year=None, fiscal_period=None, document_type="EX-99")
    generate_answer(chat, "Apple revenue?", [passage(1), eight_k], "vi")
    messages, options = chat.calls[0]
    assert options is not None and options.system is not None
    assert options.output_schema is not None
    system = options.system
    assert "Write the answer in Vietnamese" in system
    assert "insufficient" in system and "only the passages" in system
    assert "Cite every claim" in system
    assert "Cite only in the form [1] or [2, 3]" in system
    assert "in millions" in system
    user = messages[0].content[0]
    text = getattr(user, "text", "")
    assert "[1] Apple Inc. | 10-K | FY2024 FY | filed 2024-11-01 | ITEM 7. MD&A" in text
    assert "[2] Apple Inc. | EX-99 | filed 2024-11-01 | ITEM 7. MD&A" in text
    assert "Apple 10-K 2024 > Item 7 > part 2\nNet sales were $2,000 million." in text
    assert text.endswith("Question: Apple revenue?")


def test_english_language_name() -> None:
    chat = FakeChat([model_reply("Yes [1].")])
    generate_answer(chat, "Doanh thu?", [passage(1)], "en")
    options = chat.calls[0][1]
    assert options is not None and "in English" in (options.system or "")


def test_refusal_and_uncited_answer() -> None:
    chat = FakeChat(
        [
            model_reply("Không có dữ liệu về cổ tức.", insufficient=True),
            model_reply("Net sales rose."),
        ]
    )
    refusal = generate_answer(chat, "Cổ tức?", [passage(1)], "vi")
    assert refusal.insufficient and refusal.citations == [] and refusal.notes == []
    uncited = generate_answer(chat, "Sales?", [passage(1)], "en")
    assert uncited.notes == ["answer cites no passage"]


@pytest.mark.parametrize("language", ["vi", "en", "fr"])
def test_no_passages_refuses_without_calling_the_model(language: str) -> None:
    chat = FakeChat([])
    answer = generate_answer(chat, "Anything?", [], language)
    assert answer.insufficient and answer.citations == [] and answer.text
    assert chat.calls == []


def test_bad_reply_propagates() -> None:
    with pytest.raises(LLMResponseError):
        generate_answer(FakeChat([reply("not json")]), "Q?", [passage(1)], "en")

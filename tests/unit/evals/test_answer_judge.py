from __future__ import annotations

import json
from typing import Any

import pytest

from src.evals.answer_judge import Judgment, judge_answer
from src.llm import Message
from src.rag_engine.generation.answer import Answer
from src.rag_engine.retrieval.expand import Passage
from tests.llm_support import FakeChat, reply

pytestmark = pytest.mark.unit


def passage(number: int) -> Passage:
    values: dict[str, Any] = {
        "rank": number,
        "score": 1.0,
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
    return Passage(**values)


def _text(message: Message) -> str:
    return "".join(getattr(block, "text", "") for block in message.content)


def test_judges_correctness_and_faithfulness_of_cited_passages() -> None:
    chat = FakeChat(
        [
            reply(json.dumps({"reason": "Same value.", "verdict": "correct"})),
            reply(
                json.dumps(
                    {"reason": "One claim.", "unsupported": [], "faithful": True}
                )
            ),
        ]
    )
    answer = Answer(
        text="Doanh thu thuần là $3,000 triệu [3].",
        citations=[3],
        insufficient=False,
        notes=[],
    )
    passages = [passage(1), passage(2), passage(3)]
    judgment = judge_answer(
        chat, "Doanh thu?", "Net sales were $3.0 billion.", answer, passages
    )
    assert judgment == Judgment(
        correctness="correct",
        correctness_reason="Same value.",
        faithful=True,
        unsupported=[],
        faithfulness_reason="One claim.",
    )
    (correct_messages, correct_options), (faith_messages, faith_options) = chat.calls
    assert correct_options is not None and faith_options is not None
    assert "Vietnamese" in (correct_options.system or "")
    assert "data is missing" in (correct_options.system or "")
    correct_text = _text(correct_messages[0])
    assert "Question: Doanh thu?" in correct_text
    assert "Reference answer: Net sales were $3.0 billion." in correct_text
    assert answer.text in correct_text
    faith_text = _text(faith_messages[0])
    # Every passage the generator saw, under its own number.
    for number in (1, 2, 3):
        assert f"[{number}] Apple Inc." in faith_text
        assert f"${number},000 million" in faith_text
    assert faith_text.endswith(f"Answer: {answer.text}")


def test_unsupported_claims_make_the_answer_unfaithful() -> None:
    chat = FakeChat(
        [
            reply(json.dumps({"reason": "Wrong year.", "verdict": "partial"})),
            reply(
                json.dumps(
                    {"reason": "x", "unsupported": ["grew 5%"], "faithful": True}
                )
            ),
        ]
    )
    answer = Answer(text="Sales grew 5% [1].", citations=[1], insufficient=False)
    judgment = judge_answer(chat, "Q?", "Ref.", answer, [passage(1)])
    assert judgment.correctness == "partial"
    assert judgment.faithful is False and judgment.unsupported == ["grew 5%"]


@pytest.mark.parametrize("empty", [[], ["", "None"], ["N/A", " none "]])
def test_none_like_unsupported_entries_are_dropped(empty: list[str]) -> None:
    chat = FakeChat(
        [
            reply(json.dumps({"reason": "ok", "verdict": "correct"})),
            reply(json.dumps({"reason": "ok", "unsupported": empty, "faithful": True})),
        ]
    )
    answer = Answer(text="Sales [1].", citations=[1], insufficient=False)
    judgment = judge_answer(chat, "Q?", "Ref.", answer, [passage(1)])
    assert judgment.faithful is True and judgment.unsupported == []


def test_model_unfaithful_without_claims_stays_unfaithful() -> None:
    chat = FakeChat(
        [
            reply(json.dumps({"reason": "ok", "verdict": "correct"})),
            reply(
                json.dumps({"reason": "x", "unsupported": ["N/A"], "faithful": False})
            ),
        ]
    )
    answer = Answer(text="Sales [1].", citations=[1], insufficient=False)
    judgment = judge_answer(chat, "Q?", "Ref.", answer, [passage(1)])
    assert judgment.faithful is False and judgment.unsupported == []


def test_refusal_skips_faithfulness() -> None:
    chat = FakeChat(
        [reply(json.dumps({"reason": "Says missing.", "verdict": "correct"}))]
    )
    answer = Answer(text="No data.", citations=[], insufficient=True)
    judgment = judge_answer(
        chat, "Q?", "The filings do not disclose this.", answer, [passage(1)]
    )
    assert len(chat.calls) == 1
    assert judgment.faithful is None and judgment.faithfulness_reason is None
    assert judgment.unsupported == [] and judgment.correctness == "correct"

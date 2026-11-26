"""Grade a generated answer with `[chat.judge]` (another model family than
the generator): correctness against the reference answer, then, unless the
answer is a refusal, faithfulness to all the passages the generator was
shown (a hallucination check; whether citations point at the right passage
is measured deterministically elsewhere). Own prompts, two structured calls;
errors of `src/llm` propagate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.llm import ChatModel, ChatOptions, Message, generate_structured
from src.rag_engine.generation.answer import Answer, format_passages
from src.rag_engine.retrieval.expand import Passage

Correctness = Literal["correct", "partial", "incorrect"]


class Judgment(BaseModel):
    model_config = ConfigDict(frozen=True)

    correctness: Correctness
    correctness_reason: str
    # None when the answer is a refusal (faithfulness not judged).
    faithful: bool | None
    unsupported: list[str]
    faithfulness_reason: str | None


class _CorrectnessReply(BaseModel):
    reason: str = Field(description="One or two sentences")
    verdict: Correctness


class _FaithfulnessReply(BaseModel):
    reason: str = Field(description="One or two sentences")
    unsupported: list[str] = Field(
        description="Claims of the answer the passages do not support"
    )
    faithful: bool


CORRECTNESS_SYSTEM = (
    "You grade answers to questions about SEC filings against a reference "
    "answer written in English. The answer may be in another language "
    "(e.g. Vietnamese): judge its meaning, not its wording or language.\n"
    "Verdicts:\n"
    "- correct: answers everything the question asks with the reference's "
    "facts for it (values, units, periods, names; the result of a "
    "calculation; a yes/no or value for each company asked; for a question "
    "built on a false premise, that the premise is wrong and the true "
    "direction), with nothing that contradicts the reference. Context the "
    "reference adds but the question does not ask for (share counts, filing "
    "dates, operands of a calculation, prior-period figures, background) may "
    "be missing. Numbers are equal when the answer rounds to the reference "
    "value or the reference rounds to the answer's (e.g. $22,528 million vs "
    "about $22.5 billion), whatever the unit or notation (e.g. $1.2 billion "
    "vs $1,200 million). Extra correct detail is fine.\n"
    "- partial: answers part of what is asked (e.g. one of two companies, "
    "some items of a list), or a minor detail of what is asked is wrong.\n"
    "- incorrect: the main fact asked is missing or wrong, or the answer "
    "contradicts the reference.\n"
    "When the reference says the information is not available (an "
    "unanswerable question), an answer that says the data is missing or "
    "cannot be found is correct, and an answer that gives a value or claim "
    "anyway is incorrect. When the reference gives an answer, a refusal is "
    "incorrect.\n"
    "Give a short reason first, then the verdict."
)

FAITHFULNESS_SYSTEM = (
    "You check whether an answer about SEC filings is supported by the "
    "numbered passages it was written from. The answer may be in another "
    "language than the passages: compare meaning. Each claim of the answer "
    "must be stated in, or directly follow from, at least one passage "
    "(whichever number it cites); a number must match the "
    "passage (same value, unit and period; rounding is fine). Do not use "
    "outside knowledge. List every unsupported claim, quoted or paraphrased "
    "briefly from the answer (an empty list when all are supported); "
    "faithful is true only when the list is empty. Give a short reason first."
)


def correctness_prompt(question: str, reference: str, answer: Answer) -> str:
    return (
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Answer to grade: {answer.text}"
    )


def faithfulness_prompt(answer: Answer, passages: Sequence[Passage]) -> str:
    """The answer and every passage the generator was shown, numbered as
    it saw them."""

    body = format_passages(passages, range(1, len(passages) + 1))
    return f"Passages:\n\n{body}\n\nAnswer: {answer.text}"


_NONE_LIKE = frozenset({"", "none", "n/a", "na", "null", "-"})


def _claims(unsupported: Sequence[str]) -> list[str]:
    """Unsupported claims without blank or none-like entries."""

    return [c.strip() for c in unsupported if c.strip().casefold() not in _NONE_LIKE]


def judge_answer(
    chat: ChatModel,
    question: str,
    reference: str,
    answer: Answer,
    passages: Sequence[Passage],
) -> Judgment:
    """`passages` are the ones the answer was generated from (its [n] are
    1-based positions in them)."""

    correctness, _ = generate_structured(
        chat,
        [Message.user(correctness_prompt(question, reference, answer))],
        _CorrectnessReply,
        ChatOptions(system=CORRECTNESS_SYSTEM),
    )
    if answer.insufficient:
        return Judgment(
            correctness=correctness.verdict,
            correctness_reason=correctness.reason,
            faithful=None,
            unsupported=[],
            faithfulness_reason=None,
        )
    faithfulness, _ = generate_structured(
        chat,
        [Message.user(faithfulness_prompt(answer, passages))],
        _FaithfulnessReply,
        ChatOptions(system=FAITHFULNESS_SYSTEM),
    )
    unsupported = _claims(faithfulness.unsupported)
    return Judgment(
        correctness=correctness.verdict,
        correctness_reason=correctness.reason,
        faithful=faithfulness.faithful and not unsupported,
        unsupported=unsupported,
        faithfulness_reason=faithfulness.reason,
    )

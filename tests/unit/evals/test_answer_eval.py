from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from src.evals.answer_eval import (
    Steps,
    collect,
    failed_metrics,
    is_vietnamese,
    rejudge,
    retrieval_split,
    row_metrics,
    stored_passage,
    to_retry,
    write_outputs,
)
from src.evals.answer_judge import Judgment
from src.evals.retrieval_eval import Groups, load_rows, pending, resume_rows
from src.evals.testset import EvalQuestion, Language
from src.llm import LLMProviderError
from src.rag_engine.generation.answer import Answer
from src.rag_engine.retrieval.expand import Passage
from src.rag_engine.retrieval.query import QueryPlan
from src.rag_engine.retrieval.retriever import RetrievalResult

pytestmark = pytest.mark.unit

GOOGL = "0001652044"
FACT = {
    "fact_id": "f1", "concept": "CostOfRevenue", "value": "33712000000",
    "unit": "iso4217:USD", "decimals": "-6", "period_start": "2024-01-01",
    "period_end": "2024-03-31", "instant": None, "block_id": "b0",
    "cell_id": None, "block_ids": ["b0"],
}  # fmt: skip
CONFIG: dict[str, Any] = {
    "label": "base",
    "languages": ["en", "vi"],
    "today": "2026-11-01",
    "questions": {"path": "q.jsonl", "sha256": "0" * 64, "count": 3},
    "models": {"generator": "fake"},
    "judge": True,
}


def make_question(
    qid: str,
    question_type: str = "numeric",
    reference: str = "Cost of revenues was $33,712 million in Q1 2024.",
) -> EvalQuestion:
    answerable = question_type != "unanswerable"
    return EvalQuestion.model_validate(
        {
            "id": qid,
            "task_id": f"t-{qid}",
            "question_en": f"What was it, {qid}?",
            "question_vi": f"Đó là bao nhiêu, {qid}?",
            "reference_answer": reference,
            "answerable": answerable,
            "question_type": question_type,
            "data_kind": "xbrl" if answerable else "none",
            "persona": "analyst",
            "mention_style": "name",
            "evidence_groups": [[{"block_id": "b0", "quote": "x"}]]
            if answerable
            else [],
            "fact": FACT if question_type == "numeric" else None,
            "expected": {"ciks": [GOOGL]},
        }
    )


def make_passage(
    rank: int, chunk: str, text: str = "Cost of revenues 33,712"
) -> Passage:
    return Passage(
        rank=rank, score=1.0, chunk_ids=(chunk,), hit_chunk_ids=(chunk,),
        context="Alphabet 10-Q", text=text, tokens=5, corpus="sec_filings",
        company_name="Alphabet Inc.", cik=GOOGL, form="10-Q", document_type="10-Q",
        fiscal_year=2024, fiscal_period="Q1", filing_date="2024-04-25",
        section_title="Income statement", sec_url="https://example.invalid",
    )  # fmt: skip


def retrieved(
    passages: Sequence[Passage] = (), *, rerank_skipped: bool = False
) -> RetrievalResult:
    plan = QueryPlan("q", "en", ("sec_filings",), {}, source="llm")
    ranking = [
        {"chunk_id": p.hit_chunk_ids[0], "duplicate_chunk_ids": ["dup"]}
        for p in passages
    ]
    return RetrievalResult(
        question="q",
        plan=plan,
        passages=list(passages),
        ranking=ranking,
        companies=[],
        unresolved_companies=[],
        filters={},
        rerank_skipped=rerank_skipped,
    )


def answer(
    text: str, citations: Sequence[int] = (1,), *, refused: bool = False
) -> Answer:
    return Answer(text=text, citations=list(citations), insufficient=refused)


def judgment(verdict: str = "correct", faithful: bool | None = True) -> Judgment:
    return Judgment.model_validate(
        {
            "correctness": verdict,
            "correctness_reason": f"judged {verdict}",
            "faithful": faithful,
            "unsupported": [],
            "faithfulness_reason": None,
        }
    )


class Fakes:
    """Queued retrieval results, answers and judgments; an exception in a
    queue is raised instead."""

    def __init__(
        self,
        results: list[RetrievalResult | Exception],
        answers: list[Answer | Exception],
        judgments: list[Judgment | Exception] | None = None,
    ) -> None:
        self.results, self.answers, self.judgments = results, answers, judgments
        self.judged: list[tuple[str, str, Answer, list[Passage]]] = []

    @staticmethod
    def _next[T](queue: list[T | Exception]) -> T:
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def retrieve(self, _question: str) -> RetrievalResult:
        return self._next(self.results)

    def generate(self, _q: str, _p: Sequence[Passage], _l: Language) -> Answer:
        return self._next(self.answers)

    def judge(
        self, question: str, reference: str, given: Answer, passages: Sequence[Passage]
    ) -> Judgment:
        self.judged.append((question, reference, given, list(passages)))
        assert self.judgments is not None
        return self._next(self.judgments)

    def steps(self) -> Steps:
        judge = self.judge if self.judgments is not None else None
        return Steps(self.retrieve, self.generate, judge, query_model=True)


def run_rows(
    fakes: Fakes,
    pairs: Sequence[tuple[EvalQuestion, Language]],
    path: Path,
    relevance: dict[str, Groups] | None = None,
) -> list[dict[str, Any]]:
    groups = relevance or {q.id: [frozenset({"c1"})] for q, _ in pairs}
    collect(fakes.steps(), pairs, groups, path)
    return load_rows(path)


def test_metrics_of_a_cited_numeric_answer(tmp_path: Path) -> None:
    q = make_question("q1")
    fakes = Fakes(
        [retrieved([make_passage(1, "c9"), make_passage(2, "c1")])] * 2,
        [
            answer("Cost of revenues was $33.7 billion [2].", [2]),
            answer("Giá vốn là 33,7 tỷ USD [1].", [1]),
        ],
    )
    en, vi = run_rows(fakes, [(q, "en"), (q, "vi")], tmp_path / "a.jsonl")
    assert en["metrics"] == {
        "answer.value": 1.0,
        "answer.refusal": None,
        "answer.false_refusal": 0.0,
        "answer.cited": 1.0,
        "answer.citation_evidence": 1.0,
        "answer.language": 1.0,
    }
    assert en["retrieval"]["groups_found"] == [True]
    assert en["passages"][1]["evidence_groups"] == [0]
    assert en["passages"][0]["duplicate_chunk_ids"] == ["dup"]
    # Cites passage 1, which holds no evidence chunk.
    assert vi["metrics"]["answer.value"] == 1.0
    assert vi["metrics"]["answer.citation_evidence"] == 0.0
    assert failed_metrics(vi["metrics"]) == ["answer.citation_evidence"]


def test_metrics_by_question_kind() -> None:
    passages: list[dict[str, Any]] = [{"evidence_groups": []}]

    def metrics(q: EvalQuestion, given: Answer, language: str = "en") -> dict[str, Any]:
        row = {"language": language, "passages": passages, "answer": given.model_dump()}
        return row_metrics(q, row)

    refusal = answer("The filings do not say.", [], refused=True)
    unanswerable = metrics(
        make_question("u", "unanswerable", "Not in filings."), refusal
    )
    assert unanswerable["answer.refusal"] == 1.0
    assert unanswerable["answer.cited"] is None
    assert unanswerable["answer.value"] is None
    refused = metrics(make_question("q"), refusal)
    assert refused["answer.false_refusal"] == 1.0 and refused["answer.value"] is None
    assert failed_metrics(refused) == ["answer.false_refusal"]
    # Facts in any sign; references stated more precisely.
    outflow = metrics(make_question("q"), answer("Chi phí là (33.712) triệu USD [1]."))
    assert outflow["answer.value"] == 1.0
    rounded = make_question("t3", "table_lookup", "About $33.7 billion remained.")
    assert (
        metrics(rounded, answer("$33,712 million remained [1]."))["answer.value"] == 1.0
    )
    assert (
        metrics(rounded, answer("$33,812 million remained [1]."))["answer.value"] == 0.0
    )
    no_amount = make_question("t", "table_lookup", "Bradford L. Smith (age 64).")
    assert metrics(no_amount, answer("Smith, 64 [1]."))["answer.value"] is None
    lookup = make_question("t2", "table_lookup", "Net income was 56.6% of revenue.")
    assert metrics(lookup, answer("It was 56.6% [1]."))["answer.value"] == 1.0
    # A citation marker is not a stated value.
    assert metrics(lookup, answer("Not stated [56]."))["answer.value"] == 0.0
    wrong_language = metrics(make_question("q"), answer("$33,712 million [1]."), "vi")
    assert wrong_language["answer.language"] == 0.0
    uncited = metrics(make_question("q"), answer("$33,712 million.", []))
    assert uncited["answer.cited"] == 0.0


def test_is_vietnamese() -> None:
    assert is_vietnamese("Doanh thu của Alphabet là 80 tỷ USD.")
    assert not is_vietnamese("Revenue was $80 billion.")
    assert not is_vietnamese(
        "Nestlé reported revenue of CHF 93 billion in its annual report."
    )


def test_errors_and_degraded_rows_are_retried(tmp_path: Path) -> None:
    q1, q2 = make_question("q1"), make_question("q2")
    path = tmp_path / "answers.jsonl"
    fakes = Fakes(
        [
            OSError("qdrant down"),
            retrieved([make_passage(1, "c1")]),
            retrieved([make_passage(1, "c1")]),
            retrieved([make_passage(1, "c1")], rerank_skipped=True),
        ],
        [
            LLMProviderError("generator down"),
            answer("$33,712 million [1]."),
            answer("$33,712 million [1]."),
        ],
        [LLMProviderError("judge down"), judgment()],
    )
    pairs: list[tuple[EvalQuestion, Language]] = [
        (q1, "en"), (q1, "vi"), (q2, "en"), (q2, "vi"),
    ]  # fmt: skip
    failed, no_answer, unjudged, skipped = run_rows(fakes, pairs, path)
    assert failed["error"] == "OSError: qdrant down" and "answer" not in failed
    assert no_answer["degraded"] == ["generation failed"]
    assert no_answer["answer"] is None and no_answer["metrics"] == {}
    assert unjudged["degraded"] == ["judge failed"]
    assert unjudged["judge_error"] == "LLMProviderError: judge down"
    assert unjudged["metrics"]["answer.value"] == 1.0
    assert skipped["degraded"] == ["reranker skipped"]
    assert skipped["judgment"]["correctness"] == "correct"
    kept = resume_rows(path)
    assert kept == [] and pending([q1, q2], ["en", "vi"], kept) == pairs


def test_rejudge_stored_answers(tmp_path: Path) -> None:
    q, q2 = make_question("q1"), make_question("q2")
    source = tmp_path / "source.jsonl"
    rows = run_rows(
        Fakes(
            [
                retrieved([make_passage(1, "c1")]),
                OSError("down"),
                retrieved([make_passage(1, "c1")], rerank_skipped=True),
                retrieved([make_passage(1, "c1")], rerank_skipped=True),
            ],
            [answer("$33,712 million [1].")] * 3,
            [LLMProviderError("judge down"), judgment(), judgment()],
        ),
        [(q, "en"), (q, "vi"), (q2, "en"), (q2, "vi")],
        source,
    )
    fakes = Fakes(
        [], [], [judgment("partial", faithful=False), judgment(), LLMProviderError("x")]
    )
    run_dir = tmp_path / "rejudged"
    run_dir.mkdir()
    rejudge(fakes.judge, [q, q2], rows, run_dir / "answers.jsonl")
    judged, copied, kept_degraded, failed = load_rows(run_dir / "answers.jsonl")
    question, reference, given, passages = fakes.judged[0]
    assert (question, reference) == (q.question_en, q.reference_answer)
    assert given.text == "$33,712 million [1]."
    assert passages == [make_passage(1, "c1")]
    assert judged["degraded"] == [] and judged["judge_error"] is None
    assert judged["metrics"]["judge.partial"] == 1.0
    assert judged["metrics"]["judge.faithful"] == 0.0
    assert copied == rows[1]
    assert kept_degraded["degraded"] == ["reranker skipped"]
    assert failed["degraded"] == ["reranker skipped", "judge failed"]
    # Only judge failures count in a rejudge run.
    rows = [judged, copied, kept_degraded, failed]
    assert [to_retry(r, rejudged=True) for r in rows] == [False, False, False, True]
    assert [to_retry(r) for r in rows] == [False, True, True, True]
    write_outputs(run_dir, CONFIG | {"rejudged_from": "src_run"}, [q, q2])
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "to retry 1" in report and "## Judge errors" in report
    assert (
        "- q2 (vi): ['reranker skipped', 'judge failed'] LLMProviderError: x" in report
    )
    assert stored_passage(json.loads(json.dumps(rows[0]["passages"][0]))) == (
        make_passage(1, "c1")
    )


def test_report_and_scores(tmp_path: Path) -> None:
    numeric = make_question("q1")
    unanswerable = make_question("q2", "unanswerable", "The filings lack it.")
    fakes = Fakes(
        [
            retrieved([make_passage(1, "c1")]),
            retrieved([make_passage(1, "c9")]),
            retrieved([make_passage(1, "c9")]),
        ],
        [
            answer("Cost of revenues was $33,712 million [1]."),
            answer("Chi phí là 30 tỷ USD [1]."),
            answer("The filings do not say.", [], refused=True),
        ],
        [judgment(), judgment("incorrect", faithful=False), judgment(faithful=None)],
    )
    pairs: list[tuple[EvalQuestion, Language]] = [
        (numeric, "en"), (numeric, "vi"), (unanswerable, "en"),
    ]  # fmt: skip
    collect(
        fakes.steps(), pairs, {"q1": [frozenset({"c1"})]}, tmp_path / "answers.jsonl"
    )
    scored = write_outputs(tmp_path, CONFIG, [numeric, unanswerable])
    assert [retrieval_split(s.row) for s in scored] == ["found", "missed", None]
    scores = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))
    assert scores["en"]["answer.refusal"] == 1.0
    assert scores["en"]["judge.correct"] == 1.0
    assert scores["vi"]["judge.incorrect"] == 1.0
    assert scores["all"]["answer.value"] == 0.5
    assert scores["by_retrieval"]["en"] == {
        "found": scores["by_retrieval"]["en"]["found"],
        "missed": {},
    }
    assert scores["by_retrieval"]["vi"]["missed"]["answer.value"] == 0.0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| answer.value | 1.000 (1) | 0.000 (1) | 0.500 (2) |" in report
    assert "- **q1** (vi, numeric, evidence missed): answer.value" in report
    assert "judge.correct, judge.faithful" in report
    assert "  - answer: Chi phí là 30 tỷ USD [1]." in report
    assert "  - judge: judged incorrect" in report
    assert "q2" not in report.split("## Failing rows")[1]

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from src.data_pipeline.storage.native_rag.chunks import CHUNKS_SPEC
from src.evals.retrieval_eval import (
    Groups,
    Probe,
    collect,
    compare_runs,
    dimension_values,
    evidence_chunks,
    failures,
    load_rows,
    match_groups,
    pending,
    resume_rows,
    score,
    write_outputs,
)
from src.evals.testset import EvalQuestion, ExpectedFilters, Language
from src.llm import LLMProviderError
from src.rag_engine.retrieval.expand import Passage
from src.rag_engine.retrieval.query import QueryPlan
from src.rag_engine.retrieval.retriever import RetrievalResult, SearchUsed
from src.rag_engine.retrieval.search import SearchFilter
from tests.native_rag_support import gold_chunks, write_chunks

pytestmark = pytest.mark.unit

APPLE = "0000320193"
MSFT = "0000789019"
CONFIG: dict[str, Any] = {
    "label": "base",
    "languages": ["en", "vi"],
    "limit": None,
    "today": "2026-11-01",
    "questions": {"path": "q.jsonl", "sha256": "0" * 64, "count": 2},
    "chunks_rules_version": 8,
}


def question(
    qid: str,
    groups: int = 1,
    *,
    expected: ExpectedFilters | None = None,
    cell: dict[str, str] | None = None,
    companies: tuple[dict[str, str], ...] = (),
) -> EvalQuestion:
    evidence = [[{"block_id": f"b{i}", "quote": "net sales"}] for i in range(groups)]
    return EvalQuestion.model_validate(
        {
            "id": qid,
            "task_id": f"t-{qid}",
            "question_en": f"English question {qid}?",
            "question_vi": f"Câu hỏi tiếng Việt {qid}?",
            "reference_answer": "answer",
            "answerable": bool(groups),
            "question_type": "multi_hop" if groups else "unanswerable",
            "data_kind": "text" if groups else "none",
            "persona": "analyst",
            "mention_style": "name",
            "evidence_groups": evidence,
            "expected": expected or ExpectedFilters(ciks=(APPLE,)),
            "cell": cell or {},
            "companies": companies,
        }
    )


def passage(rank: int, chunk_ids: tuple[str, ...]) -> Passage:
    return Passage(
        rank, 1.0, chunk_ids, chunk_ids[:1], "ctx", "text", 10, "sec_filings",
        "Apple Inc.", APPLE, "10-K", "10-K", 2024, "FY", "2024-11-01",
        f"Section {rank}", "https://example.invalid",
    )  # fmt: skip


def result(
    ranking: Sequence[tuple[str, list[str], int, bool]] = (),
    *,
    language: str = "en",
    passages: list[Passage] | None = None,
    ciks: tuple[str, ...] = (APPLE,),
    unresolved: list[str] | None = None,
    source: str = "llm",
    rerank_skipped: bool = False,
    notes: tuple[str, ...] = (),
) -> RetrievalResult:
    """`ranking`: (chunk_id, duplicates, pre_rank, selected) in final order."""

    plan = QueryPlan(
        "q",
        language,
        ("sec_filings",),
        {},
        source="llm" if source == "llm" else "rules",
        notes=notes,
    )
    return RetrievalResult(
        question="q",
        plan=plan,
        passages=passages or [],
        ranking=[
            {"chunk_id": c, "duplicate_chunk_ids": d, "pre_rank": p, "selected": s}
            for c, d, p, s in ranking
        ],
        companies=[{"cik": c, "company_name": f"Co {c[-4:]}"} for c in ciks],
        unresolved_companies=unresolved or [],
        filters={},
        timings={"total_s": 1.0},
        searches=[SearchUsed("sec_filings", "col", SearchFilter(ciks=ciks))],
        rerank_skipped=rerank_skipped,
    )


def probe(answers: Sequence[RetrievalResult], *, admits: bool = True) -> Probe:
    queue = iter(answers)
    return Probe(lambda _: next(queue), lambda _s, _c: admits, query_model=True)


def run_rows(
    pairs: Sequence[tuple[EvalQuestion, Language, RetrievalResult]],
    relevance: dict[str, Groups],
    path: Path,
    *,
    admits: bool = True,
) -> list[dict[str, Any]]:
    collect(
        probe([r for _, _, r in pairs], admits=admits),
        [(q, lang) for q, lang, _ in pairs],
        relevance,
        path,
    )
    return load_rows(path)


def test_match_groups_needs_block_and_quote() -> None:
    q = question("q1", 2)
    holders = {"b0": {"c1", "c2"}, "b1": {"c3"}}
    texts = {"c1": "totalnetsales", "c2": "otherpart", "c3": "x"}  # normalized
    assert match_groups(q, holders, texts) == [frozenset({"c1"}), frozenset()]


def test_group_qrels_pre_post_and_duplicates(tmp_path: Path) -> None:
    q = question("q1", 2)
    relevance = {"q1": [frozenset({"a", "a2"}), frozenset({"b"})]}
    # Delivered: n (selected), then x whose duplicate is `a`; a2 adds nothing.
    res = result(
        [("n", [], 3, True), ("x", ["a"], 0, False), ("a2", [], 1, False),
         ("b", [], 2, False)]
    )  # fmt: skip
    rows = run_rows([(q, "en", res)], relevance, tmp_path / "r.jsonl")
    assert [e[0] for e in rows[0]["post"]] == ["n", "x", "a2", "b"]
    assert [e[0] for e in rows[0]["pre"]] == ["x", "a2", "b", "n"]
    m = score([q], rows, relevance).scored[0].metrics
    assert m["post.hit_rate@1"] == 0.0
    assert m["post.mrr@10"] == 0.5
    assert m["pre.hit_rate@1"] == 1.0
    assert m["pre.recall@8"] == 1.0  # both groups; a2 is not a third hit
    assert (m["pre.ndcg@10"] or 1.0) < 1.0  # b at rank 3, not 2
    assert m["post.recall@8"] == 1.0


def test_context_query_and_unanswerable_metrics(tmp_path: Path) -> None:
    answerable = question("q1", 2)
    outside = question(
        "q2",
        0,
        expected=ExpectedFilters(unknown_companies=("Tesla",)),
        cell={"reason": "company_outside_corpus"},
    )
    no_period = question("q3", 0, cell={"reason": "period_not_covered"})
    relevance = {"q1": [frozenset({"c1"}), frozenset({"c3"})]}
    rows = run_rows(
        [
            (answerable, "vi", result([("k", ["c3"], 0, True)], language="en",
                                      passages=[passage(1, ("k", "c0"))])),
            (outside, "en", result(ciks=(), unresolved=["Tesla"], language="vi")),
            (no_period, "en", result([("z", [], 0, True)], passages=[passage(1, ("z",))])),
        ],
        relevance,
        tmp_path / "r.jsonl",
        admits=False,
    )  # fmt: skip
    first, second, third = (s.metrics for s in score(
        [answerable, outside, no_period], rows, relevance).scored)  # fmt: skip
    assert first["context.recall"] == 0.5  # through the hit's duplicate c3
    assert first["query.filter_admits_evidence"] == 0.0
    assert first["query.language"] == 0.0
    assert first["query.company"] == 1.0
    # An English plan searches its own words; a Vietnamese one without an
    # English text searched the question itself.
    assert first["query.search_text"] == 1.0
    assert second["query.search_text"] == 0.0
    assert second["query.unknown_flagged"] == 1.0
    assert second["unanswerable.rejected"] == 1.0
    assert "post.hit_rate@8" not in second
    assert third["unanswerable.rejected"] is None  # judged at generation


def test_degraded_and_errors_are_retried(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    q1, q2, q3 = question("q1"), question("q2"), question("q3")
    bad = ("query model failed (LLMResponseError: no JSON): rules",)
    answers = iter(
        [
            result(),
            result(source="rules"),
            result(rerank_skipped=True),
            result(source="rules", notes=bad),
        ]
    )

    def retrieve(text: str) -> RetrievalResult:
        if "q2" in text and "Việt" in text:
            raise LLMProviderError("down")
        return next(answers)

    relevance = {q.id: [frozenset({"c"})] for q in (q1, q2, q3)}
    pairs = pending([q1, q2, q3], ["en"], []) + [(q2, "vi"), (q3, "vi")]
    collect(
        Probe(retrieve, lambda _s, _c: True, query_model=True), pairs, relevance, path
    )
    rows = load_rows(path)
    assert [r["degraded"] for r in rows[:3]] == [
        [], ["query model fell back to rules"], ["reranker skipped"]
    ]  # fmt: skip
    assert rows[3]["error"].startswith("LLMProviderError")
    assert rows[4]["degraded"] == []  # a bad reply would fail again
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"question_id": "q')  # cut by a crash
    with pytest.warns(UserWarning, match="unparsable last line"):
        kept = resume_rows(path)
    assert [(r["question_id"], r["language"]) for r in kept] == [
        ("q1", "en"),
        ("q3", "vi"),
    ]
    left = pending([q1, q2, q3], ["en", "vi"], kept)
    assert ("q1", "en") not in [(q.id, lang) for q, lang in left]


def test_breakdown_values_and_failures(tmp_path: Path) -> None:
    pair = question(
        "q1",
        cell={"cik": f"{APPLE}|{MSFT}", "reason": "-"},
        companies=({"cik": APPLE, "name": "Apple"}, {"cik": MSFT, "name": "Microsoft"}),
    )
    miss = question("q2")
    relevance: dict[str, Groups] = {"q1": [frozenset({"c1"})], "q2": [frozenset()]}
    rows = run_rows(
        [(pair, "en", result([("c1", [], 0, True)])),  # Microsoft missed
         (miss, "en", result([("c1", [], 0, True)]))],
        relevance,
        tmp_path / "r.jsonl",
    )  # fmt: skip
    scoring = score([pair, miss], rows, relevance)
    assert dimension_values(scoring.scored[0], "cik") == ["Apple", "Microsoft"]
    assert dimension_values(scoring.scored[1], "form") == ["-"]
    assert [s.question.id for s in failures(scoring)] == ["q2"]
    assert scoring.unmapped == [("q2", 0)]
    assert rows[1]["filter_admits"] == [None]
    assert scoring.scored[1].metrics["query.filter_admits_evidence"] is None


def test_one_chunk_covering_several_groups(tmp_path: Path) -> None:
    q = question("q1", 2)
    relevance = {"q1": [frozenset({"ac"}), frozenset({"ac", "x"})]}
    rows = run_rows(
        [(q, "en", result([("ac", [], 0, True)]))], relevance, tmp_path / "r"
    )
    m = score([q], rows, relevance).scored[0].metrics
    assert (m["post.recall@8"], m["post.ndcg@10"], m["post.hit_rate@1"]) == (1, 1, 1)


def test_evidence_chunks_from_gold(tmp_path: Path) -> None:
    write_chunks(tmp_path, gold_chunks())
    assert evidence_chunks(tmp_path, [question("u", 0)]) == (
        {},
        {},
        CHUNKS_SPEC.rules_version,
    )
    found = EvalQuestion.model_validate(
        question("q1").model_dump()
        | {
            "evidence_groups": [
                [
                    {"block_id": "d24:block:000002", "quote": "REVENUE  note 2"},
                    {
                        "block_id": "d24:block:000002",
                        "quote": "REVENUE",
                        "widened": True,
                    },
                ]
            ]
        }
    )
    relevance, original, _ = evidence_chunks(tmp_path, [found])
    assert [len(g) for g in relevance["q1"]] == [1]
    assert original == relevance  # the widened row is in the same chunk here
    own = EvalQuestion.model_validate(
        found.model_dump()
        | {"evidence_groups": [[found.evidence_groups[0][1].model_dump()]]}
    )
    _, original, _ = evidence_chunks(tmp_path, [own])
    assert original == {"q1": [frozenset()]}  # widened rows only: not admitted


def test_outputs_report_and_compare(tmp_path: Path) -> None:
    questions = [question("q1"), question("q2", cell={"reason": "-"})]
    relevance = {"q1": [frozenset({"c1"})], "q2": [frozenset({"c3"})]}
    run_dirs = []
    for name, top in (("a", "c1"), ("b", "zz")):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
        run_rows(
            [
                (
                    q,
                    lang,
                    result([(top, [], 0, True), ("c3", [], 1, True)], language=lang),
                )
                for q in questions
                for lang in ("en", "vi")
            ],
            relevance,
            run_dir / "results.jsonl",
        )
        write_outputs(run_dir, CONFIG, questions, relevance)
        run_dirs.append(run_dir)
    report = (run_dirs[0] / "report.md").read_text(encoding="utf-8")
    assert "| post.hit_rate@8 | 1.000 (2) | 1.000 (2) | 1.000 (4) |" in report
    assert "### reason" in report
    assert "unanswerable.rejected en" in report
    scores = json.loads((run_dirs[0] / "scores.json").read_text(encoding="utf-8"))
    assert scores["vi"]["post.mrr@10"] == 0.75
    compared = compare_runs(run_dirs, "fisher", 0.05)
    assert "## vi" in compared
    assert "MRR@10" in compared
    other = CONFIG | {"languages": ["en"]}
    (run_dirs[1] / "config.json").write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(ValueError, match="languages"):
        compare_runs(run_dirs)

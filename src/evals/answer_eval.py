"""Answer eval: questions of a frozen set -> retriever -> answer model -> scores.

Each (question, language) row keeps the answer, the passages it was
generated from (full text, so a judge can rerun on stored rows) with the
evidence groups each passage holds (`retrieval_eval.evidence_chunks`: same
relevant chunks as the retrieval eval), and an optional judgment. Rows are
scored from `answers.jsonl` alone. Deterministic metrics (None when not
applicable):

- `answer.value`: numeric/table_lookup/calc/false_premise non-refusals
  state the expected value: the XBRL fact (any sign), a computed value
  (`Computed`: a percent within half its first decimal, or as rounded as
  the answer writes it; a growth, CAGR or difference with its direction,
  `checks.states_computed_answer`), else the first amount of the reference
  answer, skipping a share count the question does not ask for, at least
  as precisely as it shows it (`numbers`); a false_premise answer without a computed value
  must also deny the premise (`checks.denies_premise`), and with no amount
  only that is checked.
- `answer.order`: ranking non-refusals name the companies of
  `Computed.order` in order, each placed by its mention nearest its value;
  reversed only when the question or answer asks for the lowest
  (`checks.ranking_holds`).
- `answer.refusal`: unanswerable questions answered with a refusal.
- `answer.false_refusal`: answerable questions refused (lower is better).
- `answer.cited`: non-refusals citing a passage.
- `answer.citation_evidence`: answerable non-refusals citing a passage that
  holds an evidence chunk.
- `answer.language`: written in the question's language.

With a judge: `judge.correct` / `judge.partial` / `judge.incorrect` shares
and `judge.faithful` (non-refusals).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from src.evals.answer_judge import Judgment
from src.evals.checks import denies_premise, ranking_holds, states_computed_answer
from src.evals.numbers import expected_amount, states_expected, text_states
from src.evals.retrieval_eval import (
    Groups,
    Row,
    Scored,
    breakdown,
    load_rows,
    means,
    retried,
)
from src.evals.testset import EvalQuestion, Language
from src.llm import LLMError
from src.rag_engine.generation.answer import Answer
from src.rag_engine.retrieval.expand import Passage
from src.rag_engine.retrieval.factory import SERVER_ERRORS
from src.rag_engine.retrieval.retriever import RetrievalResult

ANSWER_METRICS = (
    "answer.value",
    "answer.order",
    "answer.refusal",
    "answer.false_refusal",
    "answer.cited",
    "answer.citation_evidence",
    "answer.language",
)
JUDGE_METRICS = ("judge.correct", "judge.partial", "judge.incorrect", "judge.faithful")
METRIC_KEYS = (*ANSWER_METRICS, *JUDGE_METRICS)
# Shown in the report only when a row has them (ranking questions).
OPTIONAL_KEYS = frozenset({"answer.order"})
# Columns of the per-type and per-tag tables.
BREAKDOWN_KEYS = (
    "answer.value",
    "answer.refusal",
    "answer.false_refusal",
    "answer.citation_evidence",
    "judge.correct",
)
# A row fails these at 1 (the others at 0); partial/incorrect show as judge.correct.
FAIL_AT_ONE = frozenset({"answer.false_refusal"})
NOT_FAILURES = frozenset({"judge.partial", "judge.incorrect"})
VALUE_TYPES = frozenset({"numeric", "table_lookup", "calc", "false_premise"})
RETRIEVAL_SPLITS = ("found", "missed")
EXCERPT_CHARS = 160
JUDGE_FAILED = "judge failed"
# Query-model failures named in plan notes that are not transport errors
# (retrieval_eval._BAD_REPLY).
_BAD_REPLY = ("LLMResponseError", "LLMRefusalError")
_CITATION = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_VIETNAMESE = re.compile(
    "[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)
# Vietnamese text: at least this many Vietnamese letters and this share of letters.
VI_MIN_LETTERS = 2
VI_MIN_SHARE = 0.02

Generate = Callable[[str, Sequence[Passage], Language], Answer]
Judge = Callable[[str, str, Answer, Sequence[Passage]], Judgment]


@dataclass(frozen=True)
class Steps:
    """What a row needs from the servers (fakes in tests)."""

    retrieve: Callable[[str], RetrievalResult]
    generate: Generate
    judge: Judge | None
    query_model: bool  # a query model is configured


def _error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"[:500]


def _retrieval_degraded(result: RetrievalResult, *, query_model: bool) -> list[str]:
    degraded = []
    bad_reply = any(e in n for n in result.plan.notes for e in _BAD_REPLY)
    if query_model and result.plan.source != "llm" and not bad_reply:
        degraded.append("query model fell back to rules")
    if result.rerank_skipped:
        degraded.append("reranker skipped")
    return degraded


def passage_rows(result: RetrievalResult, groups: Groups) -> list[Row]:
    """Each passage as stored: every `Passage` field, the chunks dedup
    collapsed into its hits, and the evidence groups it holds."""

    duplicates = {r["chunk_id"]: r["duplicate_chunk_ids"] for r in result.ranking}
    rows = []
    for p in result.passages:
        collapsed = [d for h in p.hit_chunk_ids for d in duplicates.get(h, ())]
        held = {*p.chunk_ids, *collapsed}
        rows.append(
            p.as_dict()
            | {
                "duplicate_chunk_ids": collapsed,
                "evidence_groups": [i for i, g in enumerate(groups) if g & held],
            }
        )
    return rows


def stored_passage(row: Mapping[str, Any]) -> Passage:
    """A `Passage` back from its stored row; KeyError when a field is missing."""

    values = {f.name: row[f.name] for f in fields(Passage)}
    for name in ("chunk_ids", "hit_chunk_ids"):
        values[name] = tuple(values[name])
    return Passage(**values)


def _judge(judge: Judge, question: EvalQuestion, row: Row, answer: Answer) -> None:
    """Store the judgment in `row`, or mark the row degraded."""

    passages = [stored_passage(p) for p in row["passages"]]
    row["degraded"] = [d for d in row["degraded"] if d != JUDGE_FAILED]
    row["judge_error"] = None
    try:
        judgment = judge(row["question"], question.reference_answer, answer, passages)
    except LLMError as error:
        row["judgment"] = None
        row["degraded"].append(JUDGE_FAILED)
        row["judge_error"] = _error(error)
        return
    row["judgment"] = judgment.model_dump()


def answer_row(
    steps: Steps, question: EvalQuestion, language: Language, groups: Groups
) -> Row:
    """Retrieve, answer and judge one pair. A retrieval server error makes an
    error row; an `LLMError` of generation or judging a degraded row."""

    text = question.text(language)
    row: Row = {"question_id": question.id, "language": language, "question": text}
    try:
        result = steps.retrieve(text)
    except SERVER_ERRORS as error:
        return row | {"error": _error(error)}
    passages = passage_rows(result, groups)
    found = {i for p in passages for i in p["evidence_groups"]}
    row |= {
        "passages": passages,
        "retrieval": {
            "groups_found": [i in found for i in range(len(groups))],
            "plan_source": result.plan.source,
            "notes": result.notes,
        },
        "answer": None,
        "judgment": None,
        "degraded": _retrieval_degraded(result, query_model=steps.query_model),
        "error": None,
        "judge_error": None,
    }
    try:
        answer = steps.generate(text, result.passages, language)
    except LLMError as error:
        row["degraded"].append("generation failed")
        row["error"] = _error(error)
    else:
        row["answer"] = answer.model_dump()
        if steps.judge is not None:
            _judge(steps.judge, question, row, answer)
    row["metrics"] = row_metrics(question, row)
    return row


def collect(
    steps: Steps,
    pairs: Iterable[tuple[EvalQuestion, Language]],
    relevance: Mapping[str, Groups],
    path: Path,
) -> None:
    """Append one row per pair, flushed, so an interrupted run resumes."""

    with path.open("a", encoding="utf-8") as handle:
        for question, language in pairs:
            row = answer_row(steps, question, language, relevance.get(question.id, []))
            _write(handle, row)


def rejudge(
    judge: Judge, questions: Sequence[EvalQuestion], rows: Iterable[Row], path: Path
) -> None:
    """Judge the stored answers of `rows` again (no retrieval, no generation),
    appending every row to `path`; rows without an answer are copied."""

    by_id = {q.id: q for q in questions}
    with path.open("a", encoding="utf-8") as handle:
        for stored in rows:
            row = dict(stored)
            question = by_id.get(row["question_id"])
            if question is not None and row.get("answer") is not None:
                answer = Answer.model_validate(row["answer"])
                _judge(judge, question, row, answer)
                row["metrics"] = row_metrics(question, row)
            _write(handle, row)


def _write(handle: Any, row: Row) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    handle.flush()


# Metrics


def is_vietnamese(text: str) -> bool:
    letters = sum(c.isalpha() for c in text)
    marked = len(_VIETNAMESE.findall(text))
    return marked >= max(VI_MIN_LETTERS, VI_MIN_SHARE * letters)


# A share count in a reference answer, and an English question asking a count.
_SHARE_COUNT = re.compile(
    r"\d[\d,.]*\s*(?:(?:thousand|million|billion)\s+)?shares\b", re.IGNORECASE
)
_ASKS_COUNT = re.compile(r"\bhow many\b|\bnumber of\b", re.IGNORECASE)


def states_value(question: EvalQuestion, text: str) -> bool | None:
    """Whether `text` states the question's value (None: no value to check):
    the fact to its shown rounding, a computed value (a percent as rounded
    as `text` writes it), else the reference's first amount, at least as
    precisely as the reference shows it; a share count is skipped unless
    the question asks how many ("repurchased 3,046 shares at an average
    price of $53.80": the price)."""

    text = _CITATION.sub(" ", text)
    if question.fact is not None:
        return text_states(text, question.fact.value, question.fact.decimals)
    computed = question.computed
    if computed is not None and computed.value is not None:
        return states_computed_answer(computed, text, rounded=True)
    reference = question.reference_answer
    if not _ASKS_COUNT.search(question.question_en):
        reference = _SHARE_COUNT.sub(" ", reference)
    expected = expected_amount(reference)
    return None if expected is None else states_expected(text, expected)


def answer_value(question: EvalQuestion, text: str) -> bool | None:
    """`answer.value` of a non-refusal: `states_value` (a computed change
    with its direction); a false premise without a computed value must
    also be denied (only that when there is no value)."""

    stated = states_value(question, text)
    if question.question_type != "false_premise" or question.computed is not None:
        return stated
    return denies_premise(text) and stated is not False


def answer_order(question: EvalQuestion, text: str, asked: str) -> bool | None:
    """`answer.order`: a ranking names its companies in order, reversed
    only when the question `asked` or the answer wants the lowest (None for
    other questions)."""

    computed = question.computed
    if question.question_type != "ranking" or computed is None:
        return None
    return ranking_holds(computed, question.companies, text, asked)


def _judge_metrics(judgment: Mapping[str, Any] | None) -> dict[str, float | None]:
    if judgment is None:
        return {}
    verdict = judgment["correctness"]
    found: dict[str, float | None] = {
        f"judge.{v}": float(verdict == v) for v in ("correct", "partial", "incorrect")
    }
    faithful = judgment["faithful"]
    found["judge.faithful"] = None if faithful is None else float(faithful)
    return found


def row_metrics(question: EvalQuestion, row: Row) -> dict[str, float | None]:
    """Metrics of a row with an answer ({} without one)."""

    stored = row.get("answer")
    if stored is None:
        return {}
    answer = Answer.model_validate(stored)
    refused, answerable = answer.insufficient, question.answerable
    passages = row["passages"]
    cited = [passages[n - 1] for n in answer.citations if 0 < n <= len(passages)]
    value = (
        answer_value(question, answer.text)
        if answerable and not refused and question.question_type in VALUE_TYPES
        else None
    )
    order = (
        None
        if refused
        else answer_order(question, answer.text, row.get("question", ""))
    )
    vietnamese = is_vietnamese(answer.text)
    return {
        "answer.value": None if value is None else float(value),
        "answer.order": None if order is None else float(order),
        "answer.refusal": None if answerable else float(refused),
        "answer.false_refusal": float(refused) if answerable else None,
        "answer.cited": None if refused else float(bool(answer.citations)),
        "answer.citation_evidence": (
            float(any(p["evidence_groups"] for p in cited))
            if answerable and not refused
            else None
        ),
        "answer.language": float(vietnamese == (row["language"] == "vi")),
        **_judge_metrics(row.get("judgment")),
    }


def retrieval_split(row: Row) -> str | None:
    """`found` when the passages hold every evidence group, `missed` when
    not, None for unanswerable questions (no groups)."""

    groups = row.get("retrieval", {}).get("groups_found", [])
    if not groups:
        return None
    return "found" if all(groups) else "missed"


def score(questions: Sequence[EvalQuestion], rows: Iterable[Row]) -> list[Scored]:
    """Rows with an answer, scored (rows of other questions are dropped)."""

    by_id = {q.id: q for q in questions}
    return [
        Scored(by_id[r["question_id"]], r, row_metrics(by_id[r["question_id"]], r))
        for r in rows
        if r["question_id"] in by_id and r.get("answer") is not None
    ]


def to_retry(row: Row, *, rejudged: bool = False) -> bool:
    """A row a resume would retry; in a rejudge run, only a failed judge
    (rows copied from the source run keep their own retrieval state)."""

    return JUDGE_FAILED in row.get("degraded", []) if rejudged else retried(row)


def failed_metrics(metrics: Mapping[str, float | None]) -> list[str]:
    return [
        k
        for k in METRIC_KEYS
        if k not in NOT_FAILURES
        and (v := metrics.get(k)) is not None
        and v == (1.0 if k in FAIL_AT_ONE else 0.0)
    ]


def summary(scored: Sequence[Scored], languages: Sequence[str]) -> dict[str, Any]:
    """`scores.json`: means per language and overall, and per language split
    by whether retrieval found every evidence group."""

    def plain(items: Iterable[Scored]) -> dict[str, float]:
        return {k: v[0] for k, v in means(items).items()}

    per = {lang: [s for s in scored if s.row["language"] == lang] for lang in languages}
    return {
        **{lang: plain(items) for lang, items in per.items()},
        "all": plain(scored),
        "by_retrieval": {
            lang: {
                split: plain(s for s in items if retrieval_split(s.row) == split)
                for split in RETRIEVAL_SPLITS
            }
            for lang, items in per.items()
        },
    }


# Report


def _table(header: Sequence[str], lines: Iterable[Sequence[str]]) -> list[str]:
    rows = [header, ["---"] * len(header), *lines]
    return [*("| " + " | ".join(row) + " |" for row in rows), ""]


def _metric_table(columns: Mapping[str, Sequence[Scored]]) -> list[str]:
    found = {name: means(items) for name, items in columns.items()}
    keys = [
        k
        for k in METRIC_KEYS
        if k not in OPTIONAL_KEYS or any(k in m for m in found.values())
    ]

    def cell(value: tuple[float, int] | None) -> str:
        return "–" if value is None else f"{value[0]:.3f} ({value[1]})"

    return _table(
        ["metric", *columns],
        ([k, *(cell(m.get(k)) for m in found.values())] for k in keys),
    )


def _excerpt(text: str) -> str:
    flat = " ".join(text.split())
    short = flat if len(flat) <= EXCERPT_CHARS else flat[: EXCERPT_CHARS - 1] + "…"
    return short.replace("|", "/")


def _failure_lines(scored: Sequence[Scored]) -> list[str]:
    lines = []
    for item in scored:
        failed = failed_metrics(item.metrics)
        if not failed:
            continue
        question, row = item.question, item.row
        split = retrieval_split(row) or "unanswerable"
        lines += [
            (
                f"- **{question.id}** ({row['language']}, {question.question_type}, "
                f"evidence {split}): {', '.join(failed)}"
            ),
            f"  - answer: {_excerpt(row['answer']['text'])}",
            f"  - reference: {_excerpt(question.reference_answer)}",
        ]
        if (judgment := row.get("judgment")) is not None:
            lines.append(f"  - judge: {_excerpt(judgment['correctness_reason'])}")
    return lines


def render_report(
    config: Mapping[str, Any], scored: Sequence[Scored], rows: Sequence[Row]
) -> str:
    languages = list(config["languages"])
    q = config["questions"]
    by_language = {
        lang: [s for s in scored if s.row["language"] == lang] for lang in languages
    }
    splits = {
        f"{lang} {split}": [s for s in items if retrieval_split(s.row) == split]
        for lang, items in by_language.items()
        for split in RETRIEVAL_SPLITS
    }
    source = config.get("rejudged_from")
    retry = [r for r in rows if to_retry(r, rejudged=bool(source))]
    rejudged = f"; rejudged from {source}" if source else ""
    split = f"; split {config['split']}" if config.get("split") else ""
    lines = [
        f"# Answer eval: {config['label']}",
        "",
        f"- questions: {q['path']} (sha256 {q['sha256'][:12]}, {q['count']}){split}; today {config['today']}",
        f"- models: {config.get('models')}",
        f"- judge: {'yes' if config.get('judge') else 'no'}{rejudged}",
        f"- rows scored {len(scored)}, to retry {len(retry)}",
        "",
        (
            "Cells: mean (rows where the metric applies). `answer.false_refusal`: "
            "lower is better. `found` / `missed`: the passages held every evidence "
            "group / not (answerable questions only)."
        ),
        "",
        "## Overall",
        "",
        *_metric_table({**by_language, "all": scored}),
        "## By retrieval",
        "",
        *_metric_table(splits),
        "## Breakdowns",
        "",
        *breakdown(scored, "question_type", languages, _breakdown_keys(scored)),
        *(
            breakdown(scored, "tag", languages, _breakdown_keys(scored))
            if any(s.question.tags for s in scored)
            else []
        ),
        "## Failing rows",
        "",
        *(_failure_lines(scored) or ["None."]),
    ]
    if retry:
        title = (
            "Judge errors" if source else "Errors and degraded rows (retried on resume)"
        )
        lines += ["", f"## {title}", ""]
        lines += [
            f"- {r['question_id']} ({r['language']}): {r.get('degraded') or ''} "
            f"{r.get('error') or r.get('judge_error') or ''}".rstrip()
            for r in retry
        ]
    return "\n".join(lines) + "\n"


def _breakdown_keys(scored: Sequence[Scored]) -> list[str]:
    order = any(s.metrics.get("answer.order") is not None for s in scored)
    return [BREAKDOWN_KEYS[0], *(["answer.order"] * order), *BREAKDOWN_KEYS[1:]]


def write_outputs(
    run_dir: Path, config: Mapping[str, Any], questions: Sequence[EvalQuestion]
) -> list[Scored]:
    """Score `answers.jsonl`; write `scores.json` and `report.md`."""

    rows = load_rows(run_dir / "answers.jsonl")
    scored = score(questions, rows)
    (run_dir / "scores.json").write_text(
        json.dumps(summary(scored, config["languages"]), indent=2) + "\n",
        encoding="utf-8",
    )
    report = render_report(config, scored, rows)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return scored

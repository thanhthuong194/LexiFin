"""Retrieval eval: questions of a frozen set -> retriever -> scores.

Relevance comes from the current Gold chunks: a chunk matches an evidence
when it holds its block and shows its quote (`normalize` on both sides); a
group is found when a retrieved chunk, or a same-text chunk dedup collapsed
into it, matches any of its evidence. Each (question, language) row keeps
three lists: `pre` (the candidate pool in fused order, before rerank),
`post` (the delivered top k in final order, then the rest) and the passages
an answer model reads. ranx scores groups, not chunks: each candidate counts
for the first group it newly covers, so recall@k is group recall and a
question needing two passages has an ideal DCG the ranking can reach.
Rows are scored from `results.jsonl` alone, so a run rescores or resumes.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from qdrant_client import QdrantClient, models
from ranx import Qrels, Run, compare, evaluate

from src.data_pipeline.storage.bronze import atomic_write_text
from src.data_pipeline.storage.iceberg import (
    open_gold_catalog_read_only,
    read_rows_in,
    scan_batches,
    table_rules_version,
    values_filter,
)
from src.data_pipeline.storage.native_rag.chunks import CHUNKS_TABLE
from src.evals.testset import CELL_KEYS, EvalQuestion, Language, normalize
from src.rag_engine.indexing.qdrant_store import point_id
from src.rag_engine.retrieval.corpora import CORPORA
from src.rag_engine.retrieval.factory import SERVER_ERRORS
from src.rag_engine.retrieval.retriever import RetrievalResult, SearchUsed
from src.rag_engine.retrieval.search import qdrant_filter

METRICS = (
    "hit_rate@1",
    "hit_rate@5",
    "hit_rate@8",
    "mrr@10",
    "ndcg@10",
    "recall@8",
    "recall@50",
)
LISTS = ("pre", "post")
METRIC_KEYS = (
    *(f"{lst}.{m}" for m in METRICS for lst in LISTS),
    "context.recall",
    "query.company",
    "query.unknown_flagged",
    "query.filter_admits_evidence",
    "query.language",
    "query.llm_plan",
    "query.search_text",
    "unanswerable.rejected",
)
HEADLINE = ("post.hit_rate@8", "post.mrr@10", "post.recall@8", "context.recall")
BREAKDOWNS = ("question_type", "data_kind", *CELL_KEYS)
# Columns of a breakdown table, when not HEADLINE.
BREAKDOWN_COLUMNS = {"reason": ("unanswerable.rejected", "query.unknown_flagged")}
COMPARE_METRICS = ("hit_rate@8", "mrr@10", "ndcg@10")
FAILURE_K = 8
# Only this reason can be seen at retrieval; the others at generation.
REJECTABLE = "company_outside_corpus"
# Query-model failures named in plan notes that are not transport errors.
_BAD_REPLY = ("LLMResponseError", "LLMRefusalError")

Row = dict[str, Any]
Groups = list[frozenset[str]]  # per evidence group, the chunk IDs matching it


def evidence_chunks(
    gold_dir: Path, questions: Sequence[EvalQuestion]
) -> tuple[dict[str, Groups], dict[str, Groups], int]:
    """Question ID -> matching chunks per group, the same for the writer's
    evidence only (`match_groups`), and the chunks rules version.
    Reads `block_ids` of every chunk, `text` only of chunks holding evidence."""

    wanted = sorted({b for q in questions for b in q.evidence_blocks})
    blocks = pa.array(wanted, type=pa.string())
    # A block ID is `{document_id}:block:{ordinal}`.
    documents = values_filter(
        "document_id", {b.rsplit(":block:", 1)[0] for b in wanted}
    )
    holders: dict[str, set[str]] = defaultdict(set)  # block -> chunk IDs
    with open_gold_catalog_read_only(gold_dir) as gold:
        table = gold.load_table(CHUNKS_TABLE)
        version = table_rules_version(table)
        for batch in scan_batches(table, ["chunk_id", "block_ids"], files=documents):
            lists = batch.column("block_ids")
            flat = pc.list_flatten(lists)
            mask = pc.is_in(flat, value_set=blocks)
            if not pc.any(mask).as_py():
                continue
            parents = pc.filter(pc.list_parent_indices(lists), mask)
            chunk_ids = pc.take(batch.column("chunk_id"), parents).to_pylist()
            for block, chunk in zip(
                pc.filter(flat, mask).to_pylist(), chunk_ids, strict=True
            ):
                holders[block].add(chunk)
        ids = sorted({c for chunks in holders.values() for c in chunks})
        texts = {
            r["chunk_id"]: normalize(r["text"])
            for r in read_rows_in(
                table,
                "chunk_id",
                ids,
                columns=["text"],
                files=values_filter("chunk_id", ids),
            )
        }
    relevance = {
        q.id: match_groups(q, holders, texts) for q in questions if q.answerable
    }
    original = {
        q.id: match_groups(q, holders, texts, original_only=True)
        for q in questions
        if q.answerable
    }
    return relevance, original, version


class Evidenced(NamedTuple):
    """A question's matching chunks per evidence group: all evidence, and
    the writer's own (filters admit the question when these pass)."""

    groups: Groups
    own: Groups


def match_groups(
    question: EvalQuestion,
    holders: Mapping[str, set[str]],
    texts: Mapping[str, str],
    *,
    original_only: bool = False,
) -> Groups:
    """Per group, the chunks holding one of its blocks and showing its quote
    (`texts`: chunk ID -> `normalize`d text); with `original_only`, of the
    writer's evidence only, not the rows widening added."""

    return [
        frozenset(
            c
            for e in group
            if not (original_only and e.widened)
            for c in holders.get(e.block_id, ())
            if normalize(e.quote) in texts[c]
        )
        for group in question.evidence_groups
    ]


def filter_admits(
    client: QdrantClient, searches: Sequence[SearchUsed], chunks: frozenset[str]
) -> bool:
    """Whether a filter the retriever ended with admits one of `chunks`."""

    if not chunks:
        return False
    has_id = models.HasIdCondition(has_id=[point_id(c) for c in sorted(chunks)])
    for search in searches:
        scope = qdrant_filter(search.filter)
        found = client.count(
            search.collection,
            count_filter=models.Filter(must=[has_id, *([scope] if scope else [])]),
            exact=True,
        )
        if found.count:
            return True
    return False


# Rows


@dataclass(frozen=True)
class Probe:
    """What a row needs from the retriever (fakes in tests)."""

    retrieve: Callable[[str], RetrievalResult]
    admits: Callable[[Sequence[SearchUsed], frozenset[str]], bool]
    query_model: bool  # a query model is configured


def result_row(
    question: EvalQuestion,
    language: Language,
    result: RetrievalResult,
    probe: Probe,
    evidenced: Evidenced,
) -> Row:
    def entry(candidate: Mapping[str, Any]) -> list[str]:
        return [candidate["chunk_id"], *candidate["duplicate_chunk_ids"]]

    ranking = result.ranking
    duplicates = {r["chunk_id"]: r["duplicate_chunk_ids"] for r in ranking}
    delivered = [r for r in ranking if r["selected"]]
    plan = result.plan
    degraded = []
    # A bad reply (not a transport error) would fall back again on retry.
    bad_reply = any(e in n for n in plan.notes for e in _BAD_REPLY)
    if probe.query_model and plan.source != "llm" and not bad_reply:
        degraded.append("query model fell back to rules")
    if result.rerank_skipped:
        degraded.append("reranker skipped")
    return {
        "question_id": question.id,
        "language": language,
        "question": question.text(language),
        "plan": {
            "source": plan.source,
            "language": plan.language,
            "companies": [asdict(m) for m in plan.companies],
            "fiscal_years": list(plan.fiscal_years),
            "fiscal_periods": list(plan.fiscal_periods),
            "calendar": plan.calendar,
            "forms": list(plan.forms),
            "topics": list(plan.topics),
            "wants_latest": plan.wants_latest,
            "filed_months": list(plan.filed_months),
            "texts": dict(plan.texts),
            # Corpus languages searched with the question itself, untranslated.
            "untranslated": sorted(
                {CORPORA[c].language for c in plan.corpora if c in CORPORA}
                - {*plan.texts, plan.language}
            ),
        },
        "companies": result.companies,
        "unresolved_companies": result.unresolved_companies,
        "searches": [asdict(s) for s in result.searches],
        "filter_admits": [
            probe.admits(result.searches, g) if g else None for g in evidenced.own
        ],
        "notes": result.notes,
        "timings": result.timings,
        "degraded": degraded,
        "pre": [entry(r) for r in sorted(ranking, key=lambda r: r["pre_rank"])],
        "post": [
            entry(r) for r in delivered + [r for r in ranking if not r["selected"]]
        ],
        "passages": [
            {
                "rank": p.rank,
                "chunk_ids": list(p.chunk_ids),
                "hit_chunk_ids": list(p.hit_chunk_ids),
                "duplicate_chunk_ids": [
                    d for h in p.hit_chunk_ids for d in duplicates.get(h, ())
                ],
                "company_name": p.company_name,
                "form": p.form,
                "fiscal_year": p.fiscal_year,
                "fiscal_period": p.fiscal_period,
                "section_title": p.section_title,
            }
            for p in result.passages
        ],
        "error": None,
    }


def collect(
    probe: Probe,
    pairs: Iterable[tuple[EvalQuestion, Language]],
    relevance: Mapping[str, Groups],
    path: Path,
    original: Mapping[str, Groups] | None = None,
) -> None:
    """Append one row per pair (flushed, so an interrupted run resumes); a
    server error is recorded and the run goes on."""

    with path.open("a", encoding="utf-8") as handle:
        for question, language in pairs:
            try:
                result = probe.retrieve(question.text(language))
                groups = relevance.get(question.id, [])
                own = (original or relevance).get(question.id, groups)
                evidenced = Evidenced(groups, own)
                row = result_row(question, language, result, probe, evidenced)
            except SERVER_ERRORS as error:
                row = {
                    "question_id": question.id,
                    "language": language,
                    "question": question.text(language),
                    "error": f"{type(error).__name__}: {error}"[:500],
                }
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()


def load_rows(path: Path) -> list[Row]:
    """Rows of `results.jsonl`; a cut last line (interrupted write) is skipped."""

    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = []
    for number, line in enumerate(lines, start=1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if number < len(lines):
                raise
            warnings.warn(f"{path}: unparsable last line skipped", stacklevel=2)
    return rows


def retried(row: Row) -> bool:
    return bool(row.get("error") or row.get("degraded"))


def resume_rows(path: Path) -> list[Row]:
    """Rows a resumed run keeps; failed and degraded rows are retried."""

    kept = [r for r in load_rows(path) if not retried(r)]
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
    atomic_write_text(path, text, overwrite=True)
    return kept


def pending(
    questions: Sequence[EvalQuestion],
    languages: Sequence[Language],
    done: Iterable[Row],
) -> list[tuple[EvalQuestion, Language]]:
    seen = {(r["question_id"], r["language"]) for r in done}
    return [
        (q, lang) for q in questions for lang in languages if (q.id, lang) not in seen
    ]


# Scoring


@dataclass(frozen=True)
class Scored:
    question: EvalQuestion
    row: Row
    metrics: dict[str, float | None]


@dataclass(frozen=True)
class Scoring:
    scored: list[Scored]
    errors: list[Row]
    qrels: Any  # ranx Qrels
    runs: dict[str, Any]  # f"{list}_{language}" -> ranx Run
    unmapped: list[tuple[str, int]]  # (question ID, group index)


def _multi_group(
    prefix: str, entries: Sequence[Sequence[str]], groups: Groups
) -> dict[str, float | None]:
    """recall@k and nDCG@k when one candidate may cover several groups (ranx
    takes one relevant doc per rank): each group gains 1 at the rank of the
    first candidate covering it; ideal DCG puts one group per rank, so nDCG
    is capped at 1."""

    ranks = [
        next((r for r, e in enumerate(entries, start=1) if g & set(e)), None)
        for g in groups
    ]
    found: dict[str, float | None] = {}
    for m in METRICS:
        name, _, at = m.partition("@")
        k = int(at)
        hits = [r for r in ranks if r is not None and r <= k]
        if name == "recall":
            found[prefix + m] = len(hits) / len(groups)
        elif name == "ndcg":
            dcg = sum(1 / np.log2(r + 1) for r in hits)
            ideal = sum(1 / np.log2(r + 2) for r in range(min(len(groups), k)))
            found[prefix + m] = float(min(1.0, dcg / ideal))
    return found


def _group_docs(entries: Sequence[Sequence[str]], groups: Groups) -> dict[str, float]:
    """Each candidate as the first group it newly covers, else as its own
    chunk (not relevant, but it holds its rank); scores fall with rank."""

    docs: dict[str, float] = {}
    for rank, entry in enumerate(entries):
        found = set(entry)
        new = (
            f"g{i}" for i, g in enumerate(groups) if f"g{i}" not in docs and g & found
        )
        docs.setdefault(next(new, entry[0]), float(len(entries) - rank))
    return docs


def _row_metrics(
    question: EvalQuestion, row: Row, groups: Groups
) -> dict[str, float | None]:
    expected, plan = question.expected, row["plan"]
    flagged = bool(row["unresolved_companies"]) or not row["passages"]
    reason = question.cell.get("reason")
    metrics: dict[str, float | None] = {
        "query.company": float(
            {c["cik"] for c in row["companies"]} == set(expected.ciks)
        ),
        "query.unknown_flagged": float(flagged) if expected.unknown_companies else None,
        "query.language": float(plan["language"] == row["language"]),
        "query.llm_plan": float(plan["source"] == "llm"),
        # Every corpus searched in its own language (rows written before
        # plans kept this: none).
        "query.search_text": (
            float(not plan["untranslated"]) if "untranslated" in plan else None
        ),
        "unanswerable.rejected": float(flagged) if reason == REJECTABLE else None,
    }
    if question.answerable:
        admits = [a for a in row["filter_admits"] if a is not None]
        metrics["query.filter_admits_evidence"] = (
            sum(admits) / len(admits) if admits else None
        )
        context = {
            c
            for p in row["passages"]
            for c in (*p["chunk_ids"], *p["duplicate_chunk_ids"])
        }
        metrics["context.recall"] = sum(1 for g in groups if g & context) / len(groups)
    return metrics


def score(
    questions: Sequence[EvalQuestion],
    rows: Sequence[Row],
    relevance: Mapping[str, Groups],
) -> Scoring:
    by_id = {q.id: q for q in questions}
    rows = [r for r in rows if r["question_id"] in by_id]
    ok = [r for r in rows if not r.get("error")]
    qrels = Qrels(
        {qid: {f"g{i}": 1 for i in range(len(gs))} for qid, gs in relevance.items()}
    )
    pairs = [
        (
            r,
            _row_metrics(
                by_id[r["question_id"]], r, relevance.get(r["question_id"], [])
            ),
        )
        for r in ok
    ]
    runs: dict[str, Any] = {}
    for language in dict.fromkeys(r["language"] for r in ok):
        chosen = [
            (r, m)
            for r, m in pairs
            if r["language"] == language and r["question_id"] in relevance
        ]
        for lst in LISTS:
            docs = {
                r["question_id"]: _group_docs(r[lst], relevance[r["question_id"]])
                for r, _ in chosen
            }
            # Queries without candidates stay out; `make_comparable` scores them 0.
            kept = {qid: d for qid, d in docs.items() if d}
            run = None
            if kept:
                run = Run(kept, name=f"{lst}_{language}")
                evaluate(qrels, run, list(METRICS), make_comparable=True)
                runs[f"{lst}_{language}"] = run
            for r, row_metrics in chosen:
                qid = r["question_id"]
                for m in METRICS:
                    value = run.scores[m][qid] if run is not None else 0.0
                    row_metrics[f"{lst}.{m}"] = float(value)
                groups = relevance[qid]
                if len(groups) > 1:
                    row_metrics.update(_multi_group(f"{lst}.", r[lst], groups))
    unmapped = [
        (qid, i) for qid, gs in relevance.items() for i, g in enumerate(gs) if not g
    ]
    return Scoring(
        [Scored(by_id[r["question_id"]], r, m) for r, m in pairs],
        [r for r in rows if r.get("error")],
        qrels,
        runs,
        unmapped,
    )


def means(items: Iterable[Scored]) -> dict[str, tuple[float, int]]:
    """Metric -> (mean over the rows where it applies, row count)."""

    values: dict[str, list[float]] = defaultdict(list)
    for item in items:
        for key, value in item.metrics.items():
            if value is not None:
                values[key].append(value)
    return {k: (sum(v) / len(v), len(v)) for k, v in values.items()}


def failures(scoring: Scoring) -> list[Scored]:
    """Answerable rows with no group found in the delivered top FAILURE_K."""

    key = f"post.hit_rate@{FAILURE_K}"
    return [
        s for s in scoring.scored if s.question.answerable and not s.metrics.get(key)
    ]


# Report


def dimension_values(item: Scored, name: str) -> list[str]:
    """Values of a breakdown; a paired value `a|b` counts for each, a CIK
    shows as its company name."""

    if name in ("question_type", "data_kind"):
        return [str(getattr(item.question, name))]
    values = item.question.cell.get(name, "-").split("|")
    if name != "cik":
        return values
    names = {c.cik: c.name for c in item.question.companies}
    return [names.get(v, v) for v in values]


def _cell(value: tuple[float, int] | None, *, count: bool = True) -> str:
    if value is None:
        return "–"
    return f"{value[0]:.3f} ({value[1]})" if count else f"{value[0]:.3f}"


def _table(header: Sequence[str], lines: Iterable[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(line) + " |" for line in lines),
        "",
    ]


def _per_language(
    items: Sequence[Scored], languages: Sequence[str]
) -> dict[str, dict[str, tuple[float, int]]]:
    return {
        lang: means(s for s in items if s.row["language"] == lang) for lang in languages
    }


def _breakdown(
    scored: Sequence[Scored],
    name: str,
    languages: Sequence[str],
) -> list[str]:
    groups: dict[str, list[Scored]] = defaultdict(list)
    for item in scored:
        for value in dimension_values(item, name):
            groups[value].append(item)
    keys = BREAKDOWN_COLUMNS.get(name, HEADLINE)
    lines = []
    for value, items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        per = _per_language(items, languages)
        lines.append(
            [
                value,
                str(len({s.question.id for s in items})),
                *(
                    _cell(per[lang].get(k), count=False)
                    for k in keys
                    for lang in languages
                ),
            ]
        )
    header = [name, "questions", *(f"{k} {lang}" for k in keys for lang in languages)]
    return [f"### {name}", "", *_table(header, lines)]


def _timings(scored: Sequence[Scored], languages: Sequence[str]) -> list[str]:
    lines = []
    for stage in dict.fromkeys(k for s in scored for k in s.row["timings"]):
        line = [stage]
        for lang in languages:
            values = [
                s.row["timings"][stage]
                for s in scored
                if s.row["language"] == lang and stage in s.row["timings"]
            ]
            line.append(
                f"{np.mean(values):.2f} / {np.percentile(values, 95):.2f}"
                if values
                else "–"
            )
        lines.append(line)
    return _table(["stage (s)", *(f"{lang} mean / p95" for lang in languages)], lines)


def _failure(item: Scored) -> list[str]:
    row, plan = item.row, item.row["plan"]
    read = ", ".join(c["cik"] for c in row["companies"]) or "none"
    lines = [
        f"- **{item.question.id}** ({row['language']}, {item.question.question_type}, {item.question.data_kind}): {row['question']}",
        f"  - expected ciks {list(item.question.expected.ciks)}; read ({plan['source']}) ciks {read}, years {plan['fiscal_years']}, periods {plan['fiscal_periods']}, forms {plan['forms']}, unresolved {row['unresolved_companies']}",
        f"  - filters admit evidence groups: {row['filter_admits']}",
        *(f"  - search {s['corpus']}: {s['filter']}" for s in row["searches"]),
        *(f"  - note: {n}" for n in row["notes"]),
    ]
    for p in row["passages"][:3]:
        period = f"FY{p['fiscal_year']} {p['fiscal_period'] or ''}".strip()
        lines.append(
            f"  - [{p['rank']}] {p['company_name']} {p['form']} {period} | {p['section_title']} | {', '.join(p['hit_chunk_ids'])}"
        )
    return lines


def render_report(config: Mapping[str, Any], scoring: Scoring) -> str:
    languages = list(config["languages"])
    scored = scoring.scored
    degraded = [s for s in scored if s.row["degraded"]]
    later: dict[str, set[str]] = defaultdict(set)  # reason -> question IDs
    for s in scored:
        if (reason := s.question.cell.get("reason")) and reason != REJECTABLE:
            later[reason].add(s.question.id)
    columns = {**_per_language(scored, languages), "all": means(scored)}
    q = config["questions"]
    lines = [
        f"# Retrieval eval: {config['label']}",
        "",
        f"- questions: {q['path']} (sha256 {q['sha256'][:12]}, {q['count']}); today {config['today']}",
        f"- Gold chunks rules version {config['chunks_rules_version']}; collections {config.get('collections')}",
        f"- models: {config.get('models')}",
        f"- rows scored {len(scored)}, errors {len(scoring.errors)}, degraded {len(degraded)}",
        "",
        (
            "Scores are per evidence group. `pre` = candidate pool in fused order (before "
            "rerank), `post` = delivered top k then the rest, `context` = passage chunks. "
            "Cells: mean (rows). `unanswerable.rejected` (company outside the corpus "
            "only) = no passage or unknown company flagged; unanswerable by period or "
            f"topic is judged at generation ({ {k: len(v) for k, v in later.items()} } questions)."
        ),
        "",
        "## Overall",
        "",
        *_table(
            ["metric", *columns],
            ([k, *(_cell(c.get(k)) for c in columns.values())] for k in METRIC_KEYS),
        ),
        "## Breakdowns",
        "",
    ]
    for name in BREAKDOWNS:
        lines += _breakdown(scored, name, languages)
    lines += ["## Timings", "", *_timings(scored, languages)]
    failed = failures(scoring)
    lines += [
        f"## Failures: no group found in post top {FAILURE_K} ({len(failed)})",
        "",
    ]
    for item in failed:
        lines += _failure(item)
    lists = {
        "Evidence no current chunk matches": [
            f"{qid} group {i}" for qid, i in scoring.unmapped
        ],
        "Degraded rows (retried on resume)": [
            f"{s.question.id} ({s.row['language']}): {s.row['degraded']}"
            for s in degraded
        ],
        "Errors": [
            f"{r['question_id']} ({r['language']}): {r['error']}"
            for r in scoring.errors
        ],
    }
    for title, items in lists.items():
        if items:
            lines += ["", f"## {title}", "", *(f"- {i}" for i in items)]
    return "\n".join(lines) + "\n"


def write_outputs(
    run_dir: Path,
    config: Mapping[str, Any],
    questions: Sequence[EvalQuestion],
    relevance: Mapping[str, Groups],
) -> Scoring:
    """Score `results.jsonl`; write qrels, ranx runs, scores and the report."""

    scoring = score(questions, load_rows(run_dir / "results.jsonl"), relevance)
    scoring.qrels.save(str(run_dir / "qrels.json"))
    for name, run in scoring.runs.items():
        run.save(str(run_dir / f"run_{name}.json"))
    summary = {
        lang: {k: v[0] for k, v in found.items()}
        for lang, found in _per_language(scoring.scored, config["languages"]).items()
    }
    (run_dir / "scores.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "report.md").write_text(render_report(config, scoring), encoding="utf-8")
    return scoring


# Compare


def compare_runs(
    run_dirs: Sequence[Path], stat_test: str = "student", max_p: float = 0.05
) -> str:
    """Paired tests of the delivered rankings (`post`) per language. Runs must
    share the question set, languages and limit, and have scored the same
    (question, language) pairs; otherwise ValueError."""

    configs = [
        json.loads((d / "config.json").read_text(encoding="utf-8")) for d in run_dirs
    ]
    shared = {
        json.dumps([c["questions"]["sha256"], c["languages"], c.get("limit")])
        for c in configs
    }
    if len(shared) > 1:
        raise ValueError("runs differ in question set, languages or limit")
    scored = [
        {
            (r["question_id"], r["language"])
            for r in load_rows(d / "results.jsonl")
            if not r.get("error")
        }
        for d in run_dirs
    ]
    if any(s != scored[0] for s in scored[1:]):
        raise ValueError("runs scored different (question, language) pairs")
    qrels = Qrels.from_file(str(run_dirs[0] / "qrels.json"))
    lines = [
        "# Retrieval runs compared",
        "",
        (
            f"Delivered ranking (`post`), group-level; {stat_test} paired test, max p "
            f"{max_p}. Superscripts mark the runs a row beats."
        ),
        "",
        *(f"- {d.name}" for d in run_dirs),
        "",
    ]
    for language in configs[0]["languages"]:
        files = [d / f"run_post_{language}.json" for d in run_dirs]
        if not all(f.exists() for f in files):
            lines += [f"## {language}", "", "a run has no candidates", ""]
            continue
        runs = [Run.from_file(str(f), name=f.parent.name) for f in files]
        report = compare(
            qrels,
            runs,
            list(COMPARE_METRICS),
            stat_test=stat_test,
            max_p=max_p,
            make_comparable=True,
        )
        lines += [f"## {language}", "", "```", report.to_table(), "```", ""]
    return "\n".join(lines)

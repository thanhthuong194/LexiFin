"""Sample eval tasks from Gold chunks and Silver XBRL facts.

Each task gets passages (and for numeric tasks one XBRL fact) chosen to cover
many cells: companies, section topics (weighted by how often users ask about
them), forms, chunk kinds, split table parts, years and periods. Each pick
takes the least-used value of each coverage dimension in turn, ties broken by
a hash of (seed, key), so the same data and seed give the same tasks.
Metadata always comes from the data; writers only word questions.

Memory stays bounded on the 500-company Gold: the first chunk pass keeps
metadata (no text) of a hash sample of at most `max_candidates` chunks, and
texts are read only for the chunks finally picked (`sampling_data`).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, NamedTuple

from src.data_pipeline.storage.iceberg import (
    RULES_VERSION_PROPERTY,
    open_gold_catalog_read_only,
    open_silver_catalog_read_only,
)
from src.data_pipeline.storage.native_rag.chunks import CHUNKS_TABLE
from src.evals.sampling_data import (
    CHUNK_DIMS,
    FACT_DIMS,
    OUTSIDE_COMPANIES,
    TOPIC_PROBES,
    ChunkMeta,
    ChunkText,
    CorpusStats,
    FactCandidate,
    Occurrence,
    company_ref,
    exhibit_number,
    rank,
    read_candidates,
    read_chunk_texts,
    read_fact_candidates,
)
from src.evals.testset import (
    CompanyRef,
    DataKind,
    EvalTask,
    MentionStyle,
    QuestionType,
    TaskPassage,
    UnanswerableReason,
    XbrlFact,
    whole_number,
)
from src.rag_engine.retrieval.companies import normalize_name
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Quotas for 100 tasks; other totals scale by largest remainder.
DEFAULT_PLAN: dict[QuestionType, int] = {
    "numeric": 20,
    "fact": 20,
    "table_lookup": 12,
    "list": 8,
    "summary": 6,
    "compare_periods": 8,
    "compare_companies": 8,
    "multi_hop": 4,
    "unanswerable": 14,
}
PERSONAS = (
    "equity analyst",
    "retail investor",
    "Vietnamese retail investor",
    "financial journalist",
    "compliance officer",
    "finance student",
)
# How much users ask about a topic: a topic's pick count is divided by it.
TOPIC_WEIGHTS = {
    "mdna": 3.0,
    "risk_factors": 3.0,
    "financial_statements": 3.0,
    "business": 2.0,
    "ex99": 2.0,
    "8k_event": 2.0,
    "legal_proceedings": 2.0,
    "controls": 0.5,
    "other": 0.5,
    "properties": 0.5,
}
UNANSWERABLE_REASONS: tuple[UnanswerableReason, ...] = (
    "company_outside_corpus",
    "period_not_covered",
    "topic_absent",
)
EXCLUDED_TOPICS = frozenset({"cover", "signatures", "exhibits"})
MIN_TEXT_CHARS = 300
MIN_TABLE_CHARS = 200
SUMMARY_MIN_CHARS = 1500
LIST_MIN_LINES = 3
# Measures period_not_covered tasks ask for.
MEASURES = (
    "revenue",
    "net income",
    "diluted EPS",
    "operating cash flow",
    "capital expenditures",
    "R&D expense",
)
# Weight of a value's count within one question type vs across all tasks.
_TYPE_WEIGHT = 3


@dataclass(frozen=True, slots=True)
class Pair:
    first: ChunkMeta
    second: ChunkMeta
    kind: str  # how the two relate (`8k_exhibit`, `note_ref`, `table`...)
    penalty: int = 0


class _PairEntry(NamedTuple):
    """A pair with what its cost needs, computed once."""

    pair: Pair
    dims: tuple[tuple[tuple[str, str], ...], ...]  # (dim, value) per chunk
    headings: tuple[str, ...]
    rank: int


@dataclass(frozen=True, slots=True)
class PlannedTask:
    question_type: QuestionType
    chunks: tuple[ChunkMeta, ...] = ()
    fact: XbrlFact | None = None
    pair_kind: str | None = None
    companies: tuple[CompanyRef, ...] = ()
    unanswerable_reason: UnanswerableReason | None = None
    hint: str | None = None


def scale_plan(
    total: int, plan: Mapping[QuestionType, int] = DEFAULT_PLAN
) -> dict[QuestionType, int]:
    """Quotas summing to `total`, proportional to `plan` (largest remainder)."""

    weight = sum(plan.values())
    exact = {qtype: total * count / weight for qtype, count in plan.items()}
    quotas = {qtype: math.floor(value) for qtype, value in exact.items()}
    # Stable sort: equal remainders keep `plan` order.
    by_remainder = sorted(plan, key=lambda q: quotas[q] - exact[q])
    for qtype in by_remainder[: total - sum(quotas.values())]:
        quotas[qtype] += 1
    return quotas


# --- chunk filters -------------------------------------------------------------


def is_eligible(chunk: ChunkMeta) -> bool:
    """Whether a chunk can be a task passage (topic, length)."""

    if chunk.section_topic in EXCLUDED_TOPICS:
        return False
    minimum = MIN_TABLE_CHARS if chunk.chunk_kind == "table" else MIN_TEXT_CHARS
    return chunk.chars >= minimum


# --- greedy picks ----------------------------------------------------------------


@dataclass
class Picker:
    """Greedy picks spread over coverage dimensions; no chunk is used twice.

    A value's cost is its count within the question type (weighted) plus its
    count across all tasks; a topic's cost is divided by its TOPIC_WEIGHTS.
    """

    seed: int
    used: set[str] = field(default_factory=set)
    overall: Counter[tuple[str, str]] = field(default_factory=Counter)
    per_type: Counter[tuple[str, str, str]] = field(default_factory=Counter)

    def rank(self, key: str) -> int:
        return rank(self.seed, key)

    def score(self, qtype: str, dim: str, value: str) -> float:
        count = _TYPE_WEIGHT * self.per_type[(qtype, dim, value)]
        count += self.overall[(dim, value)]
        return (
            count / TOPIC_WEIGHTS.get(value, 1.0) if dim == "section_topic" else count
        )

    def add(self, qtype: str, names: Sequence[str], values: Sequence[str]) -> None:
        for dim, value in zip(names, values, strict=True):
            self.overall[(dim, value)] += 1
            self.per_type[(qtype, dim, value)] += 1

    def least_used(self, qtype: str, dim: str, values: Iterable[str]) -> str:
        return min(
            values, key=lambda v: (self.score(qtype, dim, v), self.rank(f"{dim}:{v}"))
        )

    def narrow[T](
        self,
        entries: Iterable[tuple[tuple[str, ...], T]],
        names: Sequence[str],
        qtype: str,
        key: Callable[[T], str],
    ) -> tuple[tuple[str, ...], T]:
        """Least-used value of each dimension in turn, then the lowest hash."""

        live = list(entries)
        for level, dim in enumerate(names):
            best = self.least_used(qtype, dim, {dims[level] for dims, _ in live})
            live = [entry for entry in live if entry[0][level] == best]
        return min(live, key=lambda entry: self.rank(key(entry[1])))

    def take(self, qtype: str, chunk: ChunkMeta) -> None:
        self.add(qtype, CHUNK_DIMS, chunk.dims())
        self.used.add(chunk.chunk_id)

    def chunks(
        self, pool: Iterable[ChunkMeta], qtype: QuestionType, quota: int
    ) -> list[ChunkMeta]:
        """Up to `quota` unused chunks; a company's heading is asked once per
        type (not the same table every year)."""

        by_company: dict[str, list[tuple[tuple[str, ...], ChunkMeta]]] = defaultdict(
            list
        )
        for chunk in pool:
            by_company[chunk.cik].append((chunk.dims(), chunk))
        asked: set[tuple[str, str]] = set()
        picked: list[ChunkMeta] = []

        def heading(chunk: ChunkMeta) -> tuple[str, str]:
            return chunk.cik, chunk.title or chunk.section_title

        while len(picked) < quota and by_company:
            cik = self.least_used(qtype, "cik", by_company)
            live = [
                entry
                for entry in by_company[cik]
                if entry[1].chunk_id not in self.used and heading(entry[1]) not in asked
            ]
            if not live:
                del by_company[cik]
                continue
            by_company[cik] = live
            chunk = self.narrow(live, CHUNK_DIMS, qtype, lambda c: c.chunk_id)[1]
            asked.add(heading(chunk))
            self.take(qtype, chunk)
            picked.append(chunk)
        return picked

    def pairs(
        self, pairs: Sequence[Pair], qtype: QuestionType, quota: int
    ) -> list[Pair]:
        """Up to `quota` pairs of unused chunks, cheapest coverage cost first."""

        def entry(pair: Pair) -> _PairEntry:
            chunks = (pair.first, pair.second)
            return _PairEntry(
                pair,
                tuple(tuple(zip(CHUNK_DIMS, c.dims(), strict=True)) for c in chunks),
                tuple(f"{c.cik}:{c.title or c.chunk_id}" for c in chunks),
                self.rank(pair.first.chunk_id + pair.second.chunk_id),
            )

        def cost(item: _PairEntry) -> tuple[float, int]:
            pair = item.pair
            total = pair.penalty + self.score(qtype, "pair_kind", pair.kind)
            total += self.score(qtype, "pair_title", pair.first.title)
            for dims in item.dims:
                total += sum(self.score(qtype, dim, value) for dim, value in dims)
            return total, item.rank

        asked: set[tuple[str, ...]] = set()  # the same two headings once per type
        picked: list[Pair] = []
        live = [entry(pair) for pair in pairs]
        while len(picked) < quota:
            live = [
                item
                for item in live
                if item.pair.first.chunk_id not in self.used
                and item.pair.second.chunk_id not in self.used
                and item.headings not in asked
            ]
            if not live:
                break
            chosen = min(live, key=cost)
            best = chosen.pair
            asked.add(chosen.headings)
            self.take(qtype, best.first)
            self.take(qtype, best.second)
            self.add(qtype, ("pair_kind", "pair_title"), (best.kind, best.first.title))
            picked.append(best)
        return picked

    def facts(
        self,
        facts: Iterable[FactCandidate],
        quota: int,
        passage_of: Callable[[FactCandidate], tuple[ChunkMeta, Occurrence] | None],
    ) -> list[tuple[XbrlFact, ChunkMeta]]:
        """Up to `quota` facts spread over company, form, concept and period;
        current-period facts first; each with an unused passage chunk."""

        # Current and other facts by position (narrowing ignores order).
        current: dict[int, tuple[tuple[str, ...], FactCandidate]] = {}
        other: dict[int, tuple[tuple[str, ...], FactCandidate]] = {}
        for index, candidate in enumerate(facts):
            (current if candidate.is_current else other)[index] = (
                candidate.dims(),
                candidate,
            )
        picked: list[tuple[XbrlFact, ChunkMeta]] = []
        while len(picked) < quota and (current or other):
            pool = current or other
            index, _ = self.narrow(
                ((dims, (index, fact)) for index, (dims, fact) in pool.items()),
                FACT_DIMS,
                "numeric",
                lambda item: item[1].fact.fact_id,
            )[1]
            choice = pool.pop(index)
            found = passage_of(choice[1])
            if found is None or found[0].chunk_id in self.used:
                continue
            chunk, occurrence = found
            # Fact dims only: numeric passages are statement tables, and counting
            # their topic would push other types off financial statements.
            self.add("numeric", FACT_DIMS, choice[0])
            self.used.add(chunk.chunk_id)
            fact = choice[1].fact.model_copy(
                update={"block_id": occurrence.block_id, "cell_id": occurrence.cell_id}
            )
            picked.append((fact, chunk))
        return picked


def fact_passage(
    fact: FactCandidate,
    chunks_of: Mapping[str, Sequence[ChunkMeta]],
    texts: Mapping[str, ChunkText],
) -> tuple[ChunkMeta, Occurrence] | None:
    """The chunk to show for a fact: an occurrence in a primary statement
    first, then a part whose text shows the value."""

    found = [
        (chunk, occurrence)
        for occurrence in fact.occurrences
        for chunk in chunks_of.get(occurrence.block_id, ())
    ]

    def order(item: tuple[ChunkMeta, Occurrence]) -> tuple[bool, bool, int, str]:
        chunk, occurrence = item
        text = texts.get(chunk.chunk_id)
        shown = (
            text is not None
            and whole_number(occurrence.raw_value).search(text.text) is not None
        )
        return not chunk.is_statement, not shown, chunk.part or 0, chunk.chunk_id

    return min(found, key=order) if found else None


# --- pairs -------------------------------------------------------------------------


def _first(chunks: Iterable[ChunkMeta], seed: int) -> ChunkMeta:
    return min(chunks, key=lambda c: rank(seed, c.chunk_id))


def period_pairs(pool: Iterable[ChunkMeta], seed: int) -> list[Pair]:
    """One company's same titled table/text in its nearest other year, same
    period and same split part (FY vs FY, an earnings release vs last year's);
    identical texts are no comparison."""

    groups: dict[tuple[Any, ...], dict[int, list[ChunkMeta]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for c in pool:
        if c.title:
            key = (c.cik, c.doc_class, c.section_topic, c.chunk_kind, c.title, c.parts)
            groups[(*key, c.period)][c.year].append(c)
    pairs: list[Pair] = []
    for by_year in groups.values():
        for older, newer in pairwise(sorted(by_year)):
            first, second = _first(by_year[older], seed), _first(by_year[newer], seed)
            if first.text_hash != second.text_hash:
                gap = newer - older
                pairs.append(Pair(first, second, first.chunk_kind, 5 * (gap - 1)))
    return pairs


def company_pairs(pool: Iterable[ChunkMeta], seed: int) -> list[Pair]:
    """Two companies' chunks with the same title, calendar period, form,
    topic and split part."""

    groups: dict[tuple[Any, ...], dict[str, list[ChunkMeta]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for c in pool:
        if c.title:
            key = (c.calendar_period, c.doc_class, c.section_topic, c.chunk_kind)
            groups[(*key, c.title, c.parts)][c.cik].append(c)
    pairs: list[Pair] = []
    for group, by_company in groups.items():
        ciks = sorted(by_company, key=lambda cik: rank(seed, f"{group}:{cik}"))
        for one, other in pairwise(ciks):
            first, second = (
                _first(by_company[one], seed),
                _first(by_company[other], seed),
            )
            pairs.append(Pair(first, second, first.chunk_kind))
    return pairs


def _note_pairs(chunks: Sequence[ChunkMeta], seed: int) -> Iterator[Pair]:
    """MD&A passages with the statement/note sharing their title, and
    passages citing "Note N" with that note."""

    by_title: dict[str, list[ChunkMeta]] = defaultdict(list)
    by_note: dict[str, list[ChunkMeta]] = defaultdict(list)
    for chunk in chunks:
        if chunk.section_topic == "financial_statements":
            by_title[chunk.title].append(chunk)
            by_note[chunk.note or ""].append(chunk)
    for chunk in chunks:
        if chunk.section_topic == "mdna" and chunk.title and chunk.title in by_title:
            yield Pair(chunk, _first(by_title[chunk.title], seed), "mdna_note")
        for note in chunk.note_refs:
            if note != chunk.note and note in by_note:
                yield Pair(chunk, _first(by_note[note], seed), "note_ref")


def filing_pairs(pool: Iterable[ChunkMeta], seed: int) -> list[Pair]:
    """Linked passages of one filing: an 8-K item and the EX-99 it cites,
    plus `_note_pairs`."""

    by_filing: dict[str, list[ChunkMeta]] = defaultdict(list)
    for chunk in pool:
        by_filing[chunk.filing_id].append(chunk)
    pairs: list[Pair] = []
    for chunks in by_filing.values():
        exhibits: dict[str, list[ChunkMeta]] = defaultdict(list)
        for chunk in chunks:
            if chunk.doc_class == "EX-99":
                exhibits[exhibit_number(chunk.document_type[3:])].append(chunk)
        for item in (c for c in chunks if c.section_topic == "8k_event"):
            for number in item.exhibit_refs:
                if number in exhibits:
                    exhibit = _first(exhibits[number], seed)
                    pairs.append(Pair(item, exhibit, "8k_exhibit"))
        pairs.extend(_note_pairs(chunks, seed))
    return pairs


# --- unanswerable --------------------------------------------------------------------


def outside_companies(stats: CorpusStats) -> list[str]:
    """OUTSIDE_COMPANIES names no corpus company has and no chunk mentions."""

    tickers = {t for company in stats.companies.values() for t in company.tickers}
    names = {normalize_name(company.name) for company in stats.companies.values()}
    return [
        name
        for index, (name, symbols, _) in enumerate(OUTSIDE_COMPANIES)
        if index not in stats.mentioned
        and not tickers.intersection(symbols)
        and normalize_name(name) not in names
    ]


def _missing_year(stats: CorpusStats, cik: str, turn: int) -> int | None:
    """A fiscal year at least 6 before the company's first, such that no
    table of the company shows it or the year before and no text says
    "fiscal <year>"."""

    first = stats.first_year[cik]
    years = [
        year
        for year in range(first - 6, first - 16, -1)
        if not {year, year - 1} & stats.years.get(cik, set())
    ]
    return years[turn % len(years)] if years else None


def plan_unanswerable(
    stats: CorpusStats, quota: int, picker: Picker
) -> list[PlannedTask]:
    """Tasks the corpus cannot answer, split evenly across the three reasons."""

    tasks: list[PlannedTask] = []
    ciks = sorted(stats.companies)
    outside = outside_companies(stats)
    seen: Counter[str] = Counter()
    probe_uses: Counter[int] = Counter()
    asked: set[tuple[str, int]] = set()

    def company(eligible: Collection[str]) -> str | None:
        if not eligible:
            return None
        cik = picker.least_used("unanswerable", "cik", eligible)
        picker.add("unanswerable", ("cik",), (cik,))
        seen[cik] += 1
        return cik

    for number in range(quota):
        reason = UNANSWERABLE_REASONS[number * 3 // quota]
        if reason == "company_outside_corpus":
            if outside:
                name = outside[len(tasks) % len(outside)]
                tasks.append(
                    PlannedTask("unanswerable", unanswerable_reason=reason, hint=name)
                )
            continue
        if reason == "period_not_covered":
            cik = company([c for c in ciks if c in stats.first_year])
            year = None if cik is None else _missing_year(stats, cik, seen[cik])
            if cik is None or year is None:
                continue
            measure = MEASURES[number % len(MEASURES)]
            quarter = number % 3
            hint = f"{measure} for {f'Q{quarter} ' if quarter else ''}FY{year}"
        else:
            options = [
                (c, p)
                for c in ciks
                for p in range(len(TOPIC_PROBES))
                if p not in stats.probe_hits.get(c, ()) and (c, p) not in asked
            ]
            cik = company({c for c, _ in options})
            if cik is None:
                continue
            # Least-used probe first, so topics vary across companies.
            probe = min(
                (p for c, p in options if c == cik),
                key=lambda p: (probe_uses[p], picker.rank(f"{cik}:{p}")),
            )
            probe_uses[probe] += 1
            asked.add((cik, probe))
            hint = TOPIC_PROBES[probe][0]
        tasks.append(
            PlannedTask(
                "unanswerable",
                companies=(stats.companies[cik],),
                unanswerable_reason=reason,
                hint=hint,
            )
        )
    return tasks


# --- planning and tasks ------------------------------------------------------------------


def plan_tasks(
    candidates: Sequence[ChunkMeta], quotas: Mapping[QuestionType, int], picker: Picker
) -> list[PlannedTask]:
    """Plan every passage-based type except numeric (planned first, from facts)."""

    pool = [c for c in candidates if is_eligible(c)]
    text = [c for c in pool if c.chunk_kind == "text"]
    planned: list[PlannedTask] = []

    def singles(qtype: QuestionType, chunks: Iterable[ChunkMeta]) -> None:
        for chunk in picker.chunks(chunks, qtype, quotas.get(qtype, 0)):
            planned.append(PlannedTask(qtype, (chunk,)))

    def doubles(qtype: QuestionType, pairs: Sequence[Pair]) -> None:
        for pair in picker.pairs(pairs, qtype, quotas.get(qtype, 0)):
            planned.append(
                PlannedTask(qtype, (pair.first, pair.second), pair_kind=pair.kind)
            )

    # Narrow pools first so broad ones do not take their chunks.
    doubles("multi_hop", filing_pairs(pool, picker.seed))
    doubles("compare_periods", period_pairs(pool, picker.seed))
    doubles("compare_companies", company_pairs(pool, picker.seed))
    singles("list", (c for c in text if c.list_lines >= LIST_MIN_LINES))
    singles("summary", (c for c in text if c.chars >= SUMMARY_MIN_CHARS))
    singles("table_lookup", (c for c in pool if c.chunk_kind == "table"))
    singles("fact", text)
    return planned


# TaskPassage fields copied from ChunkMeta (the rest come from ChunkText).
_PASSAGE_FIELDS = tuple(
    name for name in TaskPassage.model_fields if name not in ("context", "text")
)


def _passage(chunk: ChunkMeta, text: ChunkText) -> TaskPassage:
    return TaskPassage(
        **{name: getattr(chunk, name) for name in _PASSAGE_FIELDS},
        context=text.context,
        text=text.text,
    )


def _cell(task: PlannedTask) -> dict[str, str]:
    """CELL_KEYS plus type-specific keys; a pair's differing values joined by `|`."""

    if task.unanswerable_reason:
        reason: dict[str, str] = {"reason": task.unanswerable_reason}
        return reason | ({"cik": task.companies[0].cik} if task.companies else {})
    merged: dict[str, list[str]] = defaultdict(list)
    for chunk in task.chunks:
        values = {
            "cik": chunk.cik,
            "form": chunk.form,
            "document_type": chunk.document_type,
            "section_topic": chunk.section_topic,
            "chunk_kind": chunk.chunk_kind,
            "fiscal_year": str(chunk.fiscal_year or ""),
            "fiscal_period": chunk.fiscal_period or "",
            "part": "{}/{}".format(*chunk.parts) if chunk.part else "whole",
        }
        for name, value in values.items():
            if value and value not in merged[name]:
                merged[name].append(value)
    cell = {name: "|".join(values) for name, values in merged.items()}
    if task.pair_kind:
        cell["pair_kind"] = task.pair_kind
    if task.question_type == "compare_periods" and task.pair_kind == "table":
        # A table also shows prior-period columns, which may overlap the other.
        cell["ask"] = "values only one passage shows (each table's latest column)"
    if task.fact:
        cell["concept"] = task.fact.concept
    return cell


def _data_kind(task: PlannedTask) -> DataKind:
    if task.question_type == "numeric":
        return "xbrl"
    if task.question_type == "unanswerable":
        return "none"
    tables = all(chunk.chunk_kind == "table" for chunk in task.chunks)
    return "table" if task.question_type == "table_lookup" or tables else "text"


def _interleave(planned: Sequence[PlannedTask]) -> list[PlannedTask]:
    """Round-robin across types, so consecutive IDs (and batches) mix types."""

    queues: dict[QuestionType, list[PlannedTask]] = defaultdict(list)
    for task in planned:
        queues[task.question_type].append(task)
    ordered: list[PlannedTask] = []
    while any(queues.values()):
        for qtype in DEFAULT_PLAN:
            if queues[qtype]:
                ordered.append(queues[qtype].pop(0))
    return ordered


def assemble(
    planned: Sequence[PlannedTask], texts: Mapping[str, ChunkText]
) -> list[EvalTask]:
    """EvalTasks with IDs, personas and mention styles rotated per type;
    `nickname` only when every company has an alias."""

    tasks: list[EvalTask] = []
    seen: Counter[str] = Counter()
    types = list(DEFAULT_PLAN)
    for number, task in enumerate(_interleave(planned), start=1):
        j = seen[task.question_type] + types.index(task.question_type)
        seen[task.question_type] += 1
        companies = task.companies or tuple(
            dict.fromkeys(
                company_ref(c.cik, c.company_name, c.tickers) for c in task.chunks
            )
        )
        styles: tuple[MentionStyle, ...] = ("name", "ticker")
        if not companies:
            styles = ("name",)
        elif all(company.aliases for company in companies):
            styles = ("name", "ticker", "nickname")
        tasks.append(
            EvalTask(
                task_id=f"t{number:04d}",
                question_type=task.question_type,
                data_kind=_data_kind(task),
                persona=PERSONAS[j % len(PERSONAS)],
                mention_style=styles[(j + j // len(PERSONAS)) % len(styles)],
                companies=companies,
                passages=tuple(_passage(c, texts[c.chunk_id]) for c in task.chunks),
                fact=task.fact,
                unanswerable_reason=task.unanswerable_reason,
                hint=task.hint,
                cell=_cell(task),
            )
        )
    return tasks


def coverage_summary(tasks: Iterable[EvalTask]) -> dict[str, dict[str, int]]:
    """Task counts per value of each coverage dimension (pair values split)."""

    summary: dict[str, Counter[str]] = defaultdict(Counter)
    for task in tasks:
        summary["question_type"][task.question_type] += 1
        summary["data_kind"][task.data_kind] += 1
        summary["persona"][task.persona] += 1
        summary["mention_style"][task.mention_style] += 1
        for company in task.companies:
            summary["company"][
                company.tickers[0] if company.tickers else company.name
            ] += 1
        for name, values in task.cell.items():
            if name not in ("cik", "ask"):
                summary[name].update(values.split("|"))
    return {name: dict(sorted(counts.items())) for name, counts in summary.items()}


# --- entry point -------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleSettings:
    total: int = 100
    seed: int = 1
    max_candidates: int = 300_000
    max_fact_instances: int = 1_000
    plan: Mapping[QuestionType, int] = field(default_factory=lambda: dict(DEFAULT_PLAN))


def sample_tasks(
    data_dir: Path, gold_dir: Path, settings: SampleSettings
) -> tuple[list[EvalTask], int | None]:
    """Read Gold and Silver, plan every task, fetch the picked texts; also
    returns the chunks' rules version."""

    quotas = scale_plan(settings.total, settings.plan)
    picker = Picker(settings.seed)
    with open_gold_catalog_read_only(gold_dir) as gold:
        chunks = gold.load_table(CHUNKS_TABLE)
        version = chunks.properties.get(RULES_VERSION_PROPERTY)
        candidates, stats = read_candidates(
            chunks, settings.seed, settings.max_candidates
        )
        logger.info(
            "eval_sampling_candidates",
            rows=stats.rows,
            candidates=len(candidates),
            companies=len(stats.companies),
        )
        with open_silver_catalog_read_only(data_dir) as silver:
            facts = read_fact_candidates(
                silver, stats.filings, settings.seed, settings.max_fact_instances
            )
        logger.info(
            "eval_sampling_facts",
            facts=len(facts),
            current=sum(f.is_current for f in facts),
        )
        blocks = {o.block_id for fact in facts for o in fact.occurrences}
        chunks_of: dict[str, list[ChunkMeta]] = defaultdict(list)
        texts: dict[str, ChunkText] = {}
        fact_filings = {fact.filing_id: fact.filing.filing_date for fact in facts}
        for meta, text in read_chunk_texts(chunks, fact_filings, "block_ids", blocks):
            texts[meta.chunk_id] = text
            for block_id in blocks.intersection(meta.block_ids):
                chunks_of[block_id].append(meta)
        numeric = picker.facts(
            facts,
            quotas.get("numeric", 0),
            lambda fact: fact_passage(fact, chunks_of, texts),
        )
        planned = [PlannedTask("numeric", (chunk,), fact) for fact, chunk in numeric]
        planned += plan_tasks(candidates, quotas, picker)
        planned += plan_unanswerable(stats, quotas.get("unanswerable", 0), picker)
        picked = {c.chunk_id: c for task in planned for c in task.chunks}
        texts = {key: texts[key] for key in picked if key in texts}
        missing = picked.keys() - texts.keys()
        missing_filings = {
            picked[key].filing_id: picked[key].filing_date for key in missing
        }
        for meta, text in read_chunk_texts(
            chunks, missing_filings, "chunk_id", missing
        ):
            texts[meta.chunk_id] = text
    shortfall = {
        q: n - sum(t.question_type == q for t in planned) for q, n in quotas.items()
    }
    if any(shortfall.values()):
        logger.warning(
            "eval_sampling_shortfall",
            shortfall={q: n for q, n in shortfall.items() if n},
        )
    return assemble(planned, texts), int(version) if version else None


def split_batches(tasks: Sequence[EvalTask], batches: int) -> list[list[EvalTask]]:
    """Tasks dealt round-robin into `batches` lists (types already interleaved)."""

    return [list(tasks[index::batches]) for index in range(batches)]

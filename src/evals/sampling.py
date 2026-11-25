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
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from itertools import pairwise, zip_longest
from pathlib import Path
from typing import Any, NamedTuple

from pyiceberg.table import Table

from src.data_pipeline.storage.iceberg import (
    RULES_VERSION_PROPERTY,
    open_gold_catalog_read_only,
    open_silver_catalog_read_only,
)
from src.data_pipeline.storage.native_rag.chunks import CHUNKS_TABLE
from src.evals.sampling_data import (
    AGGREGATION_PROBES,
    CHUNK_DIMS,
    FACT_DIMS,
    OUTSIDE_COMPANIES,
    SECTION_TOPICS,
    TOPIC_PROBES,
    ChunkMeta,
    ChunkText,
    CompanyHeader,
    CorpusStats,
    FactCandidate,
    FilingInfo,
    Occurrence,
    OutsideName,
    company_ref,
    exhibit_number,
    rank,
    read_candidates,
    read_chunk_texts,
    read_company_headers,
    read_fact_candidates,
    sic_sector,
)
from src.evals.sampling_facts import (
    FactInputs,
    FactSettings,
    FactTask,
    fact_inputs,
    plan_fact_tasks,
)
from src.evals.testset import (
    CompanyRef,
    Computed,
    DataKind,
    EvalTask,
    MentionStyle,
    QuestionType,
    TaskPassage,
    UnanswerableReason,
    XbrlFact,
    whole_number,
)
from src.evals.universe import CompanyProfile, OutsideCompany, load_universe
from src.evals.universe import outside_companies as universe_outside
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
# v2 minimums (spec section 3), 142 of 150 tasks; `plan_quotas` gives the
# rest to the least-covered single-passage types once planned.
V2_PLAN: dict[QuestionType, int] = {
    "numeric": 12,
    "fact": 12,
    "table_lookup": 10,
    "list": 5,
    "summary": 5,
    "compare_periods": 10,
    "compare_companies": 10,
    "multi_hop": 8,
    "unanswerable": 10,
    "calc": 12,
    "ranking": 8,
    "temporal_change": 8,
    "event_8k": 10,
    "relation": 10,
    "aggregation": 6,
    "false_premise": 6,
}
V2_TOTAL = 150
PLANS: dict[str, dict[QuestionType, int]] = {"v1": DEFAULT_PLAN, "v2": V2_PLAN}
# Types extra v2 tasks go to (one passage, large pools).
EXTRA_TYPES: tuple[QuestionType, ...] = ("fact", "table_lookup", "list", "summary")
# Every type in task-ID rotation order (v1 types first, as v1 had them).
TYPE_ORDER: tuple[QuestionType, ...] = tuple(V2_PLAN)
# Company strata of v2 picks (`Picker.strata`).
STRATA_DIMS = ("sector", "size")
# 8-K items `event_8k` asks about.
EVENT_ITEMS = frozenset(
    ["1.01", "1.02", "1.05", "2.01", "2.02", "2.03", "5.02", "5.03", "5.07", "8.01"]
)
# Cell key of the value `plan_spread` spreads a type's tasks over.
SPREAD_DIMS = {"event_8k": "item_code", "relation": "relation"}
GRAPH_TYPES = frozenset({"relation", "aggregation"})
SMALL_DECILE = 8  # size deciles from here on are `small_company`
SMALL_UNIVERSE = 50  # fewer companies: no `small_company` tags
MAX_NEGATIVES = 5
AGGREGATION_WITH = 3  # companies that discuss the topic (plus one that does not)
RARITY_WEIGHT = 4.0  # cost per share of sections discussing a probe
# `temporal_change`: texts read for this many pairs per task, at most this
# many per company; pairs kept when `reworded`.
TEMPORAL_READ_FACTOR = 20
TEMPORAL_PER_COMPANY = 4
REWORD_JACCARD = 0.9
REWORD_MIN_WORDS = 15
_BALANCE_SHEET = re.compile(
    r"balance sheets?|financial (?:position|condition)", re.IGNORECASE
)
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
    computed: Computed | None = None
    tags: tuple[str, ...] = ()
    hard_negative_chunks: tuple[str, ...] = ()
    cell_extra: tuple[tuple[str, str], ...] = ()  # type-specific cell keys


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


def plan_quotas(total: int, plan_id: str) -> tuple[dict[QuestionType, int], int]:
    """Per-type quotas and the extra tasks `plan_extra` adds: v1 scales its
    plan to `total`; v2 scales its minimums to `total * 142 / 150`."""

    plan = PLANS[plan_id]
    if plan_id == "v1":
        return scale_plan(total, plan), 0
    base = scale_plan(total * sum(plan.values()) // V2_TOTAL, plan)
    return base, total - sum(base.values())


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
    With `strata` (v2: cik -> sector, size decile) companies are also spread
    over sectors and sizes.
    """

    seed: int
    used: set[str] = field(default_factory=set)
    overall: Counter[tuple[str, str]] = field(default_factory=Counter)
    per_type: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    strata: Mapping[str, tuple[str, str]] = field(default_factory=dict)

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

    def stratum(self, cik: str) -> tuple[str, str]:
        return self.strata.get(cik, ("unknown", "unknown"))

    def chunk_dims(self, chunk: ChunkMeta) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Coverage dimension names and values of a chunk."""

        if not self.strata:
            return CHUNK_DIMS, chunk.dims()
        return (*STRATA_DIMS, *CHUNK_DIMS), (*self.stratum(chunk.cik), *chunk.dims())

    def company(self, qtype: str, ciks: Iterable[str]) -> str:
        """Least-used company (after its sector and size with `strata`)."""

        if not self.strata:
            return self.least_used(qtype, "cik", ciks)
        entries = [((*self.stratum(cik), cik), cik) for cik in ciks]
        return self.narrow(entries, (*STRATA_DIMS, "cik"), qtype, lambda c: c)[1]

    def take(self, qtype: str, chunk: ChunkMeta) -> None:
        self.add(qtype, *self.chunk_dims(chunk))
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
            cik = self.company(qtype, by_company)
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
                tuple(tuple(zip(*self.chunk_dims(c), strict=True)) for c in chunks),
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
        names = (*STRATA_DIMS, *FACT_DIMS) if self.strata else FACT_DIMS
        for index, candidate in enumerate(facts):
            strata = self.stratum(candidate.filing.cik) if self.strata else ()
            (current if candidate.is_current else other)[index] = (
                (*strata, *candidate.dims()),
                candidate,
            )
        picked: list[tuple[XbrlFact, ChunkMeta]] = []
        while len(picked) < quota and (current or other):
            pool = current or other
            index, _ = self.narrow(
                ((dims, (index, fact)) for index, (dims, fact) in pool.items()),
                names,
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
            self.add("numeric", names, choice[0])
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


def temporal_pairs(pool: Iterable[ChunkMeta], seed: int) -> list[Pair]:
    """One company's 10-K text under the same heading of a SECTION_TOPICS
    section in two consecutive fiscal years, same split part, the same
    opening (`head_hash`) with another text: one passage reworded.
    `reworded` checks the texts once read."""

    groups: dict[tuple[Any, ...], dict[int, list[ChunkMeta]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for c in pool:
        if (
            c.title
            and c.form == "10-K"
            and c.chunk_kind == "text"
            and c.section_topic in SECTION_TOPICS
            and c.fiscal_year
        ):
            groups[(c.cik, c.section_topic, c.title, c.parts)][c.fiscal_year].append(c)
    pairs: list[Pair] = []
    for key, by_year in groups.items():
        for older, newer in pairwise(sorted(by_year)):
            matched = [
                (a, b)
                for a in by_year[older]
                for b in by_year[newer]
                if a.head_hash == b.head_hash and a.text_hash != b.text_hash
            ]
            if newer - older == 1 and matched:
                first, second = min(
                    matched, key=lambda ab: rank(seed, ab[0].chunk_id + ab[1].chunk_id)
                )
                pairs.append(Pair(first, second, key[1]))
    return pairs


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def reworded(old: str, new: str) -> bool:
    """Two texts differ enough to ask what changed: word sets (digits
    dropped) with Jaccard below REWORD_JACCARD and REWORD_MIN_WORDS words
    in one only."""

    first, second = _word_set(old), _word_set(new)
    union = first | second
    changed = len(union - (first & second))
    return bool(union) and (
        len(first & second) / len(union) < REWORD_JACCARD
        and changed >= REWORD_MIN_WORDS
    )


def temporal_candidates(pairs: Sequence[Pair], quota: int, seed: int) -> list[Pair]:
    """At most TEMPORAL_READ_FACTOR * `quota` pairs (TEMPORAL_PER_COMPANY
    per company, lowest hash first), whose texts are read for `reworded`."""

    per_company: Counter[str] = Counter()
    chosen: list[Pair] = []
    ordered = sorted(
        pairs, key=lambda p: rank(seed, p.first.chunk_id + p.second.chunk_id)
    )
    for pair in ordered:
        if len(chosen) >= TEMPORAL_READ_FACTOR * quota:
            break
        if per_company[pair.first.cik] < TEMPORAL_PER_COMPANY:
            per_company[pair.first.cik] += 1
            chosen.append(pair)
    return chosen


def event_items(pool: Iterable[ChunkMeta]) -> list[ChunkMeta]:
    """One chunk per 8-K item (document and section): one stating "On
    <date>" first, then the lowest chunk ordinal."""

    best: dict[tuple[str, str], ChunkMeta] = {}
    for c in pool:
        if c.form != "8-K" or c.item_code not in EVENT_ITEMS:
            continue
        key = (c.document_id, c.section_title)
        old = best.get(key)
        if old is None or (not c.dated_event, c.chunk_ordinal) < (
            not old.dated_event,
            old.chunk_ordinal,
        ):
            best[key] = c
    return list(best.values())


def _item_code(chunk: ChunkMeta) -> str | None:
    return chunk.item_code


def plan_spread(
    pool: Iterable[ChunkMeta],
    qtype: QuestionType,
    quota: int,
    picker: Picker,
    key: Callable[[ChunkMeta], str | None],
) -> list[PlannedTask]:
    """Single-passage tasks spread over the values of `key` (8-K item code,
    relation kind; cell key SPREAD_DIMS[qtype]): the least-used value first,
    then `Picker.chunks`; one task per filing."""

    dim = SPREAD_DIMS[qtype]
    by_value: dict[str, list[ChunkMeta]] = defaultdict(list)
    for chunk in pool:
        value = key(chunk)
        if value is not None:
            by_value[value].append(chunk)
    planned: list[PlannedTask] = []
    filings: set[str] = set()
    asked: set[tuple[str, str]] = set()  # a company's heading once
    tags = ("graph",) if qtype in GRAPH_TYPES else ()
    while len(planned) < quota and by_value:
        value = picker.least_used(qtype, dim, by_value)
        live = [
            c
            for c in by_value[value]
            if c.chunk_id not in picker.used
            and c.filing_id not in filings
            and (c.cik, c.title or c.section_title) not in asked
            and (c.cik, c.relation_snippet or c.chunk_id) not in asked
        ]
        picked = picker.chunks(live, qtype, 1)
        if not picked:
            del by_value[value]
            continue
        by_value[value] = live
        filings.add(picked[0].filing_id)
        chosen = picked[0]
        asked.add((chosen.cik, chosen.title or chosen.section_title))
        asked.add((chosen.cik, chosen.relation_snippet or chosen.chunk_id))
        picker.add(qtype, (dim,), (value,))
        hint = picked[0].relation_snippet if qtype == "relation" else None
        planned.append(
            PlannedTask(
                qtype, (picked[0],), hint=hint, tags=tags, cell_extra=((dim, value),)
            )
        )
    return planned


@dataclass(frozen=True, slots=True)
class AggregationGroup:
    """Companies of one sector whose 10-K section of one year discusses a
    probe topic (one chunk each, `keys` found), and one whose section never
    does (`absent`, shown by a chunk of that section)."""

    sector: str
    topic: str
    year: int
    probe: int
    chunks: tuple[ChunkMeta, ...]
    absent: tuple[str, ...]
    keys: tuple[str, ...]


def aggregation_groups(
    pool: Iterable[ChunkMeta],
    stats: CorpusStats,
    sectors: Mapping[str, str],
    seed: int,
) -> list[AggregationGroup]:
    """AGGREGATION_WITH companies of a sector with a chunk matching a probe
    in the same section and year, plus one company of the sector whose
    whole section (every chunk, `stats.section_probes`) misses it, shown by
    a chunk of that section: one passage per company. No group without
    such a company."""

    hits: dict[tuple[str, str, int, int], dict[str, list[tuple[ChunkMeta, str]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    shown: dict[tuple[str, str, int], list[ChunkMeta]] = defaultdict(list)
    for c in pool:
        sector = sectors.get(c.cik)
        if c.form != "10-K" or c.section_topic not in SECTION_TOPICS or not sector:
            continue
        shown[(c.cik, c.section_topic, c.year)].append(c)
        for probe, key in c.probes:
            hits[(sector, c.section_topic, c.year, probe)][c.cik].append((c, key))
    sections: dict[tuple[str, str, int], list[tuple[str, set[int]]]] = defaultdict(list)
    for (cik, topic, year), found in stats.section_probes.items():
        if cik in sectors:
            sections[(sectors[cik], topic, year)].append((cik, found))
    groups: list[AggregationGroup] = []
    for (sector, topic, year, probe), by_company in sorted(hits.items()):
        absent = sorted(
            (
                cik
                for cik, found in sections[(sector, topic, year)]
                if probe not in found and (cik, topic, year) in shown
            ),
            key=lambda cik: rank(seed, f"{probe}:{cik}"),
        )[:1]
        if len(by_company) < AGGREGATION_WITH or not absent:
            continue
        ciks = sorted(by_company, key=lambda cik: rank(seed, f"{probe}:{year}:{cik}"))
        picks = [
            min(by_company[cik], key=lambda ck: rank(seed, ck[0].chunk_id))
            for cik in ciks[:AGGREGATION_WITH]
        ]
        chunks = (*(c for c, _ in picks), _first(shown[(absent[0], topic, year)], seed))
        chunks = tuple(sorted(chunks, key=lambda c: rank(seed, f"order:{c.cik}")))
        keys = tuple(dict.fromkeys(key for _, key in picks))
        groups.append(
            AggregationGroup(sector, topic, year, probe, chunks, tuple(absent), keys)
        )
    return groups


def probe_shares(stats: CorpusStats) -> dict[int, float]:
    """Share of 10-K sections (company, topic, year) matching each probe."""

    counts: Counter[int] = Counter()
    for found in stats.section_probes.values():
        counts.update(found)
    total = max(len(stats.section_probes), 1)
    return {probe: count / total for probe, count in counts.items()}


def plan_aggregation(
    groups: Sequence[AggregationGroup],
    quota: int,
    picker: Picker,
    stats: CorpusStats,
) -> list[PlannedTask]:
    """Up to `quota` groups, least-used sector and probe first, rare probes
    (few sections discuss them) before common ones; one task per (sector,
    probe)."""

    qtype: QuestionType = "aggregation"
    planned: list[PlannedTask] = []
    asked: set[tuple[str, int]] = set()
    shares = probe_shares(stats)

    def cost(group: AggregationGroup) -> tuple[float, int]:
        total = picker.score(qtype, "sector", group.sector)
        total += picker.score(qtype, "probe", str(group.probe))
        total += picker.score(qtype, "section_topic", group.topic)
        total += RARITY_WEIGHT * shares.get(group.probe, 0.0)
        return total, rank(picker.seed, f"{group.sector}:{group.probe}:{group.year}")

    live = list(groups)
    while len(planned) < quota:
        live = [
            g
            for g in live
            if (g.sector, g.probe) not in asked
            and not any(c.chunk_id in picker.used for c in g.chunks)
        ]
        if not live:
            break
        group = min(live, key=cost)
        asked.add((group.sector, group.probe))
        picker.add(
            qtype,
            ("sector", "probe", "section_topic"),
            (group.sector, str(group.probe), group.topic),
        )
        for chunk in group.chunks:
            picker.take(qtype, chunk)
        topic = AGGREGATION_PROBES[group.probe][0]
        planned.append(
            PlannedTask(
                qtype,
                group.chunks,
                companies=tuple(stats.companies[c.cik] for c in group.chunks),
                hint=topic,
                cell_extra=(
                    ("probe", topic),
                    ("probe_keys", "|".join(group.keys)),
                    ("without_topic", "|".join(group.absent)),
                ),
            )
        )
    return planned


# --- unanswerable --------------------------------------------------------------------


def outside_companies(stats: CorpusStats) -> list[str]:
    """`stats.outside` names no corpus company has and no chunk mentions."""

    tickers = {t for company in stats.companies.values() for t in company.tickers}
    names = {normalize_name(company.name) for company in stats.companies.values()}
    return [
        name
        for index, (name, symbols, _) in enumerate(stats.outside)
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


def outside_count(quota: int) -> int:
    """`company_outside_corpus` tasks among `quota` unanswerable ones."""

    return sum(number * 3 // quota == 0 for number in range(quota))


def plan_unanswerable(
    stats: CorpusStats,
    quota: int,
    picker: Picker,
    outside: Sequence[str] | None = None,
) -> list[PlannedTask]:
    """Tasks the corpus cannot answer, split evenly across the three reasons
    (`outside`: company names to ask about, default `outside_companies`)."""

    tasks: list[PlannedTask] = []
    ciks = sorted(stats.companies)
    if outside is None:
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
    candidates: Sequence[ChunkMeta],
    quotas: Mapping[QuestionType, int],
    picker: Picker,
    stats: CorpusStats | None = None,
    temporal: Sequence[Pair] | None = None,
) -> list[PlannedTask]:
    """Plan every passage-based type except numeric and `sampling_facts`
    types (planned first, from facts); `aggregation` needs `stats` and
    `picker.strata`; `temporal`: `temporal_change` pairs checked as
    `reworded` (default: every `temporal_pairs` pair)."""

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
    if quotas.get("aggregation") and stats is not None:
        sectors = {cik: sector for cik, (sector, _) in picker.strata.items()}
        groups = aggregation_groups(text, stats, sectors, picker.seed)
        planned += plan_aggregation(groups, quotas["aggregation"], picker, stats)
    if quotas.get("temporal_change"):
        if temporal is None:
            temporal = temporal_pairs(text, picker.seed)
        doubles("temporal_change", temporal)
    if quotas.get("event_8k"):
        planned += plan_spread(
            event_items(pool), "event_8k", quotas["event_8k"], picker, _item_code
        )
    if quotas.get("relation"):
        planned += plan_spread(
            text,
            "relation",
            quotas["relation"],
            picker,
            lambda c: c.relation,
        )
    doubles("multi_hop", filing_pairs(pool, picker.seed))
    doubles("compare_periods", period_pairs(pool, picker.seed))
    doubles("compare_companies", company_pairs(pool, picker.seed))
    singles("list", (c for c in text if c.list_lines >= LIST_MIN_LINES))
    singles("summary", (c for c in text if c.chars >= SUMMARY_MIN_CHARS))
    singles("table_lookup", (c for c in pool if c.chunk_kind == "table"))
    singles("fact", text)
    return planned


def plan_extra(
    candidates: Sequence[ChunkMeta],
    planned: Sequence[PlannedTask],
    extra: int,
    picker: Picker,
) -> list[PlannedTask]:
    """`extra` more single-passage tasks (v2: past the minimums, plus any
    shortfall), each to the EXTRA_TYPES type with the fewest tasks so far;
    a company's heading still once per type."""

    pool = [c for c in candidates if is_eligible(c)]
    text = [c for c in pool if c.chunk_kind == "text"]
    pools: dict[QuestionType, list[ChunkMeta]] = {
        "fact": text,
        "table_lookup": [c for c in pool if c.chunk_kind == "table"],
        "list": [c for c in text if c.list_lines >= LIST_MIN_LINES],
        "summary": [c for c in text if c.chars >= SUMMARY_MIN_CHARS],
    }
    counts = Counter(task.question_type for task in planned)
    asked = {
        (task.question_type, c.cik, c.title or c.section_title)
        for task in planned
        for c in task.chunks
    }
    added: list[PlannedTask] = []
    live = list(EXTRA_TYPES)
    while len(added) < extra and live:
        qtype = min(live, key=lambda q: (counts[q], EXTRA_TYPES.index(q)))
        fresh = [
            c
            for c in pools[qtype]
            if (qtype, c.cik, c.title or c.section_title) not in asked
        ]
        picked = picker.chunks(fresh, qtype, 1)
        if not picked:
            live.remove(qtype)
            continue
        chunk = picked[0]
        asked.add((qtype, chunk.cik, chunk.title or chunk.section_title))
        counts[qtype] += 1
        added.append(PlannedTask(qtype, (chunk,)))
    return added


def fact_planned(task: FactTask) -> PlannedTask:
    """A `sampling_facts` task as a PlannedTask."""

    return PlannedTask(
        task.question_type,
        task.chunks,
        companies=task.companies,
        hint=task.hint,
        computed=task.computed,
        hard_negative_chunks=task.hard_negative_chunks,
    )


# --- hard negatives and tags -------------------------------------------------------


class NegativeIndex:
    """Chunks in memory by heading, to find chunks easy to confuse with a
    task's passages: the same company's heading in another year or period,
    then the same heading and section in another company of the sector."""

    def __init__(self, chunks: Iterable[ChunkMeta], sectors: Mapping[str, str]):
        self.sectors = sectors
        self.by_company: dict[tuple[str, str], list[ChunkMeta]] = defaultdict(list)
        self.by_sector: dict[tuple[str, str, str], list[ChunkMeta]] = defaultdict(list)
        self.by_id: dict[str, ChunkMeta] = {}
        for chunk in chunks:
            if not chunk.heading or chunk.chunk_id in self.by_id:
                continue
            self.by_id[chunk.chunk_id] = chunk
            self.by_company[(chunk.cik, chunk.heading)].append(chunk)
            sector = sectors.get(chunk.cik)
            if sector:
                key = (sector, chunk.section_topic, chunk.heading)
                self.by_sector[key].append(chunk)

    def near(self, passages: Sequence[ChunkMeta], negatives: Iterable[str]) -> bool:
        """Whether a negative nearly duplicates a passage: the same company,
        heading and period kind, or the same opening (`head_hash`)."""

        for chunk in filter(None, map(self.by_id.get, negatives)):
            for p in passages:
                same = (chunk.cik, chunk.heading) == (p.cik, p.heading)
                if chunk.head_hash == p.head_hash or (
                    same and _period_kind(chunk) == _period_kind(p)
                ):
                    return True
        return False

    def negatives(
        self, passages: Sequence[ChunkMeta], seed: int, *, numeric: bool
    ) -> tuple[str, ...]:
        """Up to MAX_NEGATIVES chunk IDs, passages taken in turn. Chunks
        sharing a block or the text hash of a passage may hold the evidence
        and are never negatives; nor, for `numeric` (table/XBRL) tasks, the
        same company's chunks of later years (prior-year columns)."""

        ids = {c.chunk_id for c in passages}
        blocks = {b for c in passages for b in c.block_ids}
        hashes = {c.text_hash for c in passages}

        def usable(chunk: ChunkMeta) -> bool:
            return (
                chunk.chunk_id not in ids
                and chunk.text_hash not in hashes
                and blocks.isdisjoint(chunk.block_ids)
            )

        def own(passage: ChunkMeta) -> list[ChunkMeta]:
            # Same period (a release twin) may hold the answer; older years
            # first, since later filings repeat prior-year values.
            found = [
                c
                for c in self.by_company.get((passage.cik, passage.heading), ())
                if usable(c)
                and (c.year, c.period) != (passage.year, passage.period)
                and not (numeric and c.year > passage.year)
            ]
            return sorted(
                found,
                key=lambda c: (
                    c.year >= passage.year,
                    abs(c.year - passage.year),
                    c.period != passage.period,
                    rank(seed, c.chunk_id),
                ),
            )

        def peers(passage: ChunkMeta) -> list[ChunkMeta]:
            sector = self.sectors.get(passage.cik)
            key = (sector or "", passage.section_topic, passage.heading)
            found = [
                c
                for c in self.by_sector.get(key, ())
                if c.cik != passage.cik and usable(c)
            ]
            return sorted(
                found,
                key=lambda c: (
                    c.calendar_period != passage.calendar_period,
                    rank(seed, c.chunk_id),
                ),
            )

        titled = [p for p in passages if p.heading]
        near = [c for group in zip_longest(*map(own, titled)) for c in group if c]
        far = [c for group in zip_longest(*map(peers, titled)) for c in group if c]
        return tuple(dict.fromkeys(c.chunk_id for c in near + far))[:MAX_NEGATIVES]


def _period_kind(chunk: ChunkMeta) -> str:
    """`FY`, `Q` (a quarter) or `event` (8-K)."""

    period = chunk.fiscal_period or ""
    return "FY" if period == "FY" else "Q" if period.startswith("Q") else "event"


def calendar_fiscal(
    header: CompanyHeader | None, filings: Iterable[FilingInfo]
) -> bool:
    """Whether a company's fiscal year is the calendar year: its header's
    fiscal year end (`MMDD`), else its 10-K period ends; 52/53-week years
    ending in the first week of January count as calendar."""

    year_end = header.fiscal_year_end if header is not None else None
    if year_end and len(year_end) == 4 and year_end.isdigit():
        month, day = int(year_end[:2]), int(year_end[2:])
    else:
        ends = [f.period_end_date for f in filings if f.fiscal_period == "FY"]
        last = max((end for end in ends if end is not None), default=None)
        if last is None:
            return True
        month, day = last.month, last.day
    return month == 12 or (month == 1 and day <= 7)


@dataclass(frozen=True)
class TagContext:
    """What tags need beyond a task: company calendars and sizes, each
    company's newest filing per form, and the set's `as_of`."""

    off_calendar: frozenset[str]
    size_deciles: Mapping[str, int]
    newest: Mapping[tuple[str, str], str]  # (cik, form) -> filing_id
    as_of: date | None


def tag_context(
    stats: CorpusStats,
    headers: Mapping[str, CompanyHeader],
    profiles: Mapping[str, CompanyProfile],
    as_of: date | None,
) -> TagContext:
    by_company: dict[str, list[FilingInfo]] = defaultdict(list)
    newest: dict[tuple[str, str], tuple[date, str]] = {}
    for filing_id, info in stats.filings.items():
        by_company[info.cik].append(info)
        if as_of is not None and info.filing_date > as_of:
            continue
        key = (info.cik, info.form)
        if key not in newest or (info.filing_date, filing_id) > newest[key]:
            newest[key] = (info.filing_date, filing_id)
    return TagContext(
        off_calendar=frozenset(
            cik
            for cik, filings in by_company.items()
            if not calendar_fiscal(headers.get(cik), filings)
        ),
        # Deciles of a small universe say nothing about company size.
        size_deciles={cik: p.size_decile for cik, p in profiles.items()}
        if len(profiles) >= SMALL_UNIVERSE
        else {},
        newest={key: filing_id for key, (_, filing_id) in newest.items()},
        as_of=as_of,
    )


def _facts(task: PlannedTask) -> tuple[XbrlFact, ...]:
    operands = task.computed.operands if task.computed else ()
    return (task.fact, *operands) if task.fact else operands


def task_tags(task: PlannedTask, context: TagContext) -> tuple[str, ...]:
    """Trap tags (spec section 3) a task carries by its data (tags it had
    are replaced); `former_name` and `near_duplicate` are added where they
    are decided."""

    tags: list[str] = []
    if task.question_type in GRAPH_TYPES:
        tags.append("graph")
    ciks = {c.cik for c in task.chunks} | {c.cik for c in task.companies}
    dated = [c for c in task.chunks if c.fiscal_year is not None]
    if dated and ciks & context.off_calendar:
        tags.append("fiscal_calendar")
    if (
        context.as_of is not None
        and len(ciks) == 1
        and task.chunks
        and all(context.newest.get((c.cik, c.form)) == c.filing_id for c in task.chunks)
    ):
        tags.append("relative_time")
    if any(f.instant is not None for f in _facts(task)) or any(
        c.chunk_kind == "table"
        and _BALANCE_SHEET.search(f"{c.table_title or ''} {c.subheading or ''}")
        for c in task.chunks
    ):
        tags.append("balance_sheet_date")
    if any(c.chunk_kind == "table" and (c.part_count or 0) > 1 for c in task.chunks):
        tags.append("split_table")
    if any(context.size_deciles.get(cik, 0) >= SMALL_DECILE for cik in ciks):
        tags.append("small_company")
    return tuple(dict.fromkeys(tags))


def finish_v2(
    planned: Sequence[PlannedTask],
    context: TagContext,
    negatives: NegativeIndex,
    seed: int,
) -> list[PlannedTask]:
    """Tags and hard negatives of every task; negatives a task already has
    (fact tasks: the same series in other years) are kept as they are."""

    finished: list[PlannedTask] = []
    for task in planned:
        tags = task_tags(task, context)
        found = task.hard_negative_chunks
        if task.chunks and not found:
            numeric = bool(_facts(task)) or any(
                c.chunk_kind == "table" for c in task.chunks
            )
            found = negatives.negatives(task.chunks, seed, numeric=numeric)
        if negatives.near(task.chunks, found):
            tags = (*tags, "near_duplicate")
        finished.append(replace(task, tags=tags, hard_negative_chunks=found))
    return finished


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


def _strata_cell(
    companies: Sequence[CompanyRef], size_deciles: Mapping[str, int]
) -> dict[str, str]:
    """v2 cell keys `sector` and `size_decile`, companies' values joined by `|`."""

    sectors = dict.fromkeys(c.sector or "unknown" for c in companies)
    sizes = dict.fromkeys(str(size_deciles.get(c.cik, "")) for c in companies)
    cell = {"sector": "|".join(sectors), "size_decile": "|".join(s for s in sizes if s)}
    return {name: value for name, value in cell.items() if value}


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
    if task.computed:
        cell["operation"] = task.computed.operation
        concepts = dict.fromkeys(f.concept for f in task.computed.operands)
        cell["concept"] = "|".join(concepts)
    cell.update(task.cell_extra)
    return cell


def _data_kind(task: PlannedTask) -> DataKind:
    if task.question_type == "numeric" or task.computed is not None:
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
        for qtype in TYPE_ORDER:
            if queues[qtype]:
                ordered.append(queues[qtype].pop(0))
    return ordered


def _styles(companies: Sequence[CompanyRef]) -> tuple[MentionStyle, ...]:
    """Mention styles every company supports."""

    if not companies:
        return ("name",)
    styles: tuple[MentionStyle, ...] = ("name", "ticker")
    if all(company.aliases for company in companies):
        styles += ("nickname",)
    if all(company.former_names for company in companies):
        styles += ("former_name",)
    return styles


def assemble(
    planned: Sequence[PlannedTask],
    texts: Mapping[str, ChunkText],
    directory: Mapping[str, CompanyRef] | None = None,
    size_deciles: Mapping[str, int] | None = None,
) -> list[EvalTask]:
    """EvalTasks with IDs, personas and mention styles rotated per type;
    `nickname` only when every company has an alias, `former_name` when
    every one has a former name. v2 passes `directory` (companies with
    sector and former names, by CIK) and `size_deciles` (cell keys)."""

    tasks: list[EvalTask] = []
    seen: Counter[str] = Counter()
    for number, task in enumerate(_interleave(planned), start=1):
        j = seen[task.question_type] + TYPE_ORDER.index(task.question_type)
        seen[task.question_type] += 1
        companies = task.companies or tuple(
            dict.fromkeys(
                company_ref(c.cik, c.company_name, c.tickers) for c in task.chunks
            )
        )
        cell = _cell(task)
        tags = task.tags
        if directory is not None:
            companies = tuple(directory.get(c.cik, c) for c in companies)
            cell |= _strata_cell(companies, size_deciles or {})
        styles = _styles(companies)
        style = styles[(j + j // len(PERSONAS)) % len(styles)]
        if style == "former_name":
            tags = (*tags, "former_name")
        tasks.append(
            EvalTask(
                task_id=f"t{number:04d}",
                question_type=task.question_type,
                data_kind=_data_kind(task),
                persona=PERSONAS[j % len(PERSONAS)],
                mention_style=style,
                companies=companies,
                passages=tuple(_passage(c, texts[c.chunk_id]) for c in task.chunks),
                fact=task.fact,
                unanswerable_reason=task.unanswerable_reason,
                hint=task.hint,
                cell=cell,
                computed=task.computed,
                tags=tags,
                hard_negative_chunks=task.hard_negative_chunks,
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
        for tag in task.tags:
            summary["tag"][tag] += 1
    return {name: dict(sorted(counts.items())) for name, counts in summary.items()}


# --- entry point -------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleSettings:
    total: int = 100
    seed: int = 1
    max_candidates: int = 300_000
    max_fact_instances: int = 1_000
    plan_id: str = "v1"  # a PLANS key
    bronze_dir: Path | None = None  # v2: default `<data_dir>/bronze/sec_filings`
    as_of: date | None = None  # v2: `relative_time` tags


_LEGAL_TAIL = re.compile(
    r"(?:[,\s]+(?:inc|corp|corporation|co|company|ltd|limited|plc|group|"
    r"holdings?|n\.?v|s\.?a|ag|se|/[a-z]{2,3})\.?)+$",
    re.IGNORECASE,
)


def outside_name(company: OutsideCompany) -> OutsideName:
    """An outside company as `CorpusStats.outside` takes it: mention keys are
    its name without legal tail (also title-cased when all caps) and aliases."""

    short = _LEGAL_TAIL.sub("", company.name) or company.name
    keys = (short, short.title()) if short.isupper() else (short,)
    return (
        company.name,
        company.tickers,
        tuple(dict.fromkeys((*keys, *company.aliases))),
    )


def alternate(names: Sequence[str], foreign: Collection[str]) -> list[str]:
    """`names` with foreign and US-listed ones taken in turn."""

    first = [n for n in names if n in foreign]
    second = [n for n in names if n not in foreign]
    return [n for pair in zip_longest(first, second) for n in pair if n]


def latest_filings(stats: CorpusStats) -> dict[str, str]:
    """Each company's latest `filing_id` (by filing date, then ID)."""

    latest: dict[str, tuple[date, str]] = {}
    for filing_id, info in stats.filings.items():
        key = (info.filing_date, filing_id)
        if info.cik not in latest or key > latest[info.cik]:
            latest[info.cik] = key
    return {cik: filing_id for cik, (_, filing_id) in latest.items()}


def company_directory(
    stats: CorpusStats,
    profiles: Mapping[str, CompanyProfile],
    headers: Mapping[str, CompanyHeader],
) -> dict[str, CompanyRef]:
    """Corpus companies with GICS sector (SIC sector when IVV has none) and
    former names."""

    directory: dict[str, CompanyRef] = {}
    for cik, ref in stats.companies.items():
        header = headers.get(cik, CompanyHeader())
        profile = profiles.get(cik)
        sector = (profile.sector if profile else None) or sic_sector(header.sic)
        directory[cik] = ref.model_copy(
            update={"former_names": header.former_names, "sector": sector}
        )
    return directory


@dataclass
class _V2:
    """v2 inputs read before Gold: the universe and outside companies."""

    bronze: Path
    profiles: dict[str, CompanyProfile]
    outside: tuple[OutsideName, ...]
    foreign: frozenset[str]

    @classmethod
    def read(cls, data_dir: Path, settings: SampleSettings, unanswerable: int) -> _V2:
        bronze = settings.bronze_dir or data_dir / "bronze" / "sec_filings"
        profiles = load_universe(bronze)
        # Twice the need: names some chunk mentions are dropped later. Corpus
        # companies (Bronze folders, read before Gold) may have left the
        # universe.
        filings = bronze / "sec-edgar-filings"
        corpus = [d.name for d in filings.iterdir()] if filings.is_dir() else []
        picks = universe_outside(
            profiles,
            bronze,
            2 * outside_count(unanswerable),
            settings.seed,
            corpus_ciks=corpus,
        )
        return cls(
            bronze,
            profiles,
            tuple(outside_name(c) for c in picks),
            frozenset(c.name for c in picks if c.country != "US-listed"),
        )


def _strata(
    directory: Mapping[str, CompanyRef], profiles: Mapping[str, CompanyProfile]
) -> dict[str, tuple[str, str]]:
    return {
        cik: (
            ref.sector or "unknown",
            str(profiles[cik].size_decile) if cik in profiles else "unknown",
        )
        for cik, ref in directory.items()
    }


def _read_blocks(
    chunks: Table, filing_dates: Mapping[str, date], blocks: Collection[str]
) -> tuple[dict[str, list[ChunkMeta]], dict[str, ChunkText]]:
    """Chunks holding `blocks` (one Gold pass over their filings' dates),
    by block, and their texts."""

    chunks_of: dict[str, list[ChunkMeta]] = defaultdict(list)
    texts: dict[str, ChunkText] = {}
    wanted = set(blocks)
    for meta, text in read_chunk_texts(chunks, filing_dates, "block_ids", wanted):
        texts[meta.chunk_id] = text
        for block_id in wanted.intersection(meta.block_ids):
            chunks_of[block_id].append(meta)
    return chunks_of, texts


def _wanted_blocks(
    facts: Sequence[FactCandidate], inputs: FactInputs | None, temporal: Sequence[Pair]
) -> tuple[dict[str, date], set[str]]:
    """Filing dates and blocks of every chunk whose text planning needs:
    numeric facts, fact-type operands and temporal_change candidates."""

    blocks = {o.block_id for fact in facts for o in fact.occurrences}
    filing_dates = {fact.filing_id: fact.filing.filing_date for fact in facts}
    if inputs is not None:
        blocks |= inputs.blocks
        filing_dates |= inputs.filing_dates
    for chunk in (c for pair in temporal for c in (pair.first, pair.second)):
        blocks.update(chunk.block_ids)
        filing_dates[chunk.filing_id] = chunk.filing_date
    return filing_dates, blocks


def _picked_texts(
    chunks: Table, planned: Sequence[PlannedTask], texts: Mapping[str, ChunkText]
) -> dict[str, ChunkText]:
    """Texts of every planned chunk: those already read, the rest from their
    filings (`chunk_id` match)."""

    picked = {c.chunk_id: c for task in planned for c in task.chunks}
    found = {key: texts[key] for key in picked if key in texts}
    missing = picked.keys() - found.keys()
    dates = {picked[key].filing_id: picked[key].filing_date for key in missing}
    for meta, text in read_chunk_texts(chunks, dates, "chunk_id", missing):
        found[meta.chunk_id] = text
    return found


def _reworded_pairs(
    pairs: Sequence[Pair], texts: Mapping[str, ChunkText]
) -> list[Pair]:
    return [
        pair
        for pair in pairs
        if pair.first.chunk_id in texts
        and pair.second.chunk_id in texts
        and reworded(texts[pair.first.chunk_id].text, texts[pair.second.chunk_id].text)
    ]


def sample_tasks(
    data_dir: Path, gold_dir: Path, settings: SampleSettings
) -> tuple[list[EvalTask], int | None]:
    """Read Gold and Silver, plan every task, fetch the picked texts; also
    returns the chunks' rules version."""

    quotas, extra = plan_quotas(settings.total, settings.plan_id)
    v2 = (
        _V2.read(data_dir, settings, quotas.get("unanswerable", 0))
        if settings.plan_id == "v2"
        else None
    )
    directory: dict[str, CompanyRef] | None = None
    headers: dict[str, CompanyHeader] = {}
    inputs: FactInputs | None = None
    fact_settings = FactSettings(quotas, settings.seed)
    with open_gold_catalog_read_only(gold_dir) as gold:
        chunks = gold.load_table(CHUNKS_TABLE)
        version = chunks.properties.get(RULES_VERSION_PROPERTY)
        candidates, stats = read_candidates(
            chunks,
            settings.seed,
            settings.max_candidates,
            v2.outside if v2 else OUTSIDE_COMPANIES,
        )
        logger.info(
            "eval_sampling_candidates",
            rows=stats.rows,
            candidates=len(candidates),
            companies=len(stats.companies),
        )
        picker = Picker(settings.seed)
        with open_silver_catalog_read_only(data_dir) as silver:
            if v2 is not None:
                headers = read_company_headers(silver, v2.bronze, latest_filings(stats))
                directory = company_directory(stats, v2.profiles, headers)
                picker.strata = _strata(directory, v2.profiles)
                sectors = {c: r.sector for c, r in directory.items() if r.sector}
                fact_settings = replace(fact_settings, sectors=sectors)
                inputs = fact_inputs(silver, stats, v2.profiles, fact_settings)
            facts = read_fact_candidates(
                silver, stats.filings, settings.seed, settings.max_fact_instances
            )
        logger.info(
            "eval_sampling_facts",
            facts=len(facts),
            current=sum(f.is_current for f in facts),
        )
        temporal = (
            temporal_candidates(
                temporal_pairs(
                    [
                        c
                        for c in candidates
                        if is_eligible(c) and c.chunk_kind == "text"
                    ],
                    settings.seed,
                ),
                quotas.get("temporal_change", 0),
                settings.seed,
            )
            if inputs is not None
            else []
        )
        chunks_of, texts = _read_blocks(
            chunks, *_wanted_blocks(facts, inputs, temporal)
        )
        planned: list[PlannedTask] = []
        if inputs is not None:
            # Fact-type tasks first: numeric and text picks skip their chunks.
            fact_tasks = plan_fact_tasks(inputs, chunks_of, texts, fact_settings)
            planned = [fact_planned(task) for task in fact_tasks]
            for task in planned:
                for chunk in task.chunks:
                    picker.take(task.question_type, chunk)
        numeric = picker.facts(
            facts,
            quotas.get("numeric", 0),
            lambda fact: fact_passage(fact, chunks_of, texts),
        )
        planned += [PlannedTask("numeric", (chunk,), fact) for fact, chunk in numeric]
        reworded_pairs = _reworded_pairs(temporal, texts) if v2 is not None else None
        planned += plan_tasks(candidates, quotas, picker, stats, reworded_pairs)
        outside = (
            alternate(outside_companies(stats), v2.foreign) if v2 is not None else None
        )
        planned += plan_unanswerable(
            stats, quotas.get("unanswerable", 0), picker, outside
        )
        if v2 is not None:
            planned += plan_extra(
                candidates, planned, settings.total - len(planned), picker
            )
            in_memory = [
                *candidates,
                *(c for found in chunks_of.values() for c in found),
                *(c for task in planned for c in task.chunks),
            ]
            sectors = {cik: sector for cik, (sector, _) in picker.strata.items()}
            context = tag_context(stats, headers, v2.profiles, settings.as_of)
            planned = finish_v2(
                planned, context, NegativeIndex(in_memory, sectors), settings.seed
            )
        texts = _picked_texts(chunks, planned, texts)
    shortfall = {
        q: n - sum(t.question_type == q for t in planned) for q, n in quotas.items()
    }
    if any(shortfall.values()):
        logger.warning(
            "eval_sampling_shortfall",
            shortfall={q: n for q, n in shortfall.items() if n},
            extra=extra,
        )
    deciles = {cik: p.size_decile for cik, p in v2.profiles.items()} if v2 else None
    tasks = assemble(planned, texts, directory, deciles)
    return tasks, int(version) if version else None


# EvalTask / CompanyRef fields added for v2, left out of a task's JSON at
# their defaults so v1 `tasks.jsonl` stay byte-identical.
V2_TASK_FIELDS = ("computed", "tags", "hard_negative_chunks")
V2_COMPANY_FIELDS = ("former_names", "sector")


def task_json(task: EvalTask) -> str:
    """A task's JSON line without v2-only fields at their defaults."""

    exclude: dict[str, Any] = {
        name: True
        for name in V2_TASK_FIELDS
        if getattr(task, name) == EvalTask.model_fields[name].get_default()
    }
    unset = {
        index: {
            name
            for name in V2_COMPANY_FIELDS
            if getattr(company, name) == CompanyRef.model_fields[name].get_default()
        }
        for index, company in enumerate(task.companies)
    }
    if any(unset.values()):
        exclude["companies"] = {i: names for i, names in unset.items() if names}
    return task.model_dump_json(exclude=exclude)


def write_tasks(path: Path, tasks: Sequence[EvalTask]) -> None:
    """`tasks` as JSON lines (`task_json`), written to a temp file first."""

    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".tmp")
    staged.write_text("".join(f"{task_json(t)}\n" for t in tasks), encoding="utf-8")
    staged.replace(path)


def split_batches(tasks: Sequence[EvalTask], batches: int) -> list[list[EvalTask]]:
    """Tasks dealt round-robin into `batches` lists (types already interleaved)."""

    return [list(tasks[index::batches]) for index in range(batches)]

"""Question -> cited passages: the retrieval steps wired together.

1. Read the question once (`query.analyze_query`): language, search text per
   corpus language, corpora, companies, periods, forms, topics.
2. Companies -> `cik` from each corpus's directory. A company the corpus does
   not hold is reported; when every named company is missing, the corpus is
   not searched (other companies' passages would answer the wrong question).
3. Hybrid search per corpus (and per company for comparisons), filters
   relaxed step by step when too few chunks match: period, then form; the
   company filter is never dropped.
4. Same-text candidates collapse (`dedup`).
5. Rerank with the cross-encoder; without one, or when it fails (the
   reranker is then skipped for the rest of the session), RRF order stays.
6. Without a year in the question, a small bonus for newer filings
   (larger when the user asks for the latest).
7. Top k, at least an equal share per compared company, grown into passages
   within a token budget (`expand`).
Online, only Qdrant and the model servers are read; Gold is for rebuilds.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from typing import Any

from qdrant_client import QdrantClient, models

from src.data_pipeline.transforms.native_rag.context import embedded_text
from src.llm import ChatModel, EmbeddingModel, LLMError, LLMResponseError, Reranker
from src.rag_engine.indexing.qdrant_store import CollectionSettings, search_params
from src.rag_engine.indexing.sparse import query_vector
from src.rag_engine.retrieval.companies import Company, CompanyIndex
from src.rag_engine.retrieval.corpora import CORPORA, CorpusProfile
from src.rag_engine.retrieval.dedup import dedupe
from src.rag_engine.retrieval.expand import ExpandSettings, Passage, expand
from src.rag_engine.retrieval.query import QueryPlan, analyze_query
from src.rag_engine.retrieval.search import (
    Hit,
    HybridQuery,
    SearchFilter,
    Window,
    hybrid_search,
    period_windows,
    qdrant_filter,
    yearless_windows,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# "Latest" covers the last annual report and the quarters after it.
LATEST_DAYS = timedelta(days=400)


@dataclass(frozen=True)
class RetrieverSettings:
    prefetch: int = 100
    candidates: int = 50
    top_k: int = 8
    context_tokens: int = 6000
    # Fewer fused chunks than this under a filter relaxes it.
    min_results: int = 8
    neighbor_window: int = 1
    small_section_chunks: int = 4
    # Recency bonus on scores scaled to [0, 1] when no period is asked.
    recency_weight: float = 0.05
    latest_weight: float = 0.25
    # Comparisons: at least this many selected chunks per company, and the
    # context budget grows with top_k up to this factor.
    per_company: int = 2
    max_budget_factor: float = 2.0
    # A failed reranker is skipped for this long, then tried again.
    rerank_cooldown_s: float = 120.0


@dataclass(frozen=True)
class Target:
    """A corpus as this retriever reads it."""

    profile: CorpusProfile
    collection: str
    companies: CompanyIndex
    params: models.SearchParams


@dataclass(frozen=True)
class _Search:
    target: Target
    text: str
    # Filters tried in order until enough chunks match.
    ladder: list[SearchFilter]


@dataclass
class _Planned:
    searches: list[_Search] = field(default_factory=list)
    filters: dict[str, list[str]] = field(default_factory=dict)
    companies: list[dict[str, str]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Candidate:
    corpus: str
    hit: Hit
    rerank_score: float | None = None
    final_score: float = 0.0


@dataclass(frozen=True)
class SearchUsed:
    """A search as run: the filter left after relaxation (for eval)."""

    corpus: str
    collection: str
    filter: SearchFilter


@dataclass
class RetrievalResult:
    question: str
    plan: QueryPlan
    passages: list[Passage]
    # Every candidate after ranking (chunk IDs, best first), for eval.
    ranking: list[dict[str, Any]]
    companies: list[dict[str, str]]
    unresolved_companies: list[str]
    filters: dict[str, list[str]]
    notes: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    searches: list[SearchUsed] = field(default_factory=list)
    # A configured reranker was not applied (failed or cooling down).
    rerank_skipped: bool = False

    @property
    def context_tokens(self) -> int:
        return sum(p.tokens for p in self.passages)


def _collection_of(client: QdrantClient, alias: str) -> str | None:
    aliases = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
    if alias in aliases:
        return aliases[alias]
    return alias if client.collection_exists(alias) else None


def _params(client: QdrantClient, collection: str) -> models.SearchParams:
    config = client.get_collection(collection).config
    large = config.quantization_config is not None
    return search_params(CollectionSettings(dimensions=0, large=large))


@dataclass(frozen=True)
class Models:
    """Model servers retrieval calls; only the embedder is required."""

    embedder: EmbeddingModel
    chat: ChatModel | None = None
    reranker: Reranker | None = None


class Retriever:
    def __init__(
        self,
        client: QdrantClient,
        servers: Models,
        corpora: Mapping[str, CorpusProfile] = CORPORA,
        settings: RetrieverSettings | None = None,
        today: date | None = None,
    ) -> None:
        self.client = client
        self.embedder = servers.embedder
        self.chat = servers.chat
        self.reranker = servers.reranker
        self._rerank_retry_at = 0.0
        self.settings = settings or RetrieverSettings()
        self.today = today
        self.notes: list[str] = []
        self.targets: dict[str, Target] = {}
        for name, profile in corpora.items():
            collection = _collection_of(client, profile.alias)
            if collection is None:
                self.notes.append(f"{name}: no collection or alias {profile.alias}")
                continue
            companies = CompanyIndex.from_qdrant(client, collection)
            if not len(companies):
                self.notes.append(f"{name}: empty company directory")
            self.targets[name] = Target(
                profile, collection, companies, _params(client, collection)
            )
        if not self.targets:
            raise ValueError(f"no corpus to search: {'; '.join(self.notes)}")

    def retrieve(self, question: str) -> RetrievalResult:
        timings: dict[str, float] = {}
        started = time.perf_counter()
        profiles = {name: t.profile for name, t in self.targets.items()}
        plan = analyze_query(
            question, profiles, self.chat, self.today or datetime.now(UTC).date()
        )
        timings["analyze_s"] = time.perf_counter() - started
        notes = [*plan.notes]
        mark = time.perf_counter()
        planned = self._plan_searches(plan, notes)
        timings["filters_s"] = time.perf_counter() - mark
        candidates, period_kept, used = self._candidates(
            plan, planned.searches, notes, timings
        )
        pre_rank = {c.hit.point_id: i for i, c in enumerate(candidates)}
        mark = time.perf_counter()
        candidates, rerank_skipped = self._rerank(plan, candidates, notes)
        timings["rerank_s"] = time.perf_counter() - mark
        rule = YearRule(
            calendar=plan.calendar,
            quarters=bool(plan.fiscal_periods) and "FY" not in plan.fiscal_periods,
            windows=tuple(
                w for search in planned.searches for w in search.ladder[0].windows
            ),
        )
        candidates = self._recency(plan, candidates, rule, period_kept=period_kept)
        ciks = [r["cik"] for r in planned.companies]
        top_k = (
            max(self.settings.top_k, self.settings.per_company * len(ciks))
            if len(ciks) >= 2
            else self.settings.top_k
        )
        selected = _select(candidates, top_k, ciks, plan.fiscal_years, rule)
        mark = time.perf_counter()
        budget = int(
            self.settings.context_tokens
            * min(top_k / self.settings.top_k, self.settings.max_budget_factor)
        )
        passages = self._expand(selected, budget)
        timings["expand_s"] = time.perf_counter() - mark
        timings["total_s"] = time.perf_counter() - started
        result = RetrievalResult(
            question=question,
            plan=plan,
            passages=passages,
            ranking=_ranking(candidates, pre_rank, {c.hit.point_id for c in selected}),
            companies=planned.companies,
            unresolved_companies=planned.unresolved,
            filters=planned.filters,
            notes=[*self.notes, *notes],
            timings={k: round(v, 3) for k, v in timings.items()},
            searches=used,
            rerank_skipped=rerank_skipped,
        )
        logger.info(
            "retrieval_completed",
            source=plan.source,
            corpora=list(plan.corpora),
            candidates=len(candidates),
            passages=len(passages),
            context_tokens=result.context_tokens,
            **result.timings,
        )
        return result

    def _plan_searches(self, plan: QueryPlan, notes: list[str]) -> _Planned:
        planned = _Planned()
        for name in plan.corpora:
            target = self.targets[name]
            companies, missing = self._companies(target, plan)
            planned.unresolved += missing
            planned.companies += [
                {"corpus": name, "cik": c.cik, "company_name": c.company_name}
                for c in companies
            ]
            if missing and not companies:
                # Only companies this corpus does not hold: another company's
                # passages would answer the wrong question.
                notes.append(f"{name}: not searched, no named company is in it")
                planned.filters[name] = []
                continue
            ladders = self._scopes(target, plan, companies)
            planned.searches += [
                _Search(target, plan.text_for(target.profile.language), ladder)
                for ladder in ladders
            ]
            planned.filters[name] = _describe(plan, companies, ladders[0][0].since)
        if planned.unresolved:
            notes.append(
                f"companies not in the corpus: {', '.join(planned.unresolved)}"
            )
        return planned

    def _candidates(
        self,
        plan: QueryPlan,
        searches: list[_Search],
        notes: list[str],
        timings: dict[str, float],
    ) -> tuple[list[Candidate], bool, list[SearchUsed]]:
        """Embed each distinct search text once, search, collapse same texts
        (fused order). Also whether every search kept its period filter, and
        the filter each search ended with."""

        mark = time.perf_counter()
        texts = list(dict.fromkeys(s.text for s in searches))
        vectors = (
            dict(zip(texts, self.embedder.embed_queries(texts), strict=True))
            if texts
            else {}
        )
        timings["embed_s"] = time.perf_counter() - mark
        mark = time.perf_counter()
        candidates: list[Candidate] = []
        # Comparisons share the candidates (the reranker cost follows them).
        per_search = max(
            self.settings.candidates // max(len(searches), 1), self.settings.top_k
        )
        period_kept = True
        used: list[SearchUsed] = []
        for search in searches:
            hits, relaxed, kept, scope = self._search(
                search, vectors[search.text], plan, per_search
            )
            used.append(
                SearchUsed(search.target.profile.name, search.target.collection, scope)
            )
            period_kept = period_kept and kept
            if relaxed:
                notes.append(relaxed)
            candidates += [Candidate(search.target.profile.name, h) for h in hits]
        unique, removed = dedupe(
            [c.hit for c in candidates],
            prefer_newest=not (plan.fiscal_years and period_kept),
        )
        corpus_of = {c.hit.point_id: c.corpus for c in candidates}
        if removed:
            notes.append(f"{removed} same-text candidates collapsed")
        timings["search_s"] = time.perf_counter() - mark
        return (
            [Candidate(corpus_of[h.point_id], h) for h in unique],
            period_kept,
            used,
        )

    def _companies(
        self, target: Target, plan: QueryPlan
    ) -> tuple[list[Company], list[str]]:
        found: dict[str, Company] = {}
        missing: list[str] = []
        for mention in plan.companies:
            company = target.companies.resolve(mention.name, mention.ticker)
            if company is None:
                missing.append(
                    f"{mention.name} ({mention.ticker})"
                    if mention.ticker
                    else mention.name
                )
            else:
                found.setdefault(company.cik, company)
        if plan.source == "rules":
            for company in target.companies.find_in_text(plan.question):
                found.setdefault(company.cik, company)
        return list(found.values()), missing

    def _scopes(
        self, target: Target, plan: QueryPlan, companies: Sequence[Company]
    ) -> list[list[SearchFilter]]:
        """One relaxation ladder per search: one search per company when
        several are compared (each gets its share of candidates)."""

        groups = (
            [[c.cik] for c in companies]
            if len(companies) > 1
            else [[c.cik for c in companies]]
        )
        forms = tuple(f for f in plan.forms if f in target.profile.forms)
        ladders = []
        for ciks in groups:
            asked = SearchFilter(
                tuple(ciks),
                forms,
                plan.fiscal_years,
                plan.fiscal_periods,
                calendar=plan.calendar,
                filed_months=plan.filed_months,
            )
            events = "8-K" in forms or "8k_event" in plan.topics
            full = replace(
                asked,
                windows=period_windows(
                    self.client, target.collection, ciks, asked, events=events
                )
                if plan.fiscal_years
                else yearless_windows(
                    self.client, target.collection, ciks, asked, events=events
                ),
                since=self._since(target, plan, ciks, forms),
            )
            ladder = [full]
            if full.has_period:
                ladder.append(full.without_period())
            if forms:
                ladder.append(ladder[-1].without_forms())
            ladders.append(ladder)
        return ladders

    def _since(
        self,
        target: Target,
        plan: QueryPlan,
        ciks: Sequence[str],
        forms: Sequence[str],
    ) -> date | None:
        """For the latest information: the newest filing of the asked form
        (or the newest earnings release) of the companies; otherwise filings
        of the last LATEST_DAYS before their latest filing."""

        if not plan.wants_latest or plan.fiscal_years or plan.fiscal_periods:
            return None
        if forms or "ex99" in plan.topics:
            # An earnings release is an EX-99 of an 8-K: its topic, not the
            # newest 8-K of any kind (a debt offering), decides.
            ex99 = "ex99" in plan.topics and (not forms or "8-K" in forms)
            wanted = () if ex99 else forms
            newest = self._newest(target, ciks, wanted)
            if newest is not None:
                return newest
        latest = [
            c.latest_filing_date
            for c in target.companies.companies
            if c.latest_filing_date and (not ciks or c.cik in ciks)
        ]
        return (
            min(latest) - LATEST_DAYS
            if ciks and latest
            else (max(latest) - LATEST_DAYS if latest else None)
        )

    def _newest(
        self, target: Target, ciks: Sequence[str], forms: Sequence[str]
    ) -> date | None:
        """Filing date of the oldest among each company's newest filing of
        `forms`, or with none of an EX-99 exhibit."""

        found: list[date] = []
        scopes: list[str | None] = list(ciks) or [None]
        for cik in scopes:
            must: list[models.Condition] = []
            if cik:
                must.append(
                    models.FieldCondition(key="cik", match=models.MatchValue(value=cik))
                )
            must.append(
                models.FieldCondition(
                    key="form", match=models.MatchAny(any=list(forms))
                )
                if forms
                else models.FieldCondition(
                    key="section_topic", match=models.MatchValue(value="ex99")
                )
            )
            points, _ = self.client.scroll(
                target.collection,
                scroll_filter=models.Filter(must=must),
                order_by=models.OrderBy(
                    key="filing_date", direction=models.Direction.DESC
                ),
                limit=1,
                with_payload=["filing_date"],
            )
            if points and points[0].payload:
                found.append(
                    date.fromisoformat(str(points[0].payload["filing_date"])[:10])
                )
        return min(found) if found else None

    def _search(
        self, search: _Search, dense: Sequence[float], plan: QueryPlan, limit: int
    ) -> tuple[list[Hit], str | None, bool, SearchFilter]:
        target, ladder = search.target, search.ladder
        sparse = query_vector(search.text)
        hits: list[Hit] = []
        for step, scope in enumerate(ladder):
            hits = hybrid_search(
                self.client,
                target.collection,
                HybridQuery(
                    dense,
                    sparse,
                    qdrant_filter(scope),
                    topics=[t for t in plan.topics if t in target.profile.topics],
                    prefetch=self.settings.prefetch,
                    limit=limit,
                    params=target.params,
                ),
            )
            if len(hits) >= self.settings.min_results or step == len(ladder) - 1:
                kept = scope.has_period or not ladder[0].has_period
                if step == 0:
                    return hits, None, kept, scope
                dropped = _dropped(ladder[0], scope)
                note = (
                    f"{target.profile.name} {','.join(scope.ciks) or 'all'}: "
                    f"{dropped} filter dropped ({len(hits)} chunks)"
                )
                return hits, note, kept, scope
        return hits, None, True, ladder[-1]

    def _rerank(
        self, plan: QueryPlan, candidates: list[Candidate], notes: list[str]
    ) -> tuple[list[Candidate], bool]:
        """Candidates by rerank score (fused order without a reranker); also
        whether a configured reranker was skipped (failed or cooling)."""

        cooling = time.monotonic() < self._rerank_retry_at
        if self.reranker is None or cooling or not candidates:
            if self.reranker is None or cooling:
                notes.append(
                    "reranker cooling down after a failure: fused order"
                    if cooling
                    else "no reranker: fused order"
                )
            fused = [replace(c, final_score=c.hit.score) for c in candidates]
            return fused, self.reranker is not None and cooling
        scored: list[Candidate] = []
        try:
            for corpus in dict.fromkeys(c.corpus for c in candidates):
                group = [c for c in candidates if c.corpus == corpus]
                profile = self.targets[corpus].profile
                documents = [
                    embedded_text(
                        str(c.hit.payload["context"]), str(c.hit.payload["text"])
                    )
                    for c in group
                ]
                query = plan.text_for(profile.language)
                for result in self.reranker.rerank(query, documents):
                    if not 0 <= result.index < len(group):
                        raise LLMResponseError(
                            f"rerank index {result.index} out of {len(group)}"
                        )
                    candidate = group[result.index]
                    scored.append(
                        replace(
                            candidate,
                            rerank_score=round(result.score, 6),
                            final_score=result.score,
                        )
                    )
        except LLMError as error:
            cooldown = self.settings.rerank_cooldown_s
            notes.append(
                f"reranker failed, skipped for {cooldown:.0f} s ({error})"[:300]
            )
            logger.warning("retrieval_rerank_failed", error=str(error)[:300])
            self._rerank_retry_at = time.monotonic() + cooldown
            return [replace(c, final_score=c.hit.score) for c in candidates], True
        # Candidates a `top_n` reranker left out rank below every scored one.
        reranked = {c.hit.point_id for c in scored}
        floor = min((c.final_score for c in scored), default=0.0) - 1.0
        scored += [
            replace(c, final_score=floor - i / len(candidates))
            for i, c in enumerate(
                c for c in candidates if c.hit.point_id not in reranked
            )
        ]
        scored.sort(key=lambda c: -c.final_score)
        return scored, False

    def _recency(
        self,
        plan: QueryPlan,
        candidates: list[Candidate],
        rule: YearRule,
        *,
        period_kept: bool,
    ) -> list[Candidate]:
        """Scores scaled to [0, 1] plus a small time bonus: without a year,
        for newer filings; with a year whose filter had to be dropped (no
        such filing), for filings closest to it. Nothing when the year
        filter held."""

        if (plan.fiscal_years and period_kept) or len(candidates) < 2:
            return candidates
        weight = (
            self.settings.latest_weight
            if plan.wants_latest
            else self.settings.recency_weight
        )
        scored = [c.final_score for c in candidates if c.rerank_score is not None]
        scores = scored or [c.final_score for c in candidates]
        if not scores or weight <= 0:
            return candidates
        bonus = (
            _closeness(candidates, plan.fiscal_years, rule)
            if plan.fiscal_years
            else _newness(candidates)
        )
        low, high = min(scores), max(scores)
        adjusted = [
            replace(
                c,
                final_score=(c.final_score - low) / (high - low or 1.0)
                + weight * bonus[c.hit.point_id],
            )
            for c in candidates
        ]
        adjusted.sort(key=lambda c: -c.final_score)
        return adjusted

    def _expand(self, selected: list[Candidate], budget_tokens: int) -> list[Passage]:
        if not selected:
            return []
        order = {c.hit.point_id: i for i, c in enumerate(selected, start=1)}
        passages: list[Passage] = []
        for corpus in dict.fromkeys(c.corpus for c in selected):
            group = [c for c in selected if c.corpus == corpus]
            budget = budget_tokens * len(group) // len(selected)
            found = expand(
                self.client,
                self.targets[corpus].collection,
                [c.hit for c in group],
                ExpandSettings(
                    budget_tokens=budget,
                    window=self.settings.neighbor_window,
                    small_section_chunks=self.settings.small_section_chunks,
                ),
            )
            by_chunk = {c.hit.chunk_id: c for c in group}
            for passage in found:
                best = min(
                    order[by_chunk[i].hit.point_id] for i in passage.hit_chunk_ids
                )
                score = by_chunk[passage.hit_chunk_ids[0]].final_score
                passages.append(replace(passage, rank=best, score=round(score, 6)))
        passages.sort(key=lambda p: p.rank)
        return passages


def _ranking(
    candidates: Sequence[Candidate], pre_rank: Mapping[str, int], selected: set[str]
) -> list[dict[str, Any]]:
    """Candidates best first, with their position in the fused order before
    rerank (`pre_rank`) and whether they are among the delivered top k."""

    return [
        {
            "chunk_id": c.hit.chunk_id,
            "duplicate_chunk_ids": list(c.hit.duplicates),
            "corpus": c.corpus,
            "fused_score": round(c.hit.score, 6),
            "rerank_score": c.rerank_score,
            "score": round(c.final_score, 6),
            "pre_rank": pre_rank[c.hit.point_id],
            "selected": c.hit.point_id in selected,
        }
        for c in candidates
    ]


def _newness(candidates: Sequence[Candidate]) -> dict[str, float]:
    """1 for the newest filing date among the candidates, 0 for the oldest."""

    dates = sorted({str(c.hit.payload.get("filing_date") or "") for c in candidates})
    rank = {d: i / max(len(dates) - 1, 1) for i, d in enumerate(dates)}
    return {
        c.hit.point_id: rank[str(c.hit.payload.get("filing_date") or "")]
        for c in candidates
    }


@dataclass(frozen=True)
class YearRule:
    """How a candidate's year is read for year quotas and closeness."""

    calendar: bool = False
    quarters: bool = False
    windows: tuple[Window, ...] = ()


def _year_of(candidate: Candidate, rule: YearRule) -> int:
    """Fiscal year; for calendar plans the calendar year of the period end
    (`calendar_period`: a full year ends 15 Jun Y-14 Jun Y+1, a quarter
    within 45 days of its end); an 8-K: the asked year whose 8-K window holds
    its filing date (a results release belongs to the year it reports), else
    its filing year."""

    payload = candidate.hit.payload
    end = payload.get("period_end_date")
    if rule.calendar and end and payload.get("fiscal_year"):
        day = date.fromisoformat(str(end)[:10])
        if rule.quarters:
            return (day - timedelta(days=45)).year
        return day.year if (day.month, day.day) >= (6, 15) else day.year - 1
    year = payload.get("fiscal_year")
    if year:
        return int(year)
    filed = date.fromisoformat(str(payload["filing_date"])[:10])
    for window in rule.windows:
        if (
            window.year is not None
            and window.cik in (None, payload.get("cik"))
            and window.start <= filed <= window.end
        ):
            return window.year
    return filed.year


def _closeness(
    candidates: Sequence[Candidate], years: Sequence[int], rule: YearRule
) -> dict[str, float]:
    """1 for filings of the asked year closest to what exists, falling to 0
    for the farthest."""

    distance = {
        c.hit.point_id: min(abs(_year_of(c, rule) - y) for y in years)
        for c in candidates
    }
    far = max(distance.values()) or 1
    return {point: 1 - d / far for point, d in distance.items()}


def _select(
    candidates: list[Candidate],
    top_k: int,
    ciks: list[str],
    years: Sequence[int] = (),
    rule: YearRule | None = None,
) -> list[Candidate]:
    """Best `top_k`; when several companies are compared (else several
    years asked), each company (year) first gets `top_k // n` of its own best
    chunks, when it has them."""

    if len(ciks) >= 2:
        groups: list[object] = list(ciks)

        def key(c: Candidate) -> object:
            return c.hit.payload.get("cik")
    elif len(years) >= 2:
        groups = list(years)

        def key(c: Candidate) -> object:
            return _year_of(c, rule or YearRule())
    else:
        return candidates[:top_k]
    share = max(top_k // len(groups), 1)
    chosen: list[Candidate] = []
    for group in groups:
        chosen += [c for c in candidates if key(c) == group][:share]
    ids = {c.hit.point_id for c in chosen}
    chosen += [c for c in candidates if c.hit.point_id not in ids][
        : max(top_k - len(chosen), 0)
    ]
    rank = {c.hit.point_id: i for i, c in enumerate(candidates)}
    # More groups than slots: the best-ranked shares win.
    return sorted(chosen, key=lambda c: rank[c.hit.point_id])[:top_k]


def _dropped(full: SearchFilter, relaxed: SearchFilter) -> str:
    dropped = []
    if full.has_period and not relaxed.has_period:
        dropped.append("recency" if full.since else "period")
    dropped += ["form"] if full.forms and not relaxed.forms else []
    return " and ".join(dropped)


def _describe(
    plan: QueryPlan, companies: Sequence[Company], since: date | None
) -> list[str]:
    found = [f"cik {', '.join(c.cik for c in companies)}"] if companies else []
    if plan.fiscal_years or plan.fiscal_periods:
        found.append("calendar periods" if plan.calendar else "fiscal periods")
    if since:
        found.append(f"latest: filed since {since.isoformat()}")
    if plan.fiscal_years:
        found.append(f"fiscal_year {', '.join(map(str, plan.fiscal_years))}")
    if plan.fiscal_periods:
        found.append(f"fiscal_period {', '.join(plan.fiscal_periods)}")
    if plan.filed_months:
        found.append(f"filed in {', '.join(plan.filed_months)}")
    if plan.forms:
        found.append(f"form {', '.join(plan.forms)}")
    if plan.topics:
        found.append(f"boost topics {', '.join(plan.topics)}")
    return found

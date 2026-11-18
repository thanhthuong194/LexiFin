"""Hybrid search on one collection: dense + BM25 prefetch fused by RRF.

Hard filters (company, period, form) restrict every prefetch; topics only
boost: the query fuses the same dense + BM25 prefetches with and without the
topic filter, so a chunk of a likely section appears in more lists and ranks
higher, while a wrong guess of the query model cannot hide the answer.

Periods: 10-K/10-Q chunks carry `fiscal_year`/`fiscal_period` and
`period_end_date`; a fiscal period matches the first two (a whole year or Q4:
the 10-K), a calendar one the period end (`calendar_period`). 8-Ks carry
neither, so they match by `filing_date` around the period's end, read from
the company's own 10-K/10-Q (fiscal calendars differ: NVIDIA's FY2025 ended
2025-01-26, Microsoft's 2025-06-30); without a known end, a fallback window.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Any

from qdrant_client import QdrantClient, models

from src.rag_engine.indexing.qdrant_store import DENSE, SPARSE
from src.rag_engine.indexing.sparse import SparseTerms

# 8-K window around a period end: a quarter's last weeks (its first weeks
# carry the previous quarter's results release) to 100 days after its end
# (results release, late events). A full year gets the same window around
# its end (the annual release; v5 took the whole year, so quarterly releases
# crowded out annual answers) unless the question is about 8-K events.
_BEFORE_END = timedelta(days=47)
_YEAR_EVENTS = timedelta(days=366 - 45)
_AFTER_END = timedelta(days=100)
# A calendar year matches the fiscal year that ends inside it or up to half
# a year later (NVIDIA's FY2025 ends 2025-01-26: calendar 2024); a calendar
# quarter matches a fiscal quarter ending within 45 days of it.
_CALENDAR_QUARTER_SLACK = timedelta(days=45)
_PERIODIC_FORMS = ("10-K", "10-Q")
_EVENT_FORM = "8-K"
# Period-end lookups per search (companies x years x periods).
MAX_PERIOD_LOOKUPS = 36
# Points scanned per company and period for the ends of a yearless period.
YEARLESS_SCAN = 10_000
_QUARTER_END = {"Q1": (3, 31), "Q2": (6, 30), "Q3": (9, 30), "Q4": (12, 31)}


@dataclass(frozen=True)
class Hit:
    point_id: str
    score: float
    payload: dict[str, Any]
    # Chunks with the same text collapsed into this one (`dedup`).
    duplicates: tuple[str, ...] = ()

    @property
    def chunk_id(self) -> str:
        return str(self.payload["chunk_id"])


@dataclass(frozen=True)
class Window:
    start: date
    end: date
    # The asked year an 8-K filed in this window belongs to, and whose window
    # it is (None: any company's, a fallback).
    year: int | None = None
    cik: str | None = None


@dataclass(frozen=True)
class SearchFilter:
    """What a search is restricted to; empty means any."""

    ciks: tuple[str, ...] = ()
    forms: tuple[str, ...] = ()
    fiscal_years: tuple[int, ...] = ()
    fiscal_periods: tuple[str, ...] = ()
    # Years/quarters are calendar ones (`period_end_date`), not fiscal.
    calendar: bool = False
    # 8-K filing-date windows for the period (empty: no 8-K period filter).
    windows: tuple[Window, ...] = ()
    # Filed on or after (questions about the latest information).
    since: date | None = None

    @property
    def has_period(self) -> bool:
        return bool(self.fiscal_years or self.fiscal_periods or self.since)

    def without_period(self) -> SearchFilter:
        return SearchFilter(self.ciks, self.forms)

    def without_forms(self) -> SearchFilter:
        return replace(self, forms=())


def _match(key: str, values: Sequence[Any]) -> models.FieldCondition:
    if len(values) == 1:
        return models.FieldCondition(key=key, match=models.MatchValue(value=values[0]))
    return models.FieldCondition(key=key, match=models.MatchAny(any=list(values)))


def periodic_periods(periods: Sequence[str]) -> tuple[str, ...]:
    """`fiscal_period` values of 10-K/10-Q chunks for asked periods (Q4 -> FY)."""

    return tuple(dict.fromkeys("FY" if p == "Q4" else p for p in periods))


@dataclass(frozen=True)
class PeriodAsk:
    """One asked period: a fiscal or calendar year, whole (`FY`) or a quarter."""

    year: int
    period: str
    calendar: bool


def asked(search: SearchFilter) -> list[PeriodAsk]:
    """Each year with each asked period; a year alone means the whole year."""

    periods = search.fiscal_periods or ("FY",)
    return [
        PeriodAsk(y, p, search.calendar) for y in search.fiscal_years for p in periods
    ]


def calendar_period(ask: PeriodAsk) -> Window:
    """`period_end_date` range of the fiscal period a calendar one matches."""

    if ask.period == "FY":
        return Window(date(ask.year, 6, 15), date(ask.year + 1, 6, 14))
    month, day = _QUARTER_END[ask.period]
    end = date(ask.year, month, day)
    return Window(end - _CALENDAR_QUARTER_SLACK, end + _CALENDAR_QUARTER_SLACK)


def periodic_conditions(ask: PeriodAsk) -> list[models.Condition]:
    """10-K/10-Q chunks of one asked period. A whole year reads the 10-K
    (`FY`); a fiscal Q4 too (10-Qs stop at Q3); a calendar quarter whichever
    filing ends near it."""

    conditions: list[models.Condition] = [_match("form", _PERIODIC_FORMS)]
    if ask.calendar:
        window = calendar_period(ask)
        conditions.append(
            models.FieldCondition(
                key="period_end_date",
                range=models.DatetimeRange(gte=window.start, lte=window.end),
            )
        )
        if ask.period == "FY":
            conditions.append(_match("fiscal_period", ["FY"]))
        return conditions
    conditions.append(_match("fiscal_year", [ask.year]))
    conditions.append(_match("fiscal_period", periodic_periods([ask.period])))
    return conditions


def qdrant_filter(search: SearchFilter) -> models.Filter | None:
    must: list[models.Condition] = []
    if search.ciks:
        must.append(_match("cik", search.ciks))
    if search.forms:
        must.append(_match("form", search.forms))
    if search.since:
        must.append(
            models.FieldCondition(
                key="filing_date", range=models.DatetimeRange(gte=search.since)
            )
        )
    if search.fiscal_years or search.fiscal_periods:
        branches: list[models.Condition] = (
            [models.Filter(must=periodic_conditions(a)) for a in asked(search)]
            if search.fiscal_years
            else [
                models.Filter(
                    must=[
                        _match("form", _PERIODIC_FORMS),
                        _match(
                            "fiscal_period", periodic_periods(search.fiscal_periods)
                        ),
                    ]
                )
            ]
        )
        events = not search.forms or _EVENT_FORM in search.forms
        if events and not search.fiscal_years and not search.windows:
            # A quarter without a year and no company: any 8-K (recency ranks).
            branches.append(_match("form", [_EVENT_FORM]))
        elif events and search.windows:
            branches.append(
                models.Filter(
                    must=[
                        _match("form", [_EVENT_FORM]),
                        models.Filter(
                            should=[_window_condition(w) for w in search.windows]
                        ),
                    ]
                )
            )
        must.append(models.Filter(should=branches))
    return models.Filter(must=must) if must else None


def _window_condition(window: Window) -> models.Condition:
    """8-Ks filed in the window, by its company only when it has one."""

    filed = models.FieldCondition(
        key="filing_date",
        range=models.DatetimeRange(gte=window.start, lte=window.end),
    )
    if window.cik is None:
        return filed
    return models.Filter(must=[filed, _match("cik", [window.cik])])


def _with(base: models.Filter | None, extra: models.Condition) -> models.Filter:
    return models.Filter(must=[*(base.must if base and base.must else []), extra])


def fallback_window(ask: PeriodAsk, *, events: bool) -> Window:
    """8-K window when no filing gives the period's end (no company named,
    or none filed for it): fiscal years named `year` end between January of
    `year` and January of `year + 1`; a calendar period ends on its last
    day."""

    if ask.calendar:
        end = calendar_period(ask).end - (
            timedelta(days=166) if ask.period == "FY" else _CALENDAR_QUARTER_SLACK
        )
        return _event_window(end, ask.period, events=events)
    if ask.period == "FY" and not events:
        return Window(date(ask.year, 1, 1) - _BEFORE_END, date(ask.year + 1, 5, 15))
    return Window(date(ask.year - 1, 2, 1), date(ask.year + 1, 4, 30))


def _event_window(end: date, period: str, *, events: bool) -> Window:
    before = _YEAR_EVENTS if period == "FY" and events else _BEFORE_END
    return Window(end - before, end + _AFTER_END)


def period_windows(
    client: QdrantClient,
    collection: str,
    ciks: Sequence[str],
    search: SearchFilter,
    *,
    events: bool,
) -> tuple[Window, ...]:
    """8-K filing-date windows: for each company and asked period, around
    the period end its own 10-K/10-Q records (`_event_window`); a fallback
    window when no end is known. `events`: the question is about 8-K events,
    so a full year keeps its whole span."""

    windows: list[Window] = []
    lookups = 0
    for ask in asked(search):
        found = False
        for cik in ciks:
            lookups += 1
            if lookups > MAX_PERIOD_LOOKUPS:
                break
            end = _period_end(client, collection, cik, ask)
            if end is not None:
                found = True
                window = _event_window(end, ask.period, events=events)
                windows.append(replace(window, year=ask.year, cik=cik))
        if not found:
            windows.append(replace(fallback_window(ask, events=events), year=ask.year))
    return tuple(dict.fromkeys(windows))


def yearless_windows(
    client: QdrantClient,
    collection: str,
    ciks: Sequence[str],
    search: SearchFilter,
    *,
    events: bool,
) -> tuple[Window, ...]:
    """A period asked without a year (`Q4 results`): 8-K windows around every
    end of that period in each company's 10-K/10-Q (v6 kept every 8-K)."""

    windows: list[Window] = []
    for cik in ciks[:MAX_PERIOD_LOOKUPS]:
        for period in search.fiscal_periods:
            points, _ = client.scroll(
                collection,
                scroll_filter=models.Filter(
                    must=[
                        _match("cik", [cik]),
                        _match("form", _PERIODIC_FORMS),
                        _match("fiscal_period", periodic_periods([period])),
                    ]
                ),
                limit=YEARLESS_SCAN,
                with_payload=["period_end_date"],
            )
            ends = {
                str(p.payload["period_end_date"])[:10]
                for p in points
                if p.payload and p.payload.get("period_end_date")
            }
            windows += [
                _event_window(date.fromisoformat(end), period, events=events)
                for end in sorted(ends)
            ]
    return tuple(dict.fromkeys(windows))


def _period_end(
    client: QdrantClient, collection: str, cik: str, ask: PeriodAsk
) -> date | None:
    points, _ = client.scroll(
        collection,
        scroll_filter=models.Filter(
            must=[_match("cik", [cik]), *periodic_conditions(ask)]
        ),
        limit=1,
        with_payload=["period_end_date"],
    )
    value = (
        points[0].payload.get("period_end_date")
        if points and points[0].payload
        else None
    )
    return date.fromisoformat(value[:10]) if value else None


@dataclass(frozen=True)
class HybridQuery:
    dense: Sequence[float]
    sparse: SparseTerms
    hard: models.Filter | None = None
    topics: Sequence[str] = ()
    prefetch: int = 100
    limit: int = 50
    params: models.SearchParams | None = None


def hybrid_search(
    client: QdrantClient, collection: str, query: HybridQuery
) -> list[Hit]:
    """Up to `limit` points fused by RRF from dense and BM25 prefetches of
    `prefetch` each, all under `hard`; with `topics`, the same two again
    restricted to those topics."""

    dense, sparse, hard, topics = query.dense, query.sparse, query.hard, query.topics
    prefetch, limit, params = query.prefetch, query.limit, query.params
    sparse_query = models.SparseVector(indices=sparse.indices, values=sparse.values)
    filters = [hard]
    if topics:
        filters.append(_with(hard, _match("section_topic", list(topics))))
    prefetches = []
    for scope in filters:
        prefetches.append(
            models.Prefetch(
                query=list(dense),
                using=DENSE,
                filter=scope,
                limit=prefetch,
                params=params,
            )
        )
        if sparse.indices:
            prefetches.append(
                models.Prefetch(
                    query=sparse_query, using=SPARSE, filter=scope, limit=prefetch
                )
            )
    response = client.query_points(
        collection,
        prefetch=prefetches,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return [
        Hit(str(point.id), point.score, point.payload or {})
        for point in response.points
    ]

"""Hybrid search on one collection: dense + BM25 prefetch fused by RRF.

Hard filters (company, period, form) restrict every prefetch; topics only
boost: the query fuses the same dense + BM25 prefetches with and without the
topic filter, so a chunk of a likely section appears in more lists and ranks
higher, while a wrong guess of the query model cannot hide the answer.

Periods: 10-K/10-Q chunks carry `fiscal_year`/`fiscal_period` and
`period_end_date`; a fiscal period matches the first two (a whole year or Q4:
the 10-K; a year still in progress, without a 10-K yet: its 10-Qs), a
calendar one the period end (`calendar_period`). 8-Ks carry no fiscal
period, so they match by `filing_date` around the period's end, read from
the company's own 10-K/10-Q (fiscal calendars differ: NVIDIA's FY2025 ended
2025-01-26, Microsoft's 2025-06-30); without a known end, a fallback window.
Questions about 8-K events take the whole period (a calendar one as is).
Filing months the user names ("filed in April 2022") admit any filing of
that month.
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
_AFTER_END = timedelta(days=100)
# 8-K events: the whole period (a 53-week year or 14-week quarter at most),
# a year then up to 120 days after its end (shareholder letter, proxy
# season: Amazon's 2021 letter was filed 2022-04-14, 104 days after).
_FISCAL_YEAR = timedelta(days=371)
_FISCAL_QUARTER = timedelta(days=98)
_YEAR_EVENTS_AFTER = timedelta(days=120)
# A named filing month, plus a week (an event late in the month).
_FILED_MONTH_SLACK = timedelta(days=7)
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
    # The company has no 10-K for this asked whole year yet (in progress):
    # its 10-Qs of the year stand in for it (`in_progress_conditions`).
    in_progress: bool = False


def month_window(month: str) -> Window | None:
    """Filing dates of a "YYYY-MM" month plus a week; None if malformed."""

    try:
        first = date.fromisoformat(f"{month}-01")
    except ValueError:
        return None
    following = date(first.year + first.month // 12, first.month % 12 + 1, 1)
    return Window(first, following - timedelta(days=1) + _FILED_MONTH_SLACK)


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
    # Filing months the user named ("YYYY-MM"): any filing of that month
    # (`month_window`) matches besides the asked periods.
    filed_months: tuple[str, ...] = ()

    @property
    def has_period(self) -> bool:
        return bool(
            self.fiscal_years or self.fiscal_periods or self.since or self.month_windows
        )

    @property
    def month_windows(self) -> tuple[Window, ...]:
        found = (month_window(m) for m in self.filed_months)
        return tuple(dict.fromkeys(w for w in found if w is not None))

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


def calendar_span(ask: PeriodAsk) -> Window:
    """First and last day of a calendar year or quarter."""

    if ask.period == "FY":
        return Window(date(ask.year, 1, 1), date(ask.year, 12, 31))
    month, day = _QUARTER_END[ask.period]
    return Window(date(ask.year, month - 2, 1), date(ask.year, month, day))


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


def in_progress_conditions(ask: PeriodAsk) -> list[models.Condition]:
    """10-Qs of an asked whole year: of the fiscal year, or ending in the
    calendar year's quarters (each within 45 days, as `calendar_period`)."""

    conditions: list[models.Condition] = [_match("form", ["10-Q"])]
    if not ask.calendar:
        return [*conditions, _match("fiscal_year", [ask.year])]
    first = calendar_period(replace(ask, period="Q1"))
    last = calendar_period(replace(ask, period="Q4"))
    window = models.DatetimeRange(gte=first.start, lte=last.end)
    return [*conditions, models.FieldCondition(key="period_end_date", range=window)]


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
    # A named filing month admits any filing of it, besides the periods.
    months = [_window_condition(w) for w in search.month_windows]
    if search.fiscal_years or search.fiscal_periods:
        branches = _periodic_branches(search)
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
        must.append(models.Filter(should=[*branches, *months]))
    elif months:
        must.append(models.Filter(should=months))
    return models.Filter(must=must) if must else None


def _periodic_branches(search: SearchFilter) -> list[models.Condition]:
    """10-K/10-Q chunks of the asked periods: each asked one, plus the 10-Qs
    of a whole year a company has no 10-K for yet; a period without a year
    matches it in any year."""

    if not search.fiscal_years:
        return [
            models.Filter(
                must=[
                    _match("form", _PERIODIC_FORMS),
                    _match("fiscal_period", periodic_periods(search.fiscal_periods)),
                ]
            )
        ]
    branches: list[models.Condition] = [
        models.Filter(must=periodic_conditions(a)) for a in asked(search)
    ]
    for window in search.windows:
        if window.in_progress and window.cik is not None and window.year is not None:
            ask = PeriodAsk(window.year, "FY", search.calendar)
            branches.append(
                models.Filter(
                    must=[_match("cik", [window.cik]), *in_progress_conditions(ask)]
                )
            )
    return branches


def _window_condition(window: Window) -> models.Condition:
    """Chunks filed in the window, by its company only when it has one."""

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
    or none filed for it): fiscal years named `year` run from late January
    of `year - 1` (NVIDIA) to early February of `year + 1` (retailers); a
    calendar period ends on its last day."""

    if ask.calendar:
        if events:
            return calendar_events_window(ask)
        end = calendar_period(ask).end - (
            timedelta(days=166) if ask.period == "FY" else _CALENDAR_QUARTER_SLACK
        )
        return _event_window(end, ask.period, events=events)
    if ask.period == "FY" and not events:
        return Window(date(ask.year, 1, 1) - _BEFORE_END, date(ask.year + 1, 5, 15))
    if ask.period == "FY":
        return Window(
            date(ask.year - 1, 1, 20), date(ask.year + 1, 2, 5) + _YEAR_EVENTS_AFTER
        )
    return Window(date(ask.year - 1, 1, 20), date(ask.year + 1, 4, 30))


def calendar_events_window(ask: PeriodAsk) -> Window:
    """8-K events of a calendar period: the period itself, then the time
    for its results release (and, for a year, its shareholder letter)."""

    span = calendar_span(ask)
    return Window(span.start, span.end + _after_events(ask.period))


def _after_events(period: str) -> timedelta:
    return _YEAR_EVENTS_AFTER if period == "FY" else _AFTER_END


def _event_window(end: date, period: str, *, events: bool) -> Window:
    """Around a known period end; for 8-K events the whole period first."""

    if not events:
        return Window(end - _BEFORE_END, end + _AFTER_END)
    length = _FISCAL_YEAR if period == "FY" else _FISCAL_QUARTER
    return Window(end - length, end + _after_events(period))


def period_windows(
    client: QdrantClient,
    collection: str,
    ciks: Sequence[str],
    search: SearchFilter,
    *,
    events: bool,
) -> tuple[Window, ...]:
    """8-K filing-date windows: for each company and asked period, around
    the period end its own 10-K/10-Q records (`_event_window`); a whole year
    without a 10-K yet, from its first to its latest 10-Q (`in_progress`); a
    fallback window when no end is known. `events`: the question is about
    8-K events, so the whole period counts (a calendar one needs no end)."""

    windows: list[Window] = []
    lookups = 0
    for ask in asked(search):
        found = False
        for cik in ciks:
            lookups += 1
            if lookups > MAX_PERIOD_LOOKUPS:
                break
            window = _company_window(client, collection, cik, ask, events=events)
            if window is not None:
                found = True
                windows.append(window)
        if not found:
            windows.append(replace(fallback_window(ask, events=events), year=ask.year))
    return tuple(dict.fromkeys(windows))


def _company_window(
    client: QdrantClient, collection: str, cik: str, ask: PeriodAsk, *, events: bool
) -> Window | None:
    end = _period_end(client, collection, cik, ask)
    if end is not None:
        window = (
            calendar_events_window(ask)
            if ask.calendar and events
            else _event_window(end, ask.period, events=events)
        )
        return replace(window, year=ask.year, cik=cik)
    if ask.period != "FY":
        return None
    ends = _in_progress_ends(client, collection, cik, ask)
    if ends is None:
        return None
    window = (
        calendar_events_window(ask)
        if ask.calendar and events
        else Window(
            _event_window(ends[0], "Q1", events=events).start,
            _event_window(ends[1], "Q1", events=events).end,
        )
    )
    return replace(window, year=ask.year, cik=cik, in_progress=True)


def _in_progress_ends(
    client: QdrantClient, collection: str, cik: str, ask: PeriodAsk
) -> tuple[date, date] | None:
    """First and latest period end of the asked year's 10-Qs, if any."""

    scope = models.Filter(must=[_match("cik", [cik]), *in_progress_conditions(ask)])
    ends: list[date] = []
    for direction in (models.Direction.ASC, models.Direction.DESC):
        points, _ = client.scroll(
            collection,
            scroll_filter=scope,
            order_by=models.OrderBy(key="period_end_date", direction=direction),
            limit=1,
            with_payload=["period_end_date"],
        )
        value = (
            points[0].payload.get("period_end_date")
            if points and points[0].payload
            else None
        )
        if not value:
            return None
        ends.append(date.fromisoformat(str(value)[:10]))
    return ends[0], ends[1]


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

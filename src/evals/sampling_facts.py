"""Task plans whose answers code computes from Silver XBRL facts: `calc`
(growth, margin, share, CAGR, difference), `ranking` and `false_premise`.

Facts are the non-dimensional monetary, per-share and share facts
`read_fact_candidates` reads, for a bounded set of companies spread over
sectors and sizes (so memory does not grow with the corpus). Each fact
becomes a point of a series (company, concept, unit, period kind, fiscal
period) by fiscal year: the filing's own (current) value wins over a later
filing's comparative, a point whose filings disagree (restated) is dropped,
and a point shows wherever any filing shows its value. Every operand must
show in a Gold chunk (its raw value in the chunk text); those chunks are the
writer's passages, as few as possible.

Two steps keep Gold to one text scan: `fact_inputs` reads Silver and says
which blocks it needs; the caller reads their chunks (with its own) and
passes them to `plan_fact_tasks`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from itertools import combinations
from typing import Literal

from pyiceberg.catalog import Catalog

from src.evals.sampling_data import (
    ChunkMeta,
    ChunkText,
    CorpusStats,
    FactCandidate,
    FilingInfo,
    Occurrence,
    rank,
    read_fact_candidates,
)
from src.evals.testset import (
    CompanyRef,
    Computed,
    Operation,
    QuestionType,
    XbrlFact,
    whole_number,
)
from src.evals.universe import CompanyProfile, display_name

Kind = Literal["annual", "quarter", "instant"]
ANNUAL_DAYS = range(350, 381)
QUARTER_DAYS = range(84, 99)
# A comparative's period end lies a whole number of years before its
# filing's period end, within this many days.
YEAR_SLACK = 10
FALSE_PREMISE_MIN = Decimal(5)  # |growth| in percent
MAX_GROWTH = Decimal(300)  # larger changes (a first dividend) read oddly
RANK_GAP = Decimal("0.01")  # ranked values differ by at least 1%
MAX_HARD_NEGATIVES = 3
RANK_DAYS = 92  # ranked period ends lie within this many days
REVENUES = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
)
# Numerator -> operation, over revenue of the same period.
RATIOS: dict[str, Operation] = {
    "NetIncomeLoss": "margin",
    "OperatingIncomeLoss": "margin",
    "GrossProfit": "margin",
    "ResearchAndDevelopmentExpense": "share",
}
# Plain-English names of KEY_CONCEPTS (false-premise hints).
LABELS = {
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "SalesRevenueNet": "net sales",
    "CostOfRevenue": "cost of revenue",
    "CostOfGoodsAndServicesSold": "cost of sales",
    "GrossProfit": "gross profit",
    "ResearchAndDevelopmentExpense": "research and development expense",
    "OperatingExpenses": "operating expenses",
    "OperatingIncomeLoss": "operating income",
    "IncomeTaxExpenseBenefit": "income tax expense",
    "NetIncomeLoss": "net income",
    "EarningsPerShareBasic": "basic earnings per share",
    "EarningsPerShareDiluted": "diluted earnings per share",
    "NetCashProvidedByUsedInOperatingActivities": "operating cash flow",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capital expenditures",
    "PaymentsForRepurchaseOfCommonStock": "share repurchases",
    "PaymentsOfDividends": "dividends paid",
    "ShareBasedCompensation": "stock-based compensation",
    "AllocatedShareBasedCompensationExpense": "stock-based compensation",
}
_QUARTERS = {"Q1": "first", "Q2": "second", "Q3": "third", "Q4": "fourth"}


@dataclass(frozen=True, slots=True)
class FactTask:
    question_type: QuestionType  # calc | ranking | false_premise
    chunks: tuple[ChunkMeta, ...]  # passages showing the operands
    computed: Computed
    companies: tuple[CompanyRef, ...]  # ranking: seeded order, not the answer's
    hint: str | None = None
    # Chunks of the same series in years before every operand's.
    hard_negative_chunks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FactSettings:
    quotas: Mapping[QuestionType, int]
    seed: int
    # Companies whose facts are read (memory and time bound).
    max_companies: int = 60
    # cik -> sector (SIC fallback included); overrides universe sectors.
    sectors: Mapping[str, str] = field(default_factory=dict)


SeriesKey = tuple[str, str, str, Kind, str]  # cik, concept, unit, kind, period


@dataclass(frozen=True, slots=True)
class Point:
    """One fiscal year's value of a series: the fact that states it (`sources[0]`,
    current first) and every other fact with that value (comparatives)."""

    series: SeriesKey
    year: int
    sources: tuple[FactCandidate, ...]

    @property
    def candidate(self) -> FactCandidate:
        return self.sources[0]

    @property
    def current(self) -> bool:
        return self.candidate.is_current

    @property
    def cik(self) -> str:
        return self.series[0]

    @property
    def concept(self) -> str:
        return self.series[1]

    @property
    def value(self) -> Decimal:
        return Decimal(self.candidate.fact.value)

    @property
    def end(self) -> date | None:
        fact = self.candidate.fact
        return fact.instant or fact.period_end

    @property
    def occurrences(self) -> tuple[Occurrence, ...]:
        """Blocks showing the value, own filing first; one per block."""

        seen: dict[str, Occurrence] = {}
        for source in self.sources:
            for occurrence in source.occurrences:
                seen.setdefault(occurrence.block_id, occurrence)
        return tuple(seen.values())


Series = dict[SeriesKey, dict[int, Point]]


@dataclass(frozen=True, slots=True)
class Option:
    """A possible task before its passages are found."""

    question_type: QuestionType
    operation: Operation
    points: tuple[Point, ...]
    value: str | None
    dims: tuple[tuple[str, str], ...]  # coverage (dimension, value) pairs
    hint: str | None = None
    order: tuple[str, ...] = ()  # ranking: CIKs high to low
    single_chunk: bool = False  # every operand in one passage


# --- arithmetic ----------------------------------------------------------------


def plain(number: Decimal) -> str:
    """Decimal without exponent or trailing zeros; never `-0`."""

    text = format(number.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def percent(number: Decimal) -> str:
    """A percent to 4 decimals (half up), trailing zeros dropped; writers
    round when they word it."""

    return plain(number.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def growth(old: Decimal, new: Decimal) -> Decimal:
    return (new - old) / abs(old) * 100


def cagr(old: Decimal, new: Decimal, years: int) -> Decimal:
    ratio = float(new / old) ** (1 / years)
    return (Decimal(repr(ratio)) - 1) * 100


# --- series --------------------------------------------------------------------


def _kind(fact: XbrlFact) -> Kind | None:
    if fact.instant is not None:
        return "instant"
    if fact.period_start is None or fact.period_end is None:
        return None
    days = (fact.period_end - fact.period_start).days
    if days in ANNUAL_DAYS:
        return "annual"
    return "quarter" if days in QUARTER_DAYS else None


def _years_back(info: FilingInfo, fact: XbrlFact) -> int | None:
    """Whole fiscal years between the fact's period end and its filing's."""

    end = fact.instant or fact.period_end
    if end is None or info.period_end_date is None:
        return None
    days = (info.period_end_date - end).days
    years = round(days / 365.25)
    return years if years >= 0 and abs(days - years * 365.25) <= YEAR_SLACK else None


def counted_unit(unit: str | None) -> bool:
    """Money (`iso4217:*`, per share included) or shares."""

    return unit is not None and (unit.startswith("iso4217:") or unit == "xbrli:shares")


def _series_key(candidate: FactCandidate) -> tuple[SeriesKey, int] | None:
    info, fact = candidate.filing, candidate.fact
    kind = _kind(fact)
    back = _years_back(info, fact)
    if (
        kind is None
        or back is None
        or fact.unit is None
        or not counted_unit(fact.unit)
        or info.fiscal_year is None
        or info.fiscal_period is None
        or (kind != "instant" and (kind == "annual") != (info.fiscal_period == "FY"))
    ):
        return None
    key: SeriesKey = (info.cik, fact.concept, fact.unit, kind, info.fiscal_period)
    return key, info.fiscal_year - back


def build_series(candidates: Iterable[FactCandidate]) -> Series:
    """Points by series and fiscal year; restated points dropped."""

    found: dict[tuple[SeriesKey, int], list[FactCandidate]] = defaultdict(list)
    for candidate in candidates:
        slot = _series_key(candidate)
        if slot is not None:
            found[slot].append(candidate)
    series: Series = defaultdict(dict)
    for (key, year), sources in sorted(found.items(), key=lambda item: item[0]):
        if len({Decimal(c.fact.value) for c in sources}) != 1:
            continue
        ordered = sorted(
            sources,
            key=lambda c: (not c.is_current, -c.filing.filing_date.toordinal()),
        )
        series[key][year] = Point(key, year, tuple(ordered))
    return series


# --- options -------------------------------------------------------------------


def period_phrase(year: int, period: str) -> str:
    quarter = _QUARTERS.get(period)
    return f"the {quarter} quarter of fiscal {year}" if quarter else f"fiscal {year}"


def _calc_dims(
    operation: str, point: Point, concept: str | None = None
) -> tuple[tuple[str, str], ...]:
    return (
        ("operation", operation),
        ("cik", point.cik),
        ("year", str(point.year)),
        ("concept", concept or point.concept),
        ("period", point.series[4]),
    )


def calc_options(series: Series) -> list[Option]:
    """Growth and CAGR of flows, difference of fiscal year-end balances,
    margin and share over revenue of the same period."""

    options: list[Option] = []
    for key, points in series.items():
        for new in points.values():
            options.extend(_change_options(key, points, new))
    options.extend(_ratio_options(series))
    return options


def _change_options(
    key: SeriesKey, points: Mapping[int, Point], new: Point
) -> Iterator[Option]:
    kind, year = key[3], new.year
    old = points.get(year - 1)
    if kind == "instant":
        if old is not None and key[4] == "FY":
            delta = plain(new.value - old.value)
            dims = _calc_dims("difference", new)
            yield Option("calc", "difference", (old, new), delta, dims)
        return
    if new.value <= 0:
        return
    if old is not None and old.value > 0:
        change = growth(old.value, new.value)
        if abs(change) <= MAX_GROWTH:
            dims = _calc_dims("growth", new)
            yield Option("calc", "growth", (old, new), percent(change), dims)
    # CAGR over the longest span of 3-5 fiscal years.
    first = next((points[y] for y in range(year - 4, year - 1) if y in points), None)
    if kind == "annual" and first is not None and first.value > 0:
        rate = percent(cagr(first.value, new.value, year - first.year))
        yield Option("calc", "cagr", (first, new), rate, _calc_dims("cagr", new))


def _ratio_options(series: Series) -> list[Option]:
    revenue: dict[tuple[str, str, str, int], Point] = {}
    for key, points in series.items():
        if key[1] in REVENUES and key[3] != "instant":
            for year, point in points.items():
                slot = (key[0], key[2], key[4], year)
                held = revenue.get(slot)
                if held is None or _revenue_order(point) < _revenue_order(held):
                    revenue[slot] = point
    options: list[Option] = []
    for key, points in series.items():
        operation = RATIOS.get(key[1])
        if operation is None or key[3] == "instant":
            continue
        for year, point in points.items():
            base = revenue.get((key[0], key[2], key[4], year))
            if base is None or base.value <= 0:
                continue
            ratio = point.value / base.value * 100
            if abs(ratio) <= 100:
                dims = _calc_dims(operation, point, f"{point.concept}/revenue")
                options.append(
                    Option("calc", operation, (point, base), percent(ratio), dims)
                )
    return options


def false_premise_options(
    series: Series, companies: Mapping[str, CompanyRef]
) -> list[Option]:
    """Year-over-year changes of flows of at least FALSE_PREMISE_MIN percent,
    with a hint stating the opposite direction; both values in one passage."""

    options: list[Option] = []
    for key, points in series.items():
        label = LABELS.get(key[1])
        if label is None or key[3] == "instant" or key[0] not in companies:
            continue
        name = display_name(companies[key[0]].name)
        for year, new in points.items():
            old = points.get(year - 1)
            if old is None or min(old.value, new.value) <= 0:
                continue
            change = growth(old.value, new.value)
            if not FALSE_PREMISE_MIN <= abs(change) <= MAX_GROWTH:
                continue
            verb = "fell" if change > 0 else "rose"
            hint = (
                f"{name}'s {label} {verb} in {period_phrase(year, key[4])} "
                f"compared with {period_phrase(year - 1, key[4])}"
            )
            dims = _calc_dims("growth", new)[1:]
            options.append(
                Option(
                    "false_premise",
                    "growth",
                    (old, new),
                    percent(change),
                    dims,
                    hint,
                    single_chunk=True,
                )
            )
    return options


def ranking_options(
    series: Series, sectors: Mapping[str, str], seed: int
) -> list[Option]:
    """One concept and fiscal year across 3-4 companies of one sector whose
    values differ by at least RANK_GAP and whose period ends lie within
    RANK_DAYS; points in seeded order, `order` high to low."""

    # Revenue concepts count as one (each company's preferred one).
    groups: dict[tuple[str, str, str, int], dict[str, Point]] = defaultdict(dict)
    for key, points in series.items():
        sector = sectors.get(key[0])
        if sector is None or key[4] != "FY":
            continue
        concept = REVENUES[0] if key[1] in REVENUES else key[1]
        for year, point in points.items():
            group = groups[(sector, concept, key[2], year)]
            held = group.get(point.cik)
            if held is None or _revenue_order(point) < _revenue_order(held):
                group[point.cik] = point
    options: list[Option] = []
    for (sector, concept, _, year), by_cik in sorted(groups.items()):
        ordered = sorted(
            by_cik.values(), key=lambda p: rank(seed, f"{concept}:{year}:{p.cik}")
        )
        chosen = _spread(ordered, 4)
        if len(chosen) < 3:
            continue
        order = tuple(p.cik for p in sorted(chosen, key=lambda p: -p.value))
        dims = (("sector", sector), ("concept", concept), ("year", str(year)))
        options.append(
            Option("ranking", "ranking", tuple(chosen), None, dims, order=order)
        )
    return options


def _revenue_order(point: Point) -> int:
    concept = point.concept
    return REVENUES.index(concept) if concept in REVENUES else 0


def _comparable(pair: tuple[Point, Point]) -> bool:
    """Values at least RANK_GAP apart, period ends within RANK_DAYS."""

    one, other = pair
    low, high = sorted((abs(one.value), abs(other.value)))
    if one.end is None or other.end is None or high == 0:
        return False
    close = abs((one.end - other.end).days) <= RANK_DAYS
    return close and high - low >= RANK_GAP * high


def _spread(points: Sequence[Point], size: int) -> list[Point]:
    """The first subset of `size` (else size - 1) points, in order, every two
    of them `_comparable`."""

    for width in (size, size - 1):
        for subset in combinations(points[:8], width):
            if all(_comparable(pair) for pair in combinations(subset, 2)):
                return list(subset)
    return []


# --- passages ------------------------------------------------------------------


def _shows(text: ChunkText | None, raw: str) -> bool:
    return text is not None and whole_number(raw).search(text.text) is not None


def cover(
    points: Sequence[Point],
    chunks_of: Mapping[str, Sequence[ChunkMeta]],
    texts: Mapping[str, ChunkText],
) -> tuple[list[ChunkMeta], list[XbrlFact]] | None:
    """Fewest chunks showing every point (statements first), and each fact
    with `block_id`/`cell_id` of the occurrence shown and `block_ids` of all
    its occurrences; None if a point shows nowhere."""

    found: list[dict[str, tuple[ChunkMeta, Occurrence]]] = []
    for point in points:
        shown: dict[str, tuple[ChunkMeta, Occurrence]] = {}
        for occurrence in point.occurrences:
            for chunk in chunks_of.get(occurrence.block_id, ()):
                if _shows(texts.get(chunk.chunk_id), occurrence.raw_value):
                    shown.setdefault(chunk.chunk_id, (chunk, occurrence))
        if not shown:
            return None
        found.append(shown)
    chosen: list[ChunkMeta] = []
    left = set(range(len(points)))
    while left:
        pool = {cid: entry[0] for i in left for cid, entry in found[i].items()}
        hits = Counter(cid for i in left for cid in found[i])
        best = min(
            pool.values(),
            key=lambda c: (
                -hits[c.chunk_id],
                not c.is_statement,
                c.part or 0,
                c.chunk_id,
            ),
        )
        chosen.append(best)
        left = {i for i in left if best.chunk_id not in found[i]}
    # Passages in operand order.
    chosen.sort(key=lambda c: next(i for i, f in enumerate(found) if c.chunk_id in f))
    facts: list[XbrlFact] = []
    for point, shown in zip(points, found, strict=True):
        chunk_id = next(c.chunk_id for c in chosen if c.chunk_id in shown)
        occurrence = shown[chunk_id][1]
        update = {
            "block_id": occurrence.block_id,
            "cell_id": occurrence.cell_id,
            "block_ids": tuple(o.block_id for o in point.occurrences),
        }
        facts.append(point.candidate.fact.model_copy(update=update))
    return chosen, facts


def hard_negatives(
    points: Sequence[Point],
    series: Series,
    chunks_of: Mapping[str, Sequence[ChunkMeta]],
    texts: Mapping[str, ChunkText],
) -> tuple[str, ...]:
    """Chunks of the same series in years before every operand's, latest
    first; never one holding an operand's block or showing its raw value."""

    first = min(p.year for p in points)
    blocks = {o.block_id for p in points for o in p.occurrences}
    raws = {o.raw_value for p in points for o in p.occurrences}

    def usable(chunk: ChunkMeta) -> bool:
        text = texts.get(chunk.chunk_id)
        return (
            text is not None
            and blocks.isdisjoint(chunk.block_ids)
            and not any(_shows(text, raw) for raw in raws)
        )

    near: list[tuple[int, str]] = []
    for key in dict.fromkeys(p.series for p in points):
        for year, other in series[key].items():
            if year >= first:
                continue
            for occurrence in other.candidate.occurrences:
                for chunk in chunks_of.get(occurrence.block_id, ()):
                    if usable(chunk):
                        near.append((first - year, chunk.chunk_id))
    picked = dict.fromkeys(chunk_id for _, chunk_id in sorted(near))
    return tuple(picked)[:MAX_HARD_NEGATIVES]


# --- picking -------------------------------------------------------------------


Placed = tuple[list[ChunkMeta], list[XbrlFact]]


@dataclass
class FactPicker:
    """Greedy picks: options with current values first, then the least-used
    coverage values in dimension order, then a seeded hash. No chunk is used
    twice; no (series, year) twice outside ranking."""

    seed: int
    counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    used_chunks: set[str] = field(default_factory=set)
    used_points: set[tuple[SeriesKey, int]] = field(default_factory=set)

    def cost(self, option: Option) -> tuple[int, tuple[int, ...], int]:
        stale = sum(not p.current for p in option.points)
        counts = tuple(self.counts[dim] for dim in option.dims)
        key = "|".join(p.candidate.fact.fact_id for p in option.points)
        return stale, counts, rank(self.seed, key)

    def pick(
        self,
        options: Sequence[Option],
        quota: int,
        place: _Placer,
    ) -> list[tuple[Option, Placed]]:
        live = list(options)
        picked: list[tuple[Option, Placed]] = []
        while len(picked) < quota and live:
            option = min(live, key=self.cost)
            live.remove(option)
            keys = {(p.series, p.year) for p in option.points}
            if option.question_type != "ranking" and keys & self.used_points:
                continue
            placed = place(option.points)
            if (
                placed is None
                or (option.single_chunk and len(placed[0]) > 1)
                or any(c.chunk_id in self.used_chunks for c in placed[0])
            ):
                continue
            self.counts.update(option.dims)
            self.used_chunks.update(c.chunk_id for c in placed[0])
            self.used_points.update(keys)
            picked.append((option, placed))
        return picked


class _Placer:
    """`cover` bound to the chunks read for the sampled facts."""

    def __init__(
        self,
        chunks_of: Mapping[str, Sequence[ChunkMeta]],
        texts: Mapping[str, ChunkText],
    ) -> None:
        self.chunks_of = chunks_of
        self.texts = texts

    def __call__(self, points: Sequence[Point]) -> Placed | None:
        return cover(points, self.chunks_of, self.texts)


# --- entry points --------------------------------------------------------------


def company_sectors(
    stats: CorpusStats,
    universe: Mapping[str, CompanyProfile],
    overrides: Mapping[str, str],
) -> dict[str, str]:
    """cik -> sector of corpus companies: `overrides`, else the universe's."""

    sectors: dict[str, str] = {}
    for cik in stats.companies:
        profile = universe.get(cik)
        sector = overrides.get(cik) or (profile.sector if profile else None)
        if sector:
            sectors[cik] = sector
    return sectors


def pick_companies(
    stats: CorpusStats,
    universe: Mapping[str, CompanyProfile],
    sectors: Mapping[str, str],
    seed: int,
    limit: int,
) -> list[str]:
    """At most `limit` corpus companies: sectors in turn (seeded order), the
    least-used size decile within each, so ranking has peers per sector."""

    by_sector: dict[str, list[str]] = defaultdict(list)
    for cik in stats.companies:
        by_sector[sectors.get(cik, "")].append(cik)
    order = sorted(by_sector, key=lambda s: (s == "", rank(seed, f"sector:{s}")))

    def decile(cik: str) -> int:
        profile = universe.get(cik)
        return profile.size_decile if profile else 0

    deciles: Counter[int] = Counter()
    picked: list[str] = []
    while len(picked) < limit and any(by_sector.values()):
        for sector in order:
            ciks = by_sector[sector]
            if not ciks or len(picked) == limit:
                continue
            cik = min(ciks, key=lambda c: (deciles[decile(c)], rank(seed, c)))
            ciks.remove(cik)
            deciles[decile(cik)] += 1
            picked.append(cik)
    return picked


@dataclass(frozen=True, slots=True)
class FactInputs:
    """What `plan_fact_tasks` needs besides chunks: the series, the companies,
    and the blocks (with their filings' dates) whose chunks it must see."""

    series: Series
    companies: Mapping[str, CompanyRef]  # with sector
    sectors: Mapping[str, str]
    blocks: frozenset[str]
    filing_dates: dict[str, date]  # filing_id -> filing_date


def fact_inputs(
    silver: Catalog,
    stats: CorpusStats,
    universe: Mapping[str, CompanyProfile],
    settings: FactSettings,
) -> FactInputs:
    """Read the facts of `pick_companies` and build their series."""

    sectors = company_sectors(stats, universe, settings.sectors)
    ciks = set(
        pick_companies(stats, universe, sectors, settings.seed, settings.max_companies)
    )
    filings = {fid: info for fid, info in stats.filings.items() if info.cik in ciks}
    per_company = Counter(
        info.cik for info in filings.values() if info.form in ("10-K", "10-Q")
    )
    limit = max(per_company.values(), default=0) * len(per_company)
    series = build_series(read_fact_candidates(silver, filings, settings.seed, limit))
    sources = [
        c for by_year in series.values() for p in by_year.values() for c in p.sources
    ]
    return FactInputs(
        series=series,
        companies={
            cik: stats.companies[cik].model_copy(update={"sector": sectors.get(cik)})
            for cik in sorted(ciks)
        },
        sectors={cik: sector for cik, sector in sectors.items() if cik in ciks},
        blocks=frozenset(o.block_id for c in sources for o in c.occurrences),
        filing_dates={c.filing_id: c.filing.filing_date for c in sources},
    )


def plan_fact_tasks(
    inputs: FactInputs,
    chunks_of: Mapping[str, Sequence[ChunkMeta]],
    texts: Mapping[str, ChunkText],
    settings: FactSettings,
) -> list[FactTask]:
    """Ranking, then false-premise, then calc tasks (narrowest pool first);
    `chunks_of` (block -> chunks) and `texts` must cover `inputs.blocks`."""

    seed = settings.seed
    picker = FactPicker(seed)
    place = _Placer(chunks_of, texts)
    plans: tuple[tuple[QuestionType, list[Option]], ...] = (
        ("ranking", ranking_options(inputs.series, inputs.sectors, seed)),
        ("false_premise", false_premise_options(inputs.series, inputs.companies)),
        ("calc", calc_options(inputs.series)),
    )
    tasks: list[FactTask] = []
    for qtype, options in plans:
        quota = settings.quotas.get(qtype, 0)
        for option, (picked, facts) in picker.pick(options, quota, place):
            negatives = hard_negatives(option.points, inputs.series, chunks_of, texts)
            tasks.append(_fact_task(option, picked, facts, inputs, negatives))
    return tasks


def _fact_task(
    option: Option,
    picked: Sequence[ChunkMeta],
    facts: Sequence[XbrlFact],
    inputs: FactInputs,
    negatives: tuple[str, ...],
) -> FactTask:
    exact = option.operation in ("ranking", "difference")
    computed = Computed(
        operation=option.operation,
        value=option.value,
        unit=facts[0].unit if exact else "percent",
        operands=tuple(facts),
        order=option.order,
    )
    ciks = dict.fromkeys(p.cik for p in option.points)
    return FactTask(
        question_type=option.question_type,
        chunks=tuple(picked),
        computed=computed,
        companies=tuple(inputs.companies[cik] for cik in ciks),
        hint=option.hint,
        hard_negative_chunks=negatives,
    )

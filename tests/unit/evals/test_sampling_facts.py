from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

import pytest

from src.evals import sampling_facts
from src.evals.sampling_data import (
    ChunkMeta,
    ChunkText,
    CorpusStats,
    FactCandidate,
    FilingInfo,
    Occurrence,
    company_ref,
)
from src.evals.sampling_facts import (
    FactPicker,
    FactSettings,
    SeriesKey,
    build_series,
    calc_options,
    company_sectors,
    counted_unit,
    cover,
    fact_inputs,
    false_premise_options,
    hard_negatives,
    percent,
    pick_companies,
    plain,
    plan_fact_tasks,
    ranking_options,
)
from src.evals.testset import XbrlFact
from src.evals.universe import CompanyProfile

pytestmark = pytest.mark.unit

ALPHA, BETA, GAMMA, DELTA = (f"000000000{n}" for n in range(1, 5))
NAMES = {ALPHA: "Alpha Corp", BETA: "Beta Inc.", GAMMA: "Gamma Co", DELTA: "Delta"}


def profile(cik: str, sector: str | None, decile: int = 1) -> CompanyProfile:
    return CompanyProfile(
        cik, NAMES[cik], (NAMES[cik][:3].upper(),), sector, 1.0, decile
    )


UNIVERSE = {
    ALPHA: profile(ALPHA, "Information Technology"),
    BETA: profile(BETA, "Information Technology", 9),
    GAMMA: profile(GAMMA, "Information Technology"),
    DELTA: profile(DELTA, None),
}


def filing(cik: str, year: int, period: str = "FY") -> tuple[str, FilingInfo]:
    """A 10-K (December year end) or 10-Q of `cik` for fiscal `year`."""

    quarter = {"FY": 4, "Q1": 1, "Q2": 2, "Q3": 3}[period]
    end = date(year, 3 * quarter, 30 if quarter in (2, 3) else 31)
    form = "10-K" if period == "FY" else "10-Q"
    info = FilingInfo(cik, form, year, period, end, end + timedelta(days=40))
    return f"f-{cik}-{year}-{period}", info


def fact(
    cik: str,
    year: int,
    concept: str,
    value: str,
    *,
    back: int = 0,
    instant: bool = False,
    period: str = "FY",
    raw: str | None = None,
    unit: str = "iso4217:USD",
) -> FactCandidate:
    """A fact of the `year` filing for the period `back` years earlier."""

    filing_id, info = filing(cik, year, period)
    assert info.period_end_date is not None
    end = info.period_end_date.replace(year=year - back)
    length = 364 if period == "FY" else 90
    block = f"{filing_id}:block:{concept}:{back}"
    xbrl = XbrlFact(
        fact_id=f"{filing_id}:{concept}:{back}",
        concept=concept,
        value=value,
        unit=unit,
        decimals="-6",
        period_start=None if instant else end - timedelta(days=length),
        period_end=None if instant else end,
        instant=end if instant else None,
        block_id=block,
        cell_id=None,
        block_ids=(block,),
    )
    occurrence = Occurrence(block, None, raw or f"{int(value) // 1_000_000:,}")
    return FactCandidate(xbrl, filing_id, info, (occurrence,), back == 0)


def chunk_for(candidate: FactCandidate, number: int, **fields: Any) -> ChunkMeta:
    info = candidate.filing
    values: dict[str, Any] = {
        "chunk_id": f"c{number:04d}",
        "filing_id": candidate.filing_id,
        "document_id": f"{candidate.filing_id}:d1",
        "block_ids": (candidate.occurrences[0].block_id,),
        "cik": info.cik,
        "company_name": NAMES[info.cik],
        "tickers": (NAMES[info.cik][:3].upper(),),
        "form": info.form,
        "document_type": info.form,
        "section_topic": "financial_statements",
        "section_title": "Item 8",
        "subheading": None,
        "table_title": "Consolidated Statements of Operations",
        "chunk_kind": "table",
        "fiscal_year": info.fiscal_year,
        "fiscal_period": info.fiscal_period,
        "period_end_date": info.period_end_date,
        "filing_date": info.filing_date,
        "part": None,
        "part_count": None,
        "chars": 900,
    }
    values.update(fields)
    return ChunkMeta(**values)


def series_of(*candidates: FactCandidate) -> sampling_facts.Series:
    return build_series(candidates)


SECTORS = {ALPHA: "Information Technology", BETA: "Information Technology"}
SECTORS |= {GAMMA: "Information Technology"}
REVENUE: SeriesKey = (ALPHA, "Revenues", "iso4217:USD", "annual", "FY")


def text_of(*candidates: FactCandidate) -> ChunkText:
    return ChunkText("", " | ".join(c.occurrences[0].raw_value for c in candidates))


def test_plain_and_percent() -> None:
    assert plain(Decimal("1.20E+9")) == "1200000000"
    assert plain(Decimal("-0.000")) == "0"
    assert plain(Decimal("2.500")) == "2.5"
    assert percent(Decimal("12.34567")) == "12.3457"
    assert percent(Decimal("10.00001")) == "10"
    assert percent(Decimal("-0.00001")) == "0"


def test_units() -> None:
    assert counted_unit("iso4217:USD")
    assert counted_unit("iso4217:USD/xbrli:shares")
    assert counted_unit("xbrli:shares")
    assert not counted_unit("xbrli:pure")
    assert not counted_unit(None)
    series = series_of(fact(ALPHA, 2024, "Ratio", "5000000", unit="xbrli:pure"))
    assert not series


def test_series_prefer_current_and_drop_restated() -> None:
    series = series_of(
        fact(ALPHA, 2024, "Revenues", "120000000"),
        fact(ALPHA, 2024, "Revenues", "100000000", back=1),  # comparative
        fact(ALPHA, 2023, "Revenues", "100000000"),  # same value, current
        fact(ALPHA, 2024, "Revenues", "90000000", back=2),  # only comparative
        fact(ALPHA, 2024, "NetIncomeLoss", "10000000", back=1),
        fact(ALPHA, 2023, "NetIncomeLoss", "11000000"),  # restated later
    )
    revenue = series[REVENUE]
    assert sorted(revenue) == [2022, 2023, 2024]
    assert revenue[2023].current and revenue[2023].candidate.filing.fiscal_year == 2023
    # The 2024 filing's comparative column also shows the 2023 value.
    blocks = [o.block_id for o in revenue[2023].occurrences]
    assert blocks == [
        f"f-{ALPHA}-2023-FY:block:Revenues:0",
        f"f-{ALPHA}-2024-FY:block:Revenues:1",
    ]
    assert not revenue[2022].current
    assert (ALPHA, "NetIncomeLoss", "iso4217:USD", "annual", "FY") not in series


def test_series_kinds() -> None:
    series = series_of(
        fact(ALPHA, 2024, "Assets", "500000000", instant=True),
        fact(ALPHA, 2024, "Revenues", "30000000", period="Q2"),
        fact(ALPHA, 2024, "Revenues", "25000000", period="Q2", back=1),
    )
    quarter: SeriesKey = (ALPHA, "Revenues", "iso4217:USD", "quarter", "Q2")
    assert set(series) == {(ALPHA, "Assets", "iso4217:USD", "instant", "FY"), quarter}
    assert sorted(series[quarter]) == [2023, 2024]


def test_calc_options_values() -> None:
    series = series_of(
        fact(ALPHA, 2021, "Revenues", "100000000"),
        fact(ALPHA, 2023, "Revenues", "121000000"),
        fact(ALPHA, 2024, "Revenues", "133100000"),
        fact(ALPHA, 2024, "NetIncomeLoss", "13310000"),
        fact(ALPHA, 2024, "ResearchAndDevelopmentExpense", "26620000"),
        fact(ALPHA, 2024, "Goodwill", "70000000", instant=True),
        fact(ALPHA, 2023, "Goodwill", "75000000", instant=True),
        fact(ALPHA, 2024, "OperatingIncomeLoss", "12345678"),
    )
    found = {
        (o.operation, o.points[0].concept, o.points[-1].year): o
        for o in calc_options(series)
    }
    assert found[("growth", "Revenues", 2024)].value == "10"
    assert found[("cagr", "Revenues", 2024)].value == "10"  # 2021 -> 2024
    assert found[("cagr", "Revenues", 2023)].value == "10"  # 2021 -> 2023
    assert found[("difference", "Goodwill", 2024)].value == "-5000000"
    margin = found[("margin", "NetIncomeLoss", 2024)]
    assert margin.value == "10"
    assert [p.concept for p in margin.points] == ["NetIncomeLoss", "Revenues"]
    assert found[("share", "ResearchAndDevelopmentExpense", 2024)].value == "20"
    # Unrounded: 12345678 / 133100000 = 9.27548...%
    assert found[("margin", "OperatingIncomeLoss", 2024)].value == "9.2755"
    assert ("growth", "Revenues", 2022) not in found  # no 2022 value


def test_false_premise_states_the_opposite_with_display_name() -> None:
    companies = {ALPHA: company_ref(ALPHA, "ALPHA HOLDING CORP", ("ALP",))}
    series = series_of(
        fact(ALPHA, 2023, "NetIncomeLoss", "100000000"),
        fact(ALPHA, 2024, "NetIncomeLoss", "80000000"),
        fact(ALPHA, 2023, "Revenues", "100000000"),
        fact(ALPHA, 2024, "Revenues", "102000000"),  # under 5%
    )
    (option,) = false_premise_options(series, companies)
    assert option.value == "-20"
    assert option.single_chunk
    assert option.hint == (
        "Alpha's net income rose in fiscal 2024 compared with fiscal 2023"
    )


def test_ranking_needs_three_comparable_peers() -> None:
    series = series_of(
        fact(ALPHA, 2024, "Revenues", "300000000"),
        fact(
            BETA,
            2024,
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "500000000",
        ),
        fact(GAMMA, 2024, "Revenues", "100000000"),
        fact(DELTA, 2024, "Revenues", "200000000"),  # no sector
        fact(ALPHA, 2024, "Goodwill", "10000000", instant=True),
        fact(BETA, 2024, "Goodwill", "10000000", instant=True),  # tie
        fact(GAMMA, 2024, "Goodwill", "20000000", instant=True),
    )
    (option,) = ranking_options(series, SECTORS, seed=1)
    assert option.order == (BETA, ALPHA, GAMMA)
    assert sorted(p.cik for p in option.points) == [ALPHA, BETA, GAMMA]
    assert dict(option.dims)["sector"] == "Information Technology"
    # A period ending months apart (another fiscal calendar) is not ranked.
    late = fact(GAMMA, 2024, "Revenues", "100000000", period="Q2")
    shifted = build_series([late])
    gamma = next(iter(shifted.values()))[2024]
    moved = series_of(
        fact(ALPHA, 2024, "Revenues", "300000000"),
        fact(BETA, 2024, "Revenues", "500000000"),
    )
    gamma_key: SeriesKey = (GAMMA, "Revenues", "iso4217:USD", "annual", "FY")
    moved[gamma_key] = {2024: gamma}
    assert ranking_options(moved, SECTORS, seed=1) == []


def test_cover_uses_one_chunk_showing_both_years() -> None:
    new = fact(ALPHA, 2024, "Revenues", "120000000")
    comparative = fact(ALPHA, 2024, "Revenues", "100000000", back=1)
    old = fact(ALPHA, 2023, "Revenues", "100000000")
    series = series_of(new, comparative, old)
    points = list(series[REVENUE].values())
    both = chunk_for(new, 1, block_ids=(new.fact.block_id, comparative.fact.block_id))
    alone = chunk_for(old, 2)
    chunks_of = {
        new.fact.block_id: [both],
        comparative.fact.block_id: [both],
        old.fact.block_id: [alone],
    }
    texts = {"c0001": text_of(new, comparative), "c0002": text_of(old)}
    placed = cover(points, chunks_of, texts)
    assert placed is not None
    assert [c.chunk_id for c in placed[0]] == ["c0001"]
    old_fact = placed[1][0]
    assert old_fact.block_id == comparative.fact.block_id
    assert set(old_fact.block_ids) == {old.fact.block_id, comparative.fact.block_id}
    # A chunk whose text does not show the raw value is no passage.
    texts["c0001"] = ChunkText("", "Revenue | 120 | 1,100")
    placed = cover(points, chunks_of, texts)
    assert placed is not None
    assert [c.chunk_id for c in placed[0]] == ["c0002", "c0001"]
    assert cover(points, {}, texts) is None


def test_hard_negatives_only_older_years_without_the_answer() -> None:
    candidates = [
        fact(ALPHA, year, "Revenues", f"{100 + year - 2020}000000")
        for year in (2020, 2021, 2022, 2023, 2024)
    ]
    series = build_series(candidates)
    chunks = {c.filing.fiscal_year: chunk_for(c, n) for n, c in enumerate(candidates)}
    chunks_of = {c.fact.block_id: [chunks[c.filing.fiscal_year]] for c in candidates}
    texts = {chunks[c.filing.fiscal_year].chunk_id: text_of(c) for c in candidates}
    # The 2021 chunk also shows the 2022 value (a comparative column).
    texts[chunks[2021].chunk_id] = text_of(candidates[1], candidates[2])
    points = [series[REVENUE][2022], series[REVENUE][2023]]
    assert hard_negatives(points, series, chunks_of, texts) == (chunks[2020].chunk_id,)


def test_picker_spreads_and_never_reuses() -> None:
    series = series_of(
        *(
            fact(cik, year, "Revenues", str(100_000_000 + year * 1_000_000 + n))
            for n, cik in enumerate((ALPHA, BETA))
            for year in (2022, 2023, 2024)
        )
    )
    options = [o for o in calc_options(series) if o.operation == "growth"]

    def place(points: Any) -> tuple[list[ChunkMeta], list[XbrlFact]]:
        chunks = [chunk_for(p.candidate, int(p.cik) * 10_000 + p.year) for p in points]
        return chunks, [p.candidate.fact for p in points]

    picked = FactPicker(seed=1).pick(options, 3, place)  # type: ignore[arg-type]
    # A company's 2023 chunk is in both its growth pairs: one pair each.
    assert sorted(option.points[0].cik for option, _ in picked) == [ALPHA, BETA]


def test_pick_companies_rotates_sectors() -> None:
    stats = CorpusStats(
        companies={cik: company_ref(cik, NAMES[cik], ()) for cik in NAMES}
    )
    sectors = company_sectors(stats, UNIVERSE, {DELTA: "Utilities"})  # SIC fallback
    assert sectors[DELTA] == "Utilities"
    picked = pick_companies(stats, UNIVERSE, sectors, seed=1, limit=2)
    assert {sectors[cik] for cik in picked} == {"Information Technology", "Utilities"}
    everyone = pick_companies(stats, UNIVERSE, sectors, seed=1, limit=9)
    assert sorted(everyone) == sorted(NAMES)


def test_fact_inputs_then_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = [
        fact(
            cik,
            year,
            "Revenues",
            f"{base + 50 * (year - back - 2021)}000000",
            back=back,
        )
        for cik, base in ((ALPHA, 300), (BETA, 500), (GAMMA, 100))
        for year in (2021, 2022, 2023, 2024)
        for back in (0, 1)
    ]
    # One chunk per filing statement, showing its current and prior year.
    chunks: dict[str, ChunkMeta] = {}
    texts: dict[str, ChunkText] = {}
    for number, c in enumerate(candidates):
        if c.is_current:
            prior = candidates[number + 1]
            blocks = (c.fact.block_id, prior.fact.block_id)
            meta = chunk_for(c, number, block_ids=blocks)
            chunks[c.filing_id] = meta
            texts[meta.chunk_id] = text_of(c, prior)
    stats = CorpusStats(
        companies={
            cik: company_ref(cik, NAMES[cik], ()) for cik in (ALPHA, BETA, GAMMA)
        },
        filings=dict(
            filing(c.filing.cik, c.filing.fiscal_year or 0) for c in candidates
        ),
    )

    def read_facts(
        silver: object, filings: Mapping[str, FilingInfo], seed: int, limit: int
    ) -> list[FactCandidate]:
        assert limit == 12
        return [c for c in candidates if c.filing_id in filings]

    monkeypatch.setattr(sampling_facts, "read_fact_candidates", read_facts)
    settings = FactSettings({"ranking": 1, "false_premise": 1, "calc": 2}, seed=1)
    inputs = fact_inputs(None, stats, UNIVERSE, settings)  # type: ignore[arg-type]
    assert inputs.filing_dates.keys() == chunks.keys()
    chunks_of: dict[str, list[ChunkMeta]] = {}
    for meta in chunks.values():
        for block in meta.block_ids:
            assert block in inputs.blocks
            chunks_of.setdefault(block, []).append(meta)
    tasks = plan_fact_tasks(inputs, chunks_of, texts, settings)
    assert Counter(t.question_type for t in tasks) == {
        "ranking": 1,
        "false_premise": 1,
        "calc": 2,
    }
    ranking = tasks[0]
    assert ranking.computed.order == (BETA, ALPHA, GAMMA)
    assert [c.cik for c in ranking.companies] != list(ranking.computed.order)
    assert all(c.sector == "Information Technology" for c in ranking.companies)
    premise = tasks[1]
    assert premise.hint is not None and " fell in " in premise.hint
    assert len(premise.chunks) == 1
    for task in tasks:
        assert not set(task.hard_negative_chunks) & {c.chunk_id for c in task.chunks}
        for operand in task.computed.operands:
            assert any(operand.block_id in c.block_ids for c in task.chunks)

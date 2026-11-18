from __future__ import annotations

import json
from datetime import date
from typing import Any, get_args

import pytest

from src.data_pipeline.transforms.native_rag.section_topics import SectionTopic
from src.rag_engine.retrieval.companies import CompanyIndex, normalize_name
from src.rag_engine.retrieval.corpora import CORPORA, SEC_FILINGS_PROFILE
from src.rag_engine.retrieval.dedup import dedupe
from src.rag_engine.retrieval.query import (
    QueryAnalysis,
    analyze_query,
    checked_plan,
    rule_plan,
)
from src.rag_engine.retrieval.search import Hit, SearchFilter, Window, qdrant_filter
from tests.llm_support import FakeChat, reply

pytestmark = pytest.mark.unit

TODAY = date(2026, 11, 17)


def _row(cik: str, name: str, tickers: list[str], *names: str) -> dict[str, Any]:
    return {
        "cik": cik,
        "company_name": name,
        "tickers": tickers,
        "names": [name, *names],
        "latest_filing_date": "2026-07-31",
    }


COMPANIES = CompanyIndex(
    [
        _row("0000320193", "Apple Inc.", ["AAPL"]),
        _row("0001652044", "Alphabet Inc.", ["GOOG", "GOOGL"]),
        _row("0001018724", "AMAZON.COM, INC.", ["AMZN"]),
        _row("0001326801", "Meta Platforms, Inc.", ["META"], "Facebook Inc"),
        _row("0000070858", "BANK OF AMERICA CORP /DE/", ["BAC"]),
        _row("0001390777", "Bank of New York Mellon Corp", ["BK"]),
        _row("0001067983", "BERKSHIRE HATHAWAY INC", ["BRK.B"]),
    ]
)


@pytest.mark.parametrize(
    ("name", "normalized"),
    [
        ("AMAZON.COM, INC.", "amazon"),
        ("The Coca-Cola Company", "coca cola"),
        ("Apple Inc.", "apple"),
        ("AT&T Inc.", "at and t"),
        ("Alphabet Inc. Class A", "alphabet"),
        ("Nestlé S.A.", "nestle"),
    ],
)
def test_normalize_name(name: str, normalized: str) -> None:
    assert normalize_name(name) == normalized


@pytest.mark.parametrize(
    ("name", "ticker", "cik"),
    [
        ("Alphabet Inc.", "GOOGL", "0001652044"),
        ("Google", "GOOG", "0001652044"),  # ticker wins
        ("Amazon", None, "0001018724"),
        ("amazon.com", None, "0001018724"),
        ("Meta", None, "0001326801"),  # unique whole-word prefix
        ("Facebook", None, "0001326801"),  # a former name
        ("MSFT", None, None),  # not in this corpus
        ("AAPL", None, "0000320193"),  # an uppercase name read as a ticker
        ("Berkshire Hathaway", "BRK-B", "0001067983"),  # ticker spelling
        ("Bank", None, None),  # ambiguous prefix
        ("Bank of America", None, "0000070858"),
        ("Tesla", "TSLA", None),
        ("Apple Hospitality REIT, Inc.", None, None),  # a longer, other name
    ],
)
def test_resolve(name: str, ticker: str | None, cik: str | None) -> None:
    found = COMPANIES.resolve(name, ticker)
    assert (found.cik if found else None) == cik


def test_find_in_text() -> None:
    found = COMPANIES.find_in_text(
        "So sánh doanh thu của Amazon với AAPL; apple pie is no ticker"
    )
    assert [c.cik for c in found] == ["0001018724", "0000320193"]
    assert COMPANIES.find_in_text("What does IT mean for the US?") == []
    words = CompanyIndex(
        [
            _row("0000027419", "TARGET CORP", ["TGT"]),
            _row("0000749251", "Gartner Inc", ["IT"]),
            _row("0001564708", "News Corp", ["NWSA"]),
        ]
    )
    for question in ("Apple's target gross margin", "spend on IT", "latest news"):
        assert words.find_in_text(question) == [], question
    assert [
        c.cik for c in words.find_in_text("Compare Target vs $IT and News Corp")
    ] == [
        "0000027419",
        "0000749251",
        "0001564708",
    ]


def test_topic_descriptions_cover_every_section_topic() -> None:
    assert set(SEC_FILINGS_PROFILE.topics) == set(get_args(SectionTopic))


def _analysis(**overrides: Any) -> QueryAnalysis:
    values: dict[str, Any] = {
        "language": "vi",
        "corpora": ["sec_filings"],
        "searches": [{"language": "en", "text": "What was Apple's Q2 2024 revenue?"}],
        "companies": [{"name": "Apple Inc.", "ticker": "aapl"}],
        "fiscal_years": [2024],
        "fiscal_periods": ["q2"],
        "forms": ["10 Q"],
        "topics": ["financial_statements"],
        "wants_latest": False,
        "period_basis": "fiscal",
    }
    return QueryAnalysis.model_validate(values | overrides)


def test_checked_plan_keeps_known_values_only() -> None:
    plan = checked_plan(
        "q",
        _analysis(
            corpora=["sec_filings", "vn_law"],
            topics=["financial_statements", "weather"],
            fiscal_years=[2024, 1700],
            wants_latest=True,
        ),
        CORPORA,
    )
    assert plan.corpora == ("sec_filings",)
    assert plan.text_for("en") == "What was Apple's Q2 2024 revenue?"
    assert plan.companies[0].ticker == "AAPL"
    assert (plan.fiscal_years, plan.fiscal_periods, plan.forms) == (
        (2024,),
        ("Q2",),
        ("10-Q",),
    )
    assert plan.topics == ("financial_statements",)
    assert not plan.wants_latest  # a period was named
    assert any("vn_law" in note for note in plan.notes)
    assert any("topics" in note for note in plan.notes)
    no_corpus = checked_plan("q", _analysis(corpora=[], searches=[]), CORPORA)
    assert no_corpus.corpora == ("sec_filings",) and no_corpus.text_for("en") == "q"


def test_analyze_query_with_model_and_fallbacks() -> None:
    chat = FakeChat([reply(_analysis().model_dump_json())])
    plan = analyze_query("Doanh thu quý 2/2024 của Apple?", CORPORA, chat, TODAY)
    assert plan.source == "llm" and plan.fiscal_periods == ("Q2",)
    _, options = chat.calls[0]
    assert options is not None and options.output_schema is not None
    assert options.system is not None and "Today is 2026-11-17" in options.system
    assert "sec_filings (language en)" in options.system
    broken = analyze_query(
        "Apple 10-K 2023 risk factors", CORPORA, FakeChat([reply("not json")]), TODAY
    )
    assert broken.source == "rules" and "query model failed" in broken.notes[0]
    assert (broken.fiscal_years, broken.forms, broken.topics) == (
        (2023,),
        ("10-K",),
        ("risk_factors",),
    )
    assert analyze_query("x", CORPORA, None, TODAY).notes == ("no query model: rules",)


@pytest.mark.parametrize(
    ("question", "language", "years", "periods", "latest"),
    [
        ("Doanh thu quý 2 năm 2024 của Apple?", "vi", (2024,), ("Q2",), False),
        ("NVIDIA FY2025 revenue", "en", (2025,), ("FY",), False),
        ("third quarter 2023 results", "en", (2023,), ("Q3",), False),
        ("Rủi ro gần nhất của Amazon", "vi", (), (), True),
        ("Apple Q1 and Q3 2022", "en", (2022,), ("Q1", "Q3"), False),
        ("Apple Q2 FY24 revenue", "en", (2024,), ("Q2",), False),
        ("AAPL current liabilities now", "en", (), (), False),
    ],
)
def test_rule_plan(
    question: str,
    language: str,
    years: tuple[int, ...],
    periods: tuple[str, ...],
    latest: bool,
) -> None:
    plan = rule_plan(question, CORPORA)
    assert (plan.language, plan.fiscal_years, plan.fiscal_periods) == (
        language,
        years,
        periods,
    )
    assert plan.wants_latest is latest


def _hit(point: str, score: float, text: str, filed: str, cik: str = "1") -> Hit:
    payload = {"chunk_id": point, "text": text, "filing_date": filed, "cik": cik}
    return Hit(point, score, payload)


def test_dedupe_keeps_best_score_and_newest_copy() -> None:
    hits = [
        _hit("a", 0.9, "Forward-looking  statements.", "2023-01-01"),
        _hit("b", 0.8, "Unique", "2024-01-01"),
        _hit("c", 0.5, "forward-looking statements.", "2025-01-01"),
    ]
    newest, removed = dedupe(hits, prefer_newest=True)
    assert removed == 1
    assert [(h.point_id, h.score) for h in newest] == [("c", 0.9), ("b", 0.8)]
    assert newest[0].duplicates == ("a",)
    best, _ = dedupe(hits, prefer_newest=False)
    assert [h.point_id for h in best] == ["a", "b"]
    assert best[0].duplicates == ("c",)
    # The same text of two companies stays two candidates.
    other = [*hits, _hit("d", 0.4, "Unique", "2024-01-01", cik="2")]
    assert dedupe(other, prefer_newest=True)[1] == 1


def test_period_filter_matches_periodic_forms_and_8k_windows() -> None:
    search = SearchFilter(
        ciks=("0000320193",),
        fiscal_years=(2024,),
        fiscal_periods=("Q4",),
        windows=(Window(date(2024, 6, 30), date(2024, 12, 31)),),
    )
    found = qdrant_filter(search)
    assert found is not None
    text = json.dumps(found.model_dump(mode="json", exclude_none=True))
    assert '"value": "FY"' in text  # Q4 numbers live in the 10-K
    assert '"8-K"' in text and "2024-06-30" in text
    only_10k = qdrant_filter(SearchFilter(forms=("10-K",), fiscal_years=(2024,)))
    assert only_10k is not None and "8-K" not in json.dumps(
        only_10k.model_dump(mode="json", exclude_none=True)
    )
    assert qdrant_filter(SearchFilter()) is None
    yearless = qdrant_filter(SearchFilter(fiscal_periods=("Q1",)))
    assert yearless is not None
    should = yearless.model_dump(mode="json", exclude_none=True)["must"][0]["should"]
    # 10-K/10-Q of Q1, or any 8-K (no year, so no window).
    assert should[1] == {"key": "form", "match": {"value": "8-K"}}
    relaxed = search.without_period()
    assert relaxed.ciks == search.ciks and not relaxed.has_period


def test_a_company_window_admits_only_that_company_8ks() -> None:
    apple, msft = "0000320193", "0000789019"
    found = qdrant_filter(
        SearchFilter(
            ciks=(apple, msft),
            fiscal_years=(2024,),
            windows=(
                Window(date(2024, 8, 1), date(2024, 12, 31), 2024, apple),
                Window(date(2024, 5, 1), date(2024, 10, 31), 2024),
            ),
        )
    )
    assert found is not None
    should = found.model_dump(mode="json", exclude_none=True)["must"][1]["should"]
    windows = should[-1]["must"][1]["should"]
    assert windows[0]["must"][1] == {"key": "cik", "match": {"value": apple}}
    assert windows[1]["key"] == "filing_date"  # fallback window: any company


def test_name_beats_a_conflicting_ticker() -> None:
    meta = COMPANIES.resolve("Meta Platforms", "AAPL")
    assert meta is not None and meta.cik == "0001326801"
    misspelled = COMPANIES.resolve("Aple Incorporated", "AAPL")
    assert misspelled is not None and misspelled.cik == "0000320193"


def test_calendar_filter_uses_period_end() -> None:
    found = qdrant_filter(
        SearchFilter(fiscal_years=(2024,), fiscal_periods=("Q3",), calendar=True)
    )
    assert found is not None
    text = json.dumps(found.model_dump(mode="json", exclude_none=True))
    assert "period_end_date" in text and "2024-08-16" in text and "2024-11-14" in text
    assert "fiscal_year" not in text
    whole = qdrant_filter(SearchFilter(fiscal_years=(2024,)))
    assert whole is not None and '"FY"' in json.dumps(
        whole.model_dump(mode="json", exclude_none=True)
    )  # a year alone reads the 10-K


def test_year_of_8k_window_and_calendar_quarter() -> None:
    from src.rag_engine.retrieval.retriever import Candidate, YearRule, _year_of

    def cand(**payload: Any) -> Candidate:
        return Candidate("sec_filings", Hit("p", 1.0, payload))

    release = cand(filing_date="2024-02-01T00:00:00Z", fiscal_year=None)
    windows = (Window(date(2023, 11, 15), date(2024, 4, 9), 2023),)
    # The FY2023 results release filed in 2024 belongs to 2023.
    assert _year_of(release, YearRule(windows=windows)) == 2023
    assert _year_of(release, YearRule()) == 2024
    q1 = cand(fiscal_year=2024, period_end_date="2024-03-31T00:00:00Z",
              filing_date="2024-05-01T00:00:00Z")  # fmt: skip
    assert _year_of(q1, YearRule(calendar=True, quarters=True)) == 2024
    assert _year_of(q1, YearRule(calendar=True)) == 2023  # full-year rule


def test_multiword_first_word_names_at_sentence_start() -> None:
    index = CompanyIndex(
        [
            _row("0001326801", "Meta Platforms, Inc.", ["META"]),
            _row("0001274494", "First Solar, Inc.", ["FSLR"]),
            _row("0000320193", "Apple Inc.", ["AAPL"]),
        ]
    )
    assert [c.cik for c in index.find_in_text("Meta revenue in 2024?")] == [
        "0001326801"
    ]
    assert [c.cik for c in index.find_in_text("First quarter revenue of Apple?")] == [
        "0000320193"
    ]

from __future__ import annotations

import json
from datetime import date
from typing import Any, get_args

import pytest

from src.data_pipeline.transforms.native_rag.section_topics import SectionTopic
from src.llm import Message
from src.llm.structured import strict_json_schema
from src.rag_engine.retrieval.companies import CompanyIndex, normalize_name
from src.rag_engine.retrieval.corpora import CORPORA, SEC_FILINGS_PROFILE
from src.rag_engine.retrieval.dedup import dedupe
from src.rag_engine.retrieval.query import (
    QueryAnalysis,
    analysis_model,
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


TICKERS = CompanyIndex(
    [
        _row("0000895126", "EXPAND ENERGY CORPORATION", ["EXE"]),
        _row("0001109357", "EXELON CORPORATION", ["EXC"]),
        _row("0000063754", "McCormick & Company, Inc.", ["MKC", "MKC-V"]),
        _row("0000927653", "McKESSON CORPORATION", ["MCK"]),
        _row("0000711404", "THE COOPER COMPANIES, INC.", ["COO"]),
        _row("0001048286", "Crown Castle Inc.", ["CCI"]),
        _row("0000884887", "ROYAL CARIBBEAN CRUISES LTD.", ["RCL"]),
        _row("0000732717", "AT&T INC.", ["T"]),
        _row("0001035002", "VALERO ENERGY CORP/TX", ["VLO"]),
        _row("0001744489", "Walt Disney Co", ["DIS"]),
        _row("0001281761", "Regions Financial Corp", ["RF"]),
        _row("0000814547", "Fair Isaac Corp", ["FICO"]),
        _row("0000078003", "Pfizer Inc", ["PFE"]),
        _row("0001126328", "Principal Financial Group Inc", ["PFG"]),
        _row("0001045810", "NVIDIA CORP", ["NVDA"]),
        _row("0000860731", "Tyler Technologies Inc", ["TYL"]),
        _row("0000352915", "UNIVERSAL HEALTH SERVICES, INC.", ["UHS"]),
        _row("0000731766", "UNITEDHEALTH GROUP INCORPORATED", ["UNH"]),
        _row("0000079282", "BROWN & BROWN, INC.", ["BRO"]),
        _row("0002005951", "Smurfit WestRock plc", ["SW"]),
        _row("0000811156", "CMS ENERGY CORP", ["CMS"]),
    ]
)


@pytest.mark.parametrize(
    ("question", "mentions", "ciks", "missing"),
    [
        # The model read the written ticker as another company (t0061, t0078).
        ("In EXE's October 2024 8-K on the Southwestern deal, what happened?",
         [("Exelon Corporation", "EXC"), ("Southwestern Energy Company", "SWN")],
         ["0000895126"], ["Southwestern Energy Company (SWN)"]),
        ("Who is MKC's partner in its Mexican joint venture?",
         [("McKesson Corporation", "MCK")], ["0000063754"], []),
        # A model's ticker that the question writes wins over its name.
        ("Who is MKC's partner?", [("McKesson Corporation", "MKC")],
         ["0000063754"], []),
        # Written tickers the model missed are added (t0063, t0117).
        ("Votes for Kevin Kabat at CCI and Amy McPherson at RCL?",
         [("Crown Castle International Corp.", "CCI"), ("Ryder System, Inc.", None)],
         ["0001048286", "0000884887"], []),
        ("Which of CCI, RCL and VLO bring up the FDIC?", [],
         ["0001048286", "0000884887", "0001035002"], []),
        # A word ticker needs the model or a `$`.
        ("How much space does COO lease in Puerto Rico?", [], [], []),
        ("How much space does $COO lease in Puerto Rico?", [], ["0000711404"], []),
        ("How much space does COO lease?", [("The Cooper Companies", "COO")],
         ["0000711404"], []),
        ("Who is Valero's COO?", [("Valero Energy Corporation", "VLO")],
         ["0001035002"], []),
        # The model's only mention, unwritten, is a misread of the one
        # ticker written (x0047); not when the question names it.
        ("Who is the trustee for the notes UHS issued in August?",
         [("UnitedHealth Group Incorporated", "UNH")], ["0000352915"], []),
        ("Who is the trustee for the notes UnitedHealth issued, per UHS?",
         [("UnitedHealth Group Incorporated", "UNH")], ["0000731766"], []),
        # No company from the model: every written ticker is the user's,
        # a word ticker too when possessive (x0050, x0101).
        ("How much cash did BRO hold at June 30?", [], ["0000079282"], []),
        ("BRO nắm giữ bao nhiêu tiền vào cuối quý 2?", [("Broadcom Inc.", "AVGO")],
         ["0000079282"], []),
        ("How did SW's earnings presentation wording change?", [],
         ["0002005951"], []),
        ("How did SW wording change?", [], [], []),
        # Agencies, not companies; brands are no misread of another ticker.
        ("How did CMS's 2026 rate notice affect insurers?", [], [], []),
        ("What did Google say about UHS?", [("Alphabet Inc.", None)],
         [], ["Alphabet Inc."]),
        # Mentions the model resolved stay, whatever their names' wording.
        ("Compare Disney's and RCL's capex", [("The Walt Disney Company", "DIS")],
         ["0001744489", "0000884887"], []),
        ("AT&T (T) vs RCL's buybacks", [("AT&T Inc.", "T")],
         ["0000732717", "0000884887"], []),
        # A lone uppercase word is no ticker unless it plainly is one.
        ("Valero vs RCL", [("Valero Energy Corporation", None)],
         ["0001035002"], []),
        ("Which RF components does Disney buy?", [("The Walt Disney Company", "DIS")],
         ["0001744489"], []),
        ("What FICO score cutoffs do banks disclose?", [], [], []),
        ("Qualcomm revenue from radio frequency (RF) front-end", [], [], []),
        ("What did NVDA say about RF chips?", [("NVIDIA Corporation", "NVDA")],
         ["0001045810"], []),
        ("Which of RF, TYL and VLO mention the FDIC?", [],
         ["0001281761", "0000860731", "0001035002"], []),
        # A company the question names is never redirected to a look-alike.
        ("Compare Pfizer and PFG dividends", [("Pfizer Inc.", "PFE")],
         ["0000078003"], []),
        ("Exelon vs EXE capex", [("Exelon Corporation", "EXC")],
         ["0001109357"], []),
        ("Capex of Valero (VLO) vs RCL", [], ["0001035002", "0000884887"], []),
        # No written ticker missed: the model's reading stands.
        ("Compare Exelon with Samsung", [("Exelon Corporation", "EXC"),
         ("Samsung Electronics", None)], ["0001109357"], ["Samsung Electronics"]),
    ],
)  # fmt: skip
def test_written_tickers_overrule_the_model(
    question: str,
    mentions: list[tuple[str, str | None]],
    ciks: list[str],
    missing: list[str],
) -> None:
    found, unresolved = TICKERS.resolve_in_question(question, mentions)
    assert ([c.cik for c in found], unresolved) == (ciks, missing)


def test_topic_descriptions_cover_every_section_topic() -> None:
    assert set(SEC_FILINGS_PROFILE.topics) == set(get_args(SectionTopic))


def _analysis(**overrides: Any) -> QueryAnalysis:
    values: dict[str, Any] = {
        "language": "vi",
        "corpora": ["sec_filings"],
        "searches": {"en": "What was Apple's Q2 2024 revenue?"},
        "companies": [{"name": "Apple Inc.", "ticker": "aapl"}],
        "fiscal_years": [2024],
        "fiscal_periods": ["q2"],
        "forms": ["10 Q"],
        "topics": ["financial_statements"],
        "wants_latest": False,
        "period_basis": "fiscal",
    }
    return analysis_model(("en",)).model_validate(values | overrides)


def test_checked_plan_keeps_known_values_only() -> None:
    plan = checked_plan(
        "What was Apple's revenue in its Q2 2024 10-Q?",
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
    empty = QueryAnalysis.model_validate(
        _analysis(corpora=[]).model_dump() | {"searches": {}}
    )
    no_corpus = checked_plan("q", empty, CORPORA)
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


def test_schema_requires_a_search_text_per_corpus_language() -> None:
    schema = strict_json_schema(analysis_model(("en", "vi")))
    texts = schema["$defs"]["SearchTexts"]
    assert texts["required"] == ["en", "vi"] and "searches" in schema["required"]
    assert "Vietnamese" in texts["properties"]["vi"]["description"]


def test_untranslated_search_text_is_translated_separately() -> None:
    question = "Doanh thu quý 2/2024 của Apple là bao nhiêu?"
    english = "What were Apple Inc.'s net sales in Q2 2024?"
    # The model copies the Vietnamese question into the English field.
    chat = FakeChat(
        [reply(_analysis(searches={"en": question}).model_dump_json()), reply(english)]
    )
    plan = analyze_query(question, CORPORA, chat, TODAY)
    assert plan.source == "llm" and plan.text_for("en") == english
    assert any("not in their language dropped: ['en']" in n for n in plan.notes)
    assert any("translated separately" in n for n in plan.notes)
    messages, options = chat.calls[1]
    assert messages[0] == Message.user(question)
    assert options is not None and options.output_schema is None
    assert options.system is not None and "in English" in options.system
    # A translation still in Vietnamese is not used: the question is searched.
    failed = analyze_query(
        question,
        CORPORA,
        FakeChat(
            [reply(_analysis(searches={"en": ""}).model_dump_json()), reply(question)]
        ),
        TODAY,
    )
    assert failed.text_for("en") == question and not failed.texts
    assert any("no en search text" in n for n in failed.notes)


def test_english_question_keeps_its_own_text_without_a_second_call() -> None:
    question = "What was Apple's Q2 2024 revenue?"
    chat = FakeChat([reply(_analysis(language="en").model_dump_json())])
    plan = analyze_query(question, CORPORA, chat, TODAY)
    assert plan.texts == {"en": question} and len(chat.calls) == 1


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
        ("Doanh thu quý thứ hai năm 2024 của Apple?", "vi", (2024,), ("Q2",), False),
        ("Microsoft second-quarter 2024 revenue", "en", (2024,), ("Q2",), False),
        ("Lợi nhuận Apple tài khóa 2023", "vi", (2023,), ("FY",), False),
        ("Votes at Apple's 2025 annual meeting", "en", (2025,), (), False),
        ("Apple annual revenue 2024", "en", (2024,), ("FY",), False),
        ("Doanh thu hàng năm 2024 của Apple", "vi", (2024,), ("FY",), False),
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


GOOGL_H1_EN = "What diluted EPS did GOOGL post for the first half of 2025?"
GOOGL_H1_VI = "EPS pha loãng của GOOGL trong 6 tháng đầu năm 2025 là bao nhiêu?"
SBC_EN = (
    "How much stock-based compensation did NVIDIA record in the first half of "
    "fiscal 2027 (six months to July 26, 2026), and Apple in the first nine months "
    "of fiscal 2026 (to June 27, 2026)?"
)
SBC_VI = (
    "NVIDIA ghi nhận bao nhiêu chi phí thù lao bằng cổ phiếu trong nửa đầu năm tài "
    "chính 2027 (sáu tháng đến 26/7/2026), và Apple trong chín tháng đầu năm tài "
    "chính 2026 (đến 27/6/2026)?"
)
GOODWILL_EN = (
    "What goodwill did GOOGL carry on its balance sheet at September 30, 2022, "
    "compared with September 30, 2021?"
)
GOODWILL_VI = (
    "Lợi thế thương mại (goodwill) của GOOGL tại ngày 30/9/2022 là bao nhiêu, so "
    "với ngày 30/9/2021?"
)
LETTER_EN = (
    "In the 2021 shareholder letter filed with an April 2022 8-K, how did Amazon's "
    "workplace injury rates compare with warehousing and delivery peers?"
)
LETTER_VI = (
    "Trong thư gửi cổ đông năm 2021 (nộp kèm 8-K tháng 4/2022), tỷ lệ tai nạn lao "
    "động của Amazon so với các công ty cùng ngành kho bãi và giao hàng ra sao?"
)

NINE_MONTHS_COMPARED = (
    "Apple revenue for the nine months ended September 30, 2024 compared to 2023"
)
ABOUT_EN = (
    "How do Amazon and Microsoft describe themselves in the 'About' section of "
    "their July 2026 earnings press releases?"
)
ABOUT_VI = (
    "Amazon và Microsoft tự mô tả thế nào trong phần “About” của thông cáo báo chí "
    "công bố kết quả kinh doanh tháng 7/2026?"
)
FULFILLMENT_EN = (
    "How much did Amazon spend on fulfillment in the September 2022 quarter versus "
    "the September 2020 quarter?"
)
FULFILLMENT_VI = (
    "Amazon chi bao nhiêu cho fulfillment trong quý kết thúc tháng 9/2022 so với "
    "quý kết thúc tháng 9/2020?"
)


@pytest.mark.parametrize(
    ("question", "years", "periods", "calendar", "filed"),
    [
        # Partial years: the year-to-date figures sit in that quarter's 10-Q.
        (GOOGL_H1_EN, (2025,), ("Q2",), True, ()),
        (GOOGL_H1_VI, (2025,), ("Q2",), True, ()),
        ("Apple H1 2024 revenue", (2024,), ("Q2",), True, ()),
        ("Doanh thu nửa đầu năm 2024 của Apple", (2024,), ("Q2",), True, ()),
        ("Apple revenue for the nine months of 2024", (2024,), ("Q3",), True, ()),
        ("Doanh thu 9 tháng đầu năm 2024 của Apple", (2024,), ("Q3",), True, ()),
        (SBC_EN, (2026, 2027), ("Q2",), True, ()),
        (SBC_VI, (2026, 2027), ("Q2",), True, ()),
        # A partial year ended on a date: that date's nearest calendar quarter.
        ("Apple revenue for the six months ended March 30, 2024", (2024,), ("Q1",),
         True, ()),
        ("Doanh thu của Apple trong sáu tháng kết thúc ngày 30/3/2024", (2024,),
         ("Q1",), True, ()),
        ("Microsoft revenue for the nine months ended March 31, 2025", (2025,),
         ("Q1",), True, ()),
        ("NVIDIA revenue for the six months ended July 27, 2025", (2025,), ("Q2",),
         True, ()),
        # Bare comparative years: the same partial year, no FY.
        ("Apple revenue for the six months ended June 30, 2024 and 2023",
         (2023, 2024), ("Q2",), True, ()),
        ("Doanh thu 6 tháng kết thúc ngày 30/6/2024 và 2023", (2023, 2024), ("Q2",),
         True, ()),
        (NINE_MONTHS_COMPARED, (2023, 2024), ("Q3",), True, ()),
        ("Apple 2Q24 revenue", (2024,), ("Q2",), False, ()),
        # Dates: the calendar quarter holding them.
        (GOODWILL_EN, (2021, 2022), ("Q3",), True, ()),
        (GOODWILL_VI, (2021, 2022), ("Q3",), True, ()),
        ("Apple cash as of 2023-12-31", (2023,), ("Q4",), True, ()),
        ("Apple cash at 3/31/2024", (2024,), ("Q1",), True, ()),
        ("Tiền mặt của Apple ngày 31 tháng 3 năm 2024", (2024,), ("Q1",), True, ()),
        # A date's quarter is the calendar one whose end is nearest, even when
        # "fiscal" is said; a date without a year is no period.
        ("Apple cash at 31 March 2024, fiscal basis", (2024,), ("Q1",), True, ()),
        ("Apple cash at April 27, 2024", (2024,), ("Q1",), True, ()),
        ("Tiền mặt của Apple tại ngày 27/4/2024", (2024,), ("Q1",), True, ()),
        ("NVIDIA inventory as of January 26, 2025", (2024,), ("Q4",), True, ()),
        ("Hàng tồn kho của NVIDIA tại ngày 26/1/2025", (2024,), ("Q4",), True, ()),
        ("Apple cash as of December 31, 2024", (2024,), ("Q4",), True, ()),
        ("Microsoft cash at October 31, 2024", (2024,), ("Q3",), True, ()),
        # A year-end date: the whole (fiscal) year.
        ("Apple net sales for the fiscal year ended September 28, 2024", (2024,),
         ("FY",), False, ()),
        ("Doanh thu năm tài chính kết thúc ngày 28/9/2024 của Apple", (2024,),
         ("FY",), False, ()),
        ("Biên lợi nhuận năm tài chính 2021, kết thúc ngày 30/11/2021", (2021,),
         ("FY",), False, ()),
        ("Lợi nhuận gộp sáu tháng đầu năm tài chính 2023, kết thúc ngày 31/7/2022",
         (2022, 2023), ("Q2",), True, ()),
        ("NVIDIA results for the fiscal year ended January 26, 2025", (2025,),
         ("FY",), False, ()),
        ("Apple 10-K for the fiscal year ended September 30, 2023", (2023,),
         ("FY",), False, ()),
        # A full-year month range: the calendar year.
        ("Apple revenue January-December 2024", (2024,), ("FY",), True, ()),
        ("Doanh thu Apple tháng 1 - tháng 12/2024", (2024,), ("FY",), True, ()),
        # A partial year beside another year asked keeps that year.
        ("Alphabet revenue H1 FY2025 vs FY2024", (2024, 2025), ("Q2", "FY"), False,
         ()),
        ("Doanh thu 6 tháng đầu năm 2024 so với năm 2023 của Apple", (2023, 2024),
         ("Q2", "FY"), True, ()),
        ("Apple revenue Q1 2025 and first half 2025", (2025,), ("Q1", "Q2"), False,
         ()),
        ("Doanh thu quý 1/2025 và 6 tháng đầu năm 2025 của Apple", (2025,),
         ("Q1", "Q2"), False, ()),
        ("Apple 9M 2025 revenue", (2025,), ("Q3",), True, ()),
        ("Apple revenue for the quarter ended March 2023", (2023,), ("Q1",), True, ()),
        # A month range is a calendar period, not a filing month.
        ("Alpha net sales October-December 2024", (2024,), ("Q4",), True, ()),
        ("Doanh thu tháng 7 - tháng 9/2025 của Apple", (2025,), ("Q3",), True, ()),
        # A month next to "quarter"/"quý" ends that quarter.
        (FULFILLMENT_EN, (2020, 2022), ("Q3",), True, ()),
        (FULFILLMENT_VI, (2020, 2022), ("Q3",), True, ()),
        ("Apple revenue in the December 2023 quarter", (2023,), ("Q4",), True, ()),
        ("Doanh thu của Apple trong quý tháng 12/2023", (2023,), ("Q4",), True, ()),
        ("Apple revenue in the June quarter of 2024", (2024,), ("Q2",), True, ()),
        ("Apple revenue for the quarter ending September 2022", (2022,), ("Q3",),
         True, ()),
        # A bare month outside a filing question: the calendar quarter
        # holding it (t0067), unless it opens a span.
        ("Apple revenue in March 2023", (2023,), ("Q1",), True, ()),
        ("Doanh thu của Apple tháng 3/2023", (2023,), ("Q1",), True, ()),
        ("AT&T buybacks in February 2022", (2022,), ("Q1",), True, ()),
        ("AT&T mua lại cổ phiếu trong tháng 2/2022", (2022,), ("Q1",), True, ()),
        ("Apple revenue in January 2025", (2025,), ("Q1",), True, ()),
        ("Apple revenue since March 2020", (2020,), (), True, ()),
        ("Doanh thu Apple kể từ tháng 3/2020", (2020,), (), True, ()),
        ("Apple notes due March 2030", (2030,), (), True, ()),
        ("Apple notes maturing in March 2030", (2030,), (), True, ()),
        ("Apple revenue in March and June 2023", (2023,), ("Q2",), True, ()),
        # Beside another year the month is context: only its year counts.
        ("The plant opened in March 2022, what was revenue in 2023?",
         (2022, 2023), (), True, ()),
        ("Apple repurchases in February 2022 and in fiscal 2021", (2021, 2022),
         ("FY",), False, ()),
        # Filing months: an asked 8-K is pinned by its month alone.
        (LETTER_EN, (), (), False, ("2022-04",)),
        (LETTER_VI, (), (), False, ("2022-04",)),
        ("What did Apple's March 2023 bylaw amendment change?", (), (), False,
         ("2023-03",)),
        ("Điều lệ sửa đổi tháng 3 năm 2023 của Apple thay đổi gì?", (), (), False,
         ("2023-03",)),
        ("Báo cáo 8-K nộp tháng 1/2025 của Apple nói gì?", (), (), False,
         ("2025-01",)),
        ("Microsoft 8-K filed on April 25, 2023", (), (), False, ("2023-04",)),
        ("Microsoft's September 2024 8-K", (), (), False, ("2024-09",)),
        ("Microsoft 8-Ks from May 2024", (), (), False, ("2024-05",)),
        # A deal's month dates its 8-K (t0078); a buyback's is a period.
        ("What stake did MKC hold after its January 2026 deal?", (), (), False,
         ("2026-01",)),
        ("MKC nắm bao nhiêu phần sau thương vụ hồi tháng 1 năm 2026?", (), (),
         False, ("2026-01",)),
        ("Microsoft 8-K Item 5.02 May 2024", (), (), False, ("2024-05",)),
        # A dated announcement is a filing month.
        ("Apple announced a buyback on May 2, 2024", (), (), False, ("2024-05",)),
        ("Apple công bố mua lại cổ phiếu ngày 2/5/2024", (), (), False,
         ("2024-05",)),
        ("Báo cáo 8-K tháng 9/2024 của Microsoft", (), (), False, ("2024-09",)),
        # A year asked beside the filing month stays.
        ("Apple 2024 revenue in the 10-K filed in November 2024", (2024,), (), True,
         ("2024-11",)),
        # A month qualifying a release dates it (t0051).
        (ABOUT_EN, (), (), False, ("2026-07",)),
        (ABOUT_VI, (), (), False, ("2026-07",)),
        ("Apple's press release of July 2026", (), (), False, ("2026-07",)),
        ("Thông cáo báo chí tháng 7/2026 của Apple", (), (), False, ("2026-07",)),
        ("Apple's July 2026 earnings release", (), (), False, ("2026-07",)),
        # A release reports periods: its bare month is no filing month.
        ("What did Apple's press release say about revenue in September 2024?",
         (2024,), ("Q3",), True, ()),
        ("Thông cáo báo chí của Apple nói gì về doanh thu tháng 9/2024?", (2024,),
         ("Q3",), True, ()),
        # Not a date: a quarter written with a slash, a day-month of a date.
        ("Doanh thu quý 2/2024 của Apple?", (2024,), ("Q2",), False, ()),
        ("Lợi nhuận ngày 6 tháng 9 năm 2022", (2022,), ("Q3",), True, ()),
    ],
)  # fmt: skip
def test_rule_plan_partial_years_dates_and_filing_months(
    question: str,
    years: tuple[int, ...],
    periods: tuple[str, ...],
    calendar: bool,
    filed: tuple[str, ...],
) -> None:
    plan = rule_plan(question, CORPORA)
    assert (plan.fiscal_years, plan.fiscal_periods, plan.calendar) == (
        years,
        periods,
        calendar,
    )
    assert plan.filed_months == filed
    assert not plan.wants_latest and plan.has_period


@pytest.mark.parametrize(
    "question",
    [
        "Apple revenue over the last six months",
        "Apple plans for the next 6 months",
        "Doanh thu Apple 6 tháng qua",
        "Kế hoạch của Apple trong 6 tháng tới",
        "Apple results in the past nine months",
        "Apple spent $9M and $6M on it",
        "Apple announce on September 6",
        "Apple công bố ngày 6 tháng 9",
        "Apple cash at March 31",
    ],
)
def test_rule_plan_reads_no_period_from_relative_or_yearless_words(
    question: str,
) -> None:
    plan = rule_plan(question, CORPORA)
    assert (plan.fiscal_years, plan.fiscal_periods, plan.filed_months) == ((), (), ())


def test_rule_plan_reads_a_dated_vietnamese_day_without_ngay() -> None:
    plan = rule_plan("Sự kiện 6 tháng 9 năm 2024 của Apple", CORPORA)
    # A date, not "six months".
    assert (plan.fiscal_years, plan.fiscal_periods) == ((2024,), ("Q3",))


@pytest.mark.parametrize(
    ("given", "periods", "years"),
    [
        (["Q2 2024", "Q2 2025"], ("Q2",), (2024, 2025)),
        (["FY2026"], ("FY",), (2026,)),
        (["FY 2025"], ("FY",), (2025,)),
        (["Q3 FY24"], ("Q3",), (2024,)),
        (["2024Q1", "q4-2023"], ("Q1", "Q4"), (2023, 2024)),
        (["H1 2025"], ("Q2",), (2025,)),
        (["9M 2025"], ("Q3",), (2025,)),
        (["2Q24"], ("Q2",), (2024,)),
        (["second quarter 2024"], ("Q2",), (2024,)),
        (["Quarter 2"], ("Q2",), ()),
        (["FQ3 2025"], ("Q3",), (2025,)),
    ],
)
def test_checked_plan_parses_periods_with_years(
    given: list[str], periods: tuple[str, ...], years: tuple[int, ...]
) -> None:
    plan = checked_plan("q", _analysis(fiscal_years=[], fiscal_periods=given), CORPORA)
    assert (plan.fiscal_periods, plan.fiscal_years) == (periods, years)
    assert not any("periods dropped" in note for note in plan.notes)


def test_checked_plan_drops_unreadable_periods_and_months() -> None:
    plan = checked_plan(
        "q",
        _analysis(fiscal_periods=["Q2", "sometime"], filed_months=["2023-13", "x"]),
        CORPORA,
    )
    assert plan.fiscal_periods == ("Q2",) and plan.filed_months == ()
    assert any("unknown periods dropped: ['sometime']" in n for n in plan.notes)
    assert any("unknown filed months dropped" in n for n in plan.notes)


@pytest.mark.parametrize(
    ("question", "given", "basis", "expected"),
    [
        # No period or the spanned quarters -> the quarter ending the span.
        (GOOGL_H1_EN, [], "calendar", (("Q2",), (2025,), True)),
        (GOOGL_H1_VI, ["Q1", "Q2"], "calendar", (("Q2",), (2025,), True)),
        # Ended on stated dates: their calendar quarter (NVIDIA July 26, 2026
        # and Apple June 27, 2026 are both nearest June 30).
        (SBC_EN, ["FY2027", "FY2026"], "fiscal", (("Q2",), (2026, 2027), True)),
        (SBC_VI, ["FY2027", "FY2026"], "fiscal", (("Q2",), (2026, 2027), True)),
        ("Apple revenue for the six months ended March 30, 2024", ["Q2"], "fiscal",
         (("Q1",), (2024,), True)),
        ("Doanh thu của Apple trong sáu tháng kết thúc ngày 30/3/2024", [],
         "fiscal", (("Q1",), (2024,), True)),
        ("Microsoft revenue for the nine months ended March 31, 2025", ["Q3"],
         "calendar", (("Q1",), (2025,), True)),
        ("Doanh thu của Microsoft trong chín tháng kết thúc ngày 31/3/2025", ["Q3"],
         "fiscal", (("Q1",), (2025,), True)),
        # A whole year asked beside the half stays.
        ("Alphabet revenue in the first half of 2025 vs full year 2024",
         ["Q2", "FY"], "calendar", (("Q2", "FY"), (2024, 2025), True)),
        # Other asked periods are merged with the partial year, not replaced.
        ("Alphabet revenue H1 FY2025 vs FY2024", ["FY"], "fiscal", (("FY", "Q2"), (2024, 2025), False)),
        ("Doanh thu nửa đầu năm tài chính 2025 so với năm tài chính 2024", ["FY"],
         "fiscal", (("FY", "Q2"), (2024, 2025), False)),
        ("Apple revenue for the nine months ended March 31, 2025 vs fiscal 2024",
         ["FY"], "fiscal", (("FY", "Q1"), (2024, 2025), True)),
        ("Doanh thu 6 tháng đầu năm 2024 so với năm 2023", ["FY"], "calendar",
         (("FY", "Q2"), (2023, 2024), True)),
        ("Apple revenue Q1 2025 and first half 2025", ["Q1", "Q2"], "fiscal",
         (("Q1", "Q2"), (2025,), False)),
        ("Doanh thu quý 1/2025 và 6 tháng đầu năm 2025", ["Q1"], "fiscal",
         (("Q1", "Q2"), (2025,), False)),
        # Bare comparative years repeat the partial year: no FY.
        ("Apple revenue for the six months ended June 30, 2024 and 2023",
         ["Q2", "FY"], "calendar", (("Q2",), (2023, 2024), True)),
        ("Doanh thu 6 tháng kết thúc ngày 30/6/2024 và 2023", ["Q2", "FY"],
         "calendar", (("Q2",), (2023, 2024), True)),
    ],
)  # fmt: skip
def test_checked_plan_reads_partial_years(
    question: str,
    given: list[str],
    basis: str,
    expected: tuple[tuple[str, ...], tuple[int, ...], bool],
) -> None:
    years = list(expected[1])
    analysis = _analysis(fiscal_years=years, fiscal_periods=given, period_basis=basis)
    plan = checked_plan(question, analysis, CORPORA)
    assert (plan.fiscal_periods, plan.fiscal_years, plan.calendar) == expected


def test_checked_plan_reads_dates() -> None:
    for question in (GOODWILL_EN, GOODWILL_VI):
        analysis = _analysis(
            fiscal_years=[2021, 2022], fiscal_periods=[], period_basis="fiscal"
        )
        plan = checked_plan(question, analysis, CORPORA)
        assert (plan.fiscal_years, plan.fiscal_periods) == ((2021, 2022), ("Q3",))
        assert plan.calendar, question  # a date is a calendar one
        assert any("dates" in note for note in plan.notes)
    # Dates overrule the model's reading: a June filer's quarter ended
    # September 30 is its fiscal Q1, found by period end.
    analysis = _analysis(fiscal_years=[2021, 2022], fiscal_periods=["Q1"])
    plan = checked_plan(GOODWILL_EN, analysis, CORPORA)
    assert plan.fiscal_periods == ("Q3",) and plan.calendar
    # A quarter the question names keeps the model's reading, with the
    # fiscal years the question writes (x0096: the model shifted them).
    plan = checked_plan(
        "Gen Digital's Q1 fiscal 2026 10-Q (ended July 4, 2025) vs Q1 fiscal 2027",
        _analysis(fiscal_years=[2025, 2026], fiscal_periods=["Q1"]),
        CORPORA,
    )
    assert plan.fiscal_periods == ("Q1",) and not plan.calendar
    assert {2026, 2027} <= set(plan.fiscal_years)
    plan = checked_plan(
        "Microsoft revenue in Q4 fiscal 2025, the quarter ended June 30, 2025",
        _analysis(fiscal_years=[2025], fiscal_periods=["Q4"]),
        CORPORA,
    )
    assert (plan.fiscal_years, plan.fiscal_periods, plan.calendar) == (
        (2025,),
        ("Q4",),
        False,
    )
    # A year the model missed comes from the dates.
    plan = checked_plan(
        "Apple cash at 3/31/2024?", _analysis(fiscal_years=[], fiscal_periods=[]),
        CORPORA,
    )  # fmt: skip
    assert (plan.fiscal_years, plan.fiscal_periods) == ((2024,), ("Q1",))


def test_checked_plan_reads_filing_months() -> None:
    for question in (LETTER_EN, LETTER_VI):
        analysis = _analysis(
            fiscal_years=[2021], fiscal_periods=[], forms=["8-K"], filed_months=[]
        )
        plan = checked_plan(question, analysis, CORPORA)
        assert plan.filed_months == ("2022-04",), question
        assert (plan.fiscal_years, plan.fiscal_periods) == ((), ())
        assert plan.forms == ("8-K",) and plan.has_period
    bylaws = "What did Apple's March 2023 bylaw amendment change?"
    analysis = _analysis(
        fiscal_years=[2023], fiscal_periods=[], forms=[], filed_months=["2023-3"],
        wants_latest=True,
    )  # fmt: skip
    plan = checked_plan(bylaws, analysis, CORPORA)
    # 2023 only dates the filing.
    assert (plan.filed_months, plan.fiscal_years) == (("2023-03",), ())
    assert not plan.wants_latest
    for given in (["04/2022"], ["April 2022"], ["2022/4"], ["2022-04-14"],
                  ["Apr-2022"], ["tháng 4/2022"]):  # fmt: skip
        analysis = _analysis(fiscal_years=[], fiscal_periods=[], filed_months=given)
        plan = checked_plan("Apple 8-K", analysis, CORPORA)
        assert plan.filed_months == ("2022-04",)
    # Outside a filing question the model's month and years are no filing.
    for question in (
        "Apple revenue in the September 2022 quarter",
        "Doanh thu của Apple trong quý tháng 9/2022",
    ):
        analysis = _analysis(
            fiscal_years=[2022], fiscal_periods=["Q3"], period_basis="calendar",
            forms=[], filed_months=["2022-09"],
        )  # fmt: skip
        plan = checked_plan(question, analysis, CORPORA)
        assert plan.filed_months == () and plan.fiscal_years == (2022,), question
        assert plan.fiscal_periods == ("Q3",)


def test_checked_plan_reads_year_ends_and_keeps_relative_recency() -> None:
    for question in (
        "Apple net sales for the fiscal year ended September 28, 2024",
        "Doanh thu năm tài chính kết thúc ngày 28/9/2024 của Apple",
    ):
        analysis = _analysis(fiscal_years=[2024], fiscal_periods=["Q3"])
        plan = checked_plan(question, analysis, CORPORA)
        assert (plan.fiscal_years, plan.fiscal_periods) == ((2024,), ("FY",))
        assert not plan.calendar
    for question in (
        "Apple revenue over the last six months",
        "Doanh thu Apple 6 tháng qua",
    ):
        analysis = _analysis(fiscal_years=[], fiscal_periods=[], wants_latest=True)
        plan = checked_plan(question, analysis, CORPORA)
        assert plan.fiscal_periods == () and plan.wants_latest
    # A partial year the model asked with recency still reads its quarter
    # only when anchored by a year; otherwise recency stays.
    analysis = _analysis(fiscal_years=[], fiscal_periods=[], wants_latest=True)
    plan = checked_plan("Apple first half results, latest", analysis, CORPORA)
    assert plan.fiscal_periods == () and plan.wants_latest


def test_checked_plan_keeps_a_release_month_out_of_filings() -> None:
    for question in (
        "What did Apple's press release say about revenue in September 2024?",
        "Thông cáo báo chí của Apple nói gì về doanh thu tháng 9/2024?",
    ):
        analysis = _analysis(
            fiscal_years=[2024], fiscal_periods=[], forms=[], filed_months=["2024-09"]
        )
        plan = checked_plan(question, analysis, CORPORA)
        assert plan.filed_months == () and plan.fiscal_years == (2024,), question
    for question in (
        "Apple announced a buyback on May 2, 2024",
        "Apple công bố mua lại cổ phiếu ngày 2/5/2024",
    ):
        analysis = _analysis(fiscal_years=[2024], fiscal_periods=["Q2"], forms=[])
        plan = checked_plan(question, analysis, CORPORA)
        # The model's own period stays; the filing month joins it.
        assert plan.filed_months == ("2024-05",), question
        assert plan.fiscal_periods == ("Q2",)


def test_system_prompt_explains_periods() -> None:
    chat = FakeChat([reply(_analysis().model_dump_json())])
    analyze_query("q", CORPORA, chat, TODAY)
    _, options = chat.calls[0]
    assert options is not None and options.system is not None
    for phrase in ("6 tháng đầu năm", "nine months", "tại ngày 30/9/2022",
                   "never with a year", "filed_months", "2022-04"):  # fmt: skip
        assert phrase in options.system, phrase
    assert options.output_schema is not None
    assert "filed_months" in options.output_schema.json_schema["required"]


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


def test_ranking_marks_pre_rank_and_selected() -> None:
    from src.rag_engine.retrieval.retriever import Candidate, _ranking

    def cand(point: str, fused: float, final: float) -> Candidate:
        hit = Hit(point, fused, {"chunk_id": f"c-{point}"}, duplicates=("d",))
        return Candidate("sec_filings", hit, rerank_score=final, final_score=final)

    fused = [cand("a", 0.9, 0.1), cand("b", 0.5, 0.8), cand("c", 0.2, 0.5)]
    pre_rank = {c.hit.point_id: i for i, c in enumerate(fused)}
    final = sorted(fused, key=lambda c: -c.final_score)
    rows = _ranking(final, pre_rank, {"b"})
    assert [(r["chunk_id"], r["pre_rank"], r["selected"]) for r in rows] == [
        ("c-b", 1, True),
        ("c-c", 2, False),
        ("c-a", 0, False),
    ]
    assert rows[0]["duplicate_chunk_ids"] == ["d"]


def test_rule_plan_reads_plural_forms() -> None:
    plan = rule_plan("What did Apple's 8-Ks and 10-Qs in 2023 say?", CORPORA)
    assert plan.forms == ("8-K", "10-Q")


TODAY_REL = date(2026, 11, 24)


@pytest.mark.parametrize(
    ("question", "years", "periods", "calendar", "filed", "offsets"),
    [
        # Calendar years written into the question, then read as usual.
        ("Steel Dynamics operating income in this year's second quarter",
         (2026,), ("Q2",), True, (), ()),
        ("Discovery goodwill at the end of June this year", (2026,), ("Q2",), True,
         (), ()),
        ("Disney segment income for the nine months to the end of June this year",
         (2026,), ("Q2",), True, (), ()),
        (("Corning cost of sales in this year's second quarter vs the same "
          "quarter last year"), (2025, 2026), ("Q2",), True, (), ()),
        ("Tổng nợ dài hạn của KMI vào cuối tháng 6 năm nay", (2026,), ("Q2",), True,
         (), ()),
        (("Doanh thu của Seagate trong quý 3 năm tài chính năm nay so với "
          "cùng quý năm ngoái"), (2025, 2026), ("Q3",), False, (), ()),
        # A yearless month takes the one year written.
        ("Notes UHS issued in its August 8-K this year", (), (), False,
         ("2026-08",), ()),
        ("Trái phiếu UHS phát hành theo 8-K tháng 8 năm nay", (), (), False,
         ("2026-08",), ()),
        # Months counted back, with a month on each side.
        ("Which bonds did AIG sell about two months ago?", (), (), False,
         ("2026-08", "2026-09", "2026-10"), ()),
        ("AIG bán trái phiếu nào cách đây hai tháng?", (), (), False,
         ("2026-08", "2026-09", "2026-10"), ()),
        # Fiscal years: offsets for the retriever.
        ("STZ stock comp in the first quarter of this fiscal year vs a year earlier",
         (), ("Q1",), False, (), (-1, 0)),
        ("SNDK opex in the fiscal year that just ended vs the year before", (), (),
         False, (), (-2, -1)),
        ("Lợi nhuận ròng của DHI trong năm tài chính vừa rồi", (), ("FY",), False,
         (), (-1,)),
        # A question naming its own year: relative words are left to it.
        ("Revenue in 2023 compared with the prior year", (2023,), (), True, (), ()),
        # Capitalized months only: "may" is a verb.
        ("What may affect revenue this year?", (2026,), (), True, (), ()),
        ("May Apple terminate its credit facility this year?", (2026,), (), True,
         (), ()),
        # Comparison bases are no asked year.
        ("How did Apple's revenue compare with the prior year in its latest 10-K?",
         (), (), False, (), ()),
        ("Doanh thu quý gần nhất của Apple tăng bao nhiêu so với năm trước?", (),
         (), False, (), ()),
        ("Apple revenue in the quarter ended in September compared with last year",
         (), (), False, (), ()),
        ("Public Storage profit last year", (2025,), (), True, (), ()),
        ("Năm trước khi sáp nhập, công ty có doanh thu bao nhiêu?", (), (), False,
         (), ()),
        # "The last month of" a period and "6 months before that" count nothing.
        ("Apple repurchases in the last month of Q3 2025", (2025,), ("Q3",), False,
         (), ()),
        ("Apple có bao nhiêu nhân viên 6 tháng trước đó?", (), (), False, (), ()),
        ("Apple đã nộp 8-K nào vào tháng trước?", (), (), False,
         ("2026-09", "2026-10", "2026-11"), ()),
    ],
)  # fmt: skip
def test_rule_plan_reads_relative_time(
    question: str,
    *,
    years: tuple[int, ...],
    periods: tuple[str, ...],
    calendar: bool,
    filed: tuple[str, ...],
    offsets: tuple[int, ...],
) -> None:
    plan = rule_plan(question, CORPORA, today=TODAY_REL)
    assert (
        plan.fiscal_years,
        plan.fiscal_periods,
        plan.calendar,
        plan.filed_months,
        plan.fiscal_offsets,
    ) == (years, periods, calendar, filed, offsets)
    assert rule_plan(question, CORPORA).fiscal_offsets == ()  # no today


def test_checked_plan_overrules_the_model_on_relative_years() -> None:
    plan = checked_plan(
        "How much goodwill did Discovery carry at the end of June this year?",
        _analysis(fiscal_years=[2024], fiscal_periods=["Q2"], forms=[]),
        CORPORA,
        TODAY_REL,
    )
    assert (plan.fiscal_years, plan.fiscal_periods, plan.calendar) == (
        (2026,),
        ("Q2",),
        True,
    )
    plan = checked_plan(
        "Why did STZ's stock comp drop in the first quarter of this fiscal year "
        "versus the first quarter a year earlier?",
        _analysis(fiscal_years=[2025, 2026], fiscal_periods=["Q1"], forms=[]),
        CORPORA,
        TODAY_REL,
    )
    assert (plan.fiscal_years, plan.fiscal_offsets) == ((), (-1, 0))
    assert plan.has_period and not plan.wants_latest


def test_checked_plan_drops_forms_the_question_does_not_name() -> None:
    question = (
        "How did Iron Mountain's IT asset lifecycle wording change between its "
        "Q1 2024 and Q1 2026 supplemental financial information?"
    )
    plan = checked_plan(question, _analysis(forms=["10-Q"]), CORPORA)
    assert plan.forms == ()
    assert any("does not name" in note for note in plan.notes)
    # An event's 8-K stays (t0123: a vote at an annual meeting).
    plan = checked_plan(
        "How did shareholders vote at Gallagher's May 2026 meeting?",
        _analysis(forms=["8-K"], fiscal_years=[2026], fiscal_periods=["Q2"]),
        CORPORA,
    )
    assert plan.forms == ("8-K",) and plan.filed_months == ("2026-05",)
    for named, form in (("annual report", "10-K"), ("báo cáo quý", "10-Q"),
                        ("current report", "8-K")):  # fmt: skip
        plan = checked_plan(f"Apple {named} 2024", _analysis(forms=[form]), CORPORA)
        assert plan.forms == (form,)

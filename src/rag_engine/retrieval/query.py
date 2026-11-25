"""Read a question once before search: language, per-corpus search text,
corpora, companies, periods, forms, topics.

One structured call to the query model (`[chat.query]`) does it all; its
schema has one required search text per corpus language (`analysis_model`),
and a text missing or left in the question's language is translated by a
second short call. When there is no model or it fails, rules give the same
plan without translation (the multilingual embedding still matches; BM25
then only matches names and numbers). The model's values are checked against each corpus's vocabulary,
and an unknown value is dropped rather than failing the plan.
"""

from __future__ import annotations

import calendar as calendar_module
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from functools import cache
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from src.llm import ChatModel, ChatOptions, LLMError, Message, generate_structured
from src.rag_engine.retrieval.corpora import CorpusProfile

PERIODS = ("FY", "Q1", "Q2", "Q3", "Q4")
_MIN_YEAR, _MAX_YEAR = 1990, 2100


class CompanyMention(BaseModel):
    name: str = Field(
        description="Official company name, e.g. 'Alphabet Inc.' when the user "
        "says Google, 'Meta Platforms, Inc.' for Facebook"
    )
    ticker: str | None = Field(description="US stock ticker if known, e.g. GOOGL")


class SearchTexts(BaseModel):
    """The question per corpus language: one field per language, named by
    its ISO 639-1 code (`analysis_model` adds them)."""


_LANGUAGE_NAMES = {"en": "English", "vi": "Vietnamese"}


def _language_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code)


class QueryAnalysis(BaseModel):
    """What the query model returns."""

    language: str = Field(description="ISO 639-1 code of the question, e.g. vi, en")
    corpora: list[str] = Field(description="Collections that can answer it")
    searches: SearchTexts = Field(
        default_factory=SearchTexts,
        description="The question in each collection language",
    )
    companies: list[CompanyMention]
    fiscal_years: list[int]
    fiscal_periods: list[str] = Field(description="FY, Q1, Q2, Q3 or Q4")
    period_basis: Literal["fiscal", "calendar"] = Field(
        description="Whether the years and quarters are the company's fiscal "
        "ones or calendar ones"
    )
    forms: list[str]
    topics: list[str]
    wants_latest: bool
    filed_months: list[str] = Field(
        default_factory=list,
        description="YYYY-MM of each month the user names as when a filing was "
        "filed or an event happened",
    )


@cache
def analysis_model(languages: tuple[str, ...]) -> type[QueryAnalysis]:
    """`QueryAnalysis` whose `searches` requires one text per language, so a
    strict schema cannot leave a corpus without its search text."""

    fields: dict[str, Any] = {
        code: (
            str,
            Field(
                description=f"The question written in {_language_name(code)}, "
                "whatever language the user wrote in"
            ),
        )
        for code in languages
    }
    texts = create_model("SearchTexts", __base__=SearchTexts, **fields)
    model: type[QueryAnalysis] = create_model(
        "QueryAnalysis", __base__=QueryAnalysis, searches=(texts, ...)
    )
    return model


@dataclass(frozen=True)
class Mention:
    name: str
    ticker: str | None = None


@dataclass(frozen=True)
class QueryPlan:
    """A checked analysis: every value is one the corpora know."""

    question: str
    language: str
    corpora: tuple[str, ...]
    # Corpus language -> search text in that language.
    texts: Mapping[str, str]
    companies: tuple[Mention, ...] = ()
    fiscal_years: tuple[int, ...] = ()
    fiscal_periods: tuple[str, ...] = ()
    # Years/quarters are calendar ones (match by period end), not fiscal.
    calendar: bool = False
    forms: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    wants_latest: bool = False
    # "YYYY-MM" months a filing (mostly an 8-K) was filed in, as the user
    # names them ("8-K filed in April 2022"); search turns them into
    # filing-date windows. Years only dating such a filing are not in
    # `fiscal_years`.
    filed_months: tuple[str, ...] = ()
    source: Literal["llm", "rules"] = "rules"
    notes: tuple[str, ...] = field(default=())

    def text_for(self, language: str) -> str:
        return self.texts.get(language) or self.question

    @property
    def has_period(self) -> bool:
        return bool(self.fiscal_years or self.fiscal_periods or self.filed_months)


def _languages(corpora: Mapping[str, CorpusProfile]) -> tuple[str, ...]:
    return tuple(sorted({p.language for p in corpora.values()}))


def _system_prompt(corpora: Mapping[str, CorpusProfile], today: date) -> str:
    languages = _languages(corpora)
    lines = [
        (
            "You prepare a user's question for search over document collections. "
            f"Today is {today.isoformat()}."
        ),
        "",
        "Collections:",
    ]
    for profile in corpora.values():
        lines.append(
            f"- {profile.name} (language {profile.language}): {profile.description}. "
            f"Forms: {', '.join(profile.forms)}."
        )
        lines.append("  Topics (sections):")
        lines += [f"  - {topic}: {what}" for topic, what in profile.topics.items()]
    lines += [
        "",
        "Fill every field:",
        "- language: ISO 639-1 code of the question.",
        "- corpora: names of the collections that can answer it (all when unsure).",
        (
            "- searches: one field per collection language, named by its code "
            f"({', '.join(languages)}). Each holds the question written in THAT "
            "language, whatever language the user wrote in: a Vietnamese question "
            "gets an English text in `en`. One complete, standalone question (a "
            "question, not keywords: the reranker judges passages against it). "
            "Translate faithfully, use the words filings use (doanh thu -> revenue "
            "/ net sales, lợi nhuận -> net income, rủi ro -> risk factors), say what "
            "pronouns refer to, write official company names instead of tickers "
            "(Apple Inc., not AAPL), keep numbers and periods. Never answer the "
            "question."
        ),
        (
            "- companies: every company the question is about, with its official name "
            "and US ticker; empty when none."
        ),
        (
            "- fiscal_years: every year asked about (a range 2021-2024 -> 2021, 2022, "
            "2023, 2024); resolve relative years from today ('last year', 'năm "
            "ngoái'); not a year that only dates a filing (that goes in "
            "filed_months); empty when none."
        ),
        (
            "- fiscal_periods: exactly FY, Q1, Q2, Q3 or Q4, never with a year "
            "('Q2 2024' -> Q2 here, 2024 in fiscal_years): FY for a full year, Q1-Q4 "
            "for a quarter. Year-to-date figures are in a quarter's 10-Q: first half "
            "/ six months / H1 / 6 tháng đầu năm / nửa đầu năm -> Q2 only; nine "
            "months / 9 tháng đầu năm -> Q3 only; 'last six months' is no period. "
            "A partial period ending on a stated date ('six months ended March 30, "
            "2024') -> that date's nearest calendar quarter (Q1 2024), basis "
            "calendar. "
            "A balance at a date with a year ('as of September 30, 2022', 'tại "
            "ngày 30/9/2022') -> the calendar quarter whose end is nearest "
            "(September 30 -> Q3, January 26, 2025 -> Q4 2024), basis calendar; a "
            "year-end date ('fiscal year ended September 28, 2024') -> FY of that "
            "year. Empty when the question names no period."
        ),
        (
            "- period_basis: fiscal when the user says fiscal / FY / năm tài chính, or "
            "names a quarter the way companies report it (Q2 2024, Q2 FY24, second "
            "quarter of fiscal 2024); calendar for a plain year ('revenue in 2024', "
            "'năm 2024', 'last year'), explicit months (July-September 2025), a date "
            "('as of September 30, 2022') or part of a plain year ('first half of "
            "2025'). Part of a fiscal year stays fiscal ('first half of fiscal 2027 "
            "(six months to July 26, 2026)' -> Q2, 2027, fiscal; 'first nine months "
            "of fiscal 2026' -> Q3, 2026, fiscal). NVIDIA's fiscal 2025 ended in "
            "January 2025, so 'NVIDIA in 2024' is calendar."
        ),
        (
            "- filed_months: YYYY-MM of each month the user names as when a filing "
            "was filed or an event happened ('8-K filed in April 2022' -> 2022-04, "
            "'March 2023 bylaw amendment' -> 2023-03, '8-K nộp tháng 1/2025' -> "
            "2025-01); not a period end ('quarter ended March 2023' is Q1 2023). When "
            "it identifies an 8-K, leave fiscal_years and fiscal_periods empty ('the "
            "2021 shareholder letter filed with an April 2022 8-K' -> filed_months "
            "2022-04, no years). Empty when none."
        ),
        (
            "- forms: only forms the user explicitly asks for (annual report -> 10-K, "
            "quarterly report -> 10-Q, current report -> 8-K); otherwise empty."
        ),
        (
            "- topics: the topics most likely to hold the answer (at most 3); empty "
            "when unsure."
        ),
        (
            "- wants_latest: true when the user asks for the most recent information "
            "('latest', 'gần nhất', 'hiện nay') without naming a period."
        ),
    ]
    return "\n".join(lines)


def analyze_query(
    question: str,
    corpora: Mapping[str, CorpusProfile],
    chat: ChatModel | None,
    today: date,
) -> QueryPlan:
    """The plan from the query model, or from rules when there is no model
    or its reply fails (the reason is kept in `notes`)."""

    if chat is None:
        return rule_plan(question, corpora, notes=("no query model: rules",))
    try:
        analysis, _ = generate_structured(
            chat,
            [Message.user(question)],
            analysis_model(_languages(corpora)),
            ChatOptions(system=_system_prompt(corpora, today)),
        )
    except LLMError as error:
        reason = f"query model failed ({type(error).__name__}: {error}): rules"
        return rule_plan(question, corpora, notes=(reason[:300],))
    return with_search_texts(checked_plan(question, analysis, corpora), corpora, chat)


def _translation_prompt(language: str) -> str:
    return (
        f"Rewrite the user's question in {_language_name(language)} for searching "
        "company filings: one complete, standalone question. Translate faithfully, "
        "use the words filings use (doanh thu -> revenue / net sales, lợi nhuận -> "
        "net income, vốn chủ sở hữu -> stockholders' equity), write official "
        "company names instead of tickers, keep numbers and periods. Reply with "
        "the question only; never answer it."
    )


def with_search_texts(
    plan: QueryPlan, corpora: Mapping[str, CorpusProfile], chat: ChatModel
) -> QueryPlan:
    """`plan` with a search text for each language of its corpora: a missing
    one is translated by a short call (none when the question is already in
    that language); when that fails too, search uses the question itself
    (noted)."""

    texts = dict(plan.texts)
    notes = list(plan.notes)
    asked_in = _question_language(plan.question)
    for language in _languages({c: corpora[c] for c in plan.corpora}):
        if texts.get(language) or language == asked_in:
            continue
        text, reason = _translate(chat, plan.question, language)
        if text and _in_language(text, language, plan.question):
            texts[language] = text
            notes.append(f"{language} search text translated separately")
        else:
            notes.append(
                f"no {language} search text ({reason}): the question is searched"[:300]
            )
    if texts == dict(plan.texts):
        return replace(plan, notes=tuple(notes))
    return replace(plan, texts=texts, notes=tuple(notes))


def _translate(chat: ChatModel, question: str, language: str) -> tuple[str, str]:
    """The model's translation (empty when the call failed), and how the
    call ended."""

    try:
        reply = chat.generate(
            [Message.user(question)], ChatOptions(system=_translation_prompt(language))
        )
    except LLMError as error:
        return "", f"{type(error).__name__}: {error}"
    text = reply.text.strip() if reply.stop_reason == "end" else ""
    return text, f"stop {reply.stop_reason}"


def _in_language(text: str, language: str, question: str) -> bool:
    """False when `text` is plainly not in `language`: Vietnamese letters in
    a text for an English corpus, or the question copied unchanged into
    another language's field."""

    if language == "en" and _VIETNAMESE.search(text):
        return False
    return language == _question_language(question) or (
        text.strip() != question.strip()
    )


def _question_language(question: str) -> str:
    return "vi" if _VIETNAMESE.search(question) else "en"


def checked_plan(
    question: str, analysis: QueryAnalysis, corpora: Mapping[str, CorpusProfile]
) -> QueryPlan:
    """Keep only values the corpora know; no corpus left means all of them.
    Periods the model misreads most (partial years, dates, filing months)
    are completed from the question's own words."""

    notes: list[str] = []
    chosen = tuple(dict.fromkeys(c for c in analysis.corpora if c in corpora))
    dropped = sorted(set(analysis.corpora) - set(chosen))
    if dropped:
        notes.append(f"unknown corpora dropped: {', '.join(dropped)}")
    chosen = chosen or tuple(corpora)
    profiles = [corpora[c] for c in chosen]
    languages = {p.language for p in profiles}
    proposed = {
        code.lower(): str(text).strip()
        for code, text in analysis.searches.model_dump().items()
        if str(text).strip()
    }
    texts = {
        code: text
        for code, text in proposed.items()
        if code in languages and _in_language(text, code, question)
    }
    if wrong := sorted((set(proposed) & languages) - set(texts)):
        notes.append(f"search texts not in their language dropped: {wrong}")
    forms = {f for p in profiles for f in p.forms}
    topics = {t for p in profiles for t in p.topics}
    plan_forms = _known((_form(f) for f in analysis.forms), forms)
    plan_topics = _known((t.lower() for t in analysis.topics), topics)
    for kind, given, kept in (
        ("forms", analysis.forms, plan_forms),
        ("topics", analysis.topics, plan_topics),
    ):
        if len(kept) < len(set(given)):
            notes.append(f"unknown {kind} dropped: {given}")
    parsed = [(value, _period(value)) for value in analysis.fiscal_periods]
    if unknown := [value for value, (period, _) in parsed if period is None]:
        notes.append(f"unknown periods dropped: {unknown}")
    periods = tuple(dict.fromkeys(p for _, (p, _) in parsed if p))
    years = {*analysis.fiscal_years, *(y for _, (_, y) in parsed if y)}
    words = _read_periods(question)
    periods, years, calendar = _completed(analysis, periods, years, words, notes)
    model_months = _model_months(question, analysis.filed_months, notes)
    filed = tuple(dict.fromkeys([*model_months, *words.filed_months]))
    asked = tuple(dict.fromkeys(_form(f) for f in analysis.forms))
    kept_years, periods = _filed_scope(years, periods, filed, asked, words)
    return QueryPlan(
        question=question,
        language=analysis.language.lower() or "und",
        corpora=chosen,
        texts=texts,
        companies=tuple(
            dict.fromkeys(
                Mention(m.name.strip(), (m.ticker or "").strip().upper() or None)
                for m in analysis.companies
                if m.name.strip() or m.ticker
            )
        ),
        fiscal_years=kept_years,
        fiscal_periods=periods,
        calendar=calendar,
        forms=plan_forms,
        topics=plan_topics,
        wants_latest=analysis.wants_latest and not (kept_years or periods or filed),
        filed_months=filed,
        source="llm",
        notes=tuple(notes),
    )


def _completed(
    analysis: QueryAnalysis,
    periods: tuple[str, ...],
    years: set[int],
    words: _Wording,
    notes: list[str],
) -> tuple[tuple[str, ...], set[int], bool]:
    """A model's periods completed from the question's words: FY for a
    year-end date, the partial years' quarters, a date's calendar quarter
    when nothing else is asked."""

    if words.year_ends and "FY" not in periods:
        # "fiscal year ended September 28, 2024": the year, not its last quarter.
        named = (*words.quarters, *words.partial)
        periods = (*(p for p in periods if p in named), "FY")
        years = years | set(words.year_ends)
        notes.append(f"FY from year-end dates in the question: {words.year_ends}")
    calendar = analysis.period_basis == "calendar"
    if words.partial and (periods or not analysis.wants_latest):
        merged = _with_partial(periods, words)
        if merged != periods:
            notes.append(f"partial-year periods from the question: {words.partial}")
        if words.partial_dated:
            # Ended on a stated date: that date's calendar quarter.
            return merged, years | set(words.date_years), True
        return merged, years, calendar
    if not periods and words.dated:
        # A date's quarter is a calendar one (search matches its period end).
        notes.append(f"periods from dates in the question: {words.dated}")
        return words.dated, years | set(words.date_years), True
    return periods, years, calendar


def _model_months(question: str, values: Iterable[str], notes: list[str]) -> list[str]:
    """The model's filed months, only in a question about a filing or event."""

    months = [(value, _month(value)) for value in values]
    if bad := [value for value, month in months if month is None]:
        notes.append(f"unknown filed months dropped: {bad}")
    kept = [m for _, m in months if m]
    if kept and not _FILING_WORDS.search(question):
        # No filing or event asked: a month the model read is a period's.
        notes.append(f"filed months outside a filing question dropped: {kept}")
        return []
    return kept


def _form(value: str) -> str:
    """`10k`, `10 K`, `Form 10-K` -> `10-K`."""

    compact = re.sub(r"^FORM\s*", "", value.strip().upper())
    return re.sub(r"^(10|8)\W*([KQ])$", r"\1-\2", compact)


def _known(values: Iterable[str], allowed: set[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(v for v in values if v in allowed))


def _period(value: str) -> tuple[str | None, int | None]:
    """A model's period with a year folded in: `Q2 2024` -> (Q2, 2024),
    `FY 2025` -> (FY, 2025), `Q3 FY24` / `2Q24` / `FQ3` -> Q3/Q2/Q3, `H1
    2025` -> (Q2, 2025); (None, year) when no period is recognized."""

    text = value.strip().upper()
    if text in PERIODS:
        return text, None
    year = None
    if found := re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text):
        year = int(found.group(1))
    elif found := re.search(r"(?:FY|[1-4]Q)\s*'?(\d{2})(?!\d)", text):
        year = 2000 + int(found.group(1))
    if year is not None and not _MIN_YEAR <= year <= _MAX_YEAR:
        year = None
    for pattern, period in _VALUE_PERIODS:
        if found := pattern.search(text):
            return period or f"Q{next(g for g in found.groups() if g)}", year
    return None, year


def _month(value: str) -> str | None:
    """A model's filed month as `YYYY-MM` (`2022-4`, `04/2022`, `2022-04-14`,
    `April 2022`, `Apr-2022`, `tháng 4/2022`)."""

    text = value.strip()
    if found := re.fullmatch(
        r"((?:19|20)\d{2})\s*[-/.]\s*(\d{1,2})(?:[-/.]\d{1,2})?", text
    ):
        year, month = int(found.group(1)), int(found.group(2))
    elif found := re.fullmatch(r"(\d{1,2})\s*[-/.]\s*((?:19|20)\d{2})", text):
        year, month = int(found.group(2)), int(found.group(1))
    elif found := re.fullmatch(
        rf"{_MONTH}\s*[-/,]?\s*(?:of\s+)?{_Y}", text, re.IGNORECASE
    ):
        year, month = int(found.group(2)), _month_number(found.group(1))
    elif found := _VI_MONTH_YEAR.fullmatch(text):
        year, month = int(found.group(2)), int(found.group(1))
    else:
        return None
    if 1 <= month <= 12 and _MIN_YEAR <= year <= _MAX_YEAR:
        return f"{year:04d}-{month:02d}"
    return None


def _filed_scope(
    years: Iterable[int],
    periods: tuple[str, ...],
    filed: tuple[str, ...],
    forms: tuple[str, ...],
    words: _Wording,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Years and periods once filing months are known: a filing month pins
    an asked 8-K by itself (the 2021 shareholder letter filed in April
    2022), and a year that only dates a filing is no period."""

    kept = {y for y in years if _MIN_YEAR <= y <= _MAX_YEAR}
    if filed and forms == ("8-K",):
        return (), ()
    kept -= {int(m[:4]) for m in filed} - set(words.years)
    return tuple(sorted(kept)), periods


def _with_partial(periods: tuple[str, ...], words: _Wording) -> tuple[str, ...]:
    """The asked periods plus the partial years' quarters, without what a
    model gives for a misread partial year: the quarters it spans, or FY
    when the question names no other year. Quarters the question names stay."""

    last = max(int(p[1]) for p in words.partial_named)
    misread = {f"Q{n}" for n in range(1, last + 1)} | {"FY"}
    misread -= set(words.quarters) | set(words.partial)
    misread -= {"FY"} if words.other_year else set()
    kept = (p for p in periods if p not in misread)
    return tuple(dict.fromkeys([*kept, *words.partial]))


# Rules: the same plan without the model (no translation, no company names;
# the retriever looks for tickers and names in the question itself).
_VIETNAMESE = re.compile(
    r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", re.IGNORECASE
)
_YEAR = re.compile(r"(?<!\d)(?:fy\s?)?((?:19|20)\d{2})(?!\d)", re.IGNORECASE)
_QUARTERS = (
    (re.compile(r"\bq([1-4])\b|\bquý\s*(?:thứ\s*)?([1-4])\b", re.IGNORECASE), None),
    (re.compile(r"\b(first|1st)[\s-]+quarter\b|\bquý\s+(thứ\s+)?(một|nhất|i)\b",
                re.IGNORECASE), "Q1"),
    (re.compile(r"\b(second|2nd)[\s-]+quarter\b|\bquý\s+(thứ\s+)?(hai|ii)\b",
                re.IGNORECASE), "Q2"),
    (re.compile(r"\b(third|3rd)[\s-]+quarter\b|\bquý\s+(thứ\s+)?(ba|iii)\b",
                re.IGNORECASE), "Q3"),
    (re.compile(r"\b(fourth|4th)[\s-]+quarter\b|\bquý\s+(thứ\s+)?(bốn|tư|iv)\b",
                re.IGNORECASE), "Q4"),
)  # fmt: skip
# Periods in a model's value, first match wins (None: the quarter number is
# the first group that matched).
_VALUE_PERIODS: tuple[tuple[re.Pattern[str], str | None], ...] = (
    (re.compile(r"(?<![A-Z])F?Q([1-4])(?!\d)|(?<!\d)([1-4])Q|\bQUARTER\s*([1-4])\b"),
     None),
    *_QUARTERS,
    (re.compile(r"(?<![A-Z0-9])(H1|1H|6M)(?![A-Z])|\bFIRST\s+HALF\b"), "Q2"),
    (re.compile(r"(?<![A-Z0-9])9M(?![A-Z])|\bNINE\s+MONTHS\b"), "Q3"),
    (re.compile(r"(?<![A-Z])FY|FULL[\s-]+YEAR|ANNUAL"), "FY"),
)  # fmt: skip

# Year-to-date periods: their figures are in the 10-Q of the quarter they
# end with. Only with a year next to them ("first half of 2025", "H1 FY25",
# "6 tháng đầu năm 2025", "six months ended June 30, 2025"), never relative
# ("last six months", "6 tháng qua"); "6 tháng 9" is a date.
_PARTIAL = (
    (re.compile(r"\b(first|1st)\s+half\b|\b(h1|1h)\b|\bhalf[\s-]+year\b|"
                r"\b(six|6)[\s-]+months?\b|(?<!ngày )\b(6|sáu)\s+tháng\b(?!\s*\d)|"
                r"\bnửa\s+đầu\b|\bbán\s+niên\b|(?<![$\d.,])\b6m(?=\s*(fy\s?)?'?\d{2})",
                re.IGNORECASE), "Q2"),
    (re.compile(r"\b(nine|9)[\s-]+months?\b|(?<!ngày )\b(9|chín)\s+tháng\b(?!\s*\d)|"
                r"(?<![$\d.,])\b9m(?=\s*(fy\s?)?'?\d{2})", re.IGNORECASE), "Q3"),
)  # fmt: skip
_ANCHOR_YEAR = r"(?:19|20)\d{2}(?!\d)|\bfy\s?'?\d{2}(?!\d)"
_ANCHOR_AFTER = re.compile(rf"^[^.;?!]{{0,45}}?(?:{_ANCHOR_YEAR})", re.IGNORECASE)
_ANCHOR_BEFORE = re.compile(rf"(?:{_ANCHOR_YEAR})\s*$", re.IGNORECASE)
_RELATIVE_BEFORE = re.compile(
    r"\b(last|past|next|previous|prior|trailing|coming|recent|within)\s+$",
    re.IGNORECASE,
)
_RELATIVE_AFTER = re.compile(
    r"^\s*(ago|later|after|before|from now|qua|tới|gần đây|vừa qua|tiếp theo|sau|"
    r"trước)\b",
    re.IGNORECASE,
)
# Bare comparative years after a partial year ("six months ended June 30,
# 2024 and 2023"): the same partial year; "so với năm 2023" is a whole year.
_COMPARATIVE_TAIL = re.compile(
    r"(?:\s*,?\s*(?:and|or|vs\.?|versus|compared\s+(?:to|with)|so\s+với|và)\s+"
    r"(?:19|20)\d{2}(?!\d))+",
    re.IGNORECASE,
)
# "2Q24": a quarter and its year.
_SHORT_QUARTER = re.compile(r"\b([1-4])q'?(\d{2})\b", re.IGNORECASE)
# A quarter with its year ("Q1 2025", "quý 1/2025", "first quarter of 2025").
_QUARTER_YEAR_TAIL = re.compile(
    r"\s*(?:[/,-]|of|in|năm(?:\s+tài\s+chính)?)?\s*(?:fiscal\s+(?:year\s+)?|fy\s?'?)?"
    r"(?:(?:19|20)\d{2}|\d{2})(?!\d)",
    re.IGNORECASE,
)
# "annual" except an annual meeting (8-K item 5.07), which is no period.
_ANNUAL = (
    r"\bannual\b(?![\s-]+(general\s+)?(shareholders?'?\s+|stockholders?'?\s+)?meeting)"
)
_FULL_YEAR = re.compile(
    rf"\bfy\s?\d{{2,4}}\b|\bfiscal (year )?\d{{4}}\b|{_ANNUAL}|"
    r"\bcả năm\b|\bhàng năm\b|\bnăm tài chính\b|\btài khóa\b|\bbáo cáo thường niên\b",
    re.IGNORECASE,
)
# A whole year named outright, so FY is no misreading of a partial year.
_WHOLE_YEAR = re.compile(
    rf"\bfull[\s-]+year\b|\bwhole\s+year\b|{_ANNUAL}|\b10-?k\b|\bcả năm\b|"
    r"\btoàn năm\b|\bbáo cáo thường niên\b",
    re.IGNORECASE,
)
_FISCAL = re.compile(r"\bfy|\bfiscal\b|năm tài chính|tài khóa", re.IGNORECASE)
_FORM = re.compile(r"\b(10-?k|10-?q|8-?k)s?\b", re.IGNORECASE)
# Not "current"/"now": current assets, current liabilities.
_LATEST = re.compile(
    r"\b(latest|most recent|newest)\b|gần nhất|gần đây nhất|mới nhất|hiện nay",
    re.IGNORECASE,
)
_SHORT_FY = re.compile(r"\bfy\s?'?(\d{2})\b", re.IGNORECASE)
_TOPIC_WORDS = (
    (re.compile(r"risk factor|rủi ro", re.IGNORECASE), "risk_factors"),
    (re.compile(r"cyber|an ninh mạng", re.IGNORECASE), "cybersecurity"),
    (re.compile(r"lawsuit|litigation|legal proceeding|kiện tụng|tố tụng", re.IGNORECASE),
     "legal_proceedings"),
    (re.compile(r"press release|earnings release|thông cáo", re.IGNORECASE), "ex99"),
    (re.compile(r"repurchase|buyback|dividend|mua lại cổ phiếu|cổ tức", re.IGNORECASE),
     "market_equity"),
    (re.compile(r"\bmd&a\b|management'?s discussion", re.IGNORECASE), "mdna"),
)  # fmt: skip

# Dates and months. A dated balance ("as of September 30, 2022") reads the
# calendar quarter whose end is nearest (search matches calendar quarters by
# period end); a year-end date ("fiscal year ended September 28, 2024") the
# whole year; a month range its span ("January-December 2024" = the year). A
# date or month is a filing month next to "filed"/"announced"/"8-K", and a
# bare month + year in a question about an 8-K or event; elsewhere a bare
# month + year reads the calendar quarter holding it ("repurchases in
# February 2022" -> Q1 2022) when no other year or date is asked, unless it
# opens a span or is a due date ("since March 2020", "notes due March 2030").
# Off-calendar filers (quarters ending Jan/Apr/Jul/Oct) can miss a quarter's
# first month. A date without a year is neither.
_MONTH = (
    r"(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)(?![a-z])\.?"
)
_Y = r"((?:19|20)\d{2})(?!\d)"
_DAY = r"(?<![\d.,/$])(\d{1,2})"
_DAY_DATES = (
    # (pattern, group of year, month, day); month groups are names or numbers.
    (re.compile(rf"\b{_MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?!\s*(?:%|[.,]\d))"
                rf"(?:,?\s*{_Y})?", re.IGNORECASE), 3, 1, 2),
    (re.compile(rf"{_DAY}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}(?:,?\s*{_Y})?",
                re.IGNORECASE), 3, 2, 1),
    (re.compile(rf"ngày\s+(\d{{1,2}})(?:\s+tháng\s+|/)(\d{{1,2}})"
                rf"(?:(?:\s+năm\s+|/){_Y})?(?![\d/])", re.IGNORECASE), 3, 2, 1),
    (re.compile(rf"{_DAY}\s+tháng\s+(\d{{1,2}})\s+năm\s+{_Y}", re.IGNORECASE), 3, 2, 1),
    (re.compile(rf"(?<!\d){_Y}-(\d{{1,2}})-(\d{{1,2}})(?!\d)"), 1, 2, 3),
)  # fmt: skip
# d/m/yyyy in Vietnamese, m/d/yyyy in English (whichever fits when one > 12).
_NUMERIC_DATE = re.compile(rf"(?<![\d/.])(\d{{1,2}})[/.](\d{{1,2}})[/.]{_Y}")
_MONTH_YEAR = re.compile(rf"\b{_MONTH},?\s+(?:of\s+)?{_Y}", re.IGNORECASE)
_VI_MONTH_YEAR = re.compile(
    rf"tháng\s+(\d{{1,2}})\s*(?:/|-|\s+năm\s+){_Y}", re.IGNORECASE
)
# "June quarter of 2024": a quarter named by its last month.
_MONTH_QUARTER = re.compile(
    rf"\b{_MONTH}\s+quarter\s+(?:of\s+|in\s+)?{_Y}", re.IGNORECASE
)
# A month next to "quarter"/"quý" ends that quarter ("the September 2022
# quarter", "quý tháng 9/2022").
_QUARTER_BEFORE = re.compile(r"\b(quarter|quý)\s+$", re.IGNORECASE)
_QUARTER_AFTER = re.compile(r"^\s*quarter\b", re.IGNORECASE)
_YEAR_END_BEFORE = re.compile(
    r"\byear\s+(ended|ending)\s+(on\s+)?$|"
    r"\bnăm(\s+tài\s+chính)?\s+kết\s+thúc(\s+vào)?(\s+ngày)?\s+$",
    re.IGNORECASE,
)
_PERIOD_END_BEFORE = re.compile(
    r"\b(ended|ending|end of|as of|as at|at|through|until|to|kết thúc|cuối|đến|"
    r"tại)(\s+(vào|ngày))?\s+$",
    re.IGNORECASE,
)
# The first month of a range ("October-December 2024", "tháng 7 - tháng 9/2025").
_RANGE_START = re.compile(
    rf"\b{_MONTH}\s*(?:-|–|to|through|until)\s*$|"
    r"\btháng\s+(\d{1,2})\s*(?:-|–|đến)\s*$",
    re.IGNORECASE,
)
# Right before a date or month: it dates a filing or an announcement.
_FILED_BEFORE = re.compile(
    r"(\b(filed|filing|submitted|dated|issued|released|announced|nộp|công bố|"
    r"thông báo|ban hành)\b|\b8-?ks?\b)[^.;?!\d]{0,25}$",
    re.IGNORECASE,
)
# A month qualifying a filing or release noun dates it ("an April 2022
# 8-K", "July 2026 earnings press releases", "press release of July 2026",
# "thông cáo báo chí tháng 7/2026"); not "release say about revenue in".
_FILED_AFTER = re.compile(
    r"^\s*(form\s+)?8-?ks?\b|^\s*((quarterly|annual|earnings|results|financial)\s+)*"
    r"(press\s+releases?|releases?|filings?|current\s+reports?)\b",
    re.IGNORECASE,
)
_RELEASE_BEFORE = re.compile(
    r"\b(press\s+)?releases?\s+((issued|published)\s+)?(of|from|in|on)?\s*$|"
    r"\bthông\s+cáo(\s+báo\s+chí)?(\s+(phát\s+hành|công\s+bố))?(\s+(vào|trong))?\s*$",
    re.IGNORECASE,
)
# A question about an 8-K or an event: a bare month + year dates its filing
# ("March 2023 bylaw amendment"). Not "press release": a release reports
# periods ("revenue in September 2024").
_EVENT_TOPIC = re.compile(
    r"\b8-?ks?\b|current report|báo cáo hiện hành|\bbylaws?\b|điều lệ|\bamend|"
    r"sửa đổi|\bappoint|bổ nhiệm|\bresign|từ nhiệm|từ chức",
    re.IGNORECASE,
)
# Any filing or event wording: only then can a model's month be a filing's.
_FILING_WORDS = re.compile(
    rf"{_EVENT_TOPIC.pattern}|\b(filed|filing|submitted|dated|issued|released|"
    r"announced?|announcement)\b|\bnộp\b|công bố|thông báo|ban hành",
    re.IGNORECASE,
)
# Right before a month that opens a span: no single period.
_OPEN_BEFORE = re.compile(
    r"\b(since|after|before|from|starting|beginning|until|till|by|due|"
    r"matur\w*|expir\w*|kể\s+từ|từ|sau|trước|đến|đáo\s+hạn|hết\s+hạn)"
    r"(\s+(in|on|of))?\s+$",
    re.IGNORECASE,
)
# Words right before a date or month that decide what it is, in order.
_BEFORE_KINDS: tuple[
    tuple[Literal["year_end", "range", "period"], re.Pattern[str]], ...
] = (
    ("year_end", _YEAR_END_BEFORE),
    ("range", _RANGE_START),
    ("period", _PERIOD_END_BEFORE),
    ("period", _QUARTER_BEFORE),
)
_MONTH_ABBREVIATIONS = (
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
)  # fmt: skip


def _month_number(value: str) -> int:
    if value.isdigit():
        return int(value)
    return _MONTH_ABBREVIATIONS.index(value[:3].lower()) + 1


@dataclass(frozen=True)
class _Mention:
    """A date or month in the question."""

    start: int
    end: int
    year: int | None
    month: int
    day: int | None


@dataclass(frozen=True)
class _Wording:
    """What a question's own words say about periods."""

    # Years outside filing dates (incl. `FY25`; a date gives its quarter's).
    years: tuple[int, ...]
    quarters: tuple[str, ...]
    # The quarter whose 10-Q holds an asked year-to-date period: the one it
    # names (first half -> Q2), or the calendar quarter nearest the date it
    # ends on ("six months ended March 30, 2024" -> Q1).
    partial: tuple[str, ...]
    # The named quarters (what a model may give for them instead).
    partial_named: tuple[str, ...]
    # Every partial year ends on a stated date: calendar quarters.
    partial_dated: bool
    # Calendar periods of asked dates, and their years.
    dated: tuple[str, ...]
    date_years: tuple[int, ...]
    # Years of "year ended <date>".
    year_ends: tuple[int, ...]
    filed_months: tuple[str, ...]
    full_year: bool
    # A whole year asked besides any partial year ("H1 FY2025 vs FY2024").
    other_year: bool
    fiscal: bool


def _mentions(question: str) -> list[_Mention]:
    """Dates first, then months without a day outside them; in text order."""

    vietnamese = bool(_VIETNAMESE.search(question))
    found: list[_Mention] = []
    taken = [False] * len(question)

    def add(
        match: re.Match[str], year: str | None, month: int, day: int | None
    ) -> None:
        start, end = match.span()
        if any(taken[start:end]) or not 1 <= month <= 12:
            return
        if day is not None and not 1 <= day <= 31:
            return
        taken[start:end] = [True] * (end - start)
        found.append(_Mention(start, end, int(year) if year else None, month, day))

    for pattern, year_group, month_group, day_group in _DAY_DATES:
        for match in pattern.finditer(question):
            day = int(match.group(day_group))
            add(match, match.group(year_group), _month_number(match.group(month_group)),
                day)  # fmt: skip
    for match in _NUMERIC_DATE.finditer(question):
        first, second = int(match.group(1)), int(match.group(2))
        day_first = vietnamese if first <= 12 and second <= 12 else first > 12
        month, day = (second, first) if day_first else (first, second)
        add(match, match.group(3), month, day)
    for match in _MONTH_QUARTER.finditer(question):
        add(match, match.group(2), _month_number(match.group(1)), None)
    for match in _MONTH_YEAR.finditer(question):
        add(match, match.group(2), _month_number(match.group(1)), None)
    for match in _VI_MONTH_YEAR.finditer(question):
        add(match, match.group(2), int(match.group(1)), None)
    return sorted(found, key=lambda m: m.start)


def _nearest_quarter(year: int, month: int, day: int | None) -> tuple[int, str]:
    """The calendar quarter whose end is nearest the date (a month: its last
    day): April 27 -> Q1, January 26, 2025 -> Q4 2024, October 31 -> Q3."""

    last = calendar_module.monthrange(year, month)[1]
    when = date(year, month, min(day or last, last))
    ends = [date(year - 1, 12, 31)] + [
        date(year, m, calendar_module.monthrange(year, m)[1]) for m in (3, 6, 9, 12)
    ]
    end = min(ends, key=lambda e: abs((when - e).days))
    return end.year, f"Q{end.month // 3}"


def _mention_kind(
    question: str, mention: _Mention
) -> Literal["filed", "year_end", "range", "period", "month", "none"]:
    """What a date or month is: a filing's, a year's end, a month range's
    end, a period's end, a bare month (its quarter), or neither (no year; a
    month opening a span)."""

    before, after = question[: mention.start], question[mention.end :]
    inside = question[mention.start : mention.end]
    if mention.year is None:
        return "none"
    for kind, pattern in _BEFORE_KINDS:
        if pattern.search(before):
            return kind
    filed = bool(
        _FILED_BEFORE.search(before)
        or _RELEASE_BEFORE.search(before)
        or _FILED_AFTER.search(after)
    )
    if (
        _QUARTER_AFTER.search(after)
        or "quarter" in inside.lower()
        or (mention.day is not None and not filed)
    ):
        return "period"
    if filed or _EVENT_TOPIC.search(question):
        return "filed"
    return "none" if _OPEN_BEFORE.search(before) else "month"


def _partial_spans(question: str) -> list[tuple[int, int, str]]:
    """Each anchored partial year: (start, end incl. its year, quarter)."""

    spans = []
    for pattern, period in _PARTIAL:
        for match in pattern.finditer(question):
            before, after = question[: match.start()], question[match.end() :]
            if _RELATIVE_BEFORE.search(before):
                continue
            if _RELATIVE_AFTER.search(after):
                continue
            if anchor := _ANCHOR_AFTER.search(after):
                spans.append((match.start(), match.end() + anchor.end(), period))
            elif anchor := _ANCHOR_BEFORE.search(before):
                spans.append((anchor.start(), match.end(), period))
    return spans


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + " " * (end - start) + text[end:]


@dataclass
class _Dates:
    """The question's dates and months, by what they are."""

    filed: list[str] = field(default_factory=list)
    dated: list[str] = field(default_factory=list)
    date_years: list[int] = field(default_factory=list)
    year_ends: list[int] = field(default_factory=list)
    # Period ends: (start, end, calendar quarter).
    period_ends: list[tuple[int, int, str]] = field(default_factory=list)
    spans: list[tuple[int, int]] = field(default_factory=list)


def _read_dates(question: str) -> _Dates:
    dates = _Dates()
    # Bare months: (start, end, year, calendar quarter holding it).
    months: list[tuple[int, int, int, str]] = []
    for mention in _mentions(question):
        kind = _mention_kind(question, mention)
        if kind == "none" or mention.year is None:
            continue
        if kind == "month":
            quarter = f"Q{(mention.month + 2) // 3}"
            months.append((mention.start, mention.end, mention.year, quarter))
            continue
        dates.spans.append((mention.start, mention.end))
        if kind == "filed":
            dates.filed.append(f"{mention.year:04d}-{mention.month:02d}")
            continue
        if kind == "year_end":
            dates.year_ends.append(mention.year)
            continue
        range_start = _RANGE_START.search(question[: mention.start])
        if kind == "range" and range_start is not None:
            first = _month_number(range_start.group(1) or range_start.group(2))
            if first == 1 and mention.month == 12:
                dates.dated.append("FY")
                dates.date_years.append(mention.year)
                continue
        year, quarter = _nearest_quarter(mention.year, mention.month, mention.day)
        dates.dated.append(quarter)
        dates.date_years.append(year)
        dates.period_ends.append((mention.start, mention.end, quarter))
    _add_months(question, months, dates)
    return dates


def _add_months(
    question: str, months: list[tuple[int, int, int, str]], dates: _Dates
) -> None:
    """Bare months read the calendar quarter holding them (the 10-Q that
    reports them, for calendar-quarter filers), only when no other date or
    year is asked: else the month is context ("closed in March 2022, revenue
    in 2023") and only its year counts, as before."""

    if not months or dates.dated or dates.year_ends:
        return
    text = question
    for start, end in dates.spans:
        text = _blank(text, start, end)
    for start, end, _, _ in months:
        text = _blank(text, start, end)
    if _YEAR.search(text) or _SHORT_FY.search(text):
        return
    for start, end, year, quarter in months:
        dates.spans.append((start, end))
        dates.dated.append(quarter)
        dates.date_years.append(year)


def _partials(question: str, dates: _Dates) -> list[tuple[int, int, str, str | None]]:
    """Each anchored partial year: (start, end incl. its year, a date it
    ends on and comparative years, named quarter, its date's calendar
    quarter or None)."""

    found = []
    for start, anchor_end, named in _partial_spans(question):
        end, quarter = anchor_end, None
        for m_start, m_end, m_quarter in dates.period_ends:
            gap = question[anchor_end:m_start]
            if start <= m_start and len(gap) <= 25 and not re.search(r"[.;?!]", gap):
                end, quarter = max(anchor_end, m_end), m_quarter
                break
        if tail := _COMPARATIVE_TAIL.match(question, end):
            end = tail.end()  # "... June 30, 2024 and 2023": no other year
        found.append((start, end, named, quarter))
    return sorted(found)


def _read_periods(question: str) -> _Wording:
    dates = _read_dates(question)
    blanked = question  # without filing dates; dates add their own years
    for start, end in dates.spans:
        blanked = _blank(blanked, start, end)
    rest = blanked  # without any date, partial year or quarter + its year
    found = {int(y) for y in _YEAR.findall(blanked)}
    found |= {2000 + int(y) for y in _SHORT_FY.findall(blanked)}
    found |= {*dates.date_years, *dates.year_ends}
    partial = _partials(question, dates)
    for start, end, _, _ in partial:
        rest = _blank(rest, start, end)
    quarters = []
    for match in _SHORT_QUARTER.finditer(question):
        quarters.append(f"Q{match.group(1)}")
        found.add(2000 + int(match.group(2)))
        rest = _blank(rest, match.start(), match.end())
    for pattern, period in _QUARTERS:
        for match in pattern.finditer(question):
            quarters.append(period or f"Q{next(g for g in match.groups() if g)}")
            tail = _QUARTER_YEAR_TAIL.match(question, match.end())
            rest = _blank(rest, match.start(), tail.end() if tail else match.end())
    other_year = bool(_YEAR.search(rest) or _SHORT_FY.search(rest))
    return _Wording(
        years=tuple(sorted(y for y in found if _MIN_YEAR <= y <= _MAX_YEAR)),
        quarters=tuple(dict.fromkeys(quarters)),
        partial=tuple(dict.fromkeys(q or n for _, _, n, q in partial)),
        partial_named=tuple(dict.fromkeys(n for _, _, n, _ in partial)),
        partial_dated=bool(partial) and all(q for *_, q in partial),
        dated=tuple(dict.fromkeys(dates.dated)),
        date_years=tuple(dict.fromkeys(dates.date_years)),
        year_ends=tuple(dict.fromkeys(dates.year_ends)),
        filed_months=tuple(dict.fromkeys(dates.filed)),
        full_year=bool(_FULL_YEAR.search(question)),
        other_year=other_year or bool(_WHOLE_YEAR.search(question)),
        fiscal=bool(_FISCAL.search(question)),
    )


def rule_plan(
    question: str,
    corpora: Mapping[str, CorpusProfile],
    notes: tuple[str, ...] = (),
) -> QueryPlan:
    """Language, years, periods, filing months, forms, topics and recency
    from patterns."""

    words = _read_periods(question)
    periods = [*words.quarters, *words.partial]
    if words.year_ends or (words.partial and words.other_year):
        periods.append("FY")
    from_dates = not periods and bool(words.dated)
    if from_dates:
        periods = list(words.dated)
    elif not periods and words.full_year:
        periods = ["FY"]
    asked = tuple(dict.fromkeys(_form(m) for m in _FORM.findall(question)))
    years, kept = _filed_scope(
        words.years, tuple(dict.fromkeys(periods)), words.filed_months, asked, words
    )
    forms = {f for p in corpora.values() for f in p.forms}
    topics = tuple(dict.fromkeys(t for rx, t in _TOPIC_WORDS if rx.search(question)))
    language = _question_language(question)
    # Dates' periods are calendar ones; otherwise a plain year or part of
    # it is, a named quarter, fiscal year or year end is not.
    named_year = bool(words.year_ends) or (words.full_year and "FY" in kept)
    return QueryPlan(
        question=question,
        language=language,
        corpora=tuple(corpora),
        texts={},
        fiscal_years=years,
        fiscal_periods=kept,
        calendar=bool(years)
        and (
            from_dates
            or (words.partial_dated and not words.quarters)
            or not (words.fiscal or words.quarters or named_year)
        ),
        forms=_known(asked, forms),
        topics=topics,
        wants_latest=not (years or kept or words.filed_months)
        and bool(_LATEST.search(question)),
        filed_months=words.filed_months,
        source="rules",
        notes=notes,
    )

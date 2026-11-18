"""Read a question once before search: language, per-corpus search text,
corpora, companies, periods, forms, topics.

One structured call to the query model (`[chat.query]`) does it all; when
there is no model or it fails, rules give the same plan without translation
(the multilingual embedding still matches; BM25 then only matches names and
numbers). The model's values are checked against each corpus's vocabulary,
and an unknown value is dropped rather than failing the plan.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

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


class SearchText(BaseModel):
    language: str = Field(description="ISO 639-1 code of a collection language")
    text: str = Field(description="The question, complete and standalone")


class QueryAnalysis(BaseModel):
    """What the query model returns."""

    language: str = Field(description="ISO 639-1 code of the question, e.g. vi, en")
    corpora: list[str] = Field(description="Collections that can answer it")
    searches: list[SearchText] = Field(
        description="One search query per language of the chosen collections"
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
    source: Literal["llm", "rules"] = "rules"
    notes: tuple[str, ...] = field(default=())

    def text_for(self, language: str) -> str:
        return self.texts.get(language) or self.question

    @property
    def has_period(self) -> bool:
        return bool(self.fiscal_years or self.fiscal_periods)


def _system_prompt(corpora: Mapping[str, CorpusProfile], today: date) -> str:
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
            "- searches: for each language of the chosen collections, the question "
            "as one complete, standalone question in that language (a question, not "
            "keywords: the reranker judges passages against it). Translate faithfully, "
            "use the words filings use (doanh thu -> revenue / net sales, lợi nhuận -> "
            "net income, rủi ro -> risk factors), say what pronouns refer to, keep "
            "names, numbers and periods. Never answer the question."
        ),
        (
            "- companies: every company the question is about, with its official name "
            "and US ticker; empty when none."
        ),
        (
            "- fiscal_years: every year asked about (a range 2021-2024 -> 2021, 2022, "
            "2023, 2024); resolve relative years from today ('last year', 'năm "
            "ngoái'); empty when none."
        ),
        (
            "- fiscal_periods: FY for a full year, Q1-Q4 for a quarter; empty when the "
            "question names no period."
        ),
        (
            "- period_basis: fiscal when the user says fiscal / FY / năm tài chính, or "
            "names a quarter the way companies report it (Q2 2024, Q2 FY24, second "
            "quarter of fiscal 2024); calendar for a plain year ('revenue in 2024', "
            "'năm 2024', 'last year') or explicit months (July-September 2025). "
            "NVIDIA's fiscal 2025 ended in January 2025, so 'NVIDIA in 2024' is "
            "calendar."
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
            QueryAnalysis,
            ChatOptions(system=_system_prompt(corpora, today)),
        )
    except LLMError as error:
        reason = f"query model failed ({type(error).__name__}: {error}): rules"
        return rule_plan(question, corpora, notes=(reason[:300],))
    return checked_plan(question, analysis, corpora)


def checked_plan(
    question: str, analysis: QueryAnalysis, corpora: Mapping[str, CorpusProfile]
) -> QueryPlan:
    """Keep only values the corpora know; no corpus left means all of them."""

    notes: list[str] = []
    chosen = tuple(dict.fromkeys(c for c in analysis.corpora if c in corpora))
    dropped = sorted(set(analysis.corpora) - set(chosen))
    if dropped:
        notes.append(f"unknown corpora dropped: {', '.join(dropped)}")
    chosen = chosen or tuple(corpora)
    profiles = [corpora[c] for c in chosen]
    languages = {p.language for p in profiles}
    texts = {
        s.language.lower(): s.text.strip()
        for s in analysis.searches
        if s.language.lower() in languages and s.text.strip()
    }
    forms = {f for p in profiles for f in p.forms}
    topics = {t for p in profiles for t in p.topics}
    plan_forms = _known((_form(f) for f in analysis.forms), forms)
    plan_topics = _known((t.lower() for t in analysis.topics), topics)
    periods = _known((p.upper() for p in analysis.fiscal_periods), set(PERIODS))
    years = tuple(
        sorted({y for y in analysis.fiscal_years if _MIN_YEAR <= y <= _MAX_YEAR})
    )
    for kind, given, kept in (
        ("forms", analysis.forms, plan_forms),
        ("topics", analysis.topics, plan_topics),
        ("periods", analysis.fiscal_periods, periods),
    ):
        if len(kept) < len(set(given)):
            notes.append(f"unknown {kind} dropped: {given}")
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
        fiscal_years=years,
        fiscal_periods=periods,
        calendar=analysis.period_basis == "calendar",
        forms=plan_forms,
        topics=plan_topics,
        wants_latest=analysis.wants_latest and not (years or periods),
        source="llm",
        notes=tuple(notes),
    )


def _form(value: str) -> str:
    """`10k`, `10 K`, `Form 10-K` -> `10-K`."""

    compact = re.sub(r"^FORM\s*", "", value.strip().upper())
    return re.sub(r"^(10|8)\W*([KQ])$", r"\1-\2", compact)


def _known(values: Iterable[str], allowed: set[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(v for v in values if v in allowed))


# Rules: the same plan without the model (no translation, no company names;
# the retriever looks for tickers and names in the question itself).
_VIETNAMESE = re.compile(
    r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", re.IGNORECASE
)
_YEAR = re.compile(r"(?<!\d)(?:fy\s?)?((?:19|20)\d{2})(?!\d)", re.IGNORECASE)
_QUARTERS = (
    (re.compile(r"\bq([1-4])\b|\bquý\s*([1-4])\b", re.IGNORECASE), None),
    (re.compile(r"\b(first|1st)\s+quarter\b|\bquý\s+(một|i)\b", re.IGNORECASE), "Q1"),
    (re.compile(r"\b(second|2nd)\s+quarter\b|\bquý\s+(hai|ii)\b", re.IGNORECASE), "Q2"),
    (re.compile(r"\b(third|3rd)\s+quarter\b|\bquý\s+(ba|iii)\b", re.IGNORECASE), "Q3"),
    (
        re.compile(r"\b(fourth|4th)\s+quarter\b|\bquý\s+(bốn|tư|iv)\b", re.IGNORECASE),
        "Q4",
    ),
)
_FULL_YEAR = re.compile(
    r"\bfy\s?\d{2,4}\b|\bfiscal (year )?\d{4}\b|\bannual\b|\bcả năm\b|"
    r"\bnăm tài chính\b",
    re.IGNORECASE,
)
_FISCAL = re.compile(r"\bfy|\bfiscal\b|năm tài chính|tài khóa", re.IGNORECASE)
_FORM = re.compile(r"\b(10-?k|10-?q|8-?k)\b", re.IGNORECASE)
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


def rule_plan(
    question: str,
    corpora: Mapping[str, CorpusProfile],
    notes: tuple[str, ...] = (),
) -> QueryPlan:
    """Language, years, periods, forms, topics and recency from patterns."""

    found = {int(y) for y in _YEAR.findall(question)}
    found |= {2000 + int(y) for y in _SHORT_FY.findall(question)}
    years = tuple(sorted(y for y in found if _MIN_YEAR <= y <= _MAX_YEAR))
    periods: list[str] = []
    for pattern, period in _QUARTERS:
        for match in pattern.finditer(question):
            periods.append(period or f"Q{next(g for g in match.groups() if g)}")
    if not periods and _FULL_YEAR.search(question):
        periods.append("FY")
    forms = {f for p in corpora.values() for f in p.forms}
    found_forms = (_form(m) for m in _FORM.findall(question))
    topics = tuple(dict.fromkeys(t for rx, t in _TOPIC_WORDS if rx.search(question)))
    language = "vi" if _VIETNAMESE.search(question) else "en"
    return QueryPlan(
        question=question,
        language=language,
        corpora=tuple(corpora),
        texts={},
        fiscal_years=years,
        fiscal_periods=tuple(dict.fromkeys(periods)),
        calendar=bool(years) and not periods and not _FISCAL.search(question),
        forms=_known(found_forms, forms),
        topics=topics,
        wants_latest=not years and not periods and bool(_LATEST.search(question)),
        source="rules",
        notes=notes,
    )

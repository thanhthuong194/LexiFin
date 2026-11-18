"""Corpora retrieval can search, and what the query model is told about each.

A corpus is one Qdrant alias (`qdrant_store.Corpus`); adding one (VN
filings, US or VN law) is one more entry here plus its collection.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.rag_engine.indexing.qdrant_store import SEC_FILINGS, Corpus


@dataclass(frozen=True)
class CorpusProfile:
    corpus: Corpus
    description: str
    forms: tuple[str, ...]
    # Section topic -> what it holds (shown to the query model).
    topics: dict[str, str]

    @property
    def name(self) -> str:
        return self.corpus.name

    @property
    def alias(self) -> str:
        return self.corpus.name

    @property
    def language(self) -> str:
        return self.corpus.language


_SEC_TOPICS = {
    "cover": "cover page: registrant, shares outstanding, filer status",
    "business": "business description, products, segments, employees, competition",
    "risk_factors": "risk factors",
    "cybersecurity": "cybersecurity risk management and governance",
    "properties": "properties, facilities",
    "legal_proceedings": "legal proceedings, litigation",
    "market_equity": "stock market, dividends, share repurchases",
    "mdna": "management's discussion and analysis: results, revenue and cost "
    "drivers, segment results, liquidity, capital resources, outlook",
    "market_risk": "quantitative disclosures about market risk (FX, interest rates)",
    "financial_statements": "financial statements and notes: income statement, "
    "balance sheet, cash flows, revenue by segment/geography, debt, tax, leases",
    "controls": "controls and procedures",
    "governance": "directors, executive officers, compensation",
    "exhibits": "exhibit index and financial statement schedules",
    "signatures": "signatures",
    "8k_event": "8-K items: events such as results announcements, executive "
    "changes, acquisitions, agreements, shareholder votes",
    "ex99": "press releases and slides attached to 8-Ks (earnings releases, "
    "guidance, announcements)",
    "other": "anything else",
}

SEC_FILINGS_PROFILE = CorpusProfile(
    corpus=SEC_FILINGS,
    description="SEC filings of large US public companies (S&P 500): 10-K annual "
    "reports, 10-Q quarterly reports, 8-K current reports with their EX-99 press "
    "releases",
    forms=("10-K", "10-Q", "8-K"),
    topics=_SEC_TOPICS,
)

CORPORA: dict[str, CorpusProfile] = {SEC_FILINGS_PROFILE.name: SEC_FILINGS_PROFILE}

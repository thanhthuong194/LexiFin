"""Map company mentions to a corpus company (`cik`) from its directory.

The query model names companies the way filers do ("Alphabet Inc.", GOOGL)
and this module checks them against the companies the collection really
holds (`indexing/companies.py`): by ticker, then by normalized name, then by
a unique whole-word prefix ("Meta" -> "Meta Platforms"). Without a query
model, `find_in_text` spots tickers and names in the question itself.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

from qdrant_client import QdrantClient

from src.rag_engine.indexing.companies import stored_companies

# Legal-form words dropped from the end of a name ("Apple Inc." -> "apple").
_SUFFIXES = frozenset(
    (
        "inc", "incorporated", "corp", "corporation", "co", "company", "companies",
        "ltd", "limited", "plc", "llc", "lp", "holdings", "holding", "group", "sa",
        "nv", "ag", "se", "com", "the", "class", "a", "b", "c",
    )
)  # fmt: skip
_TICKER = re.compile(r"(?<![\w$])\$?([A-Z]{1,5}(?:[.-][A-Z])?)(?![\w])")
_MIN_TEXT_TICKER = 2
# Tickers that are also English words or finance abbreviations: in a
# question they count only as `$IT`.
_WORD_TICKERS = frozenset(
    (
        "IT", "ON", "ALL", "NOW", "KEY", "CF", "IR", "NI", "SO", "DE", "ARE", "CAT",
        "LOW", "BE", "GO", "PM", "ES", "ED", "EL", "AN", "OR", "AM", "ANY", "FAST",
        "COST", "PEG", "DD", "CI", "HAS", "MS", "MA", "BR", "GE", "HD", "PH", "TT",
        "AI", "CPU", "EPS", "FDS", "TECH", "FIX", "LIN", "OTIS", "WELL", "BALL",
        "PAYC", "RL", "VRSK", "CE", "PNR", "EG", "J", "K", "L", "O", "T", "V", "C",
        "F", "D", "IP", "GM", "ICE", "DOW", "EA", "PG", "MO", "SPG", "CMS", "ETN",
        "NOC", "KIM", "AMP", "GL", "NRG", "PCG",
    )
)  # fmt: skip
# One-word company names that are also common words: in a question they
# count only capitalized and not at the start of a sentence.
_WORD_NAMES = frozenset(
    (
        "target", "progressive", "southern", "news", "booking", "general", "public",
        "united", "ball", "dollar", "block", "match", "waters", "best", "align",
        "ross", "fox", "carrier", "discover", "everest", "visa",
        # Common first words of longer names ("First Solar", "Global Payments").
        "first", "global", "american", "national", "international", "home",
        "texas", "western", "eastern", "live", "host", "west", "regions", "digital",
        "republic", "principal", "prudential", "consolidated", "universal",
    )
)  # fmt: skip
# Brand names the filings never use, for questions read without the query
# model (which maps them itself): alias -> normalized company name.
ALIASES = {
    "google": "alphabet",
    "youtube": "alphabet",
    "facebook": "meta platforms",
    "instagram": "meta platforms",
    "whatsapp": "meta platforms",
    "aws": "amazon",
}


def normalize_name(name: str) -> str:
    """Lowercase ASCII words of a company name without legal-form suffixes:
    `AMAZON.COM, INC.` -> `amazon`, `The Coca-Cola Company` -> `coca cola`."""

    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    text = re.sub(r"\b([a-z])\.([a-z])\.", r"\1\2", text.lower())  # S.A. -> sa
    words = re.sub(r"[^a-z0-9]+", " ", text.replace("&", " and ")).split()
    while words and words[-1] in _SUFFIXES:
        words.pop()
    while words and words[0] == "the":
        words.pop(0)
    return " ".join(words)


def _ticker_key(ticker: str) -> str:
    return re.sub(r"[.\-/ ]", "", ticker.upper())


@dataclass(frozen=True)
class Company:
    cik: str
    company_name: str
    tickers: tuple[str, ...]
    names: tuple[str, ...]
    latest_filing_date: date | None = None


class CompanyIndex:
    """The companies of one collection, looked up by ticker or name."""

    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        self.companies = [
            Company(
                cik=row["cik"],
                company_name=row["company_name"],
                tickers=tuple(row["tickers"]),
                names=tuple(row["names"]) or (row["company_name"],),
                latest_filing_date=(
                    date.fromisoformat(row["latest_filing_date"])
                    if row.get("latest_filing_date")
                    else None
                ),
            )
            for row in rows
        ]
        self._by_ticker: dict[str, Company] = {}
        self._by_name: dict[str, list[Company]] = {}
        for company in self.companies:
            for ticker in company.tickers:
                self._by_ticker[_ticker_key(ticker)] = company
            for name in {normalize_name(n) for n in company.names} - {""}:
                self._by_name.setdefault(name, []).append(company)
        self._text_names = _text_names(self._by_name)
        self._sentence_words = _WORD_NAMES

    @classmethod
    def from_qdrant(cls, client: QdrantClient, chunk_collection: str) -> CompanyIndex:
        return cls(stored_companies(client, chunk_collection))

    def __len__(self) -> int:
        return len(self.companies)

    def resolve(self, name: str | None, ticker: str | None = None) -> Company | None:
        """The one company a mention means, else None (unknown or ambiguous)."""

        by_ticker = next(
            (
                found
                for candidate in (ticker, name)
                if candidate
                and (candidate == ticker or candidate.isupper())
                and (found := self._by_ticker.get(_ticker_key(candidate)))
            ),
            None,
        )
        by_name = self._by_name_only(name)
        # A ticker the model made up must not pick another company than the
        # one the name names; a name it misspells still has the ticker.
        return by_name if by_name is not None else by_ticker

    def _by_name_only(self, name: str | None) -> Company | None:
        key = normalize_name(name or "")
        key = ALIASES.get(key, key)
        if not key:
            return None
        exact = self._by_name.get(key, [])
        if len(exact) == 1:
            return exact[0]
        if exact:
            return None
        words = key.split()
        prefixed = {
            company.cik: company
            for known, companies in self._by_name.items()
            for company in companies
            if _prefix(words, known.split())
        }
        return next(iter(prefixed.values())) if len(prefixed) == 1 else None

    def find_in_text(self, text: str) -> list[Company]:
        """Companies a question names, in order of mention (the fallback when
        no query model reads the question).

        Tickers count as written (uppercase, 2+ letters), except tickers
        that are also words or finance abbreviations (`IT`, `ON`, `CF`...),
        which need a `$`. Names count as whole words; a one-word name must
        be capitalized ("Target", not "target gross margin")."""

        found: dict[str, tuple[int, Company]] = {}
        for match in _TICKER.finditer(text):
            ticker = match.group(1)
            company = self._by_ticker.get(_ticker_key(ticker))
            if company is None or len(ticker) < _MIN_TEXT_TICKER:
                continue
            if ticker in _WORD_TICKERS and not match.group(0).startswith("$"):
                continue
            found.setdefault(company.cik, (match.start(), company))
        ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore")
        tokens = [
            (m.start(), m.group())
            for m in re.finditer(r"[A-Za-z0-9]+", ascii_text.decode())
        ]
        lowered = [token.lower() for _, token in tokens]
        for words, company in self._text_names:
            position = _find_words(
                tokens, lowered, words, ascii_text.decode(), self._sentence_words
            )
            if position is not None:
                found.setdefault(company.cik, (position, company))
        return [company for _, company in sorted(found.values(), key=lambda f: f[0])]


_MIN_PREFIX_CHARS = 4


def _text_names(
    by_name: dict[str, list[Company]],
) -> list[tuple[tuple[str, ...], Company]]:
    """Word sequences that name one company in a question, longest first:
    unambiguous normalized names, brand aliases, and the unique first word of
    a longer name ("Ford" -> "ford motor")."""

    names: dict[tuple[str, ...], Company] = {
        tuple(name.split()): companies[0]
        for name, companies in by_name.items()
        if len(companies) == 1
    }
    for alias, target in ALIASES.items():
        if len(by_name.get(target, [])) == 1:
            names[(alias,)] = by_name[target][0]
    firsts: dict[str, list[Company]] = {}
    for words, company in names.items():
        if len(words) > 1 and len(words[0]) >= _MIN_PREFIX_CHARS:
            firsts.setdefault(words[0], []).append(company)
    for word, companies in firsts.items():
        if len({c.cik for c in companies}) == 1 and (word,) not in names:
            names[(word,)] = companies[0]
    return sorted(names.items(), key=lambda item: -len(item[0]))


def _find_words(
    tokens: list[tuple[int, str]],
    lowered: list[str],
    words: tuple[str, ...],
    text: str,
    common: frozenset[str] | set[str],
) -> int | None:
    """Where `words` first appear as a name: a one-word name capitalized,
    not followed by another capitalized word (`Apple Hospitality REIT`), and
    for common words not at a sentence start."""

    for i in range(len(lowered) - len(words) + 1):
        if tuple(lowered[i : i + len(words)]) != words:
            continue
        start, token = tokens[i]
        if len(words) == 1:
            following = tokens[i + 1][1] if i + 1 < len(tokens) else ""
            sentence_start = (
                not text[:start].strip() or text[:start].rstrip()[-1] in ".!?"
            )
            if (
                not token[0].isupper()
                or (
                    following[:1].isupper()
                    and not following.isupper()
                    and following.lower() not in _SUFFIXES
                )
                or (words[0] in common and sentence_start)
            ):
                continue
        return start
    return None


def _prefix(mention: list[str], known: list[str]) -> bool:
    """The mention is a whole-word prefix of a known name ("Meta" -> "meta
    platforms") with a first word of 3+ letters. Never the reverse: "Apple
    Hospitality REIT" is not Apple."""

    return (
        bool(mention)
        and len(mention) <= len(known)
        and known[: len(mention)] == mention
        and len(mention[0]) >= 3
    )

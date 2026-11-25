"""Map company mentions to a corpus company (`cik`) from its directory.

The query model names companies the way filers do ("Alphabet Inc.", GOOGL)
and this module checks them against the companies the collection really
holds (`indexing/companies.py`): by ticker, then by normalized name, then by
a unique whole-word prefix ("Meta" -> "Meta Platforms"). Without a query
model, `find_in_text` spots tickers and names in the question itself; with
one, `resolve_in_question` lets tickers the question writes overrule the
model's guesses.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any, NamedTuple

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
        "NOC", "KIM", "AMP", "GL", "NRG", "PCG", "COO", "CRM", "AMT", "PTC", "USB",
        "DOC", "FICO", "MTD", "MSCI", "ESS", "PSA", "XYZ", "IEX", "CNC", "RF", "DG",
        "ECL", "SLB", "SW", "CB", "GIS",
    )
)  # fmt: skip
# Word tickers that, possessive, are mostly an agency or term ("CMS's rate
# notice", "ICE's").
_ACRONYM_TICKERS = frozenset(
    ("CMS", "ICE", "DOC", "USB", "EPS", "CPU", "AI", "IT", "IR", "PM", "AM", "CI",
     "MS", "ESS", "IP", "ON", "ALL", "NOW", "KEY", "ARE", "BE", "GO", "AN", "OR")
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


# Written as a ticker: possessive ("RCL's"), in parentheses after a
# capitalized word ("Valero (VLO)"), or in a list ("RF, RJF and MTB").
_POSSESSIVE = re.compile(r"['’]s\b")
_NAMED_TICKER = re.compile(r"\b[A-Z][\w&.]*\s*\(\s*$")
_LIST_GAP = re.compile(r"\s*(,|&|/|\band\b|\bor\b|\bvs\.?|và|hoặc)?\s*(and|và)?\s*")


@dataclass(frozen=True)
class Company:
    cik: str
    company_name: str
    tickers: tuple[str, ...]
    names: tuple[str, ...]
    latest_filing_date: date | None = None


class _Written(NamedTuple):
    """A ticker the question writes; `sure` when it plainly is one."""

    start: int
    key: str
    company: Company
    sure: bool


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

    def resolve_in_question(
        self, question: str, mentions: Iterable[tuple[str, str | None]]
    ) -> tuple[list[Company], list[str]]:
        """The query model's mentions (name, ticker) checked against the
        tickers the question writes: the companies and the mentions not in
        the corpus.

        - A mention whose ticker the question writes resolves by it.
        - A mention the question does not write (by name or ticker) whose
          ticker is one letter off a written ticker the model gave for
          nothing else is a misread of it (EXE read as Exelon's EXC,
          MKC as McKesson's MCK): it resolves by the written ticker. So
          does the model's only mention, unwritten, when the question
          writes one ticker the model did not give, starting like the
          mention's ticker or name (UHS read as UnitedHealth's UNH, BRO as
          Broadcom; not IBM for "Google").
        - A written ticker that plainly is one (`_ticker_matches`; any
          when the model names no company) adds its company; unresolved
          mentions the question never writes are then the model's guesses
          at it ("Ryder" for RCL) and are dropped."""

        mentions = list(mentions)
        confirmed = {_ticker_key(t) for _, t in mentions if t}
        written = self._ticker_matches(question, confirmed, alone=not mentions)
        by_key = {w.key: w.company for w in written}
        unread = [key for key in by_key if key not in confirmed]
        words = f" {normalize_name(question)} "
        read: list[tuple[str, str | None, Company | None]] = []
        for name, ticker in mentions:
            key = _ticker_key(ticker) if ticker else ""
            company = by_key.get(key)
            if company is None and not _written(name, ticker, words, question):
                near = next((w for w in unread if _one_letter_off(key, w)), None)
                if (
                    near is None
                    and len(mentions) == 1
                    and len(unread) == 1
                    and (
                        unread[0][:1] == key[:1]
                        or normalize_name(name).upper().startswith(unread[0][:2])
                    )
                    and not self._named(name, ticker, words)
                ):
                    near = unread[0]
                if near is not None:
                    company = by_key[near]
                    unread.remove(near)
            read.append((name, ticker, company or self.resolve(name, ticker)))
        found = {c.cik: c for *_, c in read if c is not None}
        added = [w.company for w in written if w.sure and w.company.cik not in found]
        missing = [
            f"{name} ({ticker})" if ticker else name
            for name, ticker, company in read
            if company is None
            and (not added or _written(name, ticker, words, question))
        ]
        for company in added:
            found.setdefault(company.cik, company)
        return list(found.values()), missing

    def _named(self, name: str, ticker: str | None, words: str) -> bool:
        """The question writes a distinctive word of the mention's company
        ("Disney" for The Walt Disney Company)."""

        company = self.resolve(name, ticker)
        if company is None:
            return False
        return any(
            f" {word} " in words
            for known in company.names
            for word in normalize_name(known).split()
            if len(word) >= _MIN_PREFIX_CHARS and word not in _WORD_NAMES
        )

    def find_in_text(self, text: str, confirmed: Iterable[str] = ()) -> list[Company]:
        """Companies a question names, in order of mention (the fallback when
        no query model reads the question).

        Tickers count as written (uppercase, 2+ letters), except tickers
        that are also words or finance abbreviations (`IT`, `ON`, `CF`...),
        which need a `$` or to be in `confirmed` (the query model read them
        as tickers). Names count as whole words; a one-word name must be
        capitalized ("Target", not "target gross margin")."""

        found: dict[str, tuple[int, Company]] = {}
        keys = {_ticker_key(t) for t in confirmed}
        for written in self._ticker_matches(text, keys):
            found.setdefault(written.company.cik, (written.start, written.company))
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

    def _ticker_matches(
        self, text: str, confirmed: set[str], *, alone: bool = False
    ) -> list[_Written]:
        """Tickers the text writes, in order. A word ticker (`IT`, `RF`)
        counts only with a `$`, `confirmed` by the query model, or listed
        next to another ticker ("RF, RJF and MTB"). `sure` marks those that
        plainly are tickers: these, a possessive ("RCL's"), parentheses
        after a capitalized word ("Valero (VLO)", not "radio frequency
        (RF)"), or 2+ tickers written. `alone` (the query model named no
        company): every ticker written is sure, a word ticker too when
        possessive ("SW's"), unless it is mostly an acronym ("CMS's")."""

        found: list[tuple[re.Match[str], str, Company]] = []
        for match in _TICKER.finditer(text):
            key = _ticker_key(match.group(1))
            company = self._by_ticker.get(key)
            if company is not None and len(match.group(1)) >= _MIN_TEXT_TICKER:
                found.append((match, key, company))
        written: list[_Written] = []
        for i, (match, key, company) in enumerate(found):
            plain = (
                match.group(0).startswith("$")
                or key in confirmed
                or _listed(text, found, i)
            )
            possessive = bool(_POSSESSIVE.match(text, match.end()))
            loose = alone and possessive and match.group(1) not in _ACRONYM_TICKERS
            if match.group(1) in _WORD_TICKERS and not (plain or loose):
                continue
            in_parentheses = bool(_NAMED_TICKER.search(text[: match.start()])) and text[
                match.end() :
            ].lstrip().startswith(")")
            sure = plain or possessive or in_parentheses or alone
            written.append(_Written(match.start(), key, company, sure))
        if len({w.key for w in written}) > 1:
            written = [w._replace(sure=True) for w in written]
        return written


def _listed(text: str, found: list[tuple[re.Match[str], str, Company]], i: int) -> bool:
    """The i-th ticker sits in a list with a neighbouring non-word ticker."""

    for j in (i - 1, i + 1):
        if not 0 <= j < len(found) or found[j][0].group(1) in _WORD_TICKERS:
            continue
        first, second = sorted((found[i][0], found[j][0]), key=lambda m: m.start())
        if _LIST_GAP.fullmatch(text[first.end() : second.start()]):
            return True
    return False


_MIN_PREFIX_CHARS = 4


def _one_letter_off(read: str, written: str) -> bool:
    """Same length, one letter changed or two neighbours swapped."""

    if not read or len(read) != len(written) or read == written:
        return False
    diff = [i for i, (a, b) in enumerate(zip(read, written, strict=True)) if a != b]
    return len(diff) == 1 or (
        len(diff) == 2
        and diff[1] == diff[0] + 1
        and read[diff[0]] == written[diff[1]]
        and read[diff[1]] == written[diff[0]]
    )


def _written(name: str, ticker: str | None, words: str, question: str) -> bool:
    """The question writes a mention: its normalized name, or the first word
    of it when distinctive ("Southwestern"), among the question's normalized
    words, or its ticker as an uppercase word."""

    key = normalize_name(name)
    first = key.split()[0] if key else ""
    distinctive = len(first) >= _MIN_PREFIX_CHARS and first not in _WORD_NAMES
    return bool(
        key and (f" {key} " in words or (distinctive and f" {first} " in words))
    ) or bool(ticker and re.search(rf"(?<!\w)\$?{re.escape(ticker)}(?!\w)", question))


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

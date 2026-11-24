"""Company universe of the eval sampler: GICS sector and size of each corpus
company (Bronze universe snapshot + IVV holdings), and companies outside it
for unanswerable questions."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from src.evals.sampling_data import rank
from src.rag_engine.retrieval.companies import normalize_name

SNAPSHOTS_DIR = "universe_snapshots"
# Distinct non-universe companies at the top of SEC's list (roughly by size)
# that outside picks draw from: recognisable names only.
SEC_TOP = 200
# Legal-form words dropped from the end of display names.
_LEGAL_TAIL = re.compile(
    r"(?:[\s,]+(?:inc|incorporated|corp|corporation|co|company|ltd|limited|plc|"
    r"holdings?|n\.?v|s\.?a|ag|se|public company|/[a-z]{2,3}/?)\.?)+$",
    re.IGNORECASE,
)
_LOWER_WORDS = frozenset({"and", "of", "the", "for", "&"})


@dataclass(frozen=True, slots=True)
class CompanyProfile:
    cik: str
    name: str
    tickers: tuple[str, ...]
    sector: str | None  # GICS sector from IVV holdings `Sector`
    weight_pct: float
    size_decile: int  # 1 = largest weights ... 10 = smallest, within the universe


@dataclass(frozen=True, slots=True)
class OutsideCompany:
    name: str
    tickers: tuple[str, ...]
    aliases: tuple[str, ...]
    country: str  # `US-listed` for SEC-listed picks (ADRs included)


# Well-known companies of smaller markets that file nothing with the SEC
# (no ADR registrants), with home-exchange tickers.
OUTSIDE_CURATED: tuple[OutsideCompany, ...] = (
    OutsideCompany("Vinamilk", ("VNM",), ("Vietnam Dairy Products",), "Vietnam"),
    OutsideCompany("FPT", ("FPT",), ("FPT Corporation",), "Vietnam"),
    OutsideCompany("Vingroup", ("VIC",), ("Tập đoàn Vingroup",), "Vietnam"),
    OutsideCompany("Masan Group", ("MSN",), ("Masan",), "Vietnam"),
    OutsideCompany("Hoa Phat Group", ("HPG",), ("Hòa Phát",), "Vietnam"),
    OutsideCompany("Vietcombank", ("VCB",), ("Ngân hàng Ngoại thương",), "Vietnam"),
    OutsideCompany("PTT", ("PTT",), ("PTT Public Company",), "Thailand"),
    OutsideCompany("CP All", ("CPALL",), ("7-Eleven Thailand",), "Thailand"),
    OutsideCompany("Siam Cement Group", ("SCC",), ("SCG",), "Thailand"),
    OutsideCompany("Airports of Thailand", ("AOT",), (), "Thailand"),
    OutsideCompany("Orlen", ("PKN",), ("PKN Orlen",), "Poland"),
    OutsideCompany("PKO Bank Polski", ("PKO",), ("PKO BP",), "Poland"),
    OutsideCompany("CD Projekt", ("CDR",), (), "Poland"),
    OutsideCompany("LPP", ("LPP",), (), "Poland"),
    OutsideCompany("Bank Central Asia", ("BBCA",), ("BCA",), "Indonesia"),
    OutsideCompany("Astra International", ("ASII",), ("Astra",), "Indonesia"),
    OutsideCompany("GoTo", ("GOTO",), ("Gojek", "Tokopedia"), "Indonesia"),
    OutsideCompany("SM Investments", ("SM",), (), "Philippines"),
    OutsideCompany("Ayala", ("AC",), ("Ayala Corporation",), "Philippines"),
    OutsideCompany("Jollibee Foods", ("JFC",), ("Jollibee",), "Philippines"),
    OutsideCompany("Maybank", ("1155",), ("Malayan Banking",), "Malaysia"),
    OutsideCompany("Public Bank", ("1295",), (), "Malaysia"),
    OutsideCompany("CEZ", ("CEZ",), ("ČEZ", "CEZ Group"), "Czechia"),
    OutsideCompany("Komercni banka", ("KOMB",), ("Komerční banka",), "Czechia"),
    OutsideCompany("Falabella", ("FALABELLA",), (), "Chile"),
    OutsideCompany("Cencosud", ("CENCOSUD",), (), "Chile"),
    OutsideCompany("OTP Bank", ("OTP",), (), "Hungary"),
    OutsideCompany("MOL Group", ("MOL",), (), "Hungary"),
    OutsideCompany("Banca Transilvania", ("TLV",), (), "Romania"),
)


def display_name(name: str) -> str:
    """A name as people write it: legal tail dropped (`PFIZER INC` ->
    `Pfizer`, `ASML HOLDING NV` -> `ASML`); all-caps names title-cased,
    keeping words of up to 3 letters as acronyms (`KLA`)."""

    short = _LEGAL_TAIL.sub("", name).strip(" ,") or name
    if not short.isupper() or (" " not in short and len(short) <= 4):
        return short  # mixed case as filed, or a lone acronym (`ASML`)
    words = []
    for index, word in enumerate(short.split()):
        lower = word.lower()
        if lower in _LOWER_WORDS and index:
            words.append(lower)
        elif len(word) <= 3 and word.isalpha():
            words.append(word)
        else:
            words.append(word.capitalize())
    return " ".join(words)


def ticker_key(ticker: str) -> str:
    """Share-class punctuation dropped: `BRK B`, `BRK.B`, `BRK-B` -> `BRKB`."""

    return re.sub(r"[^A-Z0-9]", "", ticker.upper())


def latest_snapshot(bronze_dir: Path) -> Path:
    """Newest `snapshot_date_<ts>` folder holding a `universe.jsonl`."""

    folders = sorted((bronze_dir / SNAPSHOTS_DIR).glob("snapshot_date_*"))
    for folder in reversed(folders):
        if (folder / "universe.jsonl").is_file():
            return folder
    raise FileNotFoundError(f"no universe snapshot under {bronze_dir / SNAPSHOTS_DIR}")


def holdings_sectors(path: Path) -> dict[str, str]:
    """`ticker_key` -> `Sector` of the IVV holdings CSV (preamble skipped)."""

    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    start = next(
        (i for i, line in enumerate(lines) if "Ticker" in next(csv.reader([line]), [])),
        None,
    )
    if start is None:
        return {}
    sectors: dict[str, str] = {}
    for row in csv.DictReader(lines[start:]):
        ticker, sector = (row.get("Ticker") or "").strip(), (row.get("Sector") or "")
        if ticker and sector.strip() and ticker_key(ticker) not in sectors:
            sectors[ticker_key(ticker)] = sector.strip()
    return sectors


def load_universe(bronze_dir: Path) -> dict[str, CompanyProfile]:
    """Companies of the latest universe snapshot by CIK, with GICS sector
    (None when no holdings ticker matches) and IVV weight decile."""

    folder = latest_snapshot(bronze_dir)
    sectors = holdings_sectors(folder / "ivv_holdings.csv")
    with (folder / "universe.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows.sort(key=lambda row: (-float(row["weight_pct"]), row["cik"]))
    profiles: dict[str, CompanyProfile] = {}
    for index, row in enumerate(rows):
        tickers = tuple(
            dict.fromkeys([row["primary_ticker"], *row.get("ticker_aliases", ())])
        )
        sector = next(
            (sectors[ticker_key(t)] for t in tickers if ticker_key(t) in sectors), None
        )
        profiles[row["cik"]] = CompanyProfile(
            cik=row["cik"],
            name=row["company_name"],
            tickers=tickers,
            sector=sector,
            weight_pct=float(row["weight_pct"]),
            size_decile=1 + index * 10 // len(rows),
        )
    return profiles


def _sec_outside(excluded: Collection[str], path: Path) -> list[OutsideCompany]:
    """The first SEC_TOP companies of `company_tickers.json` whose CIK is not
    `excluded`, with display names; every ticker of a CIK in list order."""

    if not path.is_file():
        return []
    entries = json.loads(path.read_text(encoding="utf-8"))
    names: dict[str, str] = {}
    tickers: dict[str, list[str]] = {}
    for entry in entries.values():  # file order: largest first
        cik = f"{int(entry['cik_str']):010d}"
        if cik in excluded:
            continue
        if cik not in names:
            if len(names) == SEC_TOP:
                continue
            names[cik] = display_name(entry["title"])
            tickers[cik] = []
        tickers[cik].append(entry["ticker"])
    return [
        OutsideCompany(names[cik], tuple(tickers[cik]), (), "US-listed")
        for cik in names
    ]


def outside_companies(
    universe: Mapping[str, CompanyProfile],
    bronze_dir: Path,
    count: int,
    seed: int,
    corpus_ciks: Collection[str] = (),
) -> list[OutsideCompany]:
    """`count` companies the corpus lacks: half curated foreign ones, half
    large US-listed ones outside the universe and `corpus_ciks`; never a
    universe company (by CIK, ticker or name)."""

    taken = {normalize_name(p.name) for p in universe.values()}
    symbols = {ticker_key(t) for p in universe.values() for t in p.tickers}

    def outside(company: OutsideCompany) -> bool:
        return normalize_name(company.name) not in taken and not symbols.intersection(
            ticker_key(t) for t in company.tickers
        )

    folder = latest_snapshot(bronze_dir)
    pools = (
        list(OUTSIDE_CURATED),
        _sec_outside({*universe, *corpus_ciks}, folder / "company_tickers.json"),
    )
    picked: list[OutsideCompany] = []
    for number, pool in enumerate(pools):
        wanted = (count + 1) // 2 if number == 0 else count - len(picked)
        ordered = sorted(
            (c for c in pool if outside(c)), key=lambda c: rank(seed, c.name)
        )
        picked.extend(ordered[:wanted])
    return picked

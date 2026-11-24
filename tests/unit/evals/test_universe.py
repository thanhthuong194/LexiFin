from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evals.universe import (
    OUTSIDE_CURATED,
    display_name,
    holdings_sectors,
    load_universe,
    outside_companies,
    ticker_key,
)

pytestmark = pytest.mark.unit

HOLDINGS = """iShares Core S&P 500 ETF
Fund Holdings as of,"Sep 21, 2026"
Stock,"-"

Ticker,Name,Sector,Asset Class,Weight (%)
"NVDA","NVIDIA","Information Technology","Equity","8.21"
"BRK B","BERKSHIRE HATHAWAY INC CLASS B","Financials","Equity","1.40"
"GOOG","ALPHABET CLASS C","Communication","Equity","2.00"
"ESZ6","S&P500 EMINI DEC 26","Cash and/or Derivatives","Futures","0.00"
"""
UNIVERSE = (
    ("0001045810", "NVIDIA CORP", "NVDA", ["NVDA"], 8.21),
    ("0001652044", "Alphabet Inc.", "GOOGL", ["GOOGL", "GOOG"], 5.6),
    ("0001067983", "Berkshire Hathaway", "BRK-B", ["BRK-B"], 1.4),
    ("0000000004", "Nomatch Inc.", "NOPE", ["NOPE"], 0.2),
)
SEC_TICKERS = {
    "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 1000001, "ticker": "TSM", "title": "TAIWAN SEMICONDUCTOR"},
    "2": {"cik_str": 1067983, "ticker": "BRK-A", "title": "BERKSHIRE HATHAWAY INC"},
    "3": {"cik_str": 1000002, "ticker": "KO", "title": "COCA COLA CO"},
    "4": {"cik_str": 1000002, "ticker": "KO-PB", "title": "COCA COLA CO"},
    "5": {"cik_str": 1000003, "ticker": "GOOG", "title": "Ticker Clash Co"},
}


def bronze(tmp_path: Path) -> Path:
    root = tmp_path / "bronze"
    old = root / "universe_snapshots" / "snapshot_date_20250101T000000000000Z"
    new = root / "universe_snapshots" / "snapshot_date_20260923T075749250953Z"
    empty = root / "universe_snapshots" / "snapshot_date_20270101T000000000000Z"
    for folder in (old, new, empty):
        folder.mkdir(parents=True)
    (old / "universe.jsonl").write_text("", encoding="utf-8")
    (new / "ivv_holdings.csv").write_text(HOLDINGS, encoding="utf-8")
    (new / "company_tickers.json").write_text(json.dumps(SEC_TICKERS), encoding="utf-8")
    rows = [
        {
            "cik": cik,
            "company_name": name,
            "primary_ticker": primary,
            "ticker_aliases": aliases,
            "weight_pct": weight,
        }
        for cik, name, primary, aliases, weight in UNIVERSE
    ]
    (new / "universe.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(rows)), encoding="utf-8"
    )
    return root


def test_ticker_key_drops_share_class_punctuation() -> None:
    assert ticker_key("BRK B") == ticker_key("brk.b") == ticker_key("BRK-B") == "BRKB"


def test_holdings_skip_preamble(tmp_path: Path) -> None:
    sectors = holdings_sectors(bronze(tmp_path) / "missing.csv")
    assert sectors == {}
    path = tmp_path / "h.csv"
    path.write_text(HOLDINGS, encoding="utf-8")
    assert holdings_sectors(path)["BRKB"] == "Financials"


def test_load_universe_joins_sector_and_ranks_size(tmp_path: Path) -> None:
    universe = load_universe(bronze(tmp_path))
    assert list(universe) == [row[0] for row in UNIVERSE]  # by weight
    assert universe["0001045810"].sector == "Information Technology"
    assert universe["0001067983"].sector == "Financials"  # BRK-B vs BRK B
    assert universe["0001652044"].sector == "Communication"  # via alias GOOG
    assert universe["0000000004"].sector is None
    assert [p.size_decile for p in universe.values()] == [1, 3, 6, 8]
    assert universe["0001652044"].tickers == ("GOOGL", "GOOG")


def test_outside_companies_never_in_universe(tmp_path: Path) -> None:
    root = bronze(tmp_path)
    universe = load_universe(root)
    picked = outside_companies(universe, root, 4, seed=3)
    assert picked == outside_companies(universe, root, 4, seed=3)
    curated = [c for c in picked if c.country != "US-listed"]
    listed = [c for c in picked if c.country == "US-listed"]
    assert len(curated) == 2
    assert all(c in OUTSIDE_CURATED for c in curated)
    # NVDA and BRK are universe CIKs; GOOG is a universe ticker.
    assert {c.name for c in listed} == {"Taiwan Semiconductor", "Coca Cola"}
    assert next(c for c in listed if c.tickers[0] == "KO").tickers == ("KO", "KO-PB")
    assert outside_companies(universe, root, 4, seed=4) != picked
    # Corpus companies outside the universe snapshot are excluded too.
    fewer = outside_companies(universe, root, 4, seed=3, corpus_ciks={"0001000001"})
    assert {c.name for c in fewer if c.country == "US-listed"} == {"Coca Cola"}


def test_display_names() -> None:
    assert display_name("PFIZER INC") == "Pfizer"
    assert display_name("ASML HOLDING NV") == "ASML"
    assert display_name("AMAZON.COM, INC.") == "Amazon.com"
    assert display_name("NVIDIA CORP /DE/") == "Nvidia"
    assert display_name("KLA CORP") == "KLA"
    assert display_name("BANK OF AMERICA CORP /DE/") == "Bank of America"
    assert display_name("PTT Public Company") == "PTT"
    assert display_name("Meta Platforms, Inc.") == "Meta Platforms"
    assert all(c.name == display_name(c.name) for c in OUTSIDE_CURATED)

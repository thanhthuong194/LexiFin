from __future__ import annotations

from datetime import date

import pytest

from src.data_pipeline.pipelines.native_rag.filing_metadata import (
    _canonical_names,
    _Dei,
    _label,
    _tickers,
    sec_url,
)

pytestmark = pytest.mark.unit


def test_tickers_skip_debt_and_bad_symbols() -> None:
    dei = _Dei()
    dei.by_context["c1"]["TradingSymbol"] += ["GOOGL"]
    dei.by_context["c1"]["Security12bTitle"] += ["Class A Common Stock"]
    dei.by_context["c2"]["TradingSymbol"] += ["GOOG", "goog"]
    dei.by_context["c3"]["TradingSymbol"] += ["GOOG29"]
    dei.by_context["c3"]["Security12bTitle"] += ["1.250% Notes due 2029"]
    dei.by_context["c4"]["TradingSymbol"] += ["—", "N/A 1"]
    assert _tickers(dei) == ("GOOG", "GOOGL")


def test_canonical_names_are_stable_and_keep_renames() -> None:
    rows = [
        {"filing_id": "a", "cik": "1", "filing_date": date(2022, 1, 1)},
        {"filing_id": "b", "cik": "1", "filing_date": date(2023, 1, 1)},
        {"filing_id": "c", "cik": "1", "filing_date": date(2024, 1, 1)},
        {"filing_id": "d", "cik": "1", "filing_date": date(2025, 1, 1)},
    ]
    names = {
        "a": "MICROSOFT CORP",
        "b": "Microsoft Corporation",
        "c": "MICROSOFT CORPORATION",
        "d": "Meta Platforms, Inc.",
    }
    chosen = _canonical_names(rows, names)
    # The earliest filing's spelling, even upper case: a later filing never
    # renames chunks already stored.
    assert chosen == {
        "a": "MICROSOFT CORP",
        "b": "MICROSOFT CORP",
        "c": "MICROSOFT CORP",
        "d": "Meta Platforms, Inc.",
    }
    later = [*rows, {"filing_id": "e", "cik": "1", "filing_date": date(2026, 1, 1)}]
    assert _canonical_names(later, {**names, "e": "Microsoft Corp."})["a"] == (
        "MICROSOFT CORP"
    )


def test_sec_url_and_exhibit_label() -> None:
    assert sec_url("0000320193", "0000320193-25-000071", "aapl.htm") == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000071/aapl.htm"
    )
    assert sec_url("0000320193", "0000320193-25-000071", None).endswith("071/")
    row = {"document_type": "EX-99.1", "description": "EX-99.1"}
    assert _label(row, None) == "EX-99.1"
    assert _label(row, "Press release") == "EX-99.1: Press release"
    assert _label(
        {"document_type": "EX-99.2", "description": "CFO COMMENTARY"}, None
    ) == ("EX-99.2: CFO COMMENTARY")

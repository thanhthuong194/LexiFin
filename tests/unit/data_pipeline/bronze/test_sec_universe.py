from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.data_pipeline.connectors.sec_filings.sec_edgar import parse_company_tickers
from src.data_pipeline.transforms.sec_universe import (
    build_sec_universe,
    normalize_ticker,
)


@pytest.mark.unit
def test_normalizes_maps_groups_and_ranks_unique_companies() -> None:
    holdings = pl.DataFrame(
        {
            "Ticker": ["goog", "GOOGL", "BRK.B", "AAA", "MISSING"],
            "Name": ["Alphabet C", "Alphabet A", "Berkshire", "Alpha", "Missing"],
            "Asset Class": ["Equity"] * 5,
            "Weight (%)": [1.0, 1.5, 2.5, 2.5, 50.0],
        }
    )
    mapping = parse_company_tickers(
        b"{"
        b'"0":{"cik_str":1,"ticker":"GOOG","title":"Alphabet Inc."},'
        b'"1":{"cik_str":1,"ticker":"GOOGL","title":"Alphabet Inc."},'
        b'"2":{"cik_str":2,"ticker":"BRK-B","title":"Berkshire Hathaway"},'
        b'"3":{"cik_str":3,"ticker":"AAA","title":"Alpha Corp"}'
        b"}"
    )

    result = build_sec_universe(
        holdings,
        mapping,
        snapshot_id="snapshot",
        holdings_as_of_date=date(2026, 9, 22),
        company_limit=3,
    )

    assert normalize_ticker(" brk.b ") == "BRK-B"
    assert normalize_ticker("BRK B") == "BRK-B"
    assert [member.cik for member in result.members] == [
        "0000000001",
        "0000000002",
        "0000000003",
    ]
    alphabet = result.members[0]
    assert alphabet.weight_pct == 2.5
    assert alphabet.primary_ticker == "GOOGL"
    assert alphabet.ticker_aliases == ("GOOG", "GOOGL")
    assert [member.rank for member in result.members] == [1, 2, 3]
    assert result.unmapped_tickers == ("MISSING",)


@pytest.mark.unit
def test_primary_ticker_tie_breaks_alphabetically_and_limit_is_unique_ciks() -> None:
    holdings = pl.DataFrame(
        {
            "Ticker": ["ZZZ", "AAA"],
            "Name": ["One", "One"],
            "Asset Class": ["Equity", "Equity"],
            "Weight (%)": [1.0, 1.0],
        }
    )
    mapping = pl.DataFrame(
        {
            "ticker": ["ZZZ", "AAA"],
            "company_name": ["One", "One"],
            "cik": ["0000000001", "0000000001"],
        }
    )

    result = build_sec_universe(
        holdings,
        mapping,
        snapshot_id="snapshot",
        holdings_as_of_date=date(2026, 1, 1),
        company_limit=1,
    )
    assert result.members[0].primary_ticker == "AAA"

    with pytest.raises(ValueError, match="Only 1 unique"):
        build_sec_universe(
            holdings,
            mapping,
            snapshot_id="snapshot",
            holdings_as_of_date=date(2026, 1, 1),
            company_limit=2,
        )

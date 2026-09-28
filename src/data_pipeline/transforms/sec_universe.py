from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import polars as pl

from src.data_pipeline.contracts.ingestion import UniverseMember

_TICKER_SEPARATOR = re.compile(r"[./_\s]+")


@dataclass(frozen=True)
class UniverseBuildResult:
    """Ranked SEC universe plus holdings tickers that did not map."""

    members: tuple[UniverseMember, ...]
    unmapped_tickers: tuple[str, ...]


def normalize_ticker(value: str) -> str:
    """Normalize common ticker separators for an exact SEC join."""

    return _TICKER_SEPARATOR.sub("-", value.strip().upper())


def build_sec_universe(
    holdings: pl.DataFrame,
    ticker_mapping: pl.DataFrame,
    *,
    snapshot_id: str,
    holdings_as_of_date: date,
    company_limit: int,
) -> UniverseBuildResult:
    """Build a deterministic company universe aggregated by CIK."""

    if not 1 <= company_limit <= 500:
        raise ValueError("company_limit must be between 1 and 500")

    normalized_holdings = holdings.with_columns(
        pl.col("Ticker")
        .map_elements(normalize_ticker, return_dtype=pl.String)
        .alias("ticker_normalized")
    )
    normalized_mapping = ticker_mapping.with_columns(
        pl.col("ticker")
        .map_elements(normalize_ticker, return_dtype=pl.String)
        .alias("ticker_normalized")
    ).unique(subset=["ticker_normalized"], keep="first")

    joined = normalized_holdings.join(
        normalized_mapping,
        on="ticker_normalized",
        how="left",
    )
    unmapped = tuple(
        joined.filter(pl.col("cik").is_null())["ticker_normalized"]
        .drop_nulls()
        .unique()
        .sort()
        .to_list()
    )
    mapped = joined.filter(pl.col("cik").is_not_null())

    rows = mapped.select(
        "cik",
        "company_name",
        pl.col("ticker_normalized").alias("ticker"),
        pl.col("Weight (%)").alias("weight_pct"),
    ).to_dicts()

    companies: dict[str, dict[str, object]] = {}
    for row in rows:
        cik = str(row["cik"])
        ticker = str(row["ticker"])
        weight = float(row["weight_pct"])
        company = companies.setdefault(
            cik,
            {
                "company_name": str(row["company_name"]),
                "weights": {},
            },
        )
        weights = company["weights"]
        assert isinstance(weights, dict)
        weights[ticker] = float(weights.get(ticker, 0.0)) + weight

    ranked: list[tuple[str, str, str, tuple[str, ...], float]] = []
    for cik, company in companies.items():
        weights = company["weights"]
        assert isinstance(weights, dict)
        aliases = tuple(sorted(str(ticker) for ticker in weights))
        primary = min(aliases, key=lambda ticker: (-float(weights[ticker]), ticker))
        ranked.append(
            (
                cik,
                str(company["company_name"]),
                primary,
                aliases,
                sum(float(weight) for weight in weights.values()),
            )
        )

    ranked.sort(key=lambda item: (-item[4], item[0]))
    if len(ranked) < company_limit:
        raise ValueError(
            f"Only {len(ranked)} unique mapped CIKs are available; "
            f"company_limit={company_limit}"
        )

    members = tuple(
        UniverseMember(
            snapshot_id=snapshot_id,
            holdings_as_of_date=holdings_as_of_date,
            rank=rank,
            cik=cik,
            company_name=company_name,
            primary_ticker=primary_ticker,
            ticker_aliases=aliases,
            weight_pct=weight,
        )
        for rank, (cik, company_name, primary_ticker, aliases, weight) in enumerate(
            ranked[:company_limit], start=1
        )
    )
    return UniverseBuildResult(members=members, unmapped_tickers=unmapped)

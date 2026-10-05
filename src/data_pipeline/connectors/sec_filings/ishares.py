from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime

import httpx
import polars as pl

IVV_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund"
)
IVV_PRODUCT_URL = "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf"
IVV_LATEST_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239726/"
    "ishares-core-s-p-500-etf/latest-holdings.csv"
)
ISHARES_HEADERS = {
    "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8",
    "Referer": IVV_PRODUCT_URL,
    "User-Agent": "Mozilla/5.0 (compatible; LexiFin/0.1; IVV holdings ingestion)",
}
REQUIRED_COLUMNS = ("Ticker", "Name", "Asset Class", "Weight (%)")


@dataclass(frozen=True)
class ParsedIVVHoldings:
    """Parsed IVV equity holdings and their source date."""

    holdings_as_of_date: date
    rows: pl.DataFrame


def _get_holdings(client: httpx.Client) -> bytes:
    response = client.get(IVV_HOLDINGS_URL, headers=ISHARES_HEADERS)
    if response.status_code in {403, 404} or not _is_holdings_csv(response):
        response = client.get(IVV_LATEST_HOLDINGS_URL, headers=ISHARES_HEADERS)
    response.raise_for_status()
    if not _is_holdings_csv(response):
        content_type = response.headers.get("content-type", "unknown")
        raise ValueError(
            f"iShares returned a non-holdings payload (content-type={content_type!r})"
        )
    return response.content


def _is_holdings_csv(response: httpx.Response) -> bool:
    content_type = response.headers.get("content-type", "").lower()
    contents = response.content
    return (
        "html" not in content_type
        and b"Fund Holdings as of" in contents
        and b"Ticker" in contents
    )


def download_ivv_holdings(client: httpx.Client | None = None) -> bytes:
    """Download the unmodified IVV holdings CSV bytes."""

    if client is not None:
        return _get_holdings(client)

    with httpx.Client(follow_redirects=True, timeout=30.0) as owned_client:
        return _get_holdings(owned_client)


def _parse_holdings_date(rows: list[list[str]]) -> date:
    for row in rows:
        for index, value in enumerate(row):
            if value.strip().lower() != "fund holdings as of":
                continue
            candidates = row[index + 1 :]
            for candidate in candidates:
                cleaned = candidate.strip()
                for date_format in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
                    try:
                        # Only the calendar date is kept, so no timezone applies.
                        return datetime.strptime(cleaned, date_format).date()  # noqa: DTZ007
                    except ValueError:
                        continue
    raise ValueError("Could not parse 'Fund Holdings as of' from IVV CSV")


def parse_ivv_holdings(raw_csv: bytes) -> ParsedIVVHoldings:
    """Parse raw IVV CSV bytes using a dynamically discovered header."""

    try:
        text = raw_csv.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("IVV holdings CSV is not valid UTF-8") from error

    lines = text.splitlines()
    parsed_rows = list(csv.reader(lines))
    holdings_date = _parse_holdings_date(parsed_rows)

    header_index = next(
        (
            index
            for index, row in enumerate(parsed_rows)
            if any(value.strip() == "Ticker" for value in row)
        ),
        None,
    )
    if header_index is None:
        raise ValueError("Could not find a CSV header containing 'Ticker'")

    table_bytes = "\n".join(lines[header_index:]).encode()
    try:
        table = pl.read_csv(
            io.BytesIO(table_bytes),
            schema_overrides={column: pl.String for column in REQUIRED_COLUMNS},
            truncate_ragged_lines=True,
        )
    except pl.exceptions.PolarsError as error:
        raise ValueError("Could not parse the IVV holdings table") from error

    missing = sorted(set(REQUIRED_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"IVV holdings is missing required columns: {missing}")

    table = table.with_columns(
        pl.col("Ticker").str.strip_chars(),
        pl.col("Name").str.strip_chars(),
        pl.col("Asset Class").str.strip_chars(),
        pl.col("Weight (%)")
        .str.replace_all(",", "")
        .str.replace_all("%", "")
        .str.strip_chars()
        .cast(pl.Float64, strict=False),
    ).filter(pl.col("Asset Class").str.to_uppercase() == "EQUITY")

    if table.is_empty():
        raise ValueError("IVV holdings contains no Equity rows")
    if table["Weight (%)"].null_count():
        raise ValueError("IVV holdings contains an invalid Equity weight")

    return ParsedIVVHoldings(
        holdings_as_of_date=holdings_date,
        rows=table.select(REQUIRED_COLUMNS),
    )

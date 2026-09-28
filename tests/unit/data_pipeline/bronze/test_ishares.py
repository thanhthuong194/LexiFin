from __future__ import annotations

import httpx
import pytest

from src.data_pipeline.connectors.sec_filings.ishares import (
    IVV_HOLDINGS_URL,
    IVV_LATEST_HOLDINGS_URL,
    download_ivv_holdings,
    parse_ivv_holdings,
)


def _csv(metadata_lines: int = 2) -> bytes:
    metadata = [f'"Metadata {index}","value"' for index in range(metadata_lines)]
    return (
        "\n".join(
            [
                '"Fund Holdings as of","Sep 22, 2026"',
                *metadata,
                '"Ticker","Name","Asset Class","Weight (%)","Market Value"',
                '"AAA","Alpha","Equity","1.25","100"',
                '"BND","Bond","Fixed Income","9.0","200"',
                '"BBB","Beta","Equity","0.75%","50"',
            ]
        )
        + "\n"
    ).encode()


@pytest.mark.unit
@pytest.mark.parametrize("metadata_lines", [0, 1, 5])
def test_parses_dynamic_header_date_equities_and_weights(metadata_lines: int) -> None:
    parsed = parse_ivv_holdings(_csv(metadata_lines))

    assert parsed.holdings_as_of_date.isoformat() == "2026-09-22"
    assert parsed.rows["Ticker"].to_list() == ["AAA", "BBB"]
    assert parsed.rows["Weight (%)"].to_list() == [1.25, 0.75]


@pytest.mark.unit
def test_rejects_missing_header_required_columns_and_empty_equities() -> None:
    with pytest.raises(ValueError, match="header"):
        parse_ivv_holdings(b'"Fund Holdings as of","Sep 22, 2026"\nA,B\n')

    with pytest.raises(ValueError, match="required columns"):
        parse_ivv_holdings(
            b'"Fund Holdings as of","Sep 22, 2026"\nTicker,Name\nAAA,Alpha\n'
        )

    with pytest.raises(ValueError, match="no Equity"):
        parse_ivv_holdings(
            b'"Fund Holdings as of","Sep 22, 2026"\n'
            b"Ticker,Name,Asset Class,Weight (%)\nBND,Bond,Fixed Income,1\n"
        )


@pytest.mark.unit
def test_download_falls_back_when_legacy_url_returns_html() -> None:
    csv_bytes = _csv()

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == IVV_HOLDINGS_URL:
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b"<!DOCTYPE html><html></html>",
                request=request,
            )
        assert str(request.url) == IVV_LATEST_HOLDINGS_URL
        return httpx.Response(
            200,
            headers={"content-type": "text/csv"},
            content=csv_bytes,
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert download_ivv_holdings(client) == csv_bytes

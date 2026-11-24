"""8-K windows, in-progress years and filing months, on the real dates of
the eval failures (v1-t0056, v1-t0058, v1-t0100) in an embedded Qdrant."""

from __future__ import annotations

from datetime import date

import pytest
from qdrant_client import QdrantClient, models

from src.rag_engine.retrieval.search import (
    PeriodAsk,
    SearchFilter,
    Window,
    fallback_window,
    month_window,
    period_windows,
    qdrant_filter,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.filterwarnings("ignore:Payload indexes have no effect"),
]

NVIDIA, APPLE, AMAZON = "0001045810", "0000320193", "0001018724"
COLLECTION = "chunks"
# (accession, cik, form, fiscal year, fiscal period, period end, filed)
FILINGS = (
    ("0001045810-23-000039", NVIDIA, "8-K", None, None, date(2023, 3, 2), date(2023, 3, 8)),
    ("nvda-8k-2022-12", NVIDIA, "8-K", None, None, date(2022, 12, 20), date(2022, 12, 20)),
    ("nvda-10k-fy2024", NVIDIA, "10-K", 2024, "FY", date(2024, 1, 28), date(2024, 2, 21)),
    ("0001045810-25-000209", NVIDIA, "10-Q", 2026, "Q2", date(2025, 7, 27), date(2025, 8, 27)),
    ("0001045810-26-000021", NVIDIA, "10-K", 2026, "FY", date(2026, 1, 25), date(2026, 2, 25)),
    ("0001045810-26-000052", NVIDIA, "10-Q", 2027, "Q1", date(2026, 4, 26), date(2026, 5, 20)),
    ("0001045810-26-000075", NVIDIA, "10-Q", 2027, "Q2", date(2026, 7, 26), date(2026, 8, 26)),
    ("0000320193-25-000079", APPLE, "10-K", 2025, "FY", date(2025, 9, 27), date(2025, 10, 31)),
    ("0000320193-26-000006", APPLE, "10-Q", 2026, "Q1", date(2025, 12, 27), date(2026, 1, 30)),
    ("0000320193-26-000020", APPLE, "10-Q", 2026, "Q3", date(2026, 6, 27), date(2026, 7, 31)),
    ("amzn-10k-fy2021", AMAZON, "10-K", 2021, "FY", date(2021, 12, 31), date(2022, 2, 4)),
    ("0001018724-22-000002", AMAZON, "8-K", None, None, date(2022, 2, 3), date(2022, 2, 3)),
    ("0001104659-22-045579", AMAZON, "8-K", None, None, date(2022, 4, 14), date(2022, 4, 14)),
    ("amzn-8k-2022-05", AMAZON, "8-K", None, None, date(2022, 5, 26), date(2022, 5, 26)),
)  # fmt: skip


def _rfc3339(day: date) -> str:
    return f"{day.isoformat()}T00:00:00Z"


@pytest.fixture(scope="module")
def client() -> QdrantClient:
    qdrant = QdrantClient(":memory:")
    qdrant.create_collection(COLLECTION, vectors_config={})
    for key in ("filing_date", "period_end_date"):
        qdrant.create_payload_index(COLLECTION, key, models.PayloadSchemaType.DATETIME)
    qdrant.upsert(
        COLLECTION,
        [
            models.PointStruct(
                id=i,
                vector={},
                payload={
                    "accession_number": accession,
                    "cik": cik,
                    "form": form,
                    "fiscal_year": year,
                    "fiscal_period": period,
                    "period_end_date": _rfc3339(end),
                    "filing_date": _rfc3339(filed),
                },
            )
            for i, (accession, cik, form, year, period, end, filed) in enumerate(
                FILINGS
            )
        ],
    )
    return qdrant


def _windowed(
    client: QdrantClient, search: SearchFilter, *, events: bool
) -> SearchFilter:
    windows = period_windows(client, COLLECTION, search.ciks, search, events=events)
    return SearchFilter(
        search.ciks,
        search.forms,
        search.fiscal_years,
        search.fiscal_periods,
        calendar=search.calendar,
        windows=windows,
        filed_months=search.filed_months,
    )


def _matched(client: QdrantClient, search: SearchFilter) -> set[str]:
    points, _ = client.scroll(
        COLLECTION, scroll_filter=qdrant_filter(search), limit=100, with_payload=True
    )
    return {str(p.payload["accession_number"]) for p in points if p.payload}


def test_calendar_year_8k_events_cover_that_year(client: QdrantClient) -> None:
    # v1-t0056: "NVIDIA's March 2023 bylaw amendment" -> calendar 2023, 8-K.
    # v1 windowed by FY2024 (ends 2024-01-28) from 2023-03-13: missed it.
    search = _windowed(
        client,
        SearchFilter((NVIDIA,), ("8-K",), (2023,), calendar=True),
        events=True,
    )
    assert search.windows == (
        Window(date(2023, 1, 1), date(2024, 4, 29), year=2023, cik=NVIDIA),
    )
    assert _matched(client, search) == {"0001045810-23-000039"}


def test_fiscal_year_8k_events_reach_the_shareholder_letter(
    client: QdrantClient,
) -> None:
    # v1-t0100: Amazon's 2021 letter, 8-K filed 2022-04-14 (FY ended 2021-12-31).
    search = _windowed(client, SearchFilter((AMAZON,), ("8-K",), (2021,)), events=True)
    (window,) = search.windows
    assert window.start <= date(2021, 1, 1) and window.end == date(2022, 4, 30)
    assert _matched(client, search) == {
        "0001018724-22-000002",
        "0001104659-22-045579",
    }


def test_annual_results_window_is_unchanged_without_events(
    client: QdrantClient,
) -> None:
    search = _windowed(client, SearchFilter((AMAZON,), (), (2021,)), events=False)
    assert search.windows == (
        Window(date(2021, 11, 14), date(2022, 4, 10), year=2021, cik=AMAZON),
    )
    assert _matched(client, search) == {"amzn-10k-fy2021", "0001018724-22-000002"}


def test_filed_month_admits_that_month_only(client: QdrantClient) -> None:
    # "8-K filed in April 2022": no year asked, the month alone filters.
    search = SearchFilter((AMAZON,), ("8-K",), filed_months=("2022-04",))
    assert search.has_period
    assert _matched(client, search) == {"0001104659-22-045579"}
    assert search.without_period().filed_months == ()


def test_filed_month_adds_to_the_asked_year(client: QdrantClient) -> None:
    # v1-t0100 as planned: fiscal 2021 + "filed with an April 2022 8-K",
    # with a narrow (non-events) year window that ends 2022-04-10.
    search = _windowed(
        client,
        SearchFilter((AMAZON,), (), (2021,), filed_months=("2022-04",)),
        events=False,
    )
    assert _matched(client, search) == {
        "amzn-10k-fy2021",
        "0001018724-22-000002",
        "0001104659-22-045579",
    }


def test_month_window() -> None:
    assert month_window("2022-04") == Window(date(2022, 4, 1), date(2022, 5, 7))
    assert month_window("2025-12") == Window(date(2025, 12, 1), date(2026, 1, 7))
    assert month_window("April 2022") is None
    assert month_window("2022-13") is None
    assert not SearchFilter(filed_months=("2022-13",)).has_period


def test_year_in_progress_reads_its_10qs(client: QdrantClient) -> None:
    # v1-t0058: NVIDIA FY2027 has no 10-K yet; its 10-Qs (to 2026-07-26)
    # stand in, with an 8-K window from the real quarter ends. FY2026 has
    # its 10-K, so its 10-Qs stay out.
    search = _windowed(client, SearchFilter((NVIDIA,), (), (2026, 2027)), events=False)
    assert search.windows == (
        Window(date(2025, 12, 9), date(2026, 5, 5), year=2026, cik=NVIDIA),
        Window(
            date(2026, 3, 10), date(2026, 11, 3), year=2027, cik=NVIDIA,
            in_progress=True,
        ),
    )  # fmt: skip
    assert _matched(client, search) == {
        "0001045810-26-000021",
        "0001045810-26-000052",
        "0001045810-26-000075",
    }


def test_apple_nine_months_of_a_year_in_progress(client: QdrantClient) -> None:
    # v1-t0058, Apple's side: FY2026 to 2026-06-27 is only in its Q3 10-Q.
    search = _windowed(client, SearchFilter((APPLE,), (), (2026,)), events=False)
    assert [w.in_progress for w in search.windows] == [True]
    assert _matched(client, search) == {
        "0000320193-26-000006",
        "0000320193-26-000020",
    }


def test_calendar_year_in_progress_reads_its_10qs(client: QdrantClient) -> None:
    # Calendar 2026 = NVIDIA's FY2027 (ends January 2027): its 10-Qs so far.
    search = _windowed(
        client, SearchFilter((NVIDIA,), (), (2026,), calendar=True), events=False
    )
    assert _matched(client, search) == {
        "0001045810-26-000052",
        "0001045810-26-000075",
    }


def test_unknown_year_falls_back_without_a_company() -> None:
    # Fiscal years named Y start from late January Y-1 (NVIDIA FY2024:
    # 2023-01-30) and end by early February Y+1 (retailers).
    events = fallback_window(PeriodAsk(2024, "FY", calendar=False), events=True)
    assert events == Window(date(2023, 1, 20), date(2025, 6, 5))
    calendar = fallback_window(PeriodAsk(2023, "Q1", calendar=True), events=True)
    assert calendar == Window(date(2023, 1, 1), date(2023, 7, 9))

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest
from qdrant_client import QdrantClient

from src.data_pipeline.contracts.native_rag.chunks import ChunkRecord, chunk_sha256
from src.llm import LLMResponseError, RerankResult
from src.rag_engine.embedding.pipeline import run_native_rag_embeddings
from src.rag_engine.indexing.pipeline import LoadRequest, run_qdrant_load
from src.rag_engine.indexing.qdrant_store import SEC_FILINGS, switch_alias
from src.rag_engine.retrieval.query import QueryAnalysis
from src.rag_engine.retrieval.retriever import (
    Models,
    RetrievalResult,
    Retriever,
    RetrieverSettings,
)
from tests.llm_support import FakeChat, FakeReranker, reply
from tests.native_rag_support import NOW, FakeEmbedder, gold_chunk, write_chunks

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore:Payload indexes have no effect"),
    pytest.mark.filterwarnings("ignore:Local mode performs exact"),
]

ALPHA, BETA = "0000000001", "0000000002"
TODAY = date(2026, 11, 17)


def _chunk(
    document: str,
    ordinal: int,
    count: int,
    text: str,
    **fields: Any,
) -> ChunkRecord:
    filed = fields.pop("filed", date(2024, 11, 1))
    base = gold_chunk(
        document, ordinal, count, filed, text, cik=fields.pop("cik", ALPHA)
    )
    context = fields.pop(
        "context", f"{fields.get('company_name', 'Alpha Corp')} | {text[:20]}"
    )
    update: dict[str, Any] = {
        "company_name": "Alpha Corp",
        "tickers": ["ALP"],
        "filing_id": f"sec:{document}",
        "context": context,
        "chunk_sha256": chunk_sha256(context, text),
        "token_count": 100,
    } | fields
    return base.model_copy(update=update)


@dataclass(frozen=True)
class Filing:
    form: str
    year: int
    period: str
    end: date
    filed: date


def _periodic(
    document: str, filing: Filing, texts: Sequence[str], **fields: Any
) -> list[ChunkRecord]:
    return [
        _chunk(document, i, len(texts), text, form=filing.form,
               document_type=filing.form, fiscal_year=filing.year,
               fiscal_period=filing.period, period_end_date=filing.end,
               filed=filing.filed, **fields)
        for i, text in enumerate(texts, start=1)
    ]  # fmt: skip


def _event(
    document: str, filed: date, texts: Sequence[str], **fields: Any
) -> list[ChunkRecord]:
    fields = {
        "form": "8-K",
        "document_type": "8-K",
        "section_topic": "8k_event",
    } | fields
    return [
        _chunk(document, i, len(texts), text, fiscal_year=None, fiscal_period=None,
               period_end_date=None, filed=filed, **fields)
        for i, text in enumerate(texts, start=1)
    ]  # fmt: skip


def _corpus() -> list[ChunkRecord]:
    table = [
        _chunk("a10k24", i, 5, f"Revenue by segment table rows {i}", form="10-K",
               fiscal_year=2024, fiscal_period="FY", period_end_date=date(2024, 9, 28),
               chunk_kind="table", table_id="a10k24:table:1", part=i - 1, part_count=3,
               section_topic="financial_statements")
        for i in (2, 3, 4)
    ]  # fmt: skip
    annual = _periodic(
        "a10k24", Filing("10-K", 2024, "FY", date(2024, 9, 28), date(2024, 11, 1)),
        ["Overview of net sales growth", "x", "x", "x", "Liquidity and capital resources"],
    )  # fmt: skip
    annual = [annual[0], *table, annual[4]]
    quarter = _periodic(
        "a10q25", Filing("10-Q", 2025, "Q1", date(2024, 12, 28), date(2025, 1, 31)),
        ["First quarter net sales rose", "Forward-looking statements boilerplate"],
    )  # fmt: skip
    old_quarter = _periodic(
        "a10q23", Filing("10-Q", 2023, "Q1", date(2022, 12, 31), date(2023, 2, 3)),
        ["First quarter net sales fell", "Forward-looking statements boilerplate"],
    )  # fmt: skip
    results_8k = _event(
        "a8k25",
        date(2025, 1, 30),
        ["Item 2.02 results of operations", "Results table", "Guidance",
         "Item 2.02 closing remarks"],
    )  # fmt: skip
    q4_8k = _event("a8k24", date(2024, 10, 30), ["Fourth quarter results release"])
    old_8k = _event("a8k23", date(2023, 5, 1), ["Item 5.02 officer departure"])
    beta = _periodic(
        "b10k24", Filing("10-K", 2024, "FY", date(2024, 12, 31), date(2025, 2, 10)),
        ["Beta net sales by segment", "Beta liquidity"],
        cik=BETA, company_name="Beta Inc", tickers=["BET"],
    )  # fmt: skip
    return [*annual, *quarter, *old_quarter, *results_8k, *q4_8k, *old_8k, *beta]


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> QdrantClient:
    gold = tmp_path_factory.mktemp("gold")
    write_chunks(gold, _corpus())
    assert run_native_rag_embeddings(gold, FakeEmbedder(), NOW).status == "completed"
    qdrant = QdrantClient(":memory:")
    load = run_qdrant_load(gold, qdrant, LoadRequest(model_id=FakeEmbedder.model))
    assert load.status == "completed" and load.companies == 2
    switch_alias(qdrant, load.collection, SEC_FILINGS.name)
    return qdrant


def _analysis(**overrides: Any) -> str:
    values: dict[str, Any] = {
        "language": "en",
        "corpora": ["sec_filings"],
        "searches": [],
        "companies": [{"name": "Alpha Corp", "ticker": "ALP"}],
        "fiscal_years": [],
        "fiscal_periods": [],
        "forms": [],
        "topics": [],
        "wants_latest": False,
        "period_basis": "fiscal",
    }
    return QueryAnalysis.model_validate(values | overrides).model_dump_json()


@dataclass
class BrokenReranker:
    model: str = "fake-reranker"
    calls: int = 0

    def rerank(
        self, query: str, documents: Sequence[str], top_n: int | None = None
    ) -> list[RerankResult]:
        self.calls += 1
        raise LLMResponseError("rerank HTTP 503")


def _retrieve(
    client: QdrantClient,
    question: str,
    analysis: str | None = None,
    *,
    reranker: Any = None,
    settings: RetrieverSettings | None = None,
) -> RetrievalResult:
    chat = FakeChat([reply(analysis)]) if analysis is not None else None
    retriever = Retriever(
        client,
        Models(FakeEmbedder(), chat, reranker or FakeReranker()),
        settings=settings or RetrieverSettings(min_results=1),
        today=TODAY,
    )
    return retriever.retrieve(question)


def _chunks(result: RetrievalResult) -> set[str]:
    return {chunk for p in result.passages for chunk in p.chunk_ids}


def test_company_and_quarter_filters_reach_10q_and_8k_window(
    client: QdrantClient,
) -> None:
    result = _retrieve(
        client,
        "Alpha first quarter 2025 net sales results",
        _analysis(fiscal_years=[2025], fiscal_periods=["Q1"]),
    )
    assert {p.cik for p in result.passages} == {ALPHA}
    documents = {chunk.split(":chunk:")[0] for chunk in _chunks(result)}
    # 10-Q of FY2025 Q1 and the 8-K filed in its window; not FY2023 or old 8-Ks.
    assert documents == {"a10q25", "a8k25"}
    assert result.filters["sec_filings"][:4] == [
        f"cik {ALPHA}",
        "fiscal periods",
        "fiscal_year 2025",
        "fiscal_period Q1",
    ]
    assert result.companies == [
        {"corpus": "sec_filings", "cik": ALPHA, "company_name": "Alpha Corp"}
    ]
    assert result.ranking and all(r["rerank_score"] is not None for r in result.ranking)


def test_table_parts_and_short_8k_item_expand(client: QdrantClient) -> None:
    result = _retrieve(
        client,
        "Revenue by segment table rows 3",
        _analysis(fiscal_years=[2024], forms=["10-K"]),
        settings=RetrieverSettings(min_results=1, top_k=1),
    )
    passage = result.passages[0]
    assert passage.hit_chunk_ids == ("a10k24:chunk:000003",)
    # Every part of the split table, nothing outside it.
    assert passage.chunk_ids == tuple(f"a10k24:chunk:00000{i}" for i in (2, 3, 4))
    assert "[" not in passage.text.split("\n\n")[0]
    item = _retrieve(
        client,
        "Item 2.02 results of operations",
        _analysis(forms=["8-K"]),
        settings=RetrieverSettings(min_results=1, top_k=1),
    )
    # The whole 4-chunk item, not just the hit's neighbour.
    assert item.passages[0].chunk_ids == tuple(
        f"a8k25:chunk:00000{i}" for i in range(1, 5)
    )


def test_budget_keeps_selected_chunks_before_context(client: QdrantClient) -> None:
    result = _retrieve(
        client,
        "Revenue by segment table rows 3",
        _analysis(fiscal_years=[2024], forms=["10-K"]),
        settings=RetrieverSettings(min_results=1, top_k=2, context_tokens=150),
    )
    # Two selected chunks of 100 tokens each: the first always fits, the
    # second and every context chunk exceed the budget.
    assert result.context_tokens == 100 and len(result.passages) == 1


def test_missing_period_relaxes_but_keeps_company(client: QdrantClient) -> None:
    result = _retrieve(client, "Alpha net sales 2031", _analysis(fiscal_years=[2031]))
    assert any("period filter dropped" in n for n in result.notes)
    assert result.passages and {p.cik for p in result.passages} == {ALPHA}


def test_company_outside_the_corpus_returns_nothing(client: QdrantClient) -> None:
    result = _retrieve(
        client,
        "Gamma revenue",
        _analysis(companies=[{"name": "Gamma Holdings", "ticker": "GAM"}]),
    )
    assert result.passages == [] and result.unresolved_companies == [
        "Gamma Holdings (GAM)"
    ]
    assert any("not searched" in n for n in result.notes)


def test_comparison_gives_each_company_a_share(client: QdrantClient) -> None:
    result = _retrieve(
        client,
        "Compare net sales by segment",
        _analysis(companies=[{"name": "Alpha Corp", "ticker": "ALP"},
                             {"name": "Beta Inc", "ticker": "BET"}],
                  fiscal_years=[2024]),
        settings=RetrieverSettings(min_results=1, top_k=2),
    )  # fmt: skip
    assert {p.cik for p in result.passages} == {ALPHA, BETA}


def test_rules_fallback_finds_ticker_and_dedupes(client: QdrantClient) -> None:
    result = _retrieve(client, "ALP forward-looking statements boilerplate")
    assert result.plan.source == "rules"
    assert result.companies[0]["cik"] == ALPHA
    assert any("same-text candidates collapsed" in n for n in result.notes)
    hits = {chunk for p in result.passages for chunk in p.hit_chunk_ids}
    # One copy, the newest filing's (no period asked).
    assert "a10q25:chunk:000002" in hits and "a10q23:chunk:000002" not in hits


def test_latest_restricts_to_recent_filings(client: QdrantClient) -> None:
    result = _retrieve(client, "Alpha latest net sales", _analysis(wants_latest=True))
    assert "latest: filed since 2023-12-28" in result.filters["sec_filings"]
    assert all(p.filing_date >= "2023-12-28" for p in result.passages)


def test_reranker_failure_falls_back_to_fused_order(client: QdrantClient) -> None:
    broken = BrokenReranker()
    chat = FakeChat([reply(_analysis()), reply(_analysis())])
    retriever = Retriever(
        client,
        Models(FakeEmbedder(), chat, broken),
        settings=RetrieverSettings(min_results=1),
        today=TODAY,
    )
    first = retriever.retrieve("Alpha liquidity")
    assert first.passages and any("reranker failed" in n for n in first.notes)
    assert all(r["rerank_score"] is None for r in first.ranking)
    second = retriever.retrieve("Alpha liquidity")
    # Cooling down: not called again yet.
    assert broken.calls == 1 and any("cooling down" in n for n in second.notes)


def test_missing_alias_is_reported() -> None:
    with pytest.raises(ValueError, match="no corpus to search"):
        Retriever(QdrantClient(":memory:"), Models(FakeEmbedder()))


def test_q4_reads_the_10k_and_8ks_filed_after_the_year_end(
    client: QdrantClient,
) -> None:
    result = _retrieve(
        client,
        "Alpha fourth quarter 2024 results",
        _analysis(fiscal_years=[2024], fiscal_periods=["Q4"]),
    )
    documents = {chunk.split(":chunk:")[0] for chunk in _chunks(result)}
    # FY ended 2024-09-28: its 10-K and the 8-K filed 2024-10-30; the 8-K of
    # 2025-01-30 belongs to the next quarter.
    assert documents == {"a10k24", "a8k24"}


def test_quarter_without_year_keeps_8ks(client: QdrantClient) -> None:
    result = _retrieve(
        client, "Alpha first quarter results release", _analysis(fiscal_periods=["Q1"])
    )
    documents = {chunk.split(":chunk:")[0] for chunk in _chunks(result)}
    assert {"a10q25", "a10q23"} <= documents
    assert documents & {"a8k25", "a8k24", "a8k23"}


def test_relaxation_drops_period_then_form(client: QdrantClient) -> None:
    result = _retrieve(
        client,
        "Alpha quarterly report 2024",
        _analysis(fiscal_years=[2024], forms=["10-Q"]),
        settings=RetrieverSettings(min_results=5),
    )
    assert any("period and form filter dropped" in n for n in result.notes)
    assert {p.cik for p in result.passages} == {ALPHA}


@dataclass
class TopTwoReranker:
    model: str = "fake-reranker"

    def rerank(
        self, query: str, documents: Sequence[str], top_n: int | None = None
    ) -> list[RerankResult]:
        return FakeReranker().rerank(query, documents)[:2]


def test_unscored_candidates_rank_below_scored_ones(client: QdrantClient) -> None:
    result = _retrieve(
        client, "Alpha net sales", _analysis(), reranker=TopTwoReranker()
    )
    scores = [r["score"] for r in result.ranking]
    assert [r["rerank_score"] is not None for r in result.ranking][:3] == [
        True,
        True,
        False,
    ]
    assert all(math.isfinite(s) for s in scores) and scores == sorted(
        scores, reverse=True
    )


def test_comparison_raises_top_k(client: QdrantClient) -> None:
    result = _retrieve(
        client,
        "Compare net sales",
        _analysis(companies=[{"name": "Alpha Corp", "ticker": "ALP"},
                             {"name": "Beta Inc", "ticker": "BET"}]),
        settings=RetrieverSettings(min_results=1, top_k=1, neighbor_window=0),
    )  # fmt: skip
    # A comparison gets at least 2 chunks per company, whatever top_k says.
    hits = {h for p in result.passages for h in p.hit_chunk_ids}
    assert len(hits) == 4 and {p.cik for p in result.passages} == {ALPHA, BETA}


def test_calendar_year_and_quarter_match_by_period_end(client: QdrantClient) -> None:
    # Alpha's FY2024 ends 2024-09-28: calendar 2024 is its FY2024 10-K.
    year = _retrieve(
        client,
        "Alpha net sales in 2024",
        _analysis(fiscal_years=[2024], period_basis="calendar"),
    )
    documents = {chunk.split(":chunk:")[0] for chunk in _chunks(year)}
    assert "a10k24" in documents and "a10q25" not in documents
    assert "calendar periods" in year.filters["sec_filings"]
    # Calendar Q4 2024 ends 2024-12-31: the 10-Q ending 2024-12-28 (fiscal Q1).
    quarter = _retrieve(
        client,
        "Alpha net sales October-December 2024",
        _analysis(fiscal_years=[2024], fiscal_periods=["Q4"], period_basis="calendar"),
    )
    documents = {chunk.split(":chunk:")[0] for chunk in _chunks(quarter)}
    assert "a10q25" in documents and "a10k24" not in documents


def test_latest_with_a_form_reads_the_newest_filing_of_it(client: QdrantClient) -> None:
    result = _retrieve(
        client,
        "Alpha latest quarterly report",
        _analysis(wants_latest=True, forms=["10-Q"]),
    )
    assert {chunk.split(":chunk:")[0] for chunk in _chunks(result)} == {"a10q25"}
    assert "latest: filed since 2025-01-31" in result.filters["sec_filings"]


def test_quarter_without_year_windows_8ks_by_each_period_end(
    client: QdrantClient,
) -> None:
    result = _retrieve(
        client, "Alpha fourth quarter results release", _analysis(fiscal_periods=["Q4"])
    )
    documents = {chunk.split(":chunk:")[0] for chunk in _chunks(result)}
    # Q4 = the FY 10-K (ended 2024-09-28); 8-Ks only near that end: the one
    # of 2024-10-30, not those of 2025-01-30 or 2023-05-01.
    assert documents == {"a10k24", "a8k24"}


def test_latest_release_follows_the_ex99_topic_not_the_newest_8k(
    client: QdrantClient,
) -> None:
    result = _retrieve(
        client,
        "Alpha latest earnings release",
        _analysis(wants_latest=True, forms=["8-K"], topics=["ex99"]),
    )
    # No EX-99 in the fixture: falls back to the 400-day window, not "newest 8-K".
    assert not any(
        f.startswith("latest: filed since 2025-01-30")
        for f in result.filters["sec_filings"]
    )

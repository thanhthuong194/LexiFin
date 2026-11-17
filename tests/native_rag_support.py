from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

from src.data_pipeline.contracts.native_rag.chunks import (
    CHUNKS_ARROW_SCHEMA,
    ChunkRecord,
    chunk_id_for,
    chunk_sha256,
)
from src.data_pipeline.storage.iceberg import append_rows, open_gold_catalog
from src.data_pipeline.storage.native_rag.chunks import CHUNKS_SPEC, CHUNKS_TABLE

NOW = datetime(2026, 11, 17, tzinfo=UTC)

_WORD = re.compile(r"\S+\s*")


class WordCounter:
    """A `TokenCounter` for tests: one token per whitespace-separated word,
    plus one end-of-text token, like the real tokenizer's template."""

    identity = "test-words"

    def count(self, text: str) -> int:
        return len(text.split()) + 1

    def split(self, text: str, max_tokens: int) -> list[str]:
        words = _WORD.findall(text)
        head = text[: len(text) - len(text.lstrip())]
        pieces = [
            "".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)
        ]
        if pieces and head:
            pieces[0] = head + pieces[0]
        return pieces or [text]


class DigitCounter(WordCounter):
    """A `TokenCounter` for tests where digits are expensive (5 tokens each on
    top of one per word), so `(part 12/13)` costs well more than `(part 9/9)`:
    the real tokenizer also spends one token per digit."""

    identity = "test-digits"

    def count(self, text: str) -> int:
        return super().count(text) + 5 * sum(ch.isdigit() for ch in text)


class FakeEmbedder:
    """Deterministic 8-dim vectors from the text hash; counts texts embedded."""

    model = "test/fake-embedding"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [
            [b - 127.5 for b in hashlib.sha256(text.encode()).digest()[:8]]
            for text in texts
        ]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def gold_chunk(
    document: str,
    ordinal: int,
    count: int,
    filed: date,
    text: str,
    *,
    cik: str = "0000320193",
    period_end: date | None = None,
) -> ChunkRecord:
    context = f"Example Corp | 10-K | FY{filed.year}"
    return ChunkRecord(
        chunk_id=chunk_id_for(document, ordinal),
        filing_id="sec:1",
        document_id=document,
        section_id=f"{document}:section:0001",
        table_id=None,
        block_ids=[f"{document}:block:{ordinal:06d}"],
        filing_date=filed,
        cik=cik,
        tickers=["EX"],
        form="10-K",
        fiscal_year=filed.year,
        fiscal_period="FY",
        period_end_date=period_end,
        section_topic="mdna",
        document_type="10-K",
        chunk_kind="text",
        company_name="Example Corp",
        accession_number="0000320193-25-000001",
        sec_url="https://www.sec.gov/Archives/edgar/data/320193/000032019325000001/a.htm",
        section_title="Item 7",
        subheading=None,
        table_title=None,
        chunk_ordinal=ordinal,
        document_chunk_count=count,
        part=None,
        part_count=None,
        context=context,
        text=text,
        token_count=10,
        chunk_sha256=chunk_sha256(context, text),
        processed_at=NOW,
    )


def write_chunks(gold_dir: Path, chunks: list[ChunkRecord]) -> None:
    with open_gold_catalog(gold_dir) as gold:
        if gold.table_exists(CHUNKS_TABLE):
            gold.purge_table(CHUNKS_TABLE)
        table, _ = CHUNKS_SPEC.open(gold, gold_dir)
        append_rows(table, chunks, CHUNKS_ARROW_SCHEMA)


def gold_chunks(texts_2024: int = 3, texts_2025: int = 4) -> list[ChunkRecord]:
    return [
        gold_chunk("d24", i, texts_2024, date(2024, 2, 1), f"Revenue note {i}")
        for i in range(1, texts_2024 + 1)
    ] + [
        gold_chunk("d25", i, texts_2025, date(2025, 2, 1), f"Margin note {i}")
        for i in range(1, texts_2025 + 1)
    ]

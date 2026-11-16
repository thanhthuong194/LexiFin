from __future__ import annotations

import hashlib
from datetime import date
from typing import Literal

import pyarrow as pa
from pydantic import AwareDatetime, Field, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel
from src.data_pipeline.contracts.sec_filings.filings import INT32_MAX, SHA256_PATTERN
from src.data_pipeline.transforms.native_rag.context import embedded_text
from src.data_pipeline.transforms.native_rag.section_topics import SectionTopic

ChunkKind = Literal["text", "table"]

CHUNKS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("chunk_id", pa.string(), nullable=False),
        pa.field("filing_id", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("section_id", pa.string(), nullable=False),
        pa.field("table_id", pa.string(), nullable=True),
        pa.field("block_ids", pa.list_(pa.string()), nullable=False),
        pa.field("filing_date", pa.date32(), nullable=False),
        pa.field("cik", pa.string(), nullable=False),
        pa.field("tickers", pa.list_(pa.string()), nullable=False),
        pa.field("form", pa.string(), nullable=False),
        pa.field("fiscal_year", pa.int32(), nullable=True),
        pa.field("fiscal_period", pa.string(), nullable=True),
        pa.field("period_end_date", pa.date32(), nullable=True),
        pa.field("section_topic", pa.string(), nullable=False),
        pa.field("document_type", pa.string(), nullable=False),
        pa.field("chunk_kind", pa.string(), nullable=False),
        pa.field("company_name", pa.string(), nullable=False),
        pa.field("accession_number", pa.string(), nullable=False),
        pa.field("sec_url", pa.string(), nullable=False),
        pa.field("section_title", pa.string(), nullable=False),
        pa.field("subheading", pa.string(), nullable=True),
        pa.field("table_title", pa.string(), nullable=True),
        pa.field("chunk_ordinal", pa.int32(), nullable=False),
        pa.field("document_chunk_count", pa.int32(), nullable=False),
        pa.field("part", pa.int32(), nullable=True),
        pa.field("part_count", pa.int32(), nullable=True),
        pa.field("context", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("token_count", pa.int32(), nullable=False),
        pa.field("chunk_sha256", pa.string(), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def chunk_id_for(document_id: str, chunk_ordinal: int) -> str:
    """Return the key of the N-th chunk of a document."""

    return f"{document_id}:chunk:{chunk_ordinal:06d}"


def chunk_sha256(context: str, text: str) -> str:
    """SHA-256 of the exact text embedded for a chunk (prefix + body)."""

    return hashlib.sha256(embedded_text(context, text).encode("utf-8")).hexdigest()


class ChunkRecord(ContractModel):
    """One retrieval chunk of a narrative document (`native_rag.chunks`).

    Field order matches CHUNKS_ARROW_SCHEMA. A chunk never spans two
    sections; `block_ids` are the Silver blocks whose text it holds, in order.
    `part`/`part_count` number the pieces of one paragraph or one table split
    across chunks. `token_count` counts `context + "\\n\\n" + text` with the
    embedding tokenizer, special tokens included; `chunk_sha256` hashes that
    same string.
    """

    chunk_id: str
    filing_id: str
    document_id: str
    section_id: str
    table_id: str | None
    block_ids: list[str] = Field(min_length=1)
    filing_date: date
    cik: str = Field(pattern=r"^\d{10}$")
    tickers: list[str]
    form: str
    fiscal_year: int | None = Field(ge=1900, le=2200)
    fiscal_period: Literal["FY", "Q1", "Q2", "Q3", "Q4"] | None
    period_end_date: date | None
    section_topic: SectionTopic
    document_type: str
    chunk_kind: ChunkKind
    company_name: str = Field(min_length=1)
    accession_number: str
    sec_url: str = Field(pattern=r"^https://www\.sec\.gov/Archives/edgar/data/")
    section_title: str = Field(min_length=1)
    subheading: str | None
    table_title: str | None
    chunk_ordinal: int = Field(ge=1, le=INT32_MAX)
    document_chunk_count: int = Field(ge=1, le=INT32_MAX)
    part: int | None = Field(ge=1, le=INT32_MAX)
    part_count: int | None = Field(ge=2, le=INT32_MAX)
    context: str = Field(min_length=1)
    text: str = Field(min_length=1)
    token_count: int = Field(ge=1, le=INT32_MAX)
    chunk_sha256: str = Field(pattern=SHA256_PATTERN)
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_derived_fields(self) -> ChunkRecord:
        if self.chunk_id != chunk_id_for(self.document_id, self.chunk_ordinal):
            raise ValueError("chunk_id must be derived from document_id and ordinal")
        if self.chunk_ordinal > self.document_chunk_count:
            raise ValueError("chunk_ordinal exceeds document_chunk_count")
        if (self.part is None) != (self.part_count is None):
            raise ValueError("part and part_count are set together")
        if (
            self.part is not None
            and self.part_count is not None
            and self.part > self.part_count
        ):
            raise ValueError("part exceeds part_count")
        if (self.chunk_kind == "table") != (self.table_id is not None):
            raise ValueError("table_id is set exactly on table chunks")
        if self.chunk_sha256 != chunk_sha256(self.context, self.text):
            raise ValueError("chunk_sha256 does not match context and text")
        return self

from __future__ import annotations

import math
from datetime import date

import pyarrow as pa
from pydantic import AwareDatetime, Field, model_validator

from src.data_pipeline.contracts.ingestion import ContractModel
from src.data_pipeline.contracts.sec_filings.filings import INT32_MAX, SHA256_PATTERN

# |norm - 1| allowed for a stored (L2-normalized) vector.
NORM_TOLERANCE = 1e-3

EMBEDDINGS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("embedding_id", pa.string(), nullable=False),
        pa.field("chunk_id", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("filing_date", pa.date32(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("chunk_sha256", pa.string(), nullable=False),
        pa.field("dimensions", pa.int32(), nullable=False),
        pa.field("vector", pa.list_(pa.float32()), nullable=False),
        pa.field("processed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def embedding_id_for(model_id: str, chunk_id: str) -> str:
    """Return the key of one model's vector of one chunk."""

    return f"{model_id}|{chunk_id}"


class EmbeddingRecord(ContractModel):
    """One model's vector of one chunk (`native_rag.embeddings`).

    Field order matches EMBEDDINGS_ARROW_SCHEMA. `chunk_sha256` is the hash of
    the embedded text when the vector was made: a chunk whose hash changed
    needs a new vector. Vectors are L2-normalized (cosine = dot product).
    """

    embedding_id: str
    chunk_id: str
    document_id: str
    filing_date: date
    model_id: str = Field(min_length=1)
    chunk_sha256: str = Field(pattern=SHA256_PATTERN)
    dimensions: int = Field(ge=1, le=INT32_MAX)
    vector: list[float]
    processed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_derived_fields(self) -> EmbeddingRecord:
        if self.embedding_id != embedding_id_for(self.model_id, self.chunk_id):
            raise ValueError("embedding_id must be derived from model_id and chunk_id")
        if len(self.vector) != self.dimensions:
            raise ValueError("vector length must equal dimensions")
        if not all(math.isfinite(value) for value in self.vector):
            raise ValueError("vector values must be finite")
        norm = math.sqrt(sum(value * value for value in self.vector))
        if abs(norm - 1) > NORM_TOLERANCE:
            raise ValueError(f"vector must be L2-normalized, norm is {norm:.6f}")
        return self

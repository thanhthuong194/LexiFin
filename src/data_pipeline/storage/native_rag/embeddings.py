from __future__ import annotations

from src.data_pipeline.contracts.native_rag.embeddings import EMBEDDINGS_ARROW_SCHEMA
from src.data_pipeline.storage.iceberg import SilverTableSpec
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

EMBEDDINGS_TABLE = "native_rag.embeddings"

# One table for every model (`model_id` column); rows are keyed by model and
# chunk and carry the chunk hash they were made from.
EMBEDDINGS_SPEC = SilverTableSpec(
    EMBEDDINGS_TABLE, EMBEDDINGS_ARROW_SCHEMA, FILING_YEAR_PARTITION, layer="gold"
)

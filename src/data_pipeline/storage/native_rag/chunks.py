from __future__ import annotations

from src.data_pipeline.contracts.native_rag.chunks import CHUNKS_ARROW_SCHEMA
from src.data_pipeline.storage.iceberg import SilverTableSpec
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

CHUNKS_TABLE = "native_rag.chunks"

# Version of the chunking rules (method, caps, noise rules, rendering). Bump
# it whenever the same Silver rows would give different chunks, then purge
# and rebuild the table.
CHUNKS_SPEC = SilverTableSpec(
    CHUNKS_TABLE, CHUNKS_ARROW_SCHEMA, FILING_YEAR_PARTITION, layer="gold"
)

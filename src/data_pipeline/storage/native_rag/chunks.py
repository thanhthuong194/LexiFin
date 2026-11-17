from __future__ import annotations

from src.data_pipeline.contracts.native_rag.chunks import CHUNKS_ARROW_SCHEMA
from src.data_pipeline.storage.iceberg import SilverTableSpec
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

CHUNKS_TABLE = "native_rag.chunks"

# Version of the chunking rules (method, caps, noise rules, rendering). Bump
# it whenever the same Silver rows would give different chunks, then purge
# and rebuild the table.
# Version 2 (2026-11-17): long paragraphs cut in thirds of the budget with a
# short overlap, so no part exceeds the cap (v1 rejected 5 documents at 513-514
# tokens on the 500-company Silver).
# Version 3 (2026-11-17): tables inside a table-of-contents section are
# classified (data tables kept: Silver puts whole statements there when it finds
# no item heading); a section's opening block is dropped only when at most
# min(title, 200) + 40 chars (HP 8-K items whose title is the whole table).
CHUNKS_SPEC = SilverTableSpec(
    CHUNKS_TABLE,
    CHUNKS_ARROW_SCHEMA,
    FILING_YEAR_PARTITION,
    rules_version=3,
    layer="gold",
)

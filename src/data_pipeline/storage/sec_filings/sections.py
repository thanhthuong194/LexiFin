from __future__ import annotations

from src.data_pipeline.contracts.sec_filings.sections import SECTIONS_ARROW_SCHEMA
from src.data_pipeline.storage.iceberg import (
    SilverTableSpec,
)
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

SECTIONS_TABLE = "sec_filings.sections"


# Version 2 (2026-11-06): section headings that are links, 10-Q PART II
# inferred from restarting items, TOC heading lines and index PARTs dropped,
# 8-K items opened by layout tables.
SECTIONS_SPEC = SilverTableSpec(
    SECTIONS_TABLE, SECTIONS_ARROW_SCHEMA, FILING_YEAR_PARTITION, rules_version=2
)

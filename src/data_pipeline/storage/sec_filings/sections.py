from __future__ import annotations

from src.data_pipeline.contracts.sec_filings.sections import SECTIONS_ARROW_SCHEMA
from src.data_pipeline.storage.iceberg import (
    SilverTableSpec,
)
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

SECTIONS_TABLE = "sec_filings.sections"


SECTIONS_SPEC = SilverTableSpec(
    SECTIONS_TABLE, SECTIONS_ARROW_SCHEMA, FILING_YEAR_PARTITION
)

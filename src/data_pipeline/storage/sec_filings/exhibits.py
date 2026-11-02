from __future__ import annotations

from src.data_pipeline.contracts.sec_filings.exhibits import EXHIBITS_ARROW_SCHEMA
from src.data_pipeline.storage.iceberg import (
    SilverTableSpec,
)
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

EXHIBITS_TABLE = "sec_filings.exhibits"


EXHIBITS_SPEC = SilverTableSpec(
    EXHIBITS_TABLE, EXHIBITS_ARROW_SCHEMA, FILING_YEAR_PARTITION
)

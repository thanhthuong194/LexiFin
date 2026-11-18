from __future__ import annotations

from src.data_pipeline.contracts.sec_filings.document_tables import (
    DOCUMENT_TABLES_ARROW_SCHEMA,
)
from src.data_pipeline.storage.iceberg import (
    SilverTableSpec,
)
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

DOCUMENT_TABLES_TABLE = "sec_filings.document_tables"


# v2: rows without a visible cell are not counted
DOCUMENT_TABLES_SPEC = SilverTableSpec(
    DOCUMENT_TABLES_TABLE,
    DOCUMENT_TABLES_ARROW_SCHEMA,
    FILING_YEAR_PARTITION,
    rules_version=2,
)

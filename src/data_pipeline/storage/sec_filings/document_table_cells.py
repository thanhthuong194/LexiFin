from __future__ import annotations

from src.data_pipeline.contracts.sec_filings.document_table_cells import (
    DOCUMENT_TABLE_CELLS_ARROW_SCHEMA,
)
from src.data_pipeline.storage.iceberg import (
    SilverTableSpec,
)
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

DOCUMENT_TABLE_CELLS_TABLE = "sec_filings.document_table_cells"


DOCUMENT_TABLE_CELLS_SPEC = SilverTableSpec(
    DOCUMENT_TABLE_CELLS_TABLE, DOCUMENT_TABLE_CELLS_ARROW_SCHEMA, FILING_YEAR_PARTITION
)

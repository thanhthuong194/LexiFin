from __future__ import annotations

from pathlib import Path

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.document_table_cells import (
    DOCUMENT_TABLE_CELLS_ARROW_SCHEMA,
)
from src.data_pipeline.storage.iceberg import (
    create_or_load_table,
    silver_table_location,
)
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

DOCUMENT_TABLE_CELLS_TABLE = "sec_filings.document_table_cells"


def open_document_table_cells_table(
    catalog: Catalog, data_dir: Path
) -> tuple[Table, bool]:
    """Load or create `sec_filings.document_table_cells`, returning whether it was created."""

    return create_or_load_table(
        catalog,
        DOCUMENT_TABLE_CELLS_TABLE,
        arrow_schema=DOCUMENT_TABLE_CELLS_ARROW_SCHEMA,
        location=silver_table_location(data_dir, DOCUMENT_TABLE_CELLS_TABLE),
        partition_by=FILING_YEAR_PARTITION,
    )

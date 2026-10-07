from __future__ import annotations

from pathlib import Path

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.exhibits import EXHIBITS_ARROW_SCHEMA
from src.data_pipeline.storage.iceberg import (
    create_or_load_table,
    silver_table_location,
)
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

EXHIBITS_TABLE = "sec_filings.exhibits"


def open_exhibits_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create `sec_filings.exhibits`, partitioned by filing year."""

    return create_or_load_table(
        catalog,
        EXHIBITS_TABLE,
        arrow_schema=EXHIBITS_ARROW_SCHEMA,
        location=silver_table_location(data_dir, EXHIBITS_TABLE),
        partition_by=FILING_YEAR_PARTITION,
    )

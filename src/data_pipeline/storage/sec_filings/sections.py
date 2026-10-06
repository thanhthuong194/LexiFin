from __future__ import annotations

from pathlib import Path

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.sections import SECTIONS_ARROW_SCHEMA
from src.data_pipeline.storage.iceberg import (
    create_or_load_table,
    silver_table_location,
)
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

SECTIONS_TABLE = "sec_filings.sections"


def open_sections_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create `sec_filings.sections`, returning whether it was created."""

    return create_or_load_table(
        catalog,
        SECTIONS_TABLE,
        arrow_schema=SECTIONS_ARROW_SCHEMA,
        location=silver_table_location(data_dir, SECTIONS_TABLE),
        partition_by=FILING_YEAR_PARTITION,
    )

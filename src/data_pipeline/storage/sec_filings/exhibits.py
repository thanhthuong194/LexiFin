from __future__ import annotations

from pathlib import Path

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.exhibits import EXHIBITS_ARROW_SCHEMA
from src.data_pipeline.storage.iceberg import (
    SilverTableSpec,
)
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

EXHIBITS_TABLE = "sec_filings.exhibits"


EXHIBITS_SPEC = SilverTableSpec(
    EXHIBITS_TABLE, EXHIBITS_ARROW_SCHEMA, FILING_YEAR_PARTITION
)


def open_exhibits_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create `sec_filings.exhibits`, partitioned by filing year."""

    return EXHIBITS_SPEC.open(catalog, data_dir)

from __future__ import annotations

from pathlib import Path

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.sections import SECTIONS_ARROW_SCHEMA
from src.data_pipeline.storage.iceberg import (
    SilverTableSpec,
)
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

SECTIONS_TABLE = "sec_filings.sections"


SECTIONS_SPEC = SilverTableSpec(
    SECTIONS_TABLE, SECTIONS_ARROW_SCHEMA, FILING_YEAR_PARTITION
)


def open_sections_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create `sec_filings.sections`, returning whether it was created."""

    return SECTIONS_SPEC.open(catalog, data_dir)

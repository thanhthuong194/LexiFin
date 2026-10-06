from __future__ import annotations

from pathlib import Path

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.document_links import (
    DOCUMENT_LINKS_ARROW_SCHEMA,
)
from src.data_pipeline.storage.iceberg import (
    create_or_load_table,
    silver_table_location,
)
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

DOCUMENT_LINKS_TABLE = "sec_filings.document_links"


def open_document_links_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create `sec_filings.document_links`, returning whether it was created."""

    return create_or_load_table(
        catalog,
        DOCUMENT_LINKS_TABLE,
        arrow_schema=DOCUMENT_LINKS_ARROW_SCHEMA,
        location=silver_table_location(data_dir, DOCUMENT_LINKS_TABLE),
        partition_by=FILING_YEAR_PARTITION,
    )

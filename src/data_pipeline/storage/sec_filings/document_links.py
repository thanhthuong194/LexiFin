from __future__ import annotations

from pathlib import Path

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.document_links import (
    DOCUMENT_LINKS_ARROW_SCHEMA,
)
from src.data_pipeline.storage.iceberg import (
    SilverTableSpec,
)
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

DOCUMENT_LINKS_TABLE = "sec_filings.document_links"


DOCUMENT_LINKS_SPEC = SilverTableSpec(
    DOCUMENT_LINKS_TABLE, DOCUMENT_LINKS_ARROW_SCHEMA, FILING_YEAR_PARTITION
)


def open_document_links_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create `sec_filings.document_links`, returning whether it was created."""

    return DOCUMENT_LINKS_SPEC.open(catalog, data_dir)

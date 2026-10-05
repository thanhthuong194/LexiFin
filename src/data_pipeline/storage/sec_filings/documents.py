from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.documents import DOCUMENTS_ARROW_SCHEMA
from src.data_pipeline.storage.iceberg import (
    create_or_load_table,
    silver_table_location,
)
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

DOCUMENTS_TABLE = "sec_filings.documents"


@dataclass(frozen=True)
class StoredDocument:
    """The stored state of one document row needed to judge its filing group."""

    document_id: str
    document_ordinal: int
    role: str
    source_submission_sha256: str


def open_documents_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create `sec_filings.documents`, returning whether it was created."""

    return create_or_load_table(
        catalog,
        DOCUMENTS_TABLE,
        arrow_schema=DOCUMENTS_ARROW_SCHEMA,
        location=silver_table_location(data_dir, DOCUMENTS_TABLE),
        partition_by=FILING_YEAR_PARTITION,
    )


def read_document_groups(table: Table) -> dict[str, list[StoredDocument]]:
    """Return the stored documents of the current snapshot grouped by filing_id."""

    rows = table.scan(
        selected_fields=(
            "filing_id",
            "document_id",
            "document_ordinal",
            "role",
            "source_submission_sha256",
        )
    ).to_arrow()
    groups: dict[str, list[StoredDocument]] = defaultdict(list)
    for row in rows.to_pylist():
        groups[row.pop("filing_id")].append(StoredDocument(**row))
    return dict(groups)

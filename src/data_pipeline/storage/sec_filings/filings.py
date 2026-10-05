from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table
from pyiceberg.transforms import YearTransform

from src.data_pipeline.contracts.sec_filings.filings import (
    FILINGS_ARROW_SCHEMA,
    FilingRecord,
)
from src.data_pipeline.storage.iceberg import (
    PartitionColumn,
    create_or_load_table,
    silver_table_location,
)

FILINGS_TABLE = "sec_filings.filings"
# SEC Silver tables carry their filing's date and are partitioned by its year.
FILING_YEAR_PARTITION = (
    PartitionColumn("filing_date", YearTransform(), "filing_year"),
)


@dataclass(frozen=True)
class StoredChecksums:
    """Artifact checksums already recorded for one filing_id."""

    full_submission_sha256: str
    primary_document_sha256: str | None


def open_filings_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create `sec_filings.filings`, returning whether it was created."""

    return create_or_load_table(
        catalog,
        FILINGS_TABLE,
        arrow_schema=FILINGS_ARROW_SCHEMA,
        location=silver_table_location(data_dir, FILINGS_TABLE),
        partition_by=FILING_YEAR_PARTITION,
    )


def read_stored_checksums(table: Table) -> dict[str, StoredChecksums]:
    """Return the checksums of every filing_id in the table's current snapshot."""

    rows = table.scan(
        selected_fields=(
            "filing_id",
            "full_submission_sha256",
            "primary_document_sha256",
        )
    ).to_arrow()
    return {
        row["filing_id"]: StoredChecksums(
            full_submission_sha256=row["full_submission_sha256"],
            primary_document_sha256=row["primary_document_sha256"],
        )
        for row in rows.to_pylist()
    }


def read_filings(table: Table) -> list[FilingRecord]:
    """Return every row of the table's current snapshot as a FilingRecord."""

    return [
        FilingRecord.model_validate(row) for row in table.scan().to_arrow().to_pylist()
    ]

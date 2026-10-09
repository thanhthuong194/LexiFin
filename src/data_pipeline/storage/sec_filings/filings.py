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
    SilverTableSpec,
    iter_rows,
)

FILINGS_TABLE = "sec_filings.filings"
# SEC Silver tables carry their filing's date and are partitioned by its year.
FILING_YEAR_PARTITION = (
    PartitionColumn("filing_date", YearTransform(), "filing_year"),
)


FILINGS_SPEC = SilverTableSpec(
    FILINGS_TABLE, FILINGS_ARROW_SCHEMA, FILING_YEAR_PARTITION
)


@dataclass(frozen=True)
class StoredChecksums:
    """Artifact checksums already recorded for one filing_id."""

    full_submission_sha256: str
    primary_document_sha256: str | None


def open_filings_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create `sec_filings.filings`, returning whether it was created."""

    return FILINGS_SPEC.open(catalog, data_dir)


def read_stored_checksums(table: Table) -> dict[str, StoredChecksums]:
    """Return the checksums of every filing_id in the table's current snapshot."""

    rows = iter_rows(
        table, ("filing_id", "full_submission_sha256", "primary_document_sha256")
    )
    return {
        row["filing_id"]: StoredChecksums(
            full_submission_sha256=row["full_submission_sha256"],
            primary_document_sha256=row["primary_document_sha256"],
        )
        for row in rows
    }


def read_filings(table: Table) -> list[FilingRecord]:
    """Return every row of the table's current snapshot as a FilingRecord."""

    return [FilingRecord.model_validate(row) for row in iter_rows(table)]

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.filing_entities import (
    FILING_ENTITIES_ARROW_SCHEMA,
    FilingEntityRecord,
)
from src.data_pipeline.storage.iceberg import (
    create_or_load_table,
    silver_table_location,
)

FILING_ENTITIES_TABLE = "sec_filings.filing_entities"


def open_filing_entities_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create `sec_filings.filing_entities`, returning whether it was created.

    The table is unpartitioned: it is small, read by filing_id or cik, and no
    benchmark yet shows partitioning would help.
    """

    return create_or_load_table(
        catalog,
        FILING_ENTITIES_TABLE,
        arrow_schema=FILING_ENTITIES_ARROW_SCHEMA,
        location=silver_table_location(data_dir, FILING_ENTITIES_TABLE),
        partition_by=(),
    )


def read_filing_entity_groups(table: Table) -> dict[str, list[FilingEntityRecord]]:
    """Return the current snapshot's rows grouped by filing_id."""

    groups: dict[str, list[FilingEntityRecord]] = defaultdict(list)
    for row in table.scan().to_arrow().to_pylist():
        record = FilingEntityRecord.model_validate(row)
        groups[record.filing_id].append(record)
    return dict(groups)

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
    SilverTableSpec,
    iter_rows,
)

FILING_ENTITIES_TABLE = "sec_filings.filing_entities"


FILING_ENTITIES_SPEC = SilverTableSpec(
    FILING_ENTITIES_TABLE, FILING_ENTITIES_ARROW_SCHEMA
)


def open_filing_entities_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create `sec_filings.filing_entities`, returning whether it was created.

    The table is unpartitioned: it is small, read by filing_id or cik, and no
    benchmark yet shows partitioning would help.
    """

    return FILING_ENTITIES_SPEC.open(catalog, data_dir)


def read_filing_entity_groups(table: Table) -> dict[str, list[FilingEntityRecord]]:
    """Return the current snapshot's rows grouped by filing_id."""

    groups: dict[str, list[FilingEntityRecord]] = defaultdict(list)
    for row in iter_rows(table):
        record = FilingEntityRecord.model_validate(row)
        groups[record.filing_id].append(record)
    return dict(groups)

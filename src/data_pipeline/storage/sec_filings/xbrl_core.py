from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.xbrl_core import (
    XBRL_CONTEXTS_ARROW_SCHEMA,
    XBRL_INSTANCES_ARROW_SCHEMA,
    XBRL_UNITS_ARROW_SCHEMA,
)
from src.data_pipeline.storage.iceberg import (
    create_or_load_table,
    group_rows_by,
    silver_table_location,
)

XBRL_INSTANCES_TABLE = "sec_filings.xbrl_instances"
XBRL_CONTEXTS_TABLE = "sec_filings.xbrl_contexts"
XBRL_UNITS_TABLE = "sec_filings.xbrl_units"
# Unpartitioned: instances are few, and contexts and units are read by
# instance_id. Partition evolution (e.g. bucket by instance_id) can follow if
# scans grow.
XBRL_CORE_SCHEMAS: dict[str, pa.Schema] = {
    XBRL_INSTANCES_TABLE: XBRL_INSTANCES_ARROW_SCHEMA,
    XBRL_CONTEXTS_TABLE: XBRL_CONTEXTS_ARROW_SCHEMA,
    XBRL_UNITS_TABLE: XBRL_UNITS_ARROW_SCHEMA,
}


def open_xbrl_core_table(
    catalog: Catalog, data_dir: Path, identifier: str
) -> tuple[Table, bool]:
    """Load or create one XBRL core table, returning whether it was created."""

    return create_or_load_table(
        catalog,
        identifier,
        arrow_schema=XBRL_CORE_SCHEMAS[identifier],
        location=silver_table_location(data_dir, identifier),
        partition_by=(),
    )


def read_instance_groups(table: Table) -> dict[str, list[dict[str, Any]]]:
    """Stored `xbrl_instances` rows grouped by filing_id, maps as dicts.

    Arrow returns a map column as (key, value) pairs; rows are not validated.
    """

    groups = group_rows_by(table, "filing_id")
    for rows in groups.values():
        for row in rows:
            row["namespace_map"] = dict(row["namespace_map"])
    return groups

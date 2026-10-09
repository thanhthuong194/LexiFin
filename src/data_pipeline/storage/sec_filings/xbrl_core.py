from __future__ import annotations

from collections import defaultdict
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
from src.data_pipeline.storage.iceberg import SilverTableSpec, scan_batches

XBRL_INSTANCES_TABLE = "sec_filings.xbrl_instances"
XBRL_CONTEXTS_TABLE = "sec_filings.xbrl_contexts"
XBRL_UNITS_TABLE = "sec_filings.xbrl_units"
# Unpartitioned: instances are few, and contexts and units are read by
# instance_id. Partition evolution (e.g. bucket by instance_id) can follow if
# scans grow.
XBRL_INSTANCES_SPEC = SilverTableSpec(XBRL_INSTANCES_TABLE, XBRL_INSTANCES_ARROW_SCHEMA)
XBRL_CONTEXTS_SPEC = SilverTableSpec(XBRL_CONTEXTS_TABLE, XBRL_CONTEXTS_ARROW_SCHEMA)
XBRL_UNITS_SPEC = SilverTableSpec(XBRL_UNITS_TABLE, XBRL_UNITS_ARROW_SCHEMA)
# In commit order: instances, then contexts and units.
XBRL_CORE_SPECS = {
    spec.identifier: spec
    for spec in (XBRL_INSTANCES_SPEC, XBRL_CONTEXTS_SPEC, XBRL_UNITS_SPEC)
}
XBRL_CORE_SCHEMAS: dict[str, pa.Schema] = {
    identifier: spec.arrow_schema for identifier, spec in XBRL_CORE_SPECS.items()
}


def open_xbrl_core_table(
    catalog: Catalog, data_dir: Path, identifier: str
) -> tuple[Table, bool]:
    """Load or create one XBRL core table, returning whether it was created."""

    return XBRL_CORE_SPECS[identifier].open(catalog, data_dir)


def read_instance_groups(table: Table) -> dict[str, list[dict[str, Any]]]:
    """Stored `xbrl_instances` rows grouped by filing_id, maps as dicts.

    Arrow returns a map column as (key, value) pairs; rows are not validated.
    Instances are one or two per filing, so the table is read whole.
    """

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in scan_batches(table):
        for row in batch.to_pylist():
            row["namespace_map"] = dict(row["namespace_map"])
            groups[row["filing_id"]].append(row)
    return dict(groups)

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.xbrl_dimensions import (
    XBRL_DIMENSIONS_ARROW_SCHEMA,
)
from src.data_pipeline.contracts.sec_filings.xbrl_fact_footnotes import (
    XBRL_FACT_FOOTNOTES_ARROW_SCHEMA,
)
from src.data_pipeline.storage.iceberg import (
    create_or_load_table,
    silver_table_location,
)

XBRL_DIMENSIONS_TABLE = "sec_filings.xbrl_dimensions"
XBRL_FACT_FOOTNOTES_TABLE = "sec_filings.xbrl_fact_footnotes"


def open_xbrl_dimensions_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create the unpartitioned `sec_filings.xbrl_dimensions`."""

    return create_or_load_table(
        catalog,
        XBRL_DIMENSIONS_TABLE,
        arrow_schema=XBRL_DIMENSIONS_ARROW_SCHEMA,
        location=silver_table_location(data_dir, XBRL_DIMENSIONS_TABLE),
        partition_by=(),
    )


def open_xbrl_fact_footnotes_table(
    catalog: Catalog, data_dir: Path
) -> tuple[Table, bool]:
    """Load or create the unpartitioned `sec_filings.xbrl_fact_footnotes`."""

    return create_or_load_table(
        catalog,
        XBRL_FACT_FOOTNOTES_TABLE,
        arrow_schema=XBRL_FACT_FOOTNOTES_ARROW_SCHEMA,
        location=silver_table_location(data_dir, XBRL_FACT_FOOTNOTES_TABLE),
        partition_by=(),
    )


def read_contexts_by_instance(table: Table) -> dict[str, list[dict[str, Any]]]:
    """Identity, hash and dimension XML of each stored context, by instance_id."""

    fields = (
        "instance_id",
        "context_key",
        "context_ref",
        "context_hash",
        "segment_xml",
        "scenario_xml",
    )
    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table.scan(selected_fields=fields).to_arrow().to_pylist():
        contexts[row["instance_id"]].append(row)
    return dict(contexts)


def read_fact_hashes(table: Table) -> dict[str, dict[str, str]]:
    """The fact_sha256 of each stored fact, by instance_id then fact_id."""

    fields = ("instance_id", "fact_id", "fact_sha256")
    hashes: dict[str, dict[str, str]] = defaultdict(dict)
    for row in table.scan(selected_fields=fields).to_arrow().to_pylist():
        hashes[row["instance_id"]][row["fact_id"]] = row["fact_sha256"]
    return dict(hashes)

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection
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
    SilverTableSpec,
    read_rows_in,
    values_filter,
)

XBRL_DIMENSIONS_TABLE = "sec_filings.xbrl_dimensions"
XBRL_FACT_FOOTNOTES_TABLE = "sec_filings.xbrl_fact_footnotes"


XBRL_DIMENSIONS_SPEC = SilverTableSpec(
    XBRL_DIMENSIONS_TABLE, XBRL_DIMENSIONS_ARROW_SCHEMA
)
XBRL_FACT_FOOTNOTES_SPEC = SilverTableSpec(
    XBRL_FACT_FOOTNOTES_TABLE, XBRL_FACT_FOOTNOTES_ARROW_SCHEMA
)


def open_xbrl_dimensions_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create the unpartitioned `sec_filings.xbrl_dimensions`."""

    return XBRL_DIMENSIONS_SPEC.open(catalog, data_dir)


def open_xbrl_fact_footnotes_table(
    catalog: Catalog, data_dir: Path
) -> tuple[Table, bool]:
    """Load or create the unpartitioned `sec_filings.xbrl_fact_footnotes`."""

    return XBRL_FACT_FOOTNOTES_SPEC.open(catalog, data_dir)


def read_contexts_by_instance(
    table: Table, instance_ids: Collection[str]
) -> dict[str, list[dict[str, Any]]]:
    """Identity, hash and dimension XML of the given instances' contexts, by instance."""

    columns = (
        "context_key",
        "context_ref",
        "context_hash",
        "segment_xml",
        "scenario_xml",
    )
    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_rows_in(table, "instance_id", instance_ids, columns=columns):
        contexts[row["instance_id"]].append(row)
    return dict(contexts)


def read_fact_hashes(
    table: Table, instance_ids: Collection[str]
) -> dict[str, dict[str, str]]:
    """The fact_sha256 of the given instances' facts, by instance_id then fact_id.

    Only the instances' buckets of `xbrl_facts` are read.
    """

    hashes: dict[str, dict[str, str]] = defaultdict(dict)
    for row in read_rows_in(
        table,
        "instance_id",
        instance_ids,
        columns=("fact_id", "fact_sha256"),
        files=values_filter("instance_id", instance_ids),
    ):
        hashes[row["instance_id"]][row["fact_id"]] = row["fact_sha256"]
    return dict(hashes)

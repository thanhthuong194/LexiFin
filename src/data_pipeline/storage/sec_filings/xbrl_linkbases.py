from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.xbrl_labels import XBRL_LABELS_ARROW_SCHEMA
from src.data_pipeline.contracts.sec_filings.xbrl_networks import (
    XBRL_NETWORKS_ARROW_SCHEMA,
    XBRL_RELATIONSHIPS_ARROW_SCHEMA,
)
from src.data_pipeline.storage.iceberg import (
    create_or_load_table,
    silver_table_location,
)

XBRL_LABELS_TABLE = "sec_filings.xbrl_labels"
XBRL_NETWORKS_TABLE = "sec_filings.xbrl_networks"
XBRL_RELATIONSHIPS_TABLE = "sec_filings.xbrl_relationships"
# Columns that identify a relationship row and its place in its network.
RELATIONSHIP_HEAD_COLUMNS = (
    "relationship_id",
    "relationship_ordinal",
    "network_relationship_count",
    "relationship_hash",
)


def open_xbrl_linkbase_table(
    catalog: Catalog, data_dir: Path, identifier: str
) -> tuple[Table, bool]:
    """Load or create one of the unpartitioned label, network and relationship tables."""

    schemas = {
        XBRL_LABELS_TABLE: XBRL_LABELS_ARROW_SCHEMA,
        XBRL_NETWORKS_TABLE: XBRL_NETWORKS_ARROW_SCHEMA,
        XBRL_RELATIONSHIPS_TABLE: XBRL_RELATIONSHIPS_ARROW_SCHEMA,
    }
    return create_or_load_table(
        catalog,
        identifier,
        arrow_schema=schemas[identifier],
        location=silver_table_location(data_dir, identifier),
        partition_by=(),
    )


def read_label_ids(table: Table) -> set[str]:
    """IDs of the stored labels."""

    return set(
        table.scan(selected_fields=("label_id",)).to_arrow()["label_id"].to_pylist()
    )


def read_concept_references(labels: Table, relationships: Table) -> set[str]:
    """The distinct concept IDs that stored labels and relationships refer to."""

    columns = (
        (labels, "concept_id"),
        (relationships, "source_concept_id"),
        (relationships, "target_concept_id"),
    )
    return {
        concept_id
        for table, column in columns
        for concept_id in pc.unique(
            table.scan(selected_fields=(column,)).to_arrow()[column]
        ).to_pylist()
    }


def read_relationship_heads(table: Table) -> dict[str, list[dict[str, Any]]]:
    """RELATIONSHIP_HEAD_COLUMNS of the stored relationships, by network_id."""

    rows = table.scan(selected_fields=("network_id", *RELATIONSHIP_HEAD_COLUMNS))
    heads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows.to_arrow().to_pylist():
        heads[row.pop("network_id")].append(row)
    return dict(heads)

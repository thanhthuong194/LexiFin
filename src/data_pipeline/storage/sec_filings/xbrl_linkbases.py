from __future__ import annotations

import pyarrow.compute as pc
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.xbrl_labels import XBRL_LABELS_ARROW_SCHEMA
from src.data_pipeline.contracts.sec_filings.xbrl_networks import (
    XBRL_NETWORKS_ARROW_SCHEMA,
    XBRL_RELATIONSHIPS_ARROW_SCHEMA,
)
from src.data_pipeline.storage.iceberg import SilverTableSpec, iter_rows, scan_batches

XBRL_LABELS_TABLE = "sec_filings.xbrl_labels"
XBRL_NETWORKS_TABLE = "sec_filings.xbrl_networks"
XBRL_RELATIONSHIPS_TABLE = "sec_filings.xbrl_relationships"
XBRL_LABELS_SPEC = SilverTableSpec(XBRL_LABELS_TABLE, XBRL_LABELS_ARROW_SCHEMA)
XBRL_NETWORKS_SPEC = SilverTableSpec(XBRL_NETWORKS_TABLE, XBRL_NETWORKS_ARROW_SCHEMA)
XBRL_RELATIONSHIPS_SPEC = SilverTableSpec(
    XBRL_RELATIONSHIPS_TABLE, XBRL_RELATIONSHIPS_ARROW_SCHEMA
)
# Columns that identify a relationship row and its place in its network.
RELATIONSHIP_HEAD_COLUMNS = (
    "relationship_id",
    "relationship_ordinal",
    "network_relationship_count",
    "relationship_hash",
)


def read_label_ids(table: Table) -> set[str]:
    """IDs of the stored labels."""

    return {row["label_id"] for row in iter_rows(table, ("label_id",))}


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
        for batch in scan_batches(table, (column,))
        for concept_id in pc.unique(batch[column]).to_pylist()
    }

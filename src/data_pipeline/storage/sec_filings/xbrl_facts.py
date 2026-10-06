from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table
from pyiceberg.transforms import BucketTransform

from src.data_pipeline.contracts.sec_filings.xbrl_concepts import (
    XBRL_CONCEPTS_ARROW_SCHEMA,
)
from src.data_pipeline.contracts.sec_filings.xbrl_facts import XBRL_FACTS_ARROW_SCHEMA
from src.data_pipeline.storage.iceberg import (
    PartitionColumn,
    create_or_load_table,
    silver_table_location,
)

XBRL_CONCEPTS_TABLE = "sec_filings.xbrl_concepts"
XBRL_FACTS_TABLE = "sec_filings.xbrl_facts"
# Facts are many and read by instance; the concept dictionary is small.
INSTANCE_BUCKET_PARTITION = (
    PartitionColumn("instance_id", BucketTransform(32), "instance_bucket"),
)


def open_xbrl_concepts_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create the unpartitioned `sec_filings.xbrl_concepts`."""

    return create_or_load_table(
        catalog,
        XBRL_CONCEPTS_TABLE,
        arrow_schema=XBRL_CONCEPTS_ARROW_SCHEMA,
        location=silver_table_location(data_dir, XBRL_CONCEPTS_TABLE),
        partition_by=(),
    )


def open_xbrl_facts_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create `sec_filings.xbrl_facts`, bucketed by instance_id."""

    return create_or_load_table(
        catalog,
        XBRL_FACTS_TABLE,
        arrow_schema=XBRL_FACTS_ARROW_SCHEMA,
        location=silver_table_location(data_dir, XBRL_FACTS_TABLE),
        partition_by=INSTANCE_BUCKET_PARTITION,
    )


def read_concept_hashes(table: Table) -> dict[str, str]:
    """Stored concept IDs and their definition hashes."""

    rows = table.scan(selected_fields=("concept_id", "concept_definition_sha256"))
    return {
        row["concept_id"]: row["concept_definition_sha256"]
        for row in rows.to_arrow().to_pylist()
    }


def read_refs_by_instance(table: Table, ref_column: str) -> dict[str, set[str]]:
    """The context or unit refs (`ref_column`) of each instance of an XBRL core table."""

    refs: dict[str, set[str]] = defaultdict(set)
    rows = (
        table.scan(selected_fields=("instance_id", ref_column)).to_arrow().to_pylist()
    )
    for row in rows:
        refs[row["instance_id"]].add(row[ref_column])
    return dict(refs)

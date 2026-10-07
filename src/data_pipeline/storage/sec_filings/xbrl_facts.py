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
    SilverTableSpec,
)

XBRL_CONCEPTS_TABLE = "sec_filings.xbrl_concepts"
XBRL_FACTS_TABLE = "sec_filings.xbrl_facts"
# Facts are many and read by instance; the concept dictionary is small.
INSTANCE_BUCKET_PARTITION = (
    PartitionColumn("instance_id", BucketTransform(32), "instance_bucket"),
)


XBRL_CONCEPTS_SPEC = SilverTableSpec(XBRL_CONCEPTS_TABLE, XBRL_CONCEPTS_ARROW_SCHEMA)
XBRL_FACTS_SPEC = SilverTableSpec(
    XBRL_FACTS_TABLE, XBRL_FACTS_ARROW_SCHEMA, INSTANCE_BUCKET_PARTITION
)


def open_xbrl_concepts_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create the unpartitioned `sec_filings.xbrl_concepts`."""

    return XBRL_CONCEPTS_SPEC.open(catalog, data_dir)


def open_xbrl_facts_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create `sec_filings.xbrl_facts`, bucketed by instance_id."""

    return XBRL_FACTS_SPEC.open(catalog, data_dir)


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

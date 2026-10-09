from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection
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
    iter_rows,
    read_rows_in,
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

    rows = iter_rows(table, ("concept_id", "concept_definition_sha256"))
    return {row["concept_id"]: row["concept_definition_sha256"] for row in rows}


def read_refs_by_instance(
    table: Table, ref_column: str, instance_ids: Collection[str]
) -> dict[str, set[str]]:
    """The context or unit refs (`ref_column`) of the given instances, by instance."""

    refs: dict[str, set[str]] = defaultdict(set)
    for row in read_rows_in(table, "instance_id", instance_ids, columns=(ref_column,)):
        refs[row["instance_id"]].add(row[ref_column])
    return dict(refs)

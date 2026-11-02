from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.documents import (
    DOCUMENTS_ARROW_SCHEMA,
    DocumentRecord,
)
from src.data_pipeline.contracts.sec_filings.xbrl_core import (
    XBRL_CONTEXTS_ARROW_SCHEMA,
    XBRL_INSTANCES_ARROW_SCHEMA,
    XBRL_UNITS_ARROW_SCHEMA,
    XbrlInstanceRecord,
)
from src.data_pipeline.storage.iceberg import (
    SilverTableSpec,
    filing_dates_filter,
    read_groups_in,
    read_rows_in,
    scan_batches,
)
from src.data_pipeline.storage.sec_filings.documents import DOCUMENTS_SPEC

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


def read_instances_with_sources(
    catalog: Catalog, data_dir: Path
) -> tuple[list[XbrlInstanceRecord], dict[str, DocumentRecord]]:
    """Every stored instance, in instance_id order, and its source documents
    by document_id: the parents of the XBRL stages that load instances."""

    instances = sorted(
        (
            XbrlInstanceRecord.model_validate(row)
            for rows in read_instance_groups(
                XBRL_INSTANCES_SPEC.open(catalog, data_dir)[0]
            ).values()
            for row in rows
        ),
        key=lambda instance: instance.instance_id,
    )
    sources = {
        row["document_id"]: DocumentRecord.model_validate(row)
        for row in read_rows_in(
            DOCUMENTS_SPEC.open(catalog, data_dir)[0],
            "document_id",
            {d for instance in instances for d in instance.source_document_ids},
            columns=DOCUMENTS_ARROW_SCHEMA.names,
        )
    }
    return instances, sources


def read_filing_documents(
    catalog: Catalog,
    data_dir: Path,
    instances: Iterable[XbrlInstanceRecord],
    sources: Mapping[str, DocumentRecord],
) -> dict[str, DocumentRecord]:
    """Every document of the instances' filings, by document_id.

    Only the partitions of the filing dates of the instances' `sources` are
    read.
    """

    instances = list(instances)
    dates = {
        sources[d].filing_date
        for instance in instances
        for d in instance.source_document_ids
        if d in sources
    }
    return {
        row["document_id"]: DocumentRecord.model_validate(row)
        for rows in read_groups_in(
            DOCUMENTS_SPEC.open(catalog, data_dir)[0],
            "filing_id",
            {instance.filing_id for instance in instances},
            files=filing_dates_filter(dates),
        ).values()
        for row in rows
    }

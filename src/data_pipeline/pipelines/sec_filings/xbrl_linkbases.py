from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlInstanceRecord
from src.data_pipeline.contracts.sec_filings.xbrl_labels import (
    XBRL_LABELS_ARROW_SCHEMA,
    XbrlLabelRecord,
)
from src.data_pipeline.contracts.sec_filings.xbrl_networks import (
    XBRL_NETWORKS_ARROW_SCHEMA,
    XBRL_RELATIONSHIPS_ARROW_SCHEMA,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_linkbases_extraction import (
    LinkbaseExtraction,
    LinkbaseInputs,
    extract_instances,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_linkbases_state import (
    LABEL_PAYLOAD,
    classify_instances,
    new_labels,
)
from src.data_pipeline.storage.iceberg import (
    append_rows,
    current_snapshot_id,
    group_rows_by,
    open_silver_catalog,
    verify_append,
    verify_written_groups,
    verify_written_values,
)
from src.data_pipeline.storage.sec_filings.documents import (
    open_documents_table,
    read_documents,
)
from src.data_pipeline.storage.sec_filings.xbrl_core import (
    XBRL_INSTANCES_TABLE,
    open_xbrl_core_table,
    read_instance_groups,
)
from src.data_pipeline.storage.sec_filings.xbrl_facts import (
    open_xbrl_concepts_table,
    read_concept_hashes,
)
from src.data_pipeline.storage.sec_filings.xbrl_linkbases import (
    RELATIONSHIP_HEAD_COLUMNS,
    XBRL_LABELS_TABLE,
    XBRL_NETWORKS_TABLE,
    XBRL_RELATIONSHIPS_TABLE,
    open_xbrl_linkbase_table,
    read_concept_references,
    read_label_ids,
    read_relationship_heads,
)
from src.data_pipeline.transforms.sec_filings.xbrl_processor import ProcessorSettings
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)


@dataclass
class XbrlLinkbasesStageSummary:
    """Outcome of one run of `xbrl_labels`, `xbrl_networks` and `xbrl_relationships`.

    Each instance is counted once: skipped (networks and relationships
    complete, not loaded), succeeded (loaded, its rows built), conflicted or
    failed. Labels are shared between instances: discovered counts the
    distinct labels of the loaded instances, skipped those already stored
    identically, conflicted those stored with another payload.
    Relationship counts by linkbase type are of the rows appended. A rejected
    run writes nothing.
    """

    status: Literal["completed", "rejected"] = "rejected"
    instances_discovered: int = 0
    instances_succeeded: int = 0
    instances_skipped: int = 0
    instances_conflicted: int = 0
    instances_failed: int = 0
    labels_discovered: int = 0
    labels_appended: int = 0
    labels_skipped: int = 0
    labels_conflicted: int = 0
    networks_discovered: int = 0
    networks_appended: int = 0
    networks_skipped: int = 0
    relationships_discovered: int = 0
    relationships_appended: int = 0
    relationships_skipped: int = 0
    presentation_relationships: int = 0
    calculation_relationships: int = 0
    definition_relationships: int = 0
    labels_snapshot_before: int | None = None
    labels_snapshot_after: int | None = None
    networks_snapshot_before: int | None = None
    networks_snapshot_after: int | None = None
    relationships_snapshot_before: int | None = None
    relationships_snapshot_after: int | None = None
    duration_seconds: float = 0.0


def _read_parents(
    catalog: Catalog, data_dir: Path
) -> tuple[list[XbrlInstanceRecord], Mapping[str, DocumentRecord], set[str]]:
    instances = [
        XbrlInstanceRecord.model_validate(row)
        for rows in read_instance_groups(
            open_xbrl_core_table(catalog, data_dir, XBRL_INSTANCES_TABLE)[0]
        ).values()
        for row in rows
    ]
    documents = {
        d.document_id: d
        for d in read_documents(open_documents_table(catalog, data_dir)[0])
    }
    concepts = set(read_concept_hashes(open_xbrl_concepts_table(catalog, data_dir)[0]))
    return sorted(instances, key=lambda i: i.instance_id), documents, concepts


def _write(
    tables: dict[str, Table],
    labels: list[XbrlLabelRecord],
    extraction: LinkbaseExtraction,
    stored_counts: dict[str, int],
    summary: XbrlLinkbasesStageSummary,
) -> None:
    """Append and verify labels, then networks, then relationships."""

    networks = sorted(
        extraction.networks, key=lambda n: (n.instance_id, n.network_ordinal)
    )
    relationships = sorted(
        extraction.relationships, key=lambda r: (r.network_id, r.relationship_ordinal)
    )
    batches = (
        (XBRL_LABELS_TABLE, labels, XBRL_LABELS_ARROW_SCHEMA, "label_id"),
        (XBRL_NETWORKS_TABLE, networks, XBRL_NETWORKS_ARROW_SCHEMA, "network_id"),
        (
            XBRL_RELATIONSHIPS_TABLE,
            relationships,
            XBRL_RELATIONSHIPS_ARROW_SCHEMA,
            "relationship_id",
        ),
    )
    snapshots = {}
    for identifier, rows, schema, key in batches:
        table = tables[identifier]
        before = current_snapshot_id(table)
        if rows:
            with timed_operation(
                f"xbrl_linkbases.write_{identifier.split('_')[-1]}", records=len(rows)
            ):
                append_rows(table, rows, schema)
        with timed_operation("xbrl_linkbases.verify", table=identifier):
            snapshots[identifier] = verify_append(
                table,
                key_column=key,
                expected_rows=stored_counts[identifier] + len(rows),
                inserted_keys=[getattr(row, key) for row in rows],
                snapshot_before=before,
            )
            if identifier == XBRL_NETWORKS_TABLE:
                verify_written_groups(
                    table, rows, key_column=key, group_column="instance_id"
                )
            elif identifier == XBRL_LABELS_TABLE:
                verify_written_values(
                    table, rows, key_column=key, columns=LABEL_PAYLOAD
                )
            else:
                verify_written_values(
                    table,
                    rows,
                    key_column=key,
                    columns=("network_id", *RELATIONSHIP_HEAD_COLUMNS[1:]),
                )
    summary.labels_snapshot_after = snapshots[XBRL_LABELS_TABLE]
    summary.networks_snapshot_after = snapshots[XBRL_NETWORKS_TABLE]
    summary.relationships_snapshot_after = snapshots[XBRL_RELATIONSHIPS_TABLE]


def _count(
    extraction: LinkbaseExtraction,
    labels: list[XbrlLabelRecord],
    summary: XbrlLinkbasesStageSummary,
) -> None:
    discovered = extraction.discovered
    summary.labels_discovered = len(extraction.labels)
    summary.labels_appended = len(labels)
    summary.labels_skipped = len(extraction.labels) - len(labels)
    summary.networks_discovered = discovered["networks"]
    summary.networks_appended = len(extraction.networks)
    summary.networks_skipped = discovered["networks"] - len(extraction.networks)
    summary.relationships_discovered = discovered["relationships"]
    summary.relationships_appended = len(extraction.relationships)
    summary.relationships_skipped = discovered["relationships"] - len(
        extraction.relationships
    )
    summary.presentation_relationships = discovered["presentation"]
    summary.calculation_relationships = discovered["calculation"]
    summary.definition_relationships = discovered["definition"]


def _run_linkbases(
    catalog: Catalog,
    data_dir: Path,
    settings: ProcessorSettings,
    processed_at: datetime,
    summary: XbrlLinkbasesStageSummary,
) -> None:
    tables = {}
    for identifier in (
        XBRL_LABELS_TABLE,
        XBRL_NETWORKS_TABLE,
        XBRL_RELATIONSHIPS_TABLE,
    ):
        tables[identifier], created = open_xbrl_linkbase_table(
            catalog, data_dir, identifier
        )
        name = identifier.split(".", 1)[1]
        logger.info(
            f"{name}_table_{'created' if created else 'loaded'}", table=identifier
        )
    summary.labels_snapshot_before = summary.labels_snapshot_after = (
        current_snapshot_id(tables[XBRL_LABELS_TABLE])
    )
    summary.networks_snapshot_before = summary.networks_snapshot_after = (
        current_snapshot_id(tables[XBRL_NETWORKS_TABLE])
    )
    summary.relationships_snapshot_before = summary.relationships_snapshot_after = (
        current_snapshot_id(tables[XBRL_RELATIONSHIPS_TABLE])
    )
    with timed_operation("xbrl_linkbases.read_parents"):
        instances, documents, concept_ids = _read_parents(catalog, data_dir)
        stored_networks = group_rows_by(tables[XBRL_NETWORKS_TABLE], "instance_id")
        heads = read_relationship_heads(tables[XBRL_RELATIONSHIPS_TABLE])
        label_ids = read_label_ids(tables[XBRL_LABELS_TABLE])
        references = read_concept_references(
            tables[XBRL_LABELS_TABLE], tables[XBRL_RELATIONSHIPS_TABLE]
        )
    summary.instances_discovered = len(instances)
    with timed_operation("xbrl_linkbases.validate", phase="existing_state"):
        plan = classify_instances(instances, documents, stored_networks, heads)
    missing = sorted(references - concept_ids)
    if missing:
        plan.conflicts.append(
            (
                {"table": "xbrl_concepts"},
                f"stored rows refer to missing concepts {missing[:5]}",
            )
        )
    for context, reason in plan.conflicts:
        logger.error("xbrl_linkbases_conflict", reason=reason, **context)
    summary.instances_conflicted = len(plan.conflicts)
    summary.instances_skipped = plan.skipped
    logger.info("xbrl_linkbases_skipped", instances=plan.skipped)
    if plan.conflicts:
        return

    inputs = LinkbaseInputs(documents, concept_ids, data_dir, settings, processed_at)
    extraction = extract_instances(plan.candidates, inputs)
    summary.instances_succeeded = extraction.succeeded
    summary.instances_failed = extraction.failures
    summary.instances_conflicted += extraction.conflicts
    if summary.instances_failed or summary.instances_conflicted:
        return
    with timed_operation("xbrl_linkbases.validate", phase="stored_labels"):
        labels, differ = new_labels(
            tables[XBRL_LABELS_TABLE], extraction.labels, label_ids
        )
    for label_id in differ[:20]:
        logger.error(
            "xbrl_linkbases_conflict", label_id=label_id, reason="stored label differs"
        )
    summary.labels_conflicted = len(differ)
    if differ:
        return
    stored_counts = {
        XBRL_LABELS_TABLE: len(label_ids),
        XBRL_NETWORKS_TABLE: sum(map(len, stored_networks.values())),
        XBRL_RELATIONSHIPS_TABLE: sum(map(len, heads.values())),
    }
    _write(tables, labels, extraction, stored_counts, summary)
    _count(extraction, labels, summary)
    logger.info(
        "xbrl_linkbases_appended",
        labels=summary.labels_appended,
        networks=summary.networks_appended,
        relationships=summary.relationships_appended,
    )
    summary.status = "completed"


def run_xbrl_linkbases_stage(
    data_dir: Path, processed_at: datetime, settings: ProcessorSettings
) -> XbrlLinkbasesStageSummary:
    """Build `xbrl_labels`, `xbrl_networks` and `xbrl_relationships` from the DTSs.

    Instances whose networks and relationships are complete are skipped
    without loading XBRL; every other instance is loaded once and gives its
    concept labels and its presentation, calculation and definition networks.
    The whole batch is validated before labels, networks and relationships
    are appended in that order and re-read. Any conflict or failure rejects
    the batch; unexpected errors are logged and raised.
    """

    started = perf_counter()
    summary = XbrlLinkbasesStageSummary()
    with log_context(source="sec_filings", stage="xbrl_linkbases"):
        logger.info(
            "xbrl_linkbases_started",
            taxonomy_cache_dir=str(settings.cache_dir),
            allow_taxonomy_network=settings.allow_network,
        )
        try:
            with (
                timed_operation("xbrl_linkbases.total"),
                open_silver_catalog(data_dir) as catalog,
            ):
                _run_linkbases(catalog, data_dir, settings, processed_at, summary)
        except Exception as error:
            logger.error(
                "xbrl_linkbases_failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise

        summary.duration_seconds = round(perf_counter() - started, 3)
        if summary.status == "completed":
            logger.info("xbrl_linkbases_completed", **asdict(summary))
        else:
            logger.error("xbrl_linkbases_failed", **asdict(summary))
    return summary

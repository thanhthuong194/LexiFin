from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import batched
from pathlib import Path
from time import perf_counter
from typing import Literal

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.documents import (
    DOCUMENTS_ARROW_SCHEMA,
    DocumentRecord,
)
from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlInstanceRecord
from src.data_pipeline.contracts.sec_filings.xbrl_labels import (
    XBRL_LABELS_ARROW_SCHEMA,
    XbrlLabelRecord,
)
from src.data_pipeline.contracts.sec_filings.xbrl_networks import (
    XBRL_NETWORKS_ARROW_SCHEMA,
    XBRL_RELATIONSHIPS_ARROW_SCHEMA,
    XbrlNetworkRecord,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_linkbases_extraction import (
    LinkbaseExtraction,
    LinkbaseInputs,
    extract_instances,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_linkbases_state import (
    LABEL_PAYLOAD,
    InstanceState,
    StoredLinkbases,
    classify_instances,
    new_labels,
)
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.storage.iceberg import (
    append_rows,
    current_snapshot_id,
    filing_dates_filter,
    iter_rows,
    open_silver_catalog,
    read_groups_in,
    read_rows_in,
    snapshot_total,
    summarize_groups,
    verify_append,
    verify_written_groups,
    verify_written_values,
)
from src.data_pipeline.storage.sec_filings.documents import open_documents_table
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
)
from src.data_pipeline.transforms.sec_filings.xbrl_processor import ProcessorSettings
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)

# Instances loaded with their DTS, validated and committed together.
INSTANCES_PER_COMMIT = 100


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


@dataclass
class _Run:
    """What every batch of one run shares and updates."""

    catalog: Catalog
    data_dir: Path
    bronze: SecBronze
    settings: ProcessorSettings
    processed_at: datetime
    sources: Mapping[str, DocumentRecord]
    concept_ids: set[str]
    label_ids: set[str]


def _read_state(
    catalog: Catalog, data_dir: Path, tables: Mapping[str, Table]
) -> tuple[list[XbrlInstanceRecord], dict[str, DocumentRecord], StoredLinkbases]:
    """Instances, their source documents, and summaries of the stored networks."""

    instances = [
        XbrlInstanceRecord.model_validate(row)
        for rows in read_instance_groups(
            open_xbrl_core_table(catalog, data_dir, XBRL_INSTANCES_TABLE)[0]
        ).values()
        for row in rows
    ]
    sources = {
        row["document_id"]: DocumentRecord.model_validate(row)
        for row in read_rows_in(
            open_documents_table(catalog, data_dir)[0],
            "document_id",
            {d for instance in instances for d in instance.source_document_ids},
            columns=DOCUMENTS_ARROW_SCHEMA.names,
        )
    }
    heads: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in iter_rows(
        tables[XBRL_NETWORKS_TABLE], ("instance_id", "network_id", "relationship_count")
    ):
        heads[row["instance_id"]].append((row["network_id"], row["relationship_count"]))
    stored = StoredLinkbases(
        networks=summarize_groups(
            tables[XBRL_NETWORKS_TABLE],
            "instance_id",
            "network_ordinal",
            declared="instance_network_count",
        ),
        heads=dict(heads),
        relationships=summarize_groups(
            tables[XBRL_RELATIONSHIPS_TABLE],
            "network_id",
            "relationship_ordinal",
            declared="network_relationship_count",
        ),
    )
    return sorted(instances, key=lambda i: i.instance_id), sources, stored


def _write(
    tables: Mapping[str, Table],
    labels: list[XbrlLabelRecord],
    extraction: LinkbaseExtraction,
    summary: XbrlLinkbasesStageSummary,
) -> None:
    """Append and verify the batch's labels, then networks, then relationships."""

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
        stored = snapshot_total(table, "total-records")
        if rows:
            with timed_operation(
                f"xbrl_linkbases.write_{identifier.split('_')[-1]}", records=len(rows)
            ):
                append_rows(table, rows, schema)
        with timed_operation("xbrl_linkbases.verify", table=identifier):
            snapshots[identifier] = verify_append(
                table,
                key_column=key,
                expected_rows=stored + len(rows),
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
    summary.labels_discovered += len(extraction.labels)
    summary.labels_appended += len(labels)
    summary.labels_skipped += len(extraction.labels) - len(labels)
    summary.networks_discovered += discovered["networks"]
    summary.networks_appended += len(extraction.networks)
    summary.networks_skipped += discovered["networks"] - len(extraction.networks)
    summary.relationships_discovered += discovered["relationships"]
    summary.relationships_appended += len(extraction.relationships)
    summary.relationships_skipped += discovered["relationships"] - len(
        extraction.relationships
    )
    summary.presentation_relationships += discovered["presentation"]
    summary.calculation_relationships += discovered["calculation"]
    summary.definition_relationships += discovered["definition"]


def _batch_states(
    run: _Run, tables: Mapping[str, Table], batch: Sequence[InstanceState]
) -> tuple[list[InstanceState], dict[str, DocumentRecord]]:
    """The batch's instances with their stored networks, and its filings' documents."""

    instances = [state.instance for state in batch]
    dates = {
        run.sources[d].filing_date
        for instance in instances
        for d in instance.source_document_ids
        if d in run.sources
    }
    documents = {
        row["document_id"]: DocumentRecord.model_validate(row)
        for rows in read_groups_in(
            open_documents_table(run.catalog, run.data_dir)[0],
            "filing_id",
            {instance.filing_id for instance in instances},
            files=filing_dates_filter(dates),
        ).values()
        for row in rows
    }
    networks = read_groups_in(
        tables[XBRL_NETWORKS_TABLE],
        "instance_id",
        {instance.instance_id for instance in instances},
    )
    states = [
        InstanceState(
            instance,
            sorted(
                (
                    XbrlNetworkRecord.model_validate(row)
                    for row in networks.get(instance.instance_id, [])
                ),
                key=lambda network: network.network_ordinal,
            ),
        )
        for instance in instances
    ]
    return states, documents


def _run_batch(
    run: _Run,
    tables: Mapping[str, Table],
    batch: Sequence[InstanceState],
    summary: XbrlLinkbasesStageSummary,
) -> bool:
    """Load, validate and commit one batch of instances; return whether written."""

    states, documents = _batch_states(run, tables, batch)
    inputs = LinkbaseInputs(
        documents, run.concept_ids, run.bronze, run.settings, run.processed_at
    )
    extraction = extract_instances(states, inputs)
    summary.instances_succeeded += extraction.succeeded
    summary.instances_failed += extraction.failures
    summary.instances_conflicted += extraction.conflicts
    if summary.instances_failed or summary.instances_conflicted:
        return False
    with timed_operation("xbrl_linkbases.validate", phase="stored_labels"):
        labels, differ = new_labels(
            tables[XBRL_LABELS_TABLE], extraction.labels, run.label_ids
        )
    for label_id in differ[:20]:
        logger.error(
            "xbrl_linkbases_conflict", label_id=label_id, reason="stored label differs"
        )
    summary.labels_conflicted += len(differ)
    if differ:
        return False
    _write(tables, labels, extraction, summary)
    _count(extraction, labels, summary)
    run.label_ids.update(label.label_id for label in labels)
    return True


def _run_linkbases(
    catalog: Catalog,
    dirs: tuple[Path, SecBronze],
    settings: ProcessorSettings,
    processed_at: datetime,
    summary: XbrlLinkbasesStageSummary,
) -> None:
    data_dir, bronze = dirs
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
        instances, sources, stored = _read_state(catalog, data_dir, tables)
        run = _Run(
            catalog,
            data_dir,
            bronze,
            settings,
            processed_at,
            sources,
            set(read_concept_hashes(open_xbrl_concepts_table(catalog, data_dir)[0])),
            read_label_ids(tables[XBRL_LABELS_TABLE]),
        )
        references = read_concept_references(
            tables[XBRL_LABELS_TABLE], tables[XBRL_RELATIONSHIPS_TABLE]
        )
    summary.instances_discovered = len(instances)
    with timed_operation("xbrl_linkbases.validate", phase="existing_state"):
        plan = classify_instances(instances, sources, stored)
    missing = sorted(references - run.concept_ids)
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
    for batch in batched(plan.candidates, INSTANCES_PER_COMMIT, strict=False):
        if not _run_batch(run, tables, batch, summary):
            return
    logger.info(
        "xbrl_linkbases_appended",
        labels=summary.labels_appended,
        networks=summary.networks_appended,
        relationships=summary.relationships_appended,
    )
    summary.status = "completed"


def run_xbrl_linkbases_stage(
    data_dir: Path,
    processed_at: datetime,
    settings: ProcessorSettings,
    bronze: SecBronze | None = None,
) -> XbrlLinkbasesStageSummary:
    """Build `xbrl_labels`, `xbrl_networks` and `xbrl_relationships` from the DTSs.

    Instances whose networks and relationships are complete are skipped
    without loading XBRL; every other instance is loaded once and gives its
    concept labels and its presentation, calculation and definition networks.
    Stored groups are judged by their summaries; any conflict rejects the
    run before the first write. Instances are loaded and committed in batches
    of INSTANCES_PER_COMMIT: labels, networks and relationships appended in
    that order and re-read. A conflict or failure rejects its batch; earlier
    batches stay committed. Unexpected errors are logged and raised.
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
                _run_linkbases(
                    catalog,
                    (data_dir, bronze or SecBronze.under(data_dir)),
                    settings,
                    processed_at,
                    summary,
                )
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

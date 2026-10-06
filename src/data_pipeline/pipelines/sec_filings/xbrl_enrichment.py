from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlInstanceRecord
from src.data_pipeline.contracts.sec_filings.xbrl_dimensions import (
    XBRL_DIMENSIONS_ARROW_SCHEMA,
)
from src.data_pipeline.contracts.sec_filings.xbrl_fact_footnotes import (
    XBRL_FACT_FOOTNOTES_ARROW_SCHEMA,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_enrichment_extraction import (
    EnrichmentExtraction,
    EnrichmentInputs,
    extract_instances,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_enrichment_state import (
    EnrichmentParents,
    InstanceState,
    classify_instances,
    declares_footnotes,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_facts_state import InstanceRefs
from src.data_pipeline.storage.iceberg import (
    append_rows,
    current_snapshot_id,
    group_rows_by,
    open_silver_catalog,
    verify_append,
    verify_written_groups,
)
from src.data_pipeline.storage.sec_filings.documents import (
    open_documents_table,
    read_documents,
)
from src.data_pipeline.storage.sec_filings.xbrl_core import (
    XBRL_CONTEXTS_TABLE,
    XBRL_INSTANCES_TABLE,
    XBRL_UNITS_TABLE,
    open_xbrl_core_table,
    read_instance_groups,
)
from src.data_pipeline.storage.sec_filings.xbrl_enrichment import (
    XBRL_DIMENSIONS_TABLE,
    XBRL_FACT_FOOTNOTES_TABLE,
    open_xbrl_dimensions_table,
    open_xbrl_fact_footnotes_table,
    read_contexts_by_instance,
    read_fact_hashes,
)
from src.data_pipeline.storage.sec_filings.xbrl_facts import (
    open_xbrl_concepts_table,
    open_xbrl_facts_table,
    read_concept_hashes,
    read_refs_by_instance,
)
from src.data_pipeline.transforms.sec_filings.xbrl_processor import ProcessorSettings
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)


@dataclass
class XbrlEnrichmentStageSummary:
    """Outcome of one XBRL enrichment run (`xbrl_dimensions`, `xbrl_fact_footnotes`).

    Each instance is counted once: skipped (both tables complete, or nothing
    new after loading), inserted, resumed (rows added to the table that
    missed them while the other table already held its rows), conflicted or
    failed. Context and fact counts describe the instances loaded in this
    run; dimension, relationship and resource counts the rows inserted. A
    rejected run writes nothing.
    """

    status: Literal["completed", "rejected"] = "rejected"
    instances_considered: int = 0
    instances_inserted: int = 0
    instances_skipped: int = 0
    instances_resumed: int = 0
    instances_conflicted: int = 0
    instances_failed: int = 0
    contexts_considered: int = 0
    contexts_with_dimensions: int = 0
    contexts_without_dimensions: int = 0
    dimensions_inserted: int = 0
    explicit_dimensions: int = 0
    typed_dimensions: int = 0
    facts_considered: int = 0
    facts_with_footnotes: int = 0
    facts_without_footnotes: int = 0
    footnote_relationships_inserted: int = 0
    unique_footnote_resources: int = 0
    empty_footnotes: int = 0
    dimensions_snapshot_before: int | None = None
    dimensions_snapshot_after: int | None = None
    footnotes_snapshot_before: int | None = None
    footnotes_snapshot_after: int | None = None
    duration_seconds: float = 0.0


def _read_parents(
    catalog: Catalog, data_dir: Path
) -> tuple[list[XbrlInstanceRecord], EnrichmentParents, InstanceRefs]:
    instances = [
        XbrlInstanceRecord.model_validate(row)
        for rows in read_instance_groups(
            open_xbrl_core_table(catalog, data_dir, XBRL_INSTANCES_TABLE)[0]
        ).values()
        for row in rows
    ]
    contexts = read_contexts_by_instance(
        open_xbrl_core_table(catalog, data_dir, XBRL_CONTEXTS_TABLE)[0]
    )
    parents = EnrichmentParents(
        documents={
            d.document_id: d
            for d in read_documents(open_documents_table(catalog, data_dir)[0])
        },
        contexts=contexts,
        fact_hashes=read_fact_hashes(open_xbrl_facts_table(catalog, data_dir)[0]),
        concept_ids=set(
            read_concept_hashes(open_xbrl_concepts_table(catalog, data_dir)[0])
        ),
    )
    refs = InstanceRefs(
        contexts={i: {c["context_ref"] for c in rows} for i, rows in contexts.items()},
        units=read_refs_by_instance(
            open_xbrl_core_table(catalog, data_dir, XBRL_UNITS_TABLE)[0], "unit_ref"
        ),
    )
    return sorted(instances, key=lambda i: i.instance_id), parents, refs


def _is_complete(
    state: InstanceState, parents: EnrichmentParents, data_dir: Path
) -> bool:
    """Whether an instance needs no load: both tables hold all its rows.

    With no stored footnote, the instance's documents tell whether it has any.
    """

    if not state.dimensions_stored:
        return False
    if state.footnotes_stored is None:
        return not declares_footnotes(state.instance, parents.documents, data_dir)
    return True


def _write(
    tables: tuple[Table, Table],
    extraction: EnrichmentExtraction,
    stored_counts: tuple[int, int],
    summary: XbrlEnrichmentStageSummary,
) -> None:
    """Append and verify the new dimensions, then the new footnote relationships."""

    dimensions_table, footnotes_table = tables
    dimensions = sorted(
        (row for rows in extraction.dimensions.values() for row in rows),
        key=lambda d: (d.context_key, d.dimension_ordinal),
    )
    footnotes = sorted(
        (row for rows in extraction.footnotes.values() for row in rows),
        key=lambda f: (f.fact_id, f.footnote_ordinal),
    )
    if dimensions:
        with timed_operation(
            "xbrl_enrichment.write_dimensions", records=len(dimensions)
        ):
            append_rows(dimensions_table, dimensions, XBRL_DIMENSIONS_ARROW_SCHEMA)
    with timed_operation("xbrl_enrichment.verify", table=XBRL_DIMENSIONS_TABLE):
        summary.dimensions_snapshot_after = verify_append(
            dimensions_table,
            key_column="dimension_id",
            expected_rows=stored_counts[0] + len(dimensions),
            inserted_keys=[d.dimension_id for d in dimensions],
            snapshot_before=summary.dimensions_snapshot_before,
        )
        verify_written_groups(
            dimensions_table,
            dimensions,
            key_column="dimension_id",
            group_column="context_key",
        )
    if footnotes:
        with timed_operation("xbrl_enrichment.write_footnotes", records=len(footnotes)):
            append_rows(footnotes_table, footnotes, XBRL_FACT_FOOTNOTES_ARROW_SCHEMA)
    with timed_operation("xbrl_enrichment.verify", table=XBRL_FACT_FOOTNOTES_TABLE):
        summary.footnotes_snapshot_after = verify_append(
            footnotes_table,
            key_column="footnote_relation_id",
            expected_rows=stored_counts[1] + len(footnotes),
            inserted_keys=[f.footnote_relation_id for f in footnotes],
            snapshot_before=summary.footnotes_snapshot_before,
        )
        verify_written_groups(
            footnotes_table,
            footnotes,
            key_column="footnote_relation_id",
            group_column="fact_id",
        )
    summary.dimensions_inserted = len(dimensions)
    summary.explicit_dimensions = sum(d.member_type == "explicit" for d in dimensions)
    summary.typed_dimensions = len(dimensions) - summary.explicit_dimensions
    summary.footnote_relationships_inserted = len(footnotes)
    summary.unique_footnote_resources = len({f.footnote_resource_id for f in footnotes})


def _count(
    extraction: EnrichmentExtraction, summary: XbrlEnrichmentStageSummary
) -> None:
    read = extraction.read
    summary.instances_skipped += extraction.unchanged
    summary.instances_resumed = len(extraction.resumed)
    summary.instances_inserted = len(extraction.dimensions) - len(extraction.resumed)
    summary.contexts_considered = read["contexts"]
    summary.contexts_with_dimensions = read["contexts_with_dimensions"]
    summary.contexts_without_dimensions = (
        read["contexts"] - read["contexts_with_dimensions"]
    )
    summary.facts_considered = read["facts"]
    summary.facts_with_footnotes = read["facts_with_footnotes"]
    summary.facts_without_footnotes = read["facts"] - read["facts_with_footnotes"]
    summary.empty_footnotes = read["empty_footnotes"]


def _run_enrichment(
    catalog: Catalog,
    data_dir: Path,
    settings: ProcessorSettings,
    processed_at: datetime,
    summary: XbrlEnrichmentStageSummary,
) -> None:
    tables = {}
    for identifier, opener in (
        (XBRL_DIMENSIONS_TABLE, open_xbrl_dimensions_table),
        (XBRL_FACT_FOOTNOTES_TABLE, open_xbrl_fact_footnotes_table),
    ):
        tables[identifier], created = opener(catalog, data_dir)
        name = identifier.split(".", 1)[1]
        logger.info(
            f"{name}_table_{'created' if created else 'loaded'}", table=identifier
        )
    dimensions_table = tables[XBRL_DIMENSIONS_TABLE]
    footnotes_table = tables[XBRL_FACT_FOOTNOTES_TABLE]
    summary.dimensions_snapshot_before = summary.dimensions_snapshot_after = (
        current_snapshot_id(dimensions_table)
    )
    summary.footnotes_snapshot_before = summary.footnotes_snapshot_after = (
        current_snapshot_id(footnotes_table)
    )
    with timed_operation("xbrl_enrichment.read_parents"):
        instances, parents, refs = _read_parents(catalog, data_dir)
        stored_dimensions = group_rows_by(dimensions_table, "context_key")
        stored_footnotes = group_rows_by(footnotes_table, "fact_id")
    summary.instances_considered = len(instances)
    logger.info(
        "xbrl_enrichment_parents_loaded",
        instances=len(instances),
        contexts=sum(map(len, parents.contexts.values())),
        facts=sum(map(len, parents.fact_hashes.values())),
        concepts=len(parents.concept_ids),
    )
    with timed_operation("xbrl_enrichment.validate", phase="existing_state"):
        plan = classify_instances(
            instances, parents, stored_dimensions, stored_footnotes
        )
        candidates = [s for s in plan.states if not _is_complete(s, parents, data_dir)]
    for context, reason in plan.conflicts:
        logger.error("xbrl_enrichment_instance_conflict", reason=reason, **context)
    summary.instances_conflicted = len(plan.conflicts)
    summary.instances_skipped = len(plan.states) - len(candidates)
    logger.info("xbrl_enrichment_instance_skipped", instances=summary.instances_skipped)
    if plan.conflicts:
        return

    inputs = EnrichmentInputs(
        parents,
        refs,
        stored_dimensions,
        stored_footnotes,
        data_dir,
        settings,
        processed_at,
    )
    extraction = extract_instances(candidates, inputs)
    summary.instances_failed = extraction.failures
    summary.instances_conflicted += extraction.conflicts
    _count(extraction, summary)
    if summary.instances_failed or summary.instances_conflicted:
        return
    stored_counts = (
        sum(map(len, stored_dimensions.values())),
        sum(map(len, stored_footnotes.values())),
    )
    _write((dimensions_table, footnotes_table), extraction, stored_counts, summary)
    logger.info(
        "xbrl_enrichment_instance_inserted", instances=summary.instances_inserted
    )
    logger.info("xbrl_enrichment_instance_resumed", instances=summary.instances_resumed)
    summary.status = "completed"


def run_xbrl_enrichment_stage(
    data_dir: Path, processed_at: datetime, settings: ProcessorSettings
) -> XbrlEnrichmentStageSummary:
    """Build `xbrl_dimensions` and `xbrl_fact_footnotes` from the stored XBRL rows.

    Instances whose rows are complete in both tables (or that declare no
    footnote and report no dimension) are skipped without loading XBRL.
    Every other instance is loaded once with its DTS; its contexts and facts
    must match the stored rows. The whole batch is validated before new
    dimensions, then footnote relationships, are appended and re-read. Any
    conflict or failure rejects the batch; unexpected errors are logged and
    raised.
    """

    started = perf_counter()
    summary = XbrlEnrichmentStageSummary()
    with log_context(source="sec_filings"):
        logger.info(
            "xbrl_enrichment_stage_started",
            taxonomy_cache_dir=str(settings.cache_dir),
            allow_taxonomy_network=settings.allow_network,
        )
        try:
            with (
                timed_operation("xbrl_enrichment.total"),
                open_silver_catalog(data_dir) as catalog,
            ):
                _run_enrichment(catalog, data_dir, settings, processed_at, summary)
        except Exception as error:
            logger.error(
                "xbrl_enrichment_stage_failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise

        summary.duration_seconds = round(perf_counter() - started, 3)
        if summary.status == "completed":
            logger.info("xbrl_enrichment_stage_completed", **asdict(summary))
        else:
            logger.error("xbrl_enrichment_stage_failed", **asdict(summary))
    return summary

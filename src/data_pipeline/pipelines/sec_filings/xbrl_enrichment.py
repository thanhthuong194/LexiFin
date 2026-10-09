from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
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
    StoredEnrichment,
    classify_instances,
    declares_footnotes,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_facts_state import InstanceRefs
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.storage.iceberg import (
    append_rows,
    count_groups,
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
)
from src.data_pipeline.storage.sec_filings.documents import open_documents_table
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
from src.data_pipeline.transforms.sec_filings.xbrl_dimensions import (
    expected_dimension_count,
)
from src.data_pipeline.transforms.sec_filings.xbrl_processor import ProcessorSettings
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)

# Instances loaded with their DTS, validated and committed together.
INSTANCES_PER_COMMIT = 100


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


@dataclass(frozen=True)
class _Run:
    """What every batch of one run shares."""

    catalog: Catalog
    data_dir: Path
    bronze: SecBronze
    settings: ProcessorSettings
    processed_at: datetime
    sources: Mapping[str, DocumentRecord]
    concept_ids: Collection[str]


def _expected_dimensions(contexts: Table) -> dict[str, dict[str, int]]:
    """Dimension members each stored context reports, by instance then context_key."""

    expected: dict[str, dict[str, int]] = defaultdict(dict)
    columns = ("instance_id", "context_key", "segment_xml", "scenario_xml")
    for row in iter_rows(contexts, columns):
        expected[row["instance_id"]][row["context_key"]] = expected_dimension_count(
            row["segment_xml"], row["scenario_xml"]
        )
    return dict(expected)


def _read_state(
    catalog: Catalog, data_dir: Path, tables: tuple[Table, Table]
) -> tuple[list[XbrlInstanceRecord], dict[str, DocumentRecord], StoredEnrichment]:
    """Instances, their source documents and summaries of the stored rows."""

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
    stored = StoredEnrichment(
        expected=_expected_dimensions(
            open_xbrl_core_table(catalog, data_dir, XBRL_CONTEXTS_TABLE)[0]
        ),
        fact_counts=count_groups(
            open_xbrl_facts_table(catalog, data_dir)[0], "instance_id"
        ),
        dimensions=summarize_groups(
            tables[0],
            "context_key",
            "dimension_ordinal",
            declared="context_dimension_count",
        ),
        footnotes=summarize_groups(
            tables[1], "fact_id", "footnote_ordinal", declared="fact_footnote_count"
        ),
    )
    return sorted(instances, key=lambda i: i.instance_id), sources, stored


def _is_complete(state: InstanceState, run: _Run) -> bool:
    """Whether an instance needs no load: both tables hold all its rows.

    With no stored footnote, the instance's documents tell whether it has any.
    """

    if not state.dimensions_stored:
        return False
    if state.footnotes_stored is None:
        return not declares_footnotes(state.instance, run.sources, run.bronze)
    return True


def _batch_inputs(
    run: _Run, tables: tuple[Table, Table], batch: Sequence[InstanceState]
) -> EnrichmentInputs:
    """Parents, refs and the stored rows of the batch's instances."""

    instances = [state.instance for state in batch]
    ids = {instance.instance_id for instance in instances}
    dates = {
        run.sources[d].filing_date
        for instance in instances
        for d in instance.source_document_ids
        if d in run.sources
    }
    contexts = read_contexts_by_instance(
        open_xbrl_core_table(run.catalog, run.data_dir, XBRL_CONTEXTS_TABLE)[0], ids
    )
    parents = EnrichmentParents(
        documents={
            row["document_id"]: DocumentRecord.model_validate(row)
            for rows in read_groups_in(
                open_documents_table(run.catalog, run.data_dir)[0],
                "filing_id",
                {instance.filing_id for instance in instances},
                files=filing_dates_filter(dates),
            ).values()
            for row in rows
        },
        contexts=contexts,
        fact_hashes=read_fact_hashes(
            open_xbrl_facts_table(run.catalog, run.data_dir)[0], ids
        ),
        concept_ids=run.concept_ids,
    )
    refs = InstanceRefs(
        contexts={i: {c["context_ref"] for c in rows} for i, rows in contexts.items()},
        units=read_refs_by_instance(
            open_xbrl_core_table(run.catalog, run.data_dir, XBRL_UNITS_TABLE)[0],
            "unit_ref",
            ids,
        ),
    )
    stored_ids = {state.instance.instance_id for state in batch if state.has_rows}
    return EnrichmentInputs(
        parents,
        refs,
        read_groups_in(
            tables[0],
            "context_key",
            {c["context_key"] for i in stored_ids for c in contexts.get(i, [])},
        ),
        read_groups_in(
            tables[1],
            "fact_id",
            {f for i in stored_ids for f in parents.fact_hashes.get(i, {})},
        ),
        run.bronze,
        run.settings,
        run.processed_at,
    )


def _write(
    tables: tuple[Table, Table],
    extraction: EnrichmentExtraction,
    summary: XbrlEnrichmentStageSummary,
) -> None:
    """Append and verify the batch's dimensions, then its footnote relationships."""

    dimensions_table, footnotes_table = tables
    stored_counts = (
        snapshot_total(dimensions_table, "total-records"),
        snapshot_total(footnotes_table, "total-records"),
    )
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
            snapshot_before=summary.dimensions_snapshot_after,
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
            snapshot_before=summary.footnotes_snapshot_after,
        )
        verify_written_groups(
            footnotes_table,
            footnotes,
            key_column="footnote_relation_id",
            group_column="fact_id",
        )
    explicit = sum(d.member_type == "explicit" for d in dimensions)
    summary.dimensions_inserted += len(dimensions)
    summary.explicit_dimensions += explicit
    summary.typed_dimensions += len(dimensions) - explicit
    summary.footnote_relationships_inserted += len(footnotes)
    summary.unique_footnote_resources += len(
        {f.footnote_resource_id for f in footnotes}
    )


def _count(
    extraction: EnrichmentExtraction, summary: XbrlEnrichmentStageSummary
) -> None:
    read = extraction.read
    summary.instances_skipped += extraction.unchanged
    summary.instances_resumed += len(extraction.resumed)
    summary.instances_inserted += len(extraction.dimensions) - len(extraction.resumed)
    summary.contexts_considered += read["contexts"]
    summary.contexts_with_dimensions += read["contexts_with_dimensions"]
    summary.contexts_without_dimensions += (
        read["contexts"] - read["contexts_with_dimensions"]
    )
    summary.facts_considered += read["facts"]
    summary.facts_with_footnotes += read["facts_with_footnotes"]
    summary.facts_without_footnotes += read["facts"] - read["facts_with_footnotes"]
    summary.empty_footnotes += read["empty_footnotes"]


def _run_batch(
    run: _Run,
    tables: tuple[Table, Table],
    batch: Sequence[InstanceState],
    summary: XbrlEnrichmentStageSummary,
) -> bool:
    """Load, validate and commit one batch of instances; return whether written."""

    extraction = extract_instances(batch, _batch_inputs(run, tables, batch))
    summary.instances_failed += extraction.failures
    summary.instances_conflicted += extraction.conflicts
    _count(extraction, summary)
    if summary.instances_failed or summary.instances_conflicted:
        return False
    _write(tables, extraction, summary)
    return True


def _run_enrichment(
    catalog: Catalog,
    dirs: tuple[Path, SecBronze],
    settings: ProcessorSettings,
    processed_at: datetime,
    summary: XbrlEnrichmentStageSummary,
) -> None:
    data_dir, bronze = dirs
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
    own = (tables[XBRL_DIMENSIONS_TABLE], tables[XBRL_FACT_FOOTNOTES_TABLE])
    summary.dimensions_snapshot_before = summary.dimensions_snapshot_after = (
        current_snapshot_id(own[0])
    )
    summary.footnotes_snapshot_before = summary.footnotes_snapshot_after = (
        current_snapshot_id(own[1])
    )
    with timed_operation("xbrl_enrichment.read_parents"):
        instances, sources, stored = _read_state(catalog, data_dir, own)
        run = _Run(
            catalog,
            data_dir,
            bronze,
            settings,
            processed_at,
            sources,
            set(read_concept_hashes(open_xbrl_concepts_table(catalog, data_dir)[0])),
        )
    summary.instances_considered = len(instances)
    logger.info(
        "xbrl_enrichment_parents_loaded",
        instances=len(instances),
        contexts=sum(map(len, stored.expected.values())),
        facts=sum(stored.fact_counts.values()),
        concepts=len(run.concept_ids),
    )
    with timed_operation("xbrl_enrichment.validate", phase="existing_state"):
        plan = classify_instances(instances, sources, stored)
        candidates = [s for s in plan.states if not _is_complete(s, run)]
    for context, reason in plan.conflicts:
        logger.error("xbrl_enrichment_instance_conflict", reason=reason, **context)
    summary.instances_conflicted = len(plan.conflicts)
    summary.instances_skipped = len(plan.states) - len(candidates)
    logger.info("xbrl_enrichment_instance_skipped", instances=summary.instances_skipped)
    if plan.conflicts:
        return
    for batch in batched(candidates, INSTANCES_PER_COMMIT, strict=False):
        if not _run_batch(run, own, batch, summary):
            return
    logger.info(
        "xbrl_enrichment_instance_inserted", instances=summary.instances_inserted
    )
    logger.info("xbrl_enrichment_instance_resumed", instances=summary.instances_resumed)
    summary.status = "completed"


def run_xbrl_enrichment_stage(
    data_dir: Path,
    processed_at: datetime,
    settings: ProcessorSettings,
    bronze: SecBronze | None = None,
) -> XbrlEnrichmentStageSummary:
    """Build `xbrl_dimensions` and `xbrl_fact_footnotes` from the stored XBRL rows.

    Instances whose rows are complete in both tables (or that declare no
    footnote and report no dimension) are skipped without loading XBRL.
    Every other instance is loaded once with its DTS; its contexts and facts
    must match the stored rows. Stored groups are judged by their summaries;
    any conflict rejects the run before the first write. Instances are loaded
    and committed in batches of INSTANCES_PER_COMMIT: dimensions, then
    footnote relationships, appended and re-read. A conflict or failure
    rejects its batch; earlier batches stay committed. Unexpected errors are
    logged and raised.
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
                _run_enrichment(
                    catalog,
                    (data_dir, bronze or SecBronze.under(data_dir)),
                    settings,
                    processed_at,
                    summary,
                )
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

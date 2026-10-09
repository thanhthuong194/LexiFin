from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import batched
from pathlib import Path
from time import perf_counter
from typing import Literal

import pyarrow.compute as pc
from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.documents import (
    DOCUMENTS_ARROW_SCHEMA,
    DocumentRecord,
)
from src.data_pipeline.contracts.sec_filings.xbrl_concepts import (
    XBRL_CONCEPTS_ARROW_SCHEMA,
)
from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlInstanceRecord
from src.data_pipeline.contracts.sec_filings.xbrl_facts import (
    XBRL_FACTS_ARROW_SCHEMA,
    XbrlFactRecord,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_facts_extraction import (
    ExtractionInputs,
    FactsExtraction,
    extract_instances,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_facts_state import (
    InstanceRefs,
    classify_instances,
    read_content,
    validate_built,
    verify_concepts,
)
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.storage.iceberg import (
    append_rows,
    current_snapshot_id,
    filing_dates_filter,
    open_silver_catalog,
    read_groups_in,
    read_rows_in,
    scan_batches,
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
from src.data_pipeline.storage.sec_filings.xbrl_facts import (
    XBRL_CONCEPTS_TABLE,
    XBRL_FACTS_TABLE,
    open_xbrl_concepts_table,
    open_xbrl_facts_table,
    read_concept_hashes,
    read_refs_by_instance,
)
from src.data_pipeline.transforms.sec_filings.xbrl_processor import ProcessorSettings
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)

# Instances loaded with their DTS, validated and committed together.
INSTANCES_PER_COMMIT = 100


@dataclass
class XbrlFactsStageSummary:
    """Outcome of one XBRL facts run (`xbrl_concepts` and `xbrl_facts`).

    Each instance is counted once: skipped (complete fact group, not
    reloaded), inserted, recovered (inserted after a run that stored its
    extension concepts but not its facts), conflicted (including disagreement with the XBRL
    core rows) or failed (unreadable source, missing taxonomy). Concept and
    fact breakdowns describe this run; `normalization_warnings` counts facts
    kept without a normalized or numeric value. A rejected run writes nothing.
    """

    status: Literal["completed", "rejected"] = "rejected"
    instances_considered: int = 0
    instances_inserted: int = 0
    instances_skipped: int = 0
    instances_recovered: int = 0
    instances_conflicted: int = 0
    instances_failed: int = 0
    concepts_discovered: int = 0
    concepts_inserted: int = 0
    concepts_skipped: int = 0
    standard_concepts: int = 0
    extension_concepts: int = 0
    facts_discovered: int = 0
    facts_inserted: int = 0
    numeric_facts: int = 0
    nonnumeric_facts: int = 0
    fraction_facts: int = 0
    nil_facts: int = 0
    hidden_facts: int = 0
    continued_facts: int = 0
    aligned_block_facts: int = 0
    aligned_cell_facts: int = 0
    unaligned_facts: int = 0
    normalization_warnings: int = 0
    duration_seconds: float = 0.0
    concepts_snapshot_before: int | None = None
    concepts_snapshot_after: int | None = None
    facts_snapshot_before: int | None = None
    facts_snapshot_after: int | None = None


def _count(
    extraction: FactsExtraction,
    facts: Sequence[XbrlFactRecord],
    summary: XbrlFactsStageSummary,
) -> None:
    concepts = list(extraction.concepts.values())
    kinds = Counter(fact.fact_kind for fact in facts)
    summary.concepts_inserted += len(concepts)
    summary.extension_concepts += sum(c.is_extension for c in concepts)
    summary.standard_concepts += len(concepts) - sum(c.is_extension for c in concepts)
    summary.facts_inserted += len(facts)
    summary.numeric_facts += kinds["numeric"]
    summary.nonnumeric_facts += kinds["non_numeric"]
    summary.fraction_facts += kinds["fraction"]
    summary.nil_facts += sum(fact.is_nil for fact in facts)
    summary.hidden_facts += sum(fact.is_hidden for fact in facts)
    summary.continued_facts += sum(bool(fact.continuation_refs) for fact in facts)
    summary.aligned_block_facts += extraction.locations["block"]
    summary.aligned_cell_facts += extraction.locations["cell"]
    summary.unaligned_facts += extraction.locations["none"]
    summary.instances_recovered += len(extraction.recovered)
    summary.instances_inserted += len(extraction.facts) - len(extraction.recovered)


def _write(
    tables: tuple[Table, Table],
    extraction: FactsExtraction,
    summary: XbrlFactsStageSummary,
) -> None:
    """Append and verify the batch's new concepts, then its fact groups."""

    concepts_table, facts_table = tables
    concepts = sorted(
        extraction.concepts.values(), key=lambda c: (c.namespace_uri, c.local_name)
    )
    facts = sorted(
        (fact for group in extraction.facts.values() for fact in group),
        key=lambda f: (f.instance_id, f.fact_order),
    )
    stored = (
        snapshot_total(concepts_table, "total-records"),
        snapshot_total(facts_table, "total-records"),
    )
    if concepts:
        with timed_operation("xbrl_facts.write_concepts", records=len(concepts)):
            append_rows(concepts_table, concepts, XBRL_CONCEPTS_ARROW_SCHEMA)
    with timed_operation("xbrl_facts.verify", table=XBRL_CONCEPTS_TABLE):
        summary.concepts_snapshot_after = verify_concepts(
            concepts_table, concepts, stored[0], summary.concepts_snapshot_after
        )
    if facts:
        with timed_operation("xbrl_facts.write_facts", records=len(facts)):
            append_rows(facts_table, facts, XBRL_FACTS_ARROW_SCHEMA)
    with timed_operation("xbrl_facts.verify", table=XBRL_FACTS_TABLE):
        summary.facts_snapshot_after = verify_append(
            facts_table,
            key_column="fact_id",
            expected_rows=stored[1] + len(facts),
            inserted_keys=[fact.fact_id for fact in facts],
            snapshot_before=summary.facts_snapshot_after,
        )
        verify_written_groups(
            facts_table, facts, key_column="fact_id", group_column="instance_id"
        )
    _count(extraction, facts, summary)
    logger.info(
        "xbrl_instance_facts_inserted",
        instances=len(extraction.facts) - len(extraction.recovered),
        facts=len(facts),
        concepts=len(concepts),
    )
    logger.info("xbrl_instance_facts_recovered", instances=len(extraction.recovered))


@dataclass
class _Run:
    """What every batch of one run shares and updates."""

    catalog: Catalog
    data_dir: Path
    bronze: SecBronze
    settings: ProcessorSettings
    processed_at: datetime
    instances: dict[str, XbrlInstanceRecord]
    sources: dict[str, DocumentRecord]
    concept_hashes: dict[str, str]
    fact_concepts: set[str]
    discovered: set[str] = field(default_factory=set)


def _batch_inputs(run: _Run, batch: Sequence[XbrlInstanceRecord]) -> ExtractionInputs:
    """Documents, refs and stored content of the batch's filings and instances."""

    filings = {instance.filing_id for instance in batch}
    dates = {
        run.sources[document_id].filing_date
        for instance in batch
        for document_id in instance.source_document_ids
        if document_id in run.sources
    }
    documents = {
        row["document_id"]: DocumentRecord.model_validate(row)
        for rows in read_groups_in(
            open_documents_table(run.catalog, run.data_dir)[0],
            "filing_id",
            filings,
            files=filing_dates_filter(dates),
        ).values()
        for row in rows
    }
    ids = {instance.instance_id for instance in batch}
    refs = InstanceRefs(
        contexts=read_refs_by_instance(
            open_xbrl_core_table(run.catalog, run.data_dir, XBRL_CONTEXTS_TABLE)[0],
            "context_ref",
            ids,
        ),
        units=read_refs_by_instance(
            open_xbrl_core_table(run.catalog, run.data_dir, XBRL_UNITS_TABLE)[0],
            "unit_ref",
            ids,
        ),
    )
    sources = [
        documents[document_id]
        for instance in batch
        for document_id in instance.source_document_ids
        if document_id in documents
    ]
    return ExtractionInputs(
        documents,
        refs,
        run.concept_hashes,
        run.fact_concepts,
        read_content(run.catalog, run.data_dir, sources),
        run.bronze,
        run.settings,
        run.processed_at,
    )


def _run_batch(
    run: _Run,
    tables: tuple[Table, Table],
    batch: Sequence[XbrlInstanceRecord],
    summary: XbrlFactsStageSummary,
) -> bool:
    """Load, validate and commit one batch of instances; return whether written."""

    inputs = _batch_inputs(run, batch)
    with timed_operation("xbrl_facts.extract", instances=len(batch)):
        extraction = extract_instances(batch, inputs)
    run.discovered |= extraction.definitions.keys()
    summary.facts_discovered += sum(map(len, extraction.facts.values()))
    summary.normalization_warnings += extraction.unusable_values
    summary.instances_failed += extraction.failures
    summary.instances_conflicted += extraction.conflicts + extraction.incompatible
    with timed_operation("xbrl_facts.validate", phase="built_records"):
        summary.instances_conflicted += validate_built(
            extraction.facts,
            run.instances,
            set(run.concept_hashes) | set(extraction.definitions),
            inputs.refs,
        )
    if summary.instances_failed or summary.instances_conflicted:
        return False
    _write(tables, extraction, summary)
    run.concept_hashes.update(
        (c.concept_id, c.concept_definition_sha256)
        for c in extraction.concepts.values()
    )
    run.fact_concepts.update(
        fact.concept_id for group in extraction.facts.values() for fact in group
    )
    return True


def _run_facts(
    catalog: Catalog,
    dirs: tuple[Path, SecBronze],
    settings: ProcessorSettings,
    processed_at: datetime,
    summary: XbrlFactsStageSummary,
) -> None:
    data_dir, bronze = dirs
    tables = {}
    for identifier, opener in (
        (XBRL_CONCEPTS_TABLE, open_xbrl_concepts_table),
        (XBRL_FACTS_TABLE, open_xbrl_facts_table),
    ):
        tables[identifier], created = opener(catalog, data_dir)
        name = identifier.split(".", 1)[1]
        logger.info(
            f"{name}_table_{'created' if created else 'loaded'}", table=identifier
        )
    concepts_table, facts_table = tables[XBRL_CONCEPTS_TABLE], tables[XBRL_FACTS_TABLE]
    summary.concepts_snapshot_before = summary.concepts_snapshot_after = (
        current_snapshot_id(concepts_table)
    )
    summary.facts_snapshot_before = summary.facts_snapshot_after = current_snapshot_id(
        facts_table
    )
    with timed_operation("xbrl_facts.read_parents"):
        instances = {
            record.instance_id: record
            for rows in read_instance_groups(
                open_xbrl_core_table(catalog, data_dir, XBRL_INSTANCES_TABLE)[0]
            ).values()
            for record in map(XbrlInstanceRecord.model_validate, rows)
        }
        sources = {
            row["document_id"]: DocumentRecord.model_validate(row)
            for row in read_rows_in(
                open_documents_table(catalog, data_dir)[0],
                "document_id",
                {d for i in instances.values() for d in i.source_document_ids},
                columns=DOCUMENTS_ARROW_SCHEMA.names,
            )
        }
        stored_facts = summarize_groups(facts_table, "instance_id", "fact_order")
        run = _Run(
            catalog,
            data_dir,
            bronze,
            settings,
            processed_at,
            instances,
            sources,
            read_concept_hashes(concepts_table),
            {
                concept_id
                for batch in scan_batches(facts_table, ("concept_id",))
                for concept_id in pc.unique(batch["concept_id"]).to_pylist()
            },
        )
    summary.instances_considered = len(instances)
    logger.info("xbrl_instances_loaded", instances=len(instances))

    ordered = sorted(instances.values(), key=lambda i: (i.filing_id, i.instance_id))
    with timed_operation("xbrl_facts.validate", phase="existing_state"):
        plan = classify_instances(ordered, sources, stored_facts)
    for context, reason in plan.conflicts:
        logger.error("xbrl_instance_facts_conflict", reason=reason, **context)
    summary.instances_skipped = plan.skipped
    summary.instances_conflicted = len(plan.conflicts)
    logger.info("xbrl_instance_facts_skipped", instances=plan.skipped)
    if plan.conflicts:
        return
    stored_concepts = len(run.concept_hashes)
    for batch in batched(plan.candidates, INSTANCES_PER_COMMIT, strict=False):
        if not _run_batch(run, (concepts_table, facts_table), batch, summary):
            return
    summary.concepts_discovered = len(run.discovered)
    summary.concepts_skipped = len(run.discovered) - (
        len(run.concept_hashes) - stored_concepts
    )
    summary.status = "completed"


def run_xbrl_facts_stage(
    data_dir: Path,
    processed_at: datetime,
    settings: ProcessorSettings,
    bronze: SecBronze | None = None,
) -> XbrlFactsStageSummary:
    """Build `xbrl_concepts` and `xbrl_facts` from the stored XBRL instances.

    Instances whose fact group is complete are skipped without loading XBRL.
    Every other instance is loaded with its DTS by the XBRL processor from
    its verified source documents; its concepts and fact occurrences must
    agree with the XBRL core rows. Stored fact groups are judged by their
    summaries; any conflict rejects the run before the first write. Instances
    are loaded and committed in batches of INSTANCES_PER_COMMIT: new concepts,
    then facts, appended and re-read. A failure, conflict or missing taxonomy
    rejects its batch; earlier batches stay committed. Unexpected errors are
    logged and raised.
    """

    started = perf_counter()
    summary = XbrlFactsStageSummary()
    with log_context(source="sec_filings"):
        logger.info(
            "xbrl_facts_stage_started",
            taxonomy_cache_dir=str(settings.cache_dir),
            allow_taxonomy_network=settings.allow_network,
        )
        try:
            with (
                timed_operation("xbrl_facts.total"),
                open_silver_catalog(data_dir) as catalog,
            ):
                _run_facts(
                    catalog,
                    (data_dir, bronze or SecBronze.under(data_dir)),
                    settings,
                    processed_at,
                    summary,
                )
        except Exception as error:
            logger.error(
                "xbrl_facts_stage_failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise

        summary.duration_seconds = round(perf_counter() - started, 3)
        if summary.status == "completed":
            logger.info("xbrl_facts_stage_completed", **asdict(summary))
        else:
            logger.error("xbrl_facts_stage_failed", **asdict(summary))
    return summary

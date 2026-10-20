from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel

from src.data_pipeline.contracts.sec_filings.xbrl_concepts import concept_id_for
from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlInstanceRecord
from src.data_pipeline.contracts.sec_filings.xbrl_dimensions import XbrlDimensionRecord
from src.data_pipeline.contracts.sec_filings.xbrl_fact_footnotes import (
    XbrlFactFootnoteRecord,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_enrichment_state import (
    EnrichmentConflict,
    EnrichmentParents,
    InstanceState,
    StoredRows,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_facts_state import InstanceRefs
from src.data_pipeline.pipelines.sec_filings.xbrl_instance_loading import (
    InstanceLoad,
    load_ahead,
)
from src.data_pipeline.quality.sec_filings.xbrl_enrichment import (
    dimension_group_problem,
    footnote_group_problem,
    footnote_resources_problem,
)
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.transforms.sec_filings.xbrl_dimensions import (
    dimension_records,
    expected_dimension_count,
)
from src.data_pipeline.transforms.sec_filings.xbrl_facts import (
    XbrlCoreMismatch,
    fact_records,
)
from src.data_pipeline.transforms.sec_filings.xbrl_footnotes import footnote_records
from src.data_pipeline.transforms.sec_filings.xbrl_processor import (
    LoadedInstance,
    ProcessorSettings,
    TaxonomyUnavailable,
)
from src.utils.logger import get_logger, log_context
from src.utils.timer import PhaseDurations

logger = get_logger(__name__)


@dataclass(frozen=True)
class EnrichmentInputs:
    """What each instance's extraction reads from the run.

    Stored dimensions are grouped by context_key and footnotes by fact_id.
    """

    parents: EnrichmentParents
    refs: InstanceRefs
    stored_dimensions: StoredRows
    stored_footnotes: StoredRows
    bronze: SecBronze
    settings: ProcessorSettings
    processed_at: datetime


@dataclass
class EnrichmentExtraction:
    """New rows by instance_id, and what the processor read for them.

    `resumed` instances already had rows in one table; `unchanged` ones were
    loaded but need no new row. `read` counts contexts, facts and footnotes
    of the loaded instances.
    """

    dimensions: dict[str, list[XbrlDimensionRecord]] = field(default_factory=dict)
    footnotes: dict[str, list[XbrlFactFootnoteRecord]] = field(default_factory=dict)
    resumed: set[str] = field(default_factory=set)
    unchanged: int = 0
    failures: int = 0
    conflicts: int = 0
    read: Counter[str] = field(default_factory=Counter)
    durations: PhaseDurations = field(default_factory=PhaseDurations)


def _check_parents(
    instance: XbrlInstanceRecord, loaded: LoadedInstance, inputs: EnrichmentInputs
) -> None:
    """Loaded contexts must be stored ones, and the facts exactly the stored ones."""

    instance_id = instance.instance_id
    stored = {
        c["context_ref"]: c["context_hash"]
        for c in inputs.parents.contexts.get(instance_id, [])
    }
    differ = sorted(
        ref
        for ref, context_hash in loaded.context_hashes.items()
        if stored.get(ref) != context_hash
    )
    if differ:
        raise EnrichmentConflict(f"contexts differ from xbrl_contexts: {differ[:5]}")
    facts, _ = fact_records(
        instance_id,
        loaded.facts,
        inputs.refs.of(instance_id),
        [(None, None)] * len(loaded.facts),
        inputs.processed_at,
    )
    if {f.fact_id: f.fact_sha256 for f in facts} != inputs.parents.fact_hashes.get(
        instance_id, {}
    ):
        raise EnrichmentConflict("facts differ from xbrl_facts")


def _dimensions(
    instance: XbrlInstanceRecord, loaded: LoadedInstance, inputs: EnrichmentInputs
) -> list[XbrlDimensionRecord]:
    contexts = sorted(
        inputs.parents.contexts.get(instance.instance_id, []),
        key=lambda c: c["context_key"],
    )
    by_ref = defaultdict(list)
    for dimension in loaded.dimensions:
        by_ref[dimension.context_ref].append(dimension)
    concept_ids = inputs.parents.concept_ids
    records = []
    for context in contexts:
        ref, key = context["context_ref"], context["context_key"]
        missing = sorted(
            f"{{{qname[0]}}}{qname[1]}"
            for dimension in by_ref[ref]
            for qname in (dimension.dimension, dimension.member)
            if qname is not None and concept_id_for(*qname) not in concept_ids
        )
        if missing:
            logger.error(
                "xbrl_dimension_parent_missing", context_key=key, qnames=missing
            )
            raise EnrichmentConflict(f"context {ref!r}: concepts {missing} not stored")
        group = dimension_records(key, by_ref[ref], inputs.processed_at)
        expected = expected_dimension_count(
            context["segment_xml"], context["scenario_xml"]
        )
        if expected and ref not in loaded.context_hashes:
            raise EnrichmentConflict(
                f"context {ref!r} reports dimensions but no fact uses it, "
                "so the processor does not read it"
            )
        problem = dimension_group_problem(group, expected, concept_ids)
        if problem is not None:
            raise EnrichmentConflict(f"context {ref!r}: {problem}")
        records.extend(group)
    return records


def _footnotes(
    instance: XbrlInstanceRecord, loaded: LoadedInstance, inputs: EnrichmentInputs
) -> list[XbrlFactFootnoteRecord]:
    records = footnote_records(
        instance.instance_id, loaded.footnotes, inputs.processed_at
    )
    facts = inputs.parents.fact_hashes.get(instance.instance_id, {})
    orphans = sorted({r.fact_id for r in records if r.fact_id not in facts})
    if orphans:
        logger.error("xbrl_footnote_fact_parent_missing", fact_ids=orphans[:5])
        raise EnrichmentConflict(f"footnotes of unknown facts {orphans[:5]}")
    by_fact = defaultdict(list)
    for record in records:
        by_fact[record.fact_id].append(record)
    problem = next(filter(None, map(footnote_group_problem, by_fact.values())), None)
    problem = problem or footnote_resources_problem(records)
    if problem is not None:
        raise EnrichmentConflict(problem)
    return records


def _matches_stored(
    built: Sequence[BaseModel], stored: StoredRows, groups: Iterable[str], key: str
) -> bool:
    """Whether stored rows of `groups` are exactly the built rows, batch time aside."""

    found = {
        row[key]: {k: v for k, v in row.items() if k != "processed_at"}
        for group in groups
        for row in stored.get(group, [])
    }
    return found == {
        getattr(row, key): row.model_dump(exclude={"processed_at"}) for row in built
    }


def _extract_one(
    state: InstanceState,
    load: InstanceLoad,
    inputs: EnrichmentInputs,
    extraction: EnrichmentExtraction,
) -> tuple[list[XbrlDimensionRecord], list[XbrlFactFootnoteRecord]]:
    """New dimension and footnote rows of an instance, from one processor load."""

    instance = state.instance
    # With worker processes, only the wait for a load ahead of this one.
    _, loaded = extraction.durations.run("load_arelle", load)
    if loaded.downloaded_uris:
        logger.info("xbrl_taxonomy_downloaded", uris=loaded.downloaded_uris)
    logger.info(
        "xbrl_enrichment_instance_loaded",
        facts=len(loaded.facts),
        contexts=len(loaded.context_hashes),
        dimensions=len(loaded.dimensions),
        footnotes=len(loaded.footnotes),
    )
    extraction.durations.run(
        "validate", lambda: _check_parents(instance, loaded, inputs)
    )
    dimensions = extraction.durations.run(
        "extract_dimensions", lambda: _dimensions(instance, loaded, inputs)
    )
    footnotes = extraction.durations.run(
        "extract_footnotes", lambda: _footnotes(instance, loaded, inputs)
    )
    _log_read(loaded, dimensions, footnotes, extraction)
    if state.dimensions_stored:
        contexts = inputs.parents.contexts.get(instance.instance_id, [])
        keys = [c["context_key"] for c in contexts]
        if not _matches_stored(
            dimensions, inputs.stored_dimensions, keys, "dimension_id"
        ):
            raise EnrichmentConflict("stored dimensions differ from the source")
        dimensions = []
    if state.footnotes_stored:
        facts = inputs.parents.fact_hashes.get(instance.instance_id, {})
        if not _matches_stored(
            footnotes, inputs.stored_footnotes, facts, "footnote_relation_id"
        ):
            raise EnrichmentConflict("stored footnotes differ from the source")
        footnotes = []
    return dimensions, footnotes


def _log_read(
    loaded: LoadedInstance,
    dimensions: Sequence[XbrlDimensionRecord],
    footnotes: Sequence[XbrlFactFootnoteRecord],
    extraction: EnrichmentExtraction,
) -> None:
    resources = {f.footnote_resource_id for f in footnotes}
    empty = sorted({f.footnote_resource_id for f in footnotes if not f.footnote_text})
    logger.info("xbrl_dimensions_extracted", dimensions=len(dimensions))
    logger.info(
        "xbrl_fact_footnotes_extracted",
        relationships=len(footnotes),
        resources=len(resources),
    )
    if not dimensions:
        logger.info("xbrl_instance_without_dimensions")
    if not footnotes:
        logger.info("xbrl_instance_without_footnotes")
    if empty:
        logger.warning("empty_xbrl_footnote", footnote_resource_ids=empty)
    extraction.read.update(
        contexts=len(loaded.context_hashes),
        contexts_with_dimensions=len({d.context_key for d in dimensions}),
        facts=len(loaded.facts),
        facts_with_footnotes=len({f.fact_id for f in footnotes}),
        empty_footnotes=len(empty),
    )


def extract_instances(
    states: Sequence[InstanceState], inputs: EnrichmentInputs
) -> EnrichmentExtraction:
    """Load every instance needing rows once and build its new rows.

    Failures and conflicts are counted and logged per instance; the caller
    rejects the batch when any instance failed or conflicted.
    """

    extraction = EnrichmentExtraction()
    with closing(
        load_ahead(
            [state.instance for state in states],
            inputs.parents.documents,
            inputs.bronze,
            inputs.settings,
        )
    ) as loads:
        for state, load in zip(states, loads, strict=True):
            instance = state.instance
            with log_context(
                filing_id=instance.filing_id, instance_id=instance.instance_id
            ):
                try:
                    dimensions, footnotes = _extract_one(
                        state, load, inputs, extraction
                    )
                except TaxonomyUnavailable as error:
                    extraction.failures += 1
                    logger.error("xbrl_taxonomy_cache_miss", uris=list(error.uris))
                    continue
                except (EnrichmentConflict, XbrlCoreMismatch) as error:
                    extraction.conflicts += 1
                    logger.error("xbrl_enrichment_instance_conflict", reason=str(error))
                    continue
                except (OSError, ValueError) as error:
                    extraction.failures += 1
                    logger.error("xbrl_enrichment_instance_failed", error=str(error))
                    continue
            if not (dimensions or footnotes):
                extraction.unchanged += 1
                continue
            extraction.dimensions[instance.instance_id] = dimensions
            extraction.footnotes[instance.instance_id] = footnotes
            if state.has_rows:
                extraction.resumed.add(instance.instance_id)
    logger.info(
        "xbrl_enrichment_extracted",
        instances=len(extraction.dimensions),
        **extraction.durations.milliseconds(),
    )
    return extraction

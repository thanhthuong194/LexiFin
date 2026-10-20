from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.xbrl_concepts import (
    ConceptDefinition,
    XbrlConceptRecord,
    concept_id_for,
)
from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlInstanceRecord
from src.data_pipeline.contracts.sec_filings.xbrl_facts import XbrlFactRecord
from src.data_pipeline.pipelines.sec_filings.xbrl_facts_state import (
    ContentMaps,
    InstanceRefs,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_instance_loading import (
    InstanceLoad,
    load_ahead,
)
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.transforms.sec_filings.html_dom import parse_html
from src.data_pipeline.transforms.sec_filings.xbrl_facts import (
    Location,
    XbrlCoreMismatch,
    align_document,
    concept_definition,
    concept_record,
    fact_records,
)
from src.data_pipeline.transforms.sec_filings.xbrl_model_reader import LoadedConcept
from src.data_pipeline.transforms.sec_filings.xbrl_processor import (
    LoadedInstance,
    ProcessorSettings,
    SourceFile,
    TaxonomyUnavailable,
)
from src.utils.logger import get_logger, log_context
from src.utils.timer import PhaseDurations

logger = get_logger(__name__)


class ConceptConflict(ValueError):
    """One concept ID with two different definitions."""


@dataclass
class FactsExtraction:
    """Records built with the XBRL processor, and the instances that failed.

    `concepts` are concepts new to `xbrl_concepts`; `definitions` every
    concept seen in this run. `incompatible` instances disagree with the
    XBRL core rows.
    """

    concepts: dict[str, XbrlConceptRecord] = field(default_factory=dict)
    definitions: dict[str, ConceptDefinition] = field(default_factory=dict)
    facts: dict[str, list[XbrlFactRecord]] = field(default_factory=dict)
    failures: int = 0
    conflicts: int = 0
    incompatible: int = 0
    recovered: set[str] = field(default_factory=set)
    unusable_values: int = 0
    locations: Counter[str] = field(default_factory=Counter)
    durations: PhaseDurations = field(default_factory=PhaseDurations)


def _merge_concepts(
    concepts: Sequence[LoadedConcept],
    inputs: ExtractionInputs,
    extraction: FactsExtraction,
) -> bool:
    """Keep each concept once; a second, different definition is a conflict.

    Returns whether this run recovers the instance: its extension concepts are
    stored but no stored fact uses any of them, as a run that committed its
    concepts and not its facts leaves them.
    """

    stored_extension = used_extension = False
    for concept in concepts:
        concept_id = concept_id_for(concept.namespace_uri, concept.local_name)
        definition = concept_definition(concept)
        if concept.is_extension and concept_id in inputs.stored_concepts:
            stored_extension = True
            used_extension = used_extension or concept_id in inputs.fact_concepts
        seen = extraction.definitions.setdefault(concept_id, definition)
        if seen != definition:
            raise ConceptConflict(f"{concept_id} is defined twice in this run")
        if seen is not definition:
            continue
        stored_hash = inputs.stored_concepts.get(concept_id)
        if stored_hash is None:
            extraction.concepts[concept_id] = concept_record(
                concept, inputs.processed_at
            )
        elif stored_hash != definition.sha256():
            raise ConceptConflict(f"{concept_id} differs from the stored definition")
    return stored_extension and not used_extension


def _align(
    loaded: LoadedInstance,
    files: Sequence[SourceFile],
    content: Callable[[str], ContentMaps],
) -> list[Location]:
    """Locate Inline facts in their primary document's stored cells and blocks."""

    by_document: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, fact in enumerate(loaded.facts):
        if fact.position is not None:
            by_document[fact.document_id].append((index, fact.position))
    locations: list[Location] = [(None, None)] * len(loaded.facts)
    sources = {file.document_id: file.content for file in files}
    for document_id, items in by_document.items():
        maps = content(document_id)
        if not any(maps):
            continue
        found = align_document(
            parse_html(sources[document_id]),
            [position for _, position in items],
            loaded.inline_fact_totals[document_id],
            maps,
        )
        for (index, _), location in zip(items, found, strict=True):
            locations[index] = location
    return locations


@dataclass(frozen=True)
class ExtractionInputs:
    """What each instance's extraction reads from the run.

    `refs` are the stored context and unit refs, `stored_concepts` the stored
    concept hashes, `fact_concepts` the concepts that stored facts use, and
    `content` looks up a document's stored cells and blocks by XPath.
    """

    documents: Mapping[str, DocumentRecord]
    refs: InstanceRefs
    stored_concepts: Mapping[str, str]
    fact_concepts: Collection[str]
    content: Callable[[str], ContentMaps]
    bronze: SecBronze
    settings: ProcessorSettings
    processed_at: datetime


def _extract_one(
    instance: XbrlInstanceRecord,
    load: InstanceLoad,
    inputs: ExtractionInputs,
    extraction: FactsExtraction,
) -> list[XbrlFactRecord]:
    # With worker processes, only the wait for a load ahead of this one.
    files, loaded = extraction.durations.run("load_dts", load)
    logger.info("xbrl_source_documents_verified", documents=len(files))
    if loaded.downloaded_uris:
        logger.info("xbrl_taxonomy_downloaded", uris=loaded.downloaded_uris)
    logger.info(
        "xbrl_model_loaded", facts=len(loaded.facts), concepts=len(loaded.concepts)
    )
    if len(loaded.facts) != instance.fact_count:
        raise XbrlCoreMismatch(
            f"processor found {len(loaded.facts)} facts, "
            f"xbrl_instances {instance.fact_count}"
        )
    new_before = len(extraction.concepts)
    recovers = extraction.durations.run(
        "resolve_concepts", lambda: _merge_concepts(loaded.concepts, inputs, extraction)
    )
    logger.info(
        "xbrl_concepts_resolved",
        concepts=len(loaded.concepts),
        new_concepts=len(extraction.concepts) - new_before,
    )
    locations = extraction.durations.run(
        "align_content", lambda: _align(loaded, files, inputs.content)
    )
    records, unusable = extraction.durations.run(
        "extract_facts",
        lambda: fact_records(
            instance.instance_id,
            loaded.facts,
            inputs.refs.of(instance.instance_id),
            locations,
            inputs.processed_at,
        ),
    )
    if recovers:
        extraction.recovered.add(instance.instance_id)
    extraction.unusable_values += unusable
    if unusable:
        logger.warning("xbrl_fact_values_not_normalized", facts=unusable)
    return records


def extract_instances(
    candidates: Sequence[XbrlInstanceRecord], inputs: ExtractionInputs
) -> FactsExtraction:
    """Load every candidate instance with the XBRL processor and build its rows.

    Failures are counted and logged per instance; the caller rejects the
    batch when any instance failed.
    """

    extraction = FactsExtraction()
    with closing(
        load_ahead(candidates, inputs.documents, inputs.bronze, inputs.settings)
    ) as loads:
        for instance, load in zip(candidates, loads, strict=True):
            context = {
                "filing_id": instance.filing_id,
                "instance_id": instance.instance_id,
            }
            with log_context(**context):
                try:
                    records = _extract_one(instance, load, inputs, extraction)
                except TaxonomyUnavailable as error:
                    extraction.failures += 1
                    logger.error("xbrl_taxonomy_cache_miss", uris=list(error.uris))
                    continue
                except XbrlCoreMismatch as error:
                    extraction.incompatible += 1
                    logger.error("xbrl_core_semantics_incompatible", reason=str(error))
                    continue
                except ConceptConflict as error:
                    extraction.conflicts += 1
                    logger.error("xbrl_instance_facts_conflict", reason=str(error))
                    continue
                except (OSError, ValueError) as error:
                    extraction.failures += 1
                    logger.error("xbrl_facts_instance_failed", error=str(error))
                    continue
            extraction.facts[instance.instance_id] = records
            for record in records:
                extraction.locations[
                    "cell" if record.cell_id else "block" if record.block_id else "none"
                ] += 1
                visible = instance.is_inline and not record.is_hidden
                extraction.locations["unaligned_visible"] += visible and not (
                    record.cell_id or record.block_id
                )
    logger.info(
        "xbrl_fact_alignment_completed",
        cell_facts=extraction.locations["cell"],
        block_facts=extraction.locations["block"],
        unaligned_facts=extraction.locations["none"],
        unaligned_visible_facts=extraction.locations["unaligned_visible"],
    )
    logger.info(
        "xbrl_facts_extracted",
        instances=len(extraction.facts),
        facts=sum(map(len, extraction.facts.values())),
        **extraction.durations.milliseconds(),
    )
    return extraction

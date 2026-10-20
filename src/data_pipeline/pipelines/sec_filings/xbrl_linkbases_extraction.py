from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.xbrl_labels import XbrlLabelRecord
from src.data_pipeline.contracts.sec_filings.xbrl_networks import (
    XbrlNetworkRecord,
    XbrlRelationshipRecord,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_instance_loading import (
    InstanceLoad,
    load_ahead,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_linkbases_state import (
    InstanceState,
    LinkbaseConflict,
)
from src.data_pipeline.quality.sec_filings.xbrl_linkbases import (
    network_problem,
    relationship_attributes_problem,
)
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.transforms.sec_filings.xbrl_linkbases import (
    label_records,
    network_records,
)
from src.data_pipeline.transforms.sec_filings.xbrl_processor import (
    ProcessorSettings,
    TaxonomyUnavailable,
)
from src.utils.logger import get_logger, log_context
from src.utils.timer import PhaseDurations

logger = get_logger(__name__)


@dataclass(frozen=True)
class LinkbaseInputs:
    """What each instance's extraction reads from the run."""

    documents: Mapping[str, DocumentRecord]
    concept_ids: Collection[str]
    bronze: SecBronze
    settings: ProcessorSettings
    processed_at: datetime


@dataclass
class LinkbaseExtraction:
    """New rows built from the loaded instances, and how the loads went.

    `labels` holds each label of the batch once; `networks` and
    `relationships` only rows not stored yet. `recovered` instances had
    their networks stored without relationships. `discovered` counts the
    networks and relationships built, and relationships by linkbase type.
    """

    labels: dict[str, XbrlLabelRecord] = field(default_factory=dict)
    networks: list[XbrlNetworkRecord] = field(default_factory=list)
    relationships: list[XbrlRelationshipRecord] = field(default_factory=list)
    succeeded: int = 0
    recovered: int = 0
    failures: int = 0
    conflicts: int = 0
    discovered: Counter[str] = field(default_factory=Counter)
    durations: PhaseDurations = field(default_factory=PhaseDurations)


def _check_concepts(
    labels: Sequence[XbrlLabelRecord],
    relationships: Sequence[XbrlRelationshipRecord],
    concept_ids: Collection[str],
) -> None:
    """Every label and relationship end must be a stored concept."""

    ends = {label.concept_id for label in labels} | {
        concept_id
        for relationship in relationships
        for concept_id in (
            relationship.source_concept_id,
            relationship.target_concept_id,
        )
    }
    missing = sorted(ends - set(concept_ids))
    if missing:
        raise LinkbaseConflict(f"concepts missing from xbrl_concepts: {missing[:5]}")


def _check_networks(
    networks: Sequence[XbrlNetworkRecord],
    relationships: Sequence[XbrlRelationshipRecord],
) -> None:
    by_network: dict[str, list[XbrlRelationshipRecord]] = defaultdict(list)
    for relationship in relationships:
        by_network[relationship.network_id].append(relationship)
    for network in networks:
        problems = [
            network_problem(network, by_network[network.network_id]),
            *(
                relationship_attributes_problem(network, relationship)
                for relationship in by_network[network.network_id]
            ),
        ]
        problem = next(filter(None, problems), None)
        if problem is not None:
            raise LinkbaseConflict(f"network {network.network_id}: {problem}")


def _merge_labels(
    labels: Sequence[XbrlLabelRecord], extraction: LinkbaseExtraction
) -> None:
    """Keep each label once; one ID with two payloads is a conflict."""

    for label in labels:
        seen = extraction.labels.setdefault(label.label_id, label)
        if seen.model_dump(exclude={"processed_at"}) != label.model_dump(
            exclude={"processed_at"}
        ):
            raise LinkbaseConflict(f"label {label.label_id} differs within the batch")


def _extract_one(
    state: InstanceState,
    load: InstanceLoad,
    inputs: LinkbaseInputs,
    extraction: LinkbaseExtraction,
) -> None:
    """Load an instance once and add its new rows to the extraction."""

    instance = state.instance
    # With worker processes, only the wait for a load ahead of this one.
    _, loaded = extraction.durations.run("load_arelle", load)
    logger.info("xbrl_model_loaded", labels=len(loaded.labels), arcs=len(loaded.arcs))
    labels = extraction.durations.run(
        "build_rows", lambda: label_records(loaded.labels, inputs.processed_at)
    )
    networks, relationships = extraction.durations.run(
        "build_rows",
        lambda: network_records(
            instance.instance_id,
            loaded.arcs,
            loaded.role_definitions,
            inputs.processed_at,
        ),
    )
    _check_concepts(labels, relationships, inputs.concept_ids)
    _check_networks(networks, relationships)
    _log_extracted(labels, networks, relationships)
    extraction.discovered.update(
        networks=len(networks), relationships=len(relationships)
    )
    for network in networks:
        extraction.discovered[network.linkbase_type] += network.relationship_count
    if state.networks:
        stored = [n.model_dump(exclude={"processed_at"}) for n in state.networks]
        if sorted(stored, key=str) != sorted(
            (n.model_dump(exclude={"processed_at"}) for n in networks), key=str
        ):
            raise LinkbaseConflict("stored networks differ from the source")
        extraction.recovered += 1
        networks = []
    _merge_labels(labels, extraction)
    extraction.networks.extend(networks)
    extraction.relationships.extend(relationships)
    extraction.succeeded += 1


def _log_extracted(
    labels: Sequence[XbrlLabelRecord],
    networks: Sequence[XbrlNetworkRecord],
    relationships: Sequence[XbrlRelationshipRecord],
) -> None:
    empty = sum(not label.label_text for label in labels)
    if empty:
        logger.warning("xbrl_labels_empty", labels=empty)
    logger.info(
        "xbrl_labels_extracted",
        labels=len(labels),
        roles=len({label.role_uri for label in labels}),
    )
    by_type: Counter[str] = Counter(network.linkbase_type for network in networks)
    logger.info("xbrl_networks_extracted", networks=len(networks), **by_type)
    logger.info("xbrl_relationships_extracted", relationships=len(relationships))


def extract_instances(
    states: Sequence[InstanceState], inputs: LinkbaseInputs
) -> LinkbaseExtraction:
    """Load every instance needing rows once and build its rows.

    Failures and conflicts are counted and logged per instance; the caller
    rejects the batch when any instance failed or conflicted.
    """

    extraction = LinkbaseExtraction()
    with closing(
        load_ahead(
            [state.instance for state in states],
            inputs.documents,
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
                    _extract_one(state, load, inputs, extraction)
                except TaxonomyUnavailable as error:
                    extraction.failures += 1
                    logger.error("xbrl_taxonomy_cache_miss", uris=list(error.uris))
                except LinkbaseConflict as error:
                    extraction.conflicts += 1
                    logger.error("xbrl_linkbases_conflict", reason=str(error))
                except (OSError, ValueError) as error:
                    extraction.failures += 1
                    logger.error("xbrl_linkbases_failed", error=str(error))
    logger.info(
        "xbrl_linkbases_extracted",
        instances=extraction.succeeded,
        labels=len(extraction.labels),
        **extraction.discovered,
        **extraction.durations.milliseconds(),
    )
    return extraction

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlInstanceRecord
from src.data_pipeline.contracts.sec_filings.xbrl_facts import instance_of_fact
from src.data_pipeline.pipelines.sec_filings.xbrl_instance_loading import (
    entry_documents,
)
from src.data_pipeline.quality.sec_filings.xbrl_core import instance_problem
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.storage.iceberg import GroupSummary
from src.data_pipeline.storage.sec_filings.documents import read_document_content
from src.data_pipeline.transforms.sec_filings.xbrl_sources import (
    INLINE_XBRL,
    LINK,
    parse_xbrl_source,
)

StoredRows = Mapping[str, Sequence[dict[str, Any]]]
# A document without these elements declares no fact-footnote relationship.
_FOOTNOTE_MARKUP = (
    f"{{{INLINE_XBRL}}}relationship",
    f"{{{INLINE_XBRL}}}footnote",
    f"{{{LINK}}}footnoteLink",
)


class EnrichmentConflict(ValueError):
    """Stored rows, or the source, disagree with the parent or sibling rows."""


@dataclass(frozen=True)
class EnrichmentParents:
    """Parent rows the stage reads.

    `contexts` (stored context rows) and `fact_hashes` (fact_sha256 by
    fact_id) are keyed by instance_id; `concept_ids` are all stored concepts.
    """

    documents: Mapping[str, DocumentRecord]
    contexts: Mapping[str, Sequence[dict[str, Any]]]
    fact_hashes: Mapping[str, Mapping[str, str]]
    concept_ids: Collection[str]


@dataclass(frozen=True)
class InstanceState:
    """What `xbrl_dimensions` and `xbrl_fact_footnotes` hold of an instance.

    `dimensions_stored` is true when every context's dimensions are stored
    (also when it reports none); `footnotes_stored` is true when its footnote
    rows are stored and None when it has no row, which may mean no footnote.
    `has_rows` tells whether either table holds rows of it.
    """

    instance: XbrlInstanceRecord
    dimensions_stored: bool
    footnotes_stored: bool | None
    has_rows: bool


@dataclass
class EnrichmentPlan:
    """Instances without conflicts, and (context, reason) conflict pairs."""

    states: list[InstanceState] = field(default_factory=list)
    conflicts: list[tuple[dict[str, str], str]] = field(default_factory=list)


@dataclass(frozen=True)
class StoredEnrichment:
    """Summaries of what the stage's parent and own tables hold, without rows.

    `expected` maps each instance's context keys to the dimension members
    their stored segment and scenario report; `fact_counts` counts stored
    facts per instance; `dimensions` summarizes stored dimension rows by
    context_key and `footnotes` stored footnote rows by fact_id.
    """

    expected: Mapping[str, Mapping[str, int]]
    fact_counts: Mapping[str, int]
    dimensions: Mapping[str, GroupSummary]
    footnotes: Mapping[str, GroupSummary]


def _dimensions_state(
    expected: Mapping[str, int], dimensions: Mapping[str, GroupSummary]
) -> tuple[bool, bool]:
    """(complete, has_rows) of an instance's stored dimension groups."""

    reporting = present = 0
    for context_key, members in expected.items():
        reporting += members > 0
        group = dimensions.get(context_key)
        if group is None:
            continue
        if not group.complete or group.rows != members:
            raise EnrichmentConflict(
                f"context {context_key}: {group.rows} stored dimensions, "
                f"the context reports {members}"
            )
        present += 1
    if present not in (0, reporting):
        raise EnrichmentConflict(
            f"dimensions are stored for {present} of {reporting} contexts"
        )
    return present == reporting, present > 0


def _footnotes_state(groups: Sequence[tuple[str, GroupSummary]]) -> bool | None:
    """True when an instance's stored footnote groups are complete, None if none."""

    if not groups:
        return None
    for fact_id, group in groups:
        if not group.complete:
            raise EnrichmentConflict(f"fact {fact_id}: footnote ordinals are not 1..N")
    return True


def classify_instances(
    instances: Sequence[XbrlInstanceRecord],
    documents: Mapping[str, DocumentRecord],
    stored: StoredEnrichment,
) -> EnrichmentPlan:
    """Classify every instance against both tables before loading XBRL.

    Stored groups are judged by their summaries; `documents` holds at least
    the instances' source documents. Rows of unknown contexts or instances,
    partial groups, dimensions stored for only some contexts, and instances
    whose sources or facts changed are conflicts. The integrity audit checks
    the stored rows themselves.
    """

    plan = EnrichmentPlan()
    known = {key for keys in stored.expected.values() for key in keys}
    for key in sorted(stored.dimensions.keys() - known):
        plan.conflicts.append(({"context_key": key}, "dimensions have no context"))
    footnotes: dict[str, list[tuple[str, GroupSummary]]] = defaultdict(list)
    instance_ids = {instance.instance_id for instance in instances}
    for fact_id, group in sorted(stored.footnotes.items()):
        if instance_of_fact(fact_id) in instance_ids:
            footnotes[instance_of_fact(fact_id)].append((fact_id, group))
        else:
            plan.conflicts.append(({"fact_id": fact_id}, "footnotes have no fact"))
    for instance in instances:
        context = {"filing_id": instance.filing_id, "instance_id": instance.instance_id}
        facts = stored.fact_counts.get(instance.instance_id, 0)
        problem = instance_problem(instance, documents)
        if problem is None and facts != instance.fact_count:
            problem = f"xbrl_facts holds {facts} of {instance.fact_count} facts"
        if problem is not None:
            plan.conflicts.append((context, problem))
            continue
        try:
            dimensions_stored, dimension_rows = _dimensions_state(
                stored.expected.get(instance.instance_id, {}), stored.dimensions
            )
            footnotes_stored = _footnotes_state(footnotes[instance.instance_id])
        except EnrichmentConflict as error:
            plan.conflicts.append((context, str(error)))
            continue
        plan.states.append(
            InstanceState(
                instance,
                dimensions_stored,
                footnotes_stored,
                has_rows=dimension_rows or footnotes_stored is True,
            )
        )
    return plan


def declares_footnotes(
    instance: XbrlInstanceRecord,
    documents: Mapping[str, DocumentRecord],
    bronze: SecBronze,
) -> bool:
    """Whether the instance's own documents hold any footnote markup.

    Documents are read and verified. Without such markup an instance has no
    footnote, so one with no stored footnote row need not be loaded again.
    """

    return any(
        next(root.iter(*_FOOTNOTE_MARKUP), None) is not None
        for root in (
            parse_xbrl_source(read_document_content(document, bronze)).root
            for document in entry_documents(instance, documents)
        )
    )

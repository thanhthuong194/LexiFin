from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlInstanceRecord
from src.data_pipeline.contracts.sec_filings.xbrl_dimensions import XbrlDimensionRecord
from src.data_pipeline.contracts.sec_filings.xbrl_fact_footnotes import (
    XbrlFactFootnoteRecord,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_facts_extraction import (
    entry_documents,
)
from src.data_pipeline.quality.sec_filings.xbrl_core import instance_problem
from src.data_pipeline.quality.sec_filings.xbrl_enrichment import (
    dimension_group_problem,
    footnote_group_problem,
    footnote_resources_problem,
)
from src.data_pipeline.storage.sec_filings.documents import read_document_content
from src.data_pipeline.transforms.sec_filings.xbrl_dimensions import (
    expected_dimension_count,
)
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


def _records[Record: (XbrlDimensionRecord, XbrlFactFootnoteRecord)](
    model: type[Record], rows: Sequence[dict[str, Any]]
) -> list[Record]:
    try:
        return [model.model_validate(row) for row in rows]
    except ValidationError as error:
        raise EnrichmentConflict(
            f"stored row is invalid: {error.errors()[0]['msg']}"
        ) from error


def _dimensions_state(
    contexts: Sequence[dict[str, Any]],
    stored: StoredRows,
    concept_ids: Collection[str],
) -> tuple[bool, bool]:
    """(complete, has_rows) of an instance's stored dimension groups."""

    reporting = present = 0
    for context in contexts:
        expected = expected_dimension_count(
            context["segment_xml"], context["scenario_xml"]
        )
        reporting += expected > 0
        rows = stored.get(context["context_key"], [])
        if not rows:
            continue
        problem = dimension_group_problem(
            _records(XbrlDimensionRecord, rows), expected, concept_ids
        )
        if problem is not None:
            raise EnrichmentConflict(f"context {context['context_ref']!r}: {problem}")
        present += 1
    if present not in (0, reporting):
        raise EnrichmentConflict(
            f"dimensions are stored for {present} of {reporting} contexts"
        )
    return present == reporting, present > 0


def _footnotes_state(fact_ids: Collection[str], stored: StoredRows) -> bool | None:
    """True when an instance's stored footnote groups are complete, None if none."""

    groups = [
        _records(XbrlFactFootnoteRecord, stored[f]) for f in fact_ids if f in stored
    ]
    if not groups:
        return None
    for group in groups:
        problem = footnote_group_problem(group)
        if problem is not None:
            raise EnrichmentConflict(f"fact {group[0].fact_id}: {problem}")
    problem = footnote_resources_problem([row for group in groups for row in group])
    if problem is not None:
        raise EnrichmentConflict(problem)
    return True


def classify_instances(
    instances: Sequence[XbrlInstanceRecord],
    parents: EnrichmentParents,
    stored_dimensions: StoredRows,
    stored_footnotes: StoredRows,
) -> EnrichmentPlan:
    """Classify every instance against both tables before loading XBRL.

    `stored_dimensions` groups rows by context_key and `stored_footnotes` by
    fact_id. Rows of unknown contexts or facts, partial or invalid groups,
    dimensions stored for only some contexts, and instances whose sources or
    facts changed are conflicts.
    """

    plan = EnrichmentPlan()
    context_keys = {
        c["context_key"] for rows in parents.contexts.values() for c in rows
    }
    for key in sorted(stored_dimensions.keys() - context_keys):
        plan.conflicts.append(({"context_key": key}, "dimensions have no context"))
    fact_ids = {f for hashes in parents.fact_hashes.values() for f in hashes}
    for fact_id in sorted(stored_footnotes.keys() - fact_ids):
        plan.conflicts.append(({"fact_id": fact_id}, "footnotes have no fact"))
    for instance in instances:
        context = {"filing_id": instance.filing_id, "instance_id": instance.instance_id}
        facts = parents.fact_hashes.get(instance.instance_id, {})
        problem = instance_problem(instance, parents.documents)
        if problem is None and len(facts) != instance.fact_count:
            problem = f"xbrl_facts holds {len(facts)} of {instance.fact_count} facts"
        if problem is not None:
            plan.conflicts.append((context, problem))
            continue
        try:
            dimensions_stored, dimension_rows = _dimensions_state(
                parents.contexts.get(instance.instance_id, []),
                stored_dimensions,
                parents.concept_ids,
            )
            footnotes_stored = _footnotes_state(facts, stored_footnotes)
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
    data_dir: Path,
) -> bool:
    """Whether the instance's own documents hold any footnote markup.

    Documents are read and verified. Without such markup an instance has no
    footnote, so one with no stored footnote row need not be loaded again.
    """

    return any(
        next(root.iter(*_FOOTNOTE_MARKUP), None) is not None
        for root in (
            parse_xbrl_source(read_document_content(document, data_dir)).root
            for document in entry_documents(instance, documents)
        )
    )

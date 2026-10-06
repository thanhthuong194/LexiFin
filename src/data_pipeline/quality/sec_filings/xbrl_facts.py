from __future__ import annotations

from collections.abc import Collection, Sequence

from src.data_pipeline.contracts.sec_filings.xbrl_core import (
    XbrlInstanceRecord,
    context_key_for,
    unit_key_for,
)
from src.data_pipeline.contracts.sec_filings.xbrl_facts import XbrlFactRecord
from src.data_pipeline.quality.sec_filings.narrative import declared_count_problem


def facts_problem(
    instance: XbrlInstanceRecord,
    facts: Sequence[XbrlFactRecord],
    concept_ids: Collection[str],
    refs: tuple[Collection[str], Collection[str]],
) -> str | None:
    """Return why an instance's facts are not one complete group, or None.

    The group needs exactly `fact_count` facts with orders 1..N, each from a
    source document of the instance, with a known concept and the instance's
    own context and unit (`refs` holds its stored context and unit refs).
    Keys, values and hashes are enforced by XbrlFactRecord.
    """

    problem = declared_count_problem(
        [fact.fact_order for fact in facts], {instance.fact_count}, "facts"
    )
    if problem is not None:
        return problem
    contexts = {context_key_for(instance.instance_id, ref) for ref in refs[0]}
    units = {unit_key_for(instance.instance_id, ref) for ref in refs[1]}
    for fact in facts:
        problem = _fact_problem(instance, fact, concept_ids, (contexts, units))
        if problem is not None:
            return f"fact {fact.fact_order} {problem}"
    return None


def _fact_problem(
    instance: XbrlInstanceRecord,
    fact: XbrlFactRecord,
    concept_ids: Collection[str],
    keys: tuple[Collection[str], Collection[str]],
) -> str | None:
    if fact.instance_id != instance.instance_id:
        return "belongs to another instance"
    if fact.document_id not in instance.source_document_ids:
        return "comes from a document outside the instance"
    if fact.concept_id not in concept_ids:
        return "has no concept in xbrl_concepts"
    if fact.context_key not in keys[0]:
        return "has no context in xbrl_contexts"
    if fact.unit_key is not None and fact.unit_key not in keys[1]:
        return "has no unit in xbrl_units"
    return None

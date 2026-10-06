from __future__ import annotations

from collections.abc import Collection, Sequence

from src.data_pipeline.contracts.sec_filings.xbrl_dimensions import XbrlDimensionRecord
from src.data_pipeline.contracts.sec_filings.xbrl_fact_footnotes import (
    XbrlFactFootnoteRecord,
)
from src.data_pipeline.quality.sec_filings.narrative import declared_count_problem


def dimension_group_problem(
    dimensions: Sequence[XbrlDimensionRecord],
    expected: int,
    concept_ids: Collection[str],
) -> str | None:
    """Return why a context's dimension rows are not one complete group, or None.

    The group needs exactly `expected` rows (the members its stored context
    reports) with consistent count and ordinals 1..N, each dimension once,
    and dimension and member concepts present in `xbrl_concepts`.
    """

    if not dimensions:
        return (
            None if expected == 0 else f"no dimensions; the context reports {expected}"
        )
    problem = declared_count_problem(
        [d.dimension_ordinal for d in dimensions],
        {d.context_dimension_count for d in dimensions},
        "dimensions",
    )
    if problem is not None:
        return problem
    if len(dimensions) != expected:
        return f"{len(dimensions)} dimensions; the context reports {expected}"
    if len({d.dimension_concept_id for d in dimensions}) != len(dimensions):
        return "a dimension is reported twice"
    missing = sorted(
        concept_id
        for d in dimensions
        for concept_id in (d.dimension_concept_id, d.member_concept_id)
        if concept_id is not None and concept_id not in concept_ids
    )
    return f"concepts missing from xbrl_concepts: {missing[:3]}" if missing else None


def footnote_group_problem(footnotes: Sequence[XbrlFactFootnoteRecord]) -> str | None:
    """Return why a fact's footnote rows are not one complete group, or None."""

    return declared_count_problem(
        [f.footnote_ordinal for f in footnotes],
        {f.fact_footnote_count for f in footnotes},
        "footnotes",
    )


def footnote_resources_problem(
    footnotes: Sequence[XbrlFactFootnoteRecord],
) -> str | None:
    """Return why an instance's footnote rows disagree on a shared resource, or None.

    Rows of one resource, whatever fact they note, carry its one location
    and content.
    """

    seen: dict[str, tuple[str, str]] = {}
    for footnote in footnotes:
        resource = (footnote.source_xpath, footnote.footnote_content_sha256)
        if seen.setdefault(footnote.footnote_resource_id, resource) != resource:
            return f"resource {footnote.footnote_resource_id} differs between facts"
    return None

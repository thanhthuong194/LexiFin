from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

from src.data_pipeline.contracts.sec_filings.content_blocks import (
    ContentBlockRecord,
    sha256_text,
)
from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.sections import SectionRecord
from src.data_pipeline.transforms.sec_filings.narrative import (
    deepest_section_ordinals,
)


def _lineage_problem(
    parent: DocumentRecord, rows: Sequence[SectionRecord | ContentBlockRecord]
) -> str | None:
    if any(
        (row.document_id, row.filing_id, row.filing_date)
        != (parent.document_id, parent.filing_id, parent.filing_date)
        for row in rows
    ):
        return "rows do not carry their parent's document_id, filing_id, filing_date"
    hashes = {row.source_document_sha256 for row in rows}
    if hashes != {parent.content_sha256}:
        return (
            f"source_document_sha256 {sorted(hashes)} differs from the parent "
            f"content_sha256 {parent.content_sha256}"
        )
    return None


def declared_count_problem(
    ordinals: Sequence[int], declared: set[int], rows: str
) -> str | None:
    """A group is complete only when its declared count matches its 1..N ordinals."""

    if declared != {len(ordinals)}:
        return f"{len(ordinals)} {rows} but declared counts {sorted(declared)}"
    if sorted(ordinals) != list(range(1, len(ordinals) + 1)):
        return f"{rows} ordinals are not exactly 1..{len(ordinals)}"
    return None


def sections_problem(
    parent: DocumentRecord, sections: Sequence[SectionRecord]
) -> str | None:
    """Return why a document's sections are not one complete hierarchy, or None.

    Checks lineage and source hash, declared count against ordinals 1..N,
    strictly increasing starts of detected sections, and that every non-root
    section has a shallower parent in the same document whose range contains
    its own. Root shape and ID derivation are enforced by SectionRecord.
    """

    if not sections:
        return "document has no sections"
    problem = _lineage_problem(parent, sections) or declared_count_problem(
        [section.section_ordinal for section in sections],
        {section.document_section_count for section in sections},
        "sections",
    )
    if problem is not None:
        return problem

    return _hierarchy_problem(
        sorted(sections, key=lambda section: section.section_ordinal)
    )


def _hierarchy_problem(ordered: Sequence[SectionRecord]) -> str | None:
    """Check start order, parent levels and range nesting of ordered sections."""

    starts = [section.start_block_ordinal for section in ordered[1:]]
    if any(earlier >= later for earlier, later in pairwise(starts)):
        return "detected sections do not start in strictly increasing block order"
    by_id = {section.section_id: section for section in ordered}
    for section in ordered[1:]:
        owner = by_id.get(section.parent_section_id or "")
        if owner is None:
            return f"{section.section_id}: parent is not a section of this document"
        if owner.level >= section.level:
            return f"{section.section_id}: level is not below its parent's"
        if not (
            owner.start_block_ordinal <= section.start_block_ordinal
            and section.end_block_ordinal <= owner.end_block_ordinal
        ):
            return f"{section.section_id}: range lies outside its parent's range"
    return None


def narrative_problem(
    parent: DocumentRecord,
    sections: Sequence[SectionRecord],
    blocks: Sequence[ContentBlockRecord],
) -> str | None:
    """Return why a document's content blocks do not fit its sections, or None.

    `sections` must already pass `sections_problem`. Checks lineage and the
    declared block count, that the root spans exactly every block, that each
    block belongs to the deepest section containing it, and every
    `section_text_sha256` against the blocks in the section's range. Block
    text hashes and table XPaths are enforced by ContentBlockRecord.
    """

    if not blocks:
        return "document has no content blocks"
    problem = _lineage_problem(parent, blocks) or declared_count_problem(
        [block.block_ordinal for block in blocks],
        {block.document_block_count for block in blocks},
        "content blocks",
    )
    if problem is not None:
        return problem

    ordered = sorted(blocks, key=lambda block: block.block_ordinal)
    root = min(sections, key=lambda section: section.section_ordinal)
    if root.end_block_ordinal != len(ordered) + 1:
        return f"root section ends at {root.end_block_ordinal}; blocks: {len(ordered)}"
    ids = {section.section_ordinal: section.section_id for section in sections}
    owners = deepest_section_ordinals(
        [
            (s.section_ordinal, s.level, s.start_block_ordinal, s.end_block_ordinal)
            for s in sections
        ],
        len(ordered),
    )
    for block, owner in zip(ordered, owners, strict=True):
        if block.section_id != ids[owner]:
            return (
                f"{block.block_id} is not assigned to its deepest section {ids[owner]}"
            )
    texts = [block.text for block in ordered]
    for section in sections:
        section_text = "\n".join(
            texts[section.start_block_ordinal - 1 : section.end_block_ordinal - 1]
        )
        if sha256_text(section_text) != section.section_text_sha256:
            return (
                f"{section.section_id}: section_text_sha256 does not match its blocks"
            )
    return None

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceBlock:
    """A Silver content block as the chunker sees it."""

    block_id: str
    block_ordinal: int
    section_id: str
    block_type: str
    text: str


@dataclass(frozen=True)
class SourceSection:
    """A Silver section of a narrative document, with what the chunker derives.

    `path` lists titles from the root's child down to this section (empty for
    the root); `topic` is its `section_topic`.
    """

    section_id: str
    section_ordinal: int
    parent_section_id: str | None
    section_kind: str
    section_code: str | None
    title: str
    start_block_ordinal: int
    path: tuple[str, ...]
    topic: str

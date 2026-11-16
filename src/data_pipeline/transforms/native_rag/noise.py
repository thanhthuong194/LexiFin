from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal

from src.data_pipeline.transforms.native_rag.blocks import SourceBlock, SourceSection

# Why a block is not chunked. `section_title` blocks are not noise: their text
# goes into the chunk prefix instead of the body.
DropReason = Literal[
    "empty",
    "decoration",
    "page_number",
    "running_header",
    "table_of_contents",
    "section_title",
]

_PAGE_NUMBER = re.compile(
    r"^[-–—(\s]*(page\s+)?(\d{1,4}|[a-z]{1,2}-\d{1,4}"
    r"|(?=[ivxl])l?x{0,3}(ix|iv|v?i{0,3}))"
    r"[-–—).\s]*$",
    re.IGNORECASE,
)
_ALPHANUMERIC = re.compile(r"[^\W_]")
_DIGITS = re.compile(r"\d+")
COMPANY_SUFFIX = re.compile(r"\b(inc|corp|co|ltd|plc|llc|n\.a|s\.a)\.$", re.IGNORECASE)
RUNNING_HEADER_MAX_CHARS = 100
# A TOC section runs until the first PART/ITEM, so it also holds the prose
# before it (forward-looking statements...): only its tables and short
# lines are TOC entries.
TOC_LINE_MAX_CHARS = 150
# A section's opening block is dropped (its title is in the path) only when it
# holds little more than the title; a block carrying the item's text too
# (8-K items laid out as one table) stays.
SECTION_TITLE_SLACK_CHARS = 40
RUNNING_HEADER_MIN_REPEATS = 4
RUNNING_HEADER_MIN_SECTIONS = 3
# Page headers of one long section (financial statement notes) repeat on
# every page: this many repeats count without the section condition.
RUNNING_HEADER_MIN_REPEATS_ANYWHERE = 8


def _header_key(text: str) -> str:
    return _DIGITS.sub("#", " ".join(text.split()).lower())


def _could_be_running_header(text: str) -> bool:
    """Short, and not a sentence: a final period only after a company suffix
    (`Alphabet Inc.`), so repeated answers such as "None." stay."""

    stripped = " ".join(text.split())
    if not 0 < len(stripped) <= RUNNING_HEADER_MAX_CHARS:
        return False
    return not stripped.endswith(".") or bool(COMPANY_SUFFIX.search(stripped))


def drop_reasons(
    blocks: Sequence[SourceBlock], sections: Mapping[str, SourceSection]
) -> dict[str, DropReason]:
    """Blocks of one document that are not chunk text, with the reason.

    Rules: tables and lines of at most 150 chars in a table-of-contents
    section; the block that opens a
    detected section when it is about its title (the title goes into the
    prefix); empty text (an empty
    table included); text without letters or digits; page numbers (`12`,
    `Page 12`, `xii`, `F-12`, `- 12 -`, `23.`); running headers (text or a
    small table of at most 100 chars, not a sentence, whose text with digits as
    `#` occurs at least 4 times in at least 3 sections of the document, or at
    least 8 times anywhere).
    """

    occurrences: dict[str, list[str]] = defaultdict(list)
    for block in blocks:
        if _could_be_running_header(block.text):
            occurrences[_header_key(block.text)].append(block.section_id)
    running_headers = {
        key
        for key, section_ids in occurrences.items()
        if len(section_ids) >= RUNNING_HEADER_MIN_REPEATS_ANYWHERE
        or (
            len(section_ids) >= RUNNING_HEADER_MIN_REPEATS
            and len(set(section_ids)) >= RUNNING_HEADER_MIN_SECTIONS
        )
    }

    reasons: dict[str, DropReason] = {}
    for block in blocks:
        reason = _reason(block, sections[block.section_id], running_headers)
        if reason is not None:
            reasons[block.block_id] = reason
    return reasons


def _reason(
    block: SourceBlock, section: SourceSection, running_headers: set[str]
) -> DropReason | None:
    if section.section_kind == "table_of_contents" and (
        block.block_type == "table" or len(block.text.strip()) <= TOC_LINE_MAX_CHARS
    ):
        return "table_of_contents"
    if (
        section.parent_section_id is not None
        and block.block_ordinal == section.start_block_ordinal
        and len(" ".join(block.text.split()))
        <= len(section.title) + SECTION_TITLE_SLACK_CHARS
    ):
        return "section_title"
    return _text_reason(block, running_headers)


def _text_reason(block: SourceBlock, running_headers: set[str]) -> DropReason | None:
    text = block.text.strip()
    is_text = block.block_type != "table"
    if not text:
        return "empty"
    if not _ALPHANUMERIC.search(text):
        return "decoration"
    if is_text and _PAGE_NUMBER.match(text):
        return "page_number"
    if _could_be_running_header(text) and _header_key(text) in running_headers:
        return "running_header"
    return None

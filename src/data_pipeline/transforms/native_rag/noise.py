from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
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
# before it (forward-looking statements...) and, when Silver finds no item
# heading for a while, whole financial statements: only its short lines are
# TOC entries here; its tables are classified like any other (a TOC table is
# dropped as `toc_table`, a data table kept).
TOC_LINE_MAX_CHARS = 150
# A real TOC is short (median 5 blocks, p90 38 short lines on 500-company
# Silver); one with more short lines is a region Silver found no item heading
# in (Entergy 10-Qs: the whole document). There only lines shaped like TOC
# entries are dropped, so its subheadings stay.
TOC_MAX_SHORT_LINES = 100
_TOC_ENTRY = re.compile(
    r"^(part|item)\s+[ivx\d]+\b|\s\d{1,3}$|^(page|table of contents)$", re.IGNORECASE
)
# A section's opening block is dropped (its title is in the path) only when it
# holds little more than the title; a block carrying the item's text too
# (8-K items laid out as one table) stays.
SECTION_TITLE_SLACK_CHARS = 40
# Silver titles a section opened by a table with all its text (HP 8-K items
# laid out as one table): past this length a title is content, not a title.
SECTION_TITLE_MAX_CHARS = 200
RUNNING_HEADER_MIN_REPEATS = 4
RUNNING_HEADER_MIN_SECTIONS = 3
# Page headers of one long section (financial statement notes) repeat on
# every page: this many repeats count without the section condition.
RUNNING_HEADER_MIN_REPEATS_ANYWHERE = 8
# A repeated line counts as a running header only near page breaks: within
# this many blocks of a page number, a `Table of Contents` line or another
# running header, for at least half its occurrences.
PAGE_BREAK_WINDOW = 3
PAGE_COUNTER_RISING_SHARE = 0.8
_LAST_NUMBER = re.compile(r"(\d+)\D*$")


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

    Rules: lines of at most 150 chars in a table-of-contents section, or in
    one with over 100 such lines (not a real TOC) only lines shaped like TOC
    entries (its tables are classified by the chunker); the block that opens a detected
    section when it is about its title, at most min(title, 200) + 40 chars
    (the title goes into the prefix); empty text (an empty
    table included); text without letters or digits; page numbers (`12`,
    `Page 12`, `xii`, `F-12`, `- 12 -`, `23.`); running headers and footers
    (`_running_headers`: repeated short lines near page breaks).
    """

    running_headers = _running_headers(blocks)

    short_lines = Counter(
        block.section_id
        for block in blocks
        if sections[block.section_id].section_kind == "table_of_contents"
        and _is_short_line(block)
    )
    large_tocs = {s for s, n in short_lines.items() if n > TOC_MAX_SHORT_LINES}
    reasons: dict[str, DropReason] = {}
    for block in blocks:
        reason = _reason(
            block,
            sections[block.section_id],
            running_headers,
            large_toc=block.section_id in large_tocs,
        )
        if reason is not None:
            reasons[block.block_id] = reason
    return reasons


def _is_page_marker(block: SourceBlock) -> bool:
    text = " ".join(block.text.split())
    if block.block_type != "table" and _PAGE_NUMBER.match(text):
        return True
    return text.lower() == "table of contents"


def _is_page_counter(texts: Sequence[str]) -> bool:
    """Whether the last number of the texts mostly increases along the
    document, like a page number inside a footer (`Apple Inc. | 2021 Form
    10-K | 20`)."""

    numbers = [int(m.group(1)) for t in texts if (m := _LAST_NUMBER.search(t))]
    if len(set(numbers)) < RUNNING_HEADER_MIN_REPEATS - 1:
        return False
    rises = sum(b > a for a, b in itertools.pairwise(numbers))
    return rises >= PAGE_COUNTER_RISING_SHARE * (len(numbers) - 1)


def _running_headers(blocks: Sequence[SourceBlock]) -> set[str]:
    """Keys (`_header_key`) of the document's running headers and footers.

    A candidate (short, not a sentence) repeats at least 4 times in at least
    3 sections, or at least 8 times. It is a running header when it is
    `Table of Contents`, carries a page counter, or when at least half its
    occurrences lie within 3 blocks of a page break (a page number, a
    `Table of Contents` line, or another running header; two passes). Row
    labels repeated in a table laid out as text (`US, Canada, Puerto Rico -
    Named Storm and Earthquake`) are not near page breaks and stay.
    """

    positions: dict[str, list[int]] = defaultdict(list)
    sections: dict[str, set[str]] = defaultdict(set)
    texts: dict[str, list[str]] = defaultdict(list)
    for index, block in enumerate(blocks):
        if _could_be_running_header(block.text):
            key = _header_key(block.text)
            positions[key].append(index)
            sections[key].add(block.section_id)
            texts[key].append(" ".join(block.text.split()))
    repeated = [
        key
        for key, found in positions.items()
        if len(found) >= RUNNING_HEADER_MIN_REPEATS_ANYWHERE
        or (
            len(found) >= RUNNING_HEADER_MIN_REPEATS
            and len(sections[key]) >= RUNNING_HEADER_MIN_SECTIONS
        )
    ]
    headers = {
        key
        for key in repeated
        if key == "table of contents" or _is_page_counter(texts[key])
    }
    breaks = {i for i, block in enumerate(blocks) if _is_page_marker(block)}
    for _ in range(2):
        marks = breaks | {i for key in headers for i in positions[key]}
        for key in repeated:
            near = sum(
                any(
                    j in marks
                    for j in range(i - PAGE_BREAK_WINDOW, i + PAGE_BREAK_WINDOW + 1)
                    if j != i
                )
                for i in positions[key]
            )
            if near * 2 >= len(positions[key]):
                headers.add(key)
    return headers


def _is_short_line(block: SourceBlock) -> bool:
    return block.block_type != "table" and len(block.text.strip()) <= TOC_LINE_MAX_CHARS


def _reason(
    block: SourceBlock,
    section: SourceSection,
    running_headers: set[str],
    *,
    large_toc: bool,
) -> DropReason | None:
    if (
        section.section_kind == "table_of_contents"
        and _is_short_line(block)
        and (not large_toc or _TOC_ENTRY.search(block.text.strip()))
    ):
        return "table_of_contents"
    title_room = min(len(section.title), SECTION_TITLE_MAX_CHARS)
    if (
        section.parent_section_id is not None
        and block.block_ordinal == section.start_block_ordinal
        and len(" ".join(block.text.split())) <= title_room + SECTION_TITLE_SLACK_CHARS
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

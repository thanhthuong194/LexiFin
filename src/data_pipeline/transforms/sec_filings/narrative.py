from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from src.data_pipeline.contracts.sec_filings.content_blocks import (
    ContentBlockRecord,
    block_id_for,
    sha256_text,
)
from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.sections import (
    SectionKind,
    SectionRecord,
    section_id_for,
)
from src.data_pipeline.transforms.sec_filings.html_blocks import HtmlBlock
from src.data_pipeline.transforms.sec_filings.html_dom import normalize_text

# A regulatory heading is a short standalone block: the code, optionally
# followed by a separator and a title that does not continue a sentence.
_MAX_HEADING_LENGTH = 300
# A newline separates the cells of a table block ("Item 5.02\nDeparture ...").
_TITLE_SEPARATORS = " \n.:-\u2013\u2014"
_PART = re.compile(r"PART\s+(IV|I{1,3})(?![A-Z0-9])(.*)", re.IGNORECASE | re.DOTALL)
# 10-K/10-Q items: 1, 1A, 7A, 16 (never a decimal 8-K code such as 2.02).
_PERIODIC_ITEM = re.compile(
    r"ITEM\s+(\d{1,2}[A-Z]?)(?![A-Z0-9]|\.\d)(.*)", re.IGNORECASE | re.DOTALL
)
_CURRENT_ITEM = re.compile(
    r"ITEM\s+(\d{1,2}\.\d{2})(?!\d)(.*)", re.IGNORECASE | re.DOTALL
)
_STANDALONE_HEADINGS: dict[str, tuple[SectionKind, str]] = {
    "SIGNATURE": ("signature", "SIGNATURES"),
    "SIGNATURES": ("signature", "SIGNATURES"),
    "EXHIBIT INDEX": ("exhibit_index", "EXHIBIT INDEX"),
    "INDEX TO EXHIBITS": ("exhibit_index", "EXHIBIT INDEX"),
    "TABLE OF CONTENTS": ("table_of_contents", "TABLE OF CONTENTS"),
}
_PERIODIC_FORMS = frozenset({"10-K", "10-Q"})
_CODE_LINE = re.compile(r"(?:PART\s+[IV]+|ITEM\s+\d)", re.IGNORECASE)


@dataclass(frozen=True)
class SectionHeading:
    """A detected section heading: the 0-based index of its block, kind and code."""

    block_index: int
    kind: SectionKind
    code: str


@dataclass(frozen=True)
class _Candidate:
    block_index: int
    kind: SectionKind
    code: str
    titled: bool


@dataclass(frozen=True)
class DocumentNarrative:
    """The sections and content blocks built from one parsed document."""

    sections: tuple[SectionRecord, ...]
    blocks: tuple[ContentBlockRecord, ...]
    duplicate_section_codes: int


def _title_after_code(rest: str) -> str | None:
    """Return the title after a code ('' if none), or None if a sentence goes on."""

    if rest and rest[0] not in _TITLE_SEPARATORS:
        return None
    title = rest.strip(_TITLE_SEPARATORS)
    if title[:1].islower():
        return None
    return title


def _match_heading(form: str, block: HtmlBlock) -> tuple[SectionKind, str, bool] | None:
    """Return (kind, normalized code, has title) if the block reads as a heading."""

    text = block.text
    if len(text) > _MAX_HEADING_LENGTH or 2 * block.link_text_length >= len(text):
        return None  # running text, or navigation such as TOC links
    if sum(_CODE_LINE.match(line) is not None for line in text.split("\n")) > 1:
        return None  # a table listing several codes is a table of contents
    standalone = _STANDALONE_HEADINGS.get(text.upper().rstrip(".:"))
    if standalone is not None:
        return (*standalone, True)
    patterns: tuple[tuple[re.Pattern[str], SectionKind, str], ...] = (
        ((_PART, "part", "PART"), (_PERIODIC_ITEM, "item", "ITEM"))
        if form in _PERIODIC_FORMS
        else ((_CURRENT_ITEM, "item", "ITEM"),)
    )
    for pattern, kind, prefix in patterns:
        match = pattern.fullmatch(text)
        if match is not None:
            title = _title_after_code(match.group(2))
            if title is None:
                return None
            return kind, f"{prefix} {match.group(1).upper()}", bool(title)
    return None


def _select_headings(candidates: Sequence[_Candidate]) -> list[SectionHeading]:
    """Drop repeated running headers and code-only duplicates of titled headings.

    A code-only PART/ITEM (e.g. a page header "Item 1") is dropped when the
    same code, in the same PART, also appears with a title; a heading that
    repeats the section already open is a page header. Only the first table
    of contents before the body counts.
    """

    part: str | None = None
    keys: list[tuple[str | None, str]] = []
    for candidate in candidates:
        if candidate.kind == "part":
            part = candidate.code
        keys.append((part if candidate.kind == "item" else None, candidate.code))
    titled = {
        key for candidate, key in zip(candidates, keys, strict=True) if candidate.titled
    }

    headings: list[SectionHeading] = []
    open_part: tuple[str | None, str] | None = None
    open_item: tuple[str | None, str] | None = None
    seen_toc = body_started = False
    for candidate, key in zip(candidates, keys, strict=True):
        if candidate.kind in ("part", "item"):
            if key in (open_part, open_item) or (
                not candidate.titled and key in titled
            ):
                continue
            if candidate.kind == "part":
                open_part, open_item = key, None
            else:
                open_item = key
            body_started = True
        elif candidate.kind == "table_of_contents":
            if seen_toc or body_started:
                continue
            seen_toc = True
        else:
            if headings and headings[-1].kind == candidate.kind:
                continue
            open_part = open_item = None
        headings.append(
            SectionHeading(candidate.block_index, candidate.kind, candidate.code)
        )
    return headings


def detect_section_headings(
    form: str, blocks: Sequence[HtmlBlock]
) -> list[SectionHeading]:
    """Detect regulatory section headings among a document's blocks, in order.

    Rule-based and form-aware: PART and integer ITEM codes (1, 1A) for 10-K and
    10-Q, decimal ITEM codes (2.02) for 8-K, plus SIGNATURES, EXHIBIT INDEX and
    TABLE OF CONTENTS. Navigation links, blocks listing several codes (a table
    of contents) and repeated page headers do not start sections.
    """

    candidates = []
    for index, block in enumerate(blocks):
        match = _match_heading(form, block)
        if match is not None:
            kind, code, titled = match
            candidates.append(_Candidate(index, kind, code, titled))
    return _select_headings(candidates)


@dataclass(frozen=True)
class _SectionSpan:
    ordinal: int
    kind: SectionKind
    code: str | None
    title: str
    level: int
    parent_ordinal: int | None
    start: int
    source_xpath: str | None


def _section_spans(
    document_type: str, blocks: Sequence[HtmlBlock], headings: Sequence[SectionHeading]
) -> list[_SectionSpan]:
    """Place the root and each heading in the hierarchy, ordered by start block.

    PART is a root child; a 10-K/10-Q ITEM belongs to the open PART (level 2)
    or else to the root; every other section is a root child. SIGNATURES,
    EXHIBIT INDEX and TABLE OF CONTENTS close the open PART.
    """

    spans = [
        _SectionSpan(1, "document", None, f"{document_type} Document", 0, None, 1, None)
    ]
    open_part: int | None = None
    for ordinal, heading in enumerate(headings, start=2):
        block = blocks[heading.block_index]
        level, parent = 1, 1
        if heading.kind == "part":
            open_part = ordinal
        elif heading.kind == "item":
            if open_part is not None:
                level, parent = 2, open_part
        else:
            open_part = None
        spans.append(
            _SectionSpan(
                ordinal,
                heading.kind,
                heading.code,
                # A table heading's text joins its cells with "\n"; titles are
                # normalized observed text like every other heading.
                normalize_text(block.text),
                level,
                parent,
                heading.block_index + 1,
                block.source_xpath,
            )
        )
    return spans


def _section_ends(spans: Sequence[_SectionSpan], block_count: int) -> list[int]:
    """End (exclusive) of each span: the next span at the same or a higher level."""

    return [
        next(
            (
                later.start
                for later in spans[position + 1 :]
                if later.level <= span.level
            ),
            block_count + 1,
        )
        for position, span in enumerate(spans)
    ]


def deepest_section_ordinals(
    ranges: Sequence[tuple[int, int, int, int]], block_count: int
) -> list[int]:
    """Return, per block, the ordinal of the deepest section whose range contains it.

    `ranges` holds (section_ordinal, level, start_block_ordinal,
    end_block_ordinal) including the root, which spans every block. Item i of
    the result is the owner of block ordinal i + 1.
    """

    owner = [0] * block_count
    for ordinal, _, start, end in sorted(ranges, key=lambda section: section[1]):
        owner[start - 1 : end - 1] = [ordinal] * (end - start)
    return owner


def _duplicate_codes(spans: Sequence[_SectionSpan]) -> int:
    """Count sections whose code already occurred under the same parent code."""

    codes = {span.ordinal: span.code for span in spans}
    keys = [(codes[span.parent_ordinal or 1], span.code) for span in spans[1:]]
    return len(keys) - len(set(keys))


def build_document_narrative(
    parent: DocumentRecord,
    blocks: Sequence[HtmlBlock],
    headings: Sequence[SectionHeading],
    processed_at: datetime,
) -> DocumentNarrative:
    """Build the section and content-block records of one primary HTML document.

    Every block is assigned to the deepest section whose range contains it;
    a section's range and text hash include the blocks of its descendants.
    Raises ValueError when the document has no visible text block.
    """

    source_sha256 = parent.content_sha256
    if source_sha256 is None:
        raise ValueError("Parent document has no <TEXT> content")
    if not blocks:
        raise ValueError("Document has no visible text blocks")
    spans = _section_spans(parent.document_type, blocks, headings)
    ends = _section_ends(spans, len(blocks))
    owner = deepest_section_ordinals(
        [
            (span.ordinal, span.level, span.start, end)
            for span, end in zip(spans, ends, strict=True)
        ],
        len(blocks),
    )
    heading_blocks = {heading.block_index for heading in headings}

    def section_id(ordinal: int) -> str:
        return section_id_for(parent.document_id, ordinal)

    sections = tuple(
        SectionRecord(
            section_id=section_id(span.ordinal),
            filing_id=parent.filing_id,
            document_id=parent.document_id,
            filing_date=parent.filing_date,
            section_ordinal=span.ordinal,
            document_section_count=len(spans),
            parent_section_id=(
                None if span.parent_ordinal is None else section_id(span.parent_ordinal)
            ),
            section_kind=span.kind,
            section_code=span.code,
            title=span.title,
            level=span.level,
            start_block_ordinal=span.start,
            end_block_ordinal=end,
            source_xpath=span.source_xpath,
            source_document_sha256=source_sha256,
            section_text_sha256=sha256_text(
                "\n".join(block.text for block in blocks[span.start - 1 : end - 1])
            ),
            processed_at=processed_at,
        )
        for span, end in zip(spans, ends, strict=True)
    )
    records = tuple(
        ContentBlockRecord(
            block_id=block_id_for(parent.document_id, index + 1),
            filing_id=parent.filing_id,
            document_id=parent.document_id,
            filing_date=parent.filing_date,
            section_id=section_id(owner[index]),
            block_ordinal=index + 1,
            document_block_count=len(blocks),
            # A table block stays a table even when it holds a heading, so
            # it keeps its document_tables row.
            block_type=(
                "heading"
                if index in heading_blocks and block.block_type != "table"
                else block.block_type
            ),
            text=block.text,
            text_length=len(block.text),
            text_sha256=sha256_text(block.text),
            source_tag=block.source_tag,
            source_xpath=block.source_xpath,
            heading_level=block.heading_level,
            inside_table=block.table_xpath is not None,
            source_table_xpath=block.table_xpath,
            source_document_sha256=source_sha256,
            processed_at=processed_at,
        )
        for index, block in enumerate(blocks)
    )
    return DocumentNarrative(sections, records, _duplicate_codes(spans))

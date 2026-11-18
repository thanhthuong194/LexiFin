from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.data_pipeline.transforms.native_rag.noise import COMPANY_SUFFIX, MARKER_ONLY
from src.data_pipeline.transforms.native_rag.sentences import split_sentences

TITLE_MAX_CHARS = 120
TITLE_MAX_WORDS = 12
_LETTER = re.compile(r"[A-Za-z]")
JOINER = "\n"


def is_title_like(text: str) -> bool:
    """A short standalone line that reads as a subheading.

    At most 120 chars and 12 words, contains a letter, is not a lone
    footnote marker (`(c)`), and does not end in sentence or clause
    punctuation (`.`, `,`, `;`, `:`), except a company suffix
    (`Amazon.com, Inc.`).
    """

    stripped = text.strip()
    return (
        0 < len(stripped) <= TITLE_MAX_CHARS
        and not MARKER_ONLY.match(stripped)
        and len(stripped.split()) <= TITLE_MAX_WORDS
        and bool(_LETTER.search(stripped))
        and (stripped[-1] not in ".,;:" or bool(COMPANY_SUFFIX.search(stripped)))
    )


@dataclass(frozen=True)
class Paragraph:
    """One text block in reading order (a layout-table row counts as one)."""

    block_id: str
    text: str
    title_like: bool
    # Further blocks joined into this paragraph (word-fragmented documents).
    extra_block_ids: tuple[str, ...] = ()

    @property
    def block_ids(self) -> tuple[str, ...]:
        return (self.block_id, *self.extra_block_ids)


@dataclass(frozen=True)
class TextChunk:
    """Body of one text chunk. `part`/`part_count` are set on the pieces of a
    paragraph too long for one chunk (all from the same block)."""

    text: str
    block_ids: tuple[str, ...]
    subheading: str | None
    part: int | None = None
    part_count: int | None = None


@dataclass(frozen=True)
class Sizing:
    """How the chunker measures text.

    `size` counts tokens of a body without special tokens; `fits` says whether
    the embedded chunk (prefix with this subheading + body) is within the cap;
    `budget` is the body tokens left beside the prefix of a subheading;
    `split_tokens` cuts text into pieces of at most N tokens; `min_tokens` is
    the size under which a unit is merged into a neighbour.
    """

    size: Callable[[str], int]
    fits: Callable[[str | None, str], bool]
    budget: Callable[[str | None], int]
    split_tokens: Callable[[str, int], list[str]]
    min_tokens: int


@dataclass
class _Unit:
    subheading: str | None
    paragraphs: list[Paragraph]

    def text(self) -> str:
        return JOINER.join(p.text for p in self.paragraphs)


def chunk_paragraphs(
    paragraphs: Sequence[Paragraph], subheading: str | None, sizing: Sizing
) -> list[TextChunk]:
    """Chunk one run of paragraphs of a section (no table inside the run).

    1. Topic units: a title-like paragraph opens a unit and names its
       subheading; paragraphs before the first title keep `subheading` (the
       one in force where the run starts).
    2. A unit that is only title lines (a category heading over the next
       heading, "Exhibit 99.1" over the release title) joins the next unit
       even when that unit will be split, under the next unit's subheading
       (the closer one). Any other unit under `min_tokens` merges whole into
       the next unit, else into the previous one, when the result fits;
       otherwise it stays short; the merged unit keeps the first unit's
       subheading. A unit left as only its title line keeps the subheading in
       force before it.
    3. A unit that fits is one chunk; a longer one is cut between paragraphs
       into balanced parts; a paragraph that alone exceeds the budget is cut
       by sentences (`_split_paragraph`) and never packed with neighbours.
    """

    units = _units(paragraphs, subheading)
    units = _merge_short(units, sizing)
    chunks: list[TextChunk] = []
    in_force = subheading
    for unit in units:
        if unit.text().strip() == unit.subheading:
            # A lone title line names nothing below it: keep the subheading
            # in force instead of repeating the text as its own heading.
            unit.subheading = in_force
        in_force = unit.subheading
        if sizing.fits(unit.subheading, unit.text()):
            chunks.append(
                TextChunk(
                    unit.text(),
                    tuple(b for p in unit.paragraphs for b in p.block_ids),
                    unit.subheading,
                )
            )
            continue
        chunks.extend(_split_unit(unit, sizing))
    return chunks


def _units(paragraphs: Sequence[Paragraph], subheading: str | None) -> list[_Unit]:
    units: list[_Unit] = []
    for paragraph in paragraphs:
        if paragraph.title_like or not units:
            name = paragraph.text.strip() if paragraph.title_like else subheading
            units.append(_Unit(name, [paragraph]))
        else:
            units[-1].paragraphs.append(paragraph)
    return units


def _merge_short(units: list[_Unit], sizing: Sizing) -> list[_Unit]:
    index = 0
    while index < len(units):
        unit = units[index]
        if (
            index + 1 < len(units)
            and all(p.title_like for p in unit.paragraphs)
            and sizing.size(unit.text()) < sizing.min_tokens
        ):
            following = units[index + 1]
            units[index : index + 2] = [
                _Unit(following.subheading, unit.paragraphs + following.paragraphs)
            ]
            continue
        if len(units) > 1 and sizing.size(unit.text()) < sizing.min_tokens:
            if index + 1 < len(units):
                merged = _Unit(
                    unit.subheading, unit.paragraphs + units[index + 1].paragraphs
                )
                if sizing.fits(merged.subheading, merged.text()):
                    units[index : index + 2] = [merged]
                    continue
            if index > 0:
                previous = units[index - 1]
                merged = _Unit(
                    previous.subheading, previous.paragraphs + unit.paragraphs
                )
                if sizing.fits(merged.subheading, merged.text()):
                    units[index - 1 : index + 1] = [merged]
                    index -= 1
                    continue
        index += 1
    return units


def _split_unit(unit: _Unit, sizing: Sizing) -> list[TextChunk]:
    budget = sizing.budget(unit.subheading)
    chunks: list[TextChunk] = []
    run: list[Paragraph] = []

    def flush() -> None:
        for group in _balanced(run, lambda p: p.text, sizing, unit.subheading):
            chunks.append(
                TextChunk(
                    JOINER.join(p.text for p in group),
                    tuple(b for p in group for b in p.block_ids),
                    unit.subheading,
                )
            )
        run.clear()

    for paragraph in unit.paragraphs:
        if sizing.size(paragraph.text) > budget or not sizing.fits(
            unit.subheading, paragraph.text
        ):
            long = paragraph
            if run and all(p.title_like for p in run):
                # Title lines right above a paragraph cut by sentences open
                # its first part instead of standing alone.
                ids = [b for p in [*run, paragraph] for b in p.block_ids]
                long = Paragraph(
                    ids[0],
                    JOINER.join([*(p.text for p in run), paragraph.text]),
                    False,
                    tuple(dict.fromkeys(ids[1:])),
                )
                run.clear()
            flush()
            chunks.extend(_split_paragraph(long, unit.subheading, sizing))
        else:
            run.append(paragraph)
    flush()
    return chunks


def _balanced[T](
    items: Sequence[T],
    text_of: Callable[[T], str],
    sizing: Sizing,
    subheading: str | None,
) -> list[list[T]]:
    """Cut items (each fitting alone) into the fewest contiguous groups that
    fit, as even as possible: the smallest largest group for that count."""

    if not items:
        return []
    sizes = [sizing.size(text_of(item)) + 1 for item in items]
    parts = max(1, -(-sum(sizes) // max(sizing.budget(subheading), 1)))
    while True:
        groups = _partition(items, sizes, parts)
        if all(
            sizing.fits(subheading, JOINER.join(text_of(i) for i in g)) for g in groups
        ):
            return groups
        if parts >= len(items):
            return [[item] for item in items]
        parts += 1


def _partition[T](
    items: Sequence[T], sizes: Sequence[int], parts: int
) -> list[list[T]]:
    """Greedy cut under the smallest bound that yields at most `parts` groups."""

    def greedy(bound: int) -> list[list[T]]:
        groups: list[list[T]] = [[]]
        total = 0
        for item, size in zip(items, sizes, strict=True):
            if groups[-1] and total + size > bound:
                groups.append([])
                total = 0
            groups[-1].append(item)
            total += size
        return groups

    low, high = max(sizes), sum(sizes)
    while low < high:
        middle = (low + high) // 2
        if len(greedy(middle)) <= parts:
            high = middle
        else:
            low = middle + 1
    return greedy(low)


def _split_paragraph(
    paragraph: Paragraph, subheading: str | None, sizing: Sizing
) -> list[TextChunk]:
    """Cut a too-long paragraph at sentence boundaries into balanced parts.

    A sentence longer than a third of the budget is cut by tokens. Each part
    after the first starts with the previous part's last piece when that piece
    is at most a quarter of the budget (a short sentence gives its vector
    context; a long token cut would not). Parts are added until every part
    fits; if single pieces with their overlap still do not (tokens are not
    exactly additive), overlaps are dropped. Parts share the block and carry
    `part`/`part_count`.
    """

    budget = sizing.budget(subheading)
    third = max(budget // 3, 1)
    pieces: list[str] = []
    for sentence in split_sentences(paragraph.text):
        if sizing.size(sentence) > third:
            pieces.extend(sizing.split_tokens(sentence, third))
        else:
            pieces.append(sentence)
    sizes = [sizing.size(piece) for piece in pieces]
    overlap_max = budget // 4
    parts = max(2, -(-sum(sizes) // max(budget - overlap_max, 1)))
    while True:
        groups = _partition(pieces, sizes, min(parts, len(pieces)))
        texts = _overlapped(groups, sizing, overlap_max)
        if all(sizing.fits(subheading, t) for t in texts):
            break
        if parts >= len(pieces):
            texts = _overlapped(groups, sizing, -1)
            break
        parts += 1
    if len(texts) == 1:
        return [TextChunk(texts[0], paragraph.block_ids, subheading)]
    return [
        TextChunk(text, paragraph.block_ids, subheading, position, len(texts))
        for position, text in enumerate(texts, start=1)
    ]


def _overlapped(
    groups: Sequence[Sequence[str]], sizing: Sizing, overlap_max: int
) -> list[str]:
    """Join each group, prefixed by the previous group's last piece when it is
    at most `overlap_max` tokens (never when negative)."""

    texts: list[str] = []
    for index, group in enumerate(groups):
        previous = groups[index - 1][-1] if index else ""
        if previous and sizing.size(previous) > overlap_max:
            previous = ""
        texts.append((previous + "".join(group)).strip())
    return texts

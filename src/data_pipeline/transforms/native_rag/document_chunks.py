from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import groupby

from src.data_pipeline.contracts.sec_filings.document_tables import table_id_for
from src.data_pipeline.transforms.native_rag.blocks import SourceBlock, SourceSection
from src.data_pipeline.transforms.native_rag.context import embedded_text
from src.data_pipeline.transforms.native_rag.noise import drop_reasons
from src.data_pipeline.transforms.native_rag.table_chunks import (
    Cell,
    RenderedTable,
    classify_table,
    layout_paragraphs,
    render_table,
    split_rows,
)
from src.data_pipeline.transforms.native_rag.text_chunks import (
    Paragraph,
    Sizing,
    chunk_paragraphs,
    is_title_like,
)
from src.data_pipeline.transforms.native_rag.tokenizer import TokenCounter

_FOOTNOTE = re.compile(r"^\s*(\(\s*[0-9a-z]{1,2}\s*\)|[0-9]{1,2}\)|\*{1,3}|[†‡§])")


@dataclass(frozen=True)
class ChunkParams:
    """Tunables of the chunker; stored with the table, covered by its rules version."""

    max_tokens: int = 512
    min_unit_tokens: int = 64
    small_table_chars: int = 100
    table_title_max_chars: int = 150
    table_lead_in_max_chars: int = 400
    max_footnotes: int = 5
    footnote_max_chars: int = 600

    def as_property(self) -> str:
        return ",".join(f"{key}={value}" for key, value in asdict(self).items())


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk of one document before keys and filing metadata are added."""

    section_id: str
    chunk_kind: str
    context: str
    text: str
    block_ids: tuple[str, ...]
    subheading: str | None
    table_id: str | None
    table_title: str | None
    part: int | None
    part_count: int | None
    token_count: int


# Builds the prefix of a chunk from its section path, subheading and table label.
ContextBuilder = Callable[[tuple[str, ...], str | None, str | None], str]


@dataclass(frozen=True)
class DocumentChunks:
    chunks: list[ChunkDraft]
    # Blocks not in any chunk, with why (noise, section title, TOC/signature table).
    dropped: dict[str, str]


@dataclass(frozen=True)
class _Table:
    block_id: str
    table_id: str
    rendered: RenderedTable


@dataclass(frozen=True)
class DocumentInput:
    """One narrative document's Silver rows and its prefix builder.

    `blocks` in document order; `cells_by_table` keyed by `table_id`.
    """

    blocks: Sequence[SourceBlock]
    sections: Mapping[str, SourceSection]
    cells_by_table: Mapping[str, Sequence[Cell]]
    context: ContextBuilder


def chunk_document(
    document: DocumentInput, counter: TokenCounter, params: ChunkParams
) -> DocumentChunks:
    """Chunk one narrative document.

    Order: drop noise; read each table block (a TOC or signature table is
    dropped, a layout table becomes paragraphs, a small data table one inline
    paragraph, any other data table its own chunk); then per run of blocks of
    one section, chunk the text between tables (`chunk_paragraphs`) and each
    table (`_table`, taking the lead-in paragraph as title and following
    footnotes). Every block ends in exactly one chunk's `block_ids` or in
    `dropped`.
    """

    dropped: dict[str, str] = dict(drop_reasons(document.blocks, document.sections))
    chunks: list[ChunkDraft] = []
    builder = _Builder(document.context, counter, params)
    for section_id, run in groupby(
        (b for b in document.blocks if b.block_id not in dropped),
        key=lambda b: b.section_id,
    ):
        items: list[Paragraph | _Table] = []
        for block in run:
            items.extend(_read_block(block, document.cells_by_table, params, dropped))
        chunks.extend(builder.section(document.sections[section_id], items))
    return DocumentChunks(chunks, dropped)


def _read_block(
    block: SourceBlock,
    cells_by_table: Mapping[str, Sequence[Cell]],
    params: ChunkParams,
    dropped: dict[str, str],
) -> list[Paragraph | _Table]:
    if block.block_type != "table":
        return [
            Paragraph(block.block_id, block.text.strip(), is_title_like(block.text))
        ]
    table_id = table_id_for(block.block_id)
    cells = cells_by_table.get(table_id, ())
    kind = classify_table(cells)
    if kind in ("toc", "signature"):
        dropped[block.block_id] = f"{kind}_table"
        return []
    if kind == "layout":
        return [
            Paragraph(block.block_id, text, is_title_like(text))
            for text in layout_paragraphs(cells)
        ]
    rendered = render_table(cells)
    lines = [*rendered.header, *rendered.rows]
    if len(block.text.strip()) < params.small_table_chars:
        return [
            Paragraph(block.block_id, "; ".join(line for line in lines if line), False)
        ]
    return [_Table(block.block_id, table_id, rendered)]


class _Builder:
    def __init__(
        self, context: ContextBuilder, counter: TokenCounter, params: ChunkParams
    ) -> None:
        self._context = context
        self._counter = counter
        self._params = params

    def _tokens(self, context: str, text: str) -> int:
        return self._counter.count(embedded_text(context, text))

    def _sizing(self, path: tuple[str, ...]) -> Sizing:
        cap = self._params.max_tokens

        def fits(subheading: str | None, text: str) -> bool:
            return self._tokens(self._context(path, subheading, None), text) <= cap

        def budget(subheading: str | None) -> int:
            prefix = self._counter.count(self._context(path, subheading, None) + "\n\n")
            return cap - prefix

        return Sizing(
            size=lambda text: max(self._counter.count(text) - 1, 0),
            fits=fits,
            budget=budget,
            split_tokens=self._counter.split,
            min_tokens=self._params.min_unit_tokens,
        )

    def section(
        self, section: SourceSection, items: Sequence[Paragraph | _Table]
    ) -> list[ChunkDraft]:
        sizing = self._sizing(section.path)
        chunks: list[ChunkDraft] = []
        subheading: str | None = None
        run: list[Paragraph] = []
        index = 0
        while index < len(items):
            item = items[index]
            if isinstance(item, Paragraph):
                run.append(item)
                index += 1
                continue
            title = self._title_lines(run)
            chunks.extend(self._text(section, run, subheading, sizing))
            subheading = _subheading_after(run, subheading)
            heading = next((line.text for line in title if line.title_like), None)
            subheading = heading or subheading
            run = []
            footnotes, index = self._footnotes(items, index + 1)
            chunks.extend(self._table(section, item, title, footnotes, subheading))
        chunks.extend(self._text(section, run, subheading, sizing))
        return chunks

    def _title_lines(self, run: list[Paragraph]) -> list[Paragraph]:
        """Pop a table's title off the end of the text before it, in document
        order: the last paragraph when short or a lead-in ending in `:`
        (`... were as follows:`), plus up to three short title-like lines
        above it (`Amazon.com, Inc.` / `Consolidated Balance Sheets`)."""

        params = self._params
        lines: list[Paragraph] = []
        if (
            run
            and not _splits_block(run)
            and (
                len(run[-1].text) <= params.table_title_max_chars
                or (
                    run[-1].text.endswith(":")
                    and len(run[-1].text) <= params.table_lead_in_max_chars
                )
            )
        ):
            lines.append(run.pop())
        while (
            lines
            and run
            and not _splits_block(run)
            and len(lines) < 4
            and run[-1].title_like
            and len(run[-1].text) <= params.table_title_max_chars
        ):
            lines.insert(0, run.pop())
        return lines

    def _footnotes(
        self, items: Sequence[Paragraph | _Table], start: int
    ) -> tuple[list[Paragraph], int]:
        notes: list[Paragraph] = []
        index = start
        while index < len(items) and len(notes) < self._params.max_footnotes:
            paragraph = items[index]
            if (
                not isinstance(paragraph, Paragraph)
                or len(paragraph.text) > self._params.footnote_max_chars
                or not _FOOTNOTE.match(paragraph.text)
            ):
                break
            notes.append(paragraph)
            index += 1
        return notes, index

    def _text(
        self,
        section: SourceSection,
        run: Sequence[Paragraph],
        subheading: str | None,
        sizing: Sizing,
    ) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for chunk in chunk_paragraphs(run, subheading, sizing):
            context = self._context(section.path, chunk.subheading, None)
            drafts.append(
                ChunkDraft(
                    section.section_id,
                    "text",
                    context,
                    chunk.text,
                    tuple(dict.fromkeys(chunk.block_ids)),
                    chunk.subheading,
                    None,
                    None,
                    chunk.part,
                    chunk.part_count,
                    self._tokens(context, chunk.text),
                )
            )
        return drafts

    def _table(
        self,
        section: SourceSection,
        table: _Table,
        title: Sequence[Paragraph],
        footnotes: Sequence[Paragraph],
        subheading: str | None,
    ) -> list[ChunkDraft]:
        title_text = " ".join(line.text for line in title) or None
        cap = self._params.max_tokens
        rendered = RenderedTable(
            table.rendered.header,
            (*table.rendered.rows, *(note.text for note in footnotes)),
        )
        # Size parts against the widest label; token cuts can add parts and
        # lengthen `(part i/n)`, so check the real prefixes and cut finer
        # until every part fits.
        most = max(len(rendered.rows), 2)
        margin = 0
        while True:
            widest = self._context(
                section.path, subheading, _label(title_text, most, most)
            )
            parts = split_rows(rendered, "", self._fits_within(widest, cap - margin))
            parts = self._hard_split(parts, widest, margin)
            labels = _part_labels(title_text, len(parts))
            over = max(
                self._tokens(self._context(section.path, subheading, label), text) - cap
                for label, text in zip(labels, parts, strict=True)
            )
            if over <= 0:
                break
            most = max(most, len(parts))
            margin += over
        block_ids = tuple(
            dict.fromkeys(
                [
                    *(line.block_id for line in title),
                    table.block_id,
                    *(n.block_id for n in footnotes),
                ]
            )
        )
        drafts: list[ChunkDraft] = []
        count = len(parts)
        for position, (label, text) in enumerate(zip(labels, parts, strict=True), 1):
            context = self._context(section.path, subheading, label)
            drafts.append(
                ChunkDraft(
                    section.section_id,
                    "table",
                    context,
                    text,
                    block_ids,
                    subheading,
                    table.table_id,
                    title_text,
                    position if count > 1 else None,
                    count if count > 1 else None,
                    self._tokens(context, text),
                )
            )
        return drafts

    def _fits_within(self, context: str, limit: int) -> Callable[[str], bool]:
        return lambda text: self._tokens(context, text) <= limit

    def _hard_split(self, parts: list[str], context: str, margin: int) -> list[str]:
        """Cut by tokens any part still over the cap less `margin` (a huge row)."""

        cap = self._params.max_tokens - margin
        budget = cap - self._counter.count(context + "\n\n")
        result: list[str] = []
        for part in parts:
            if self._tokens(context, part) <= cap:
                result.append(part)
            else:
                result.extend(self._counter.split(part, max(budget - 8, 1)))
        return result


def _part_labels(title: str | None, count: int) -> list[str]:
    """The `Table:` label of each part: the title (or "" untitled) when the
    table is one chunk, else with `(part i/n)`."""

    if count == 1:
        return [title or ""]
    return [_label(title, position, count) for position in range(1, count + 1)]


def _splits_block(run: Sequence[Paragraph]) -> bool:
    """Whether taking the last paragraph would split a layout table's rows."""

    return len(run) > 1 and run[-1].block_id == run[-2].block_id


def _label(title: str | None, position: int, count: int) -> str:
    part = f"part {position}/{count}"
    return f"{title} ({part})" if title else part


def _subheading_after(run: Sequence[Paragraph], current: str | None) -> str | None:
    """The subheading in force after a run: its last title-like paragraph."""

    for paragraph in reversed(run):
        if paragraph.title_like:
            return paragraph.text
    return current

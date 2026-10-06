from __future__ import annotations

from dataclasses import dataclass, field

from lxml import etree

from src.data_pipeline.contracts.sec_filings.content_blocks import BlockType
from src.data_pipeline.transforms.sec_filings.html_dom import (
    HEADING_TAGS,
    element_tag,
    is_hidden,
    is_link,
    normalize_text,
)
from src.data_pipeline.transforms.sec_filings.html_tables import (
    PhysicalCell,
    physical_cells,
)

_BLOCK_TYPE_BY_TAG: dict[str, BlockType] = {
    **dict.fromkeys(HEADING_TAGS, "heading"),
    "p": "paragraph",
    "blockquote": "paragraph",
    "div": "paragraph",
    "li": "list_item",
    "pre": "preformatted",
}
# Other block-level elements own the text written directly inside them, so it
# is not merged with text of a neighbouring block. Tables are handled apart.
_OTHER_CONTAINERS = frozenset(
    {
        "body", "tr", "ul", "ol", "dl", "dt", "dd", "caption", "center",
        "section", "article", "header", "footer", "address", "figcaption",
    }
)  # fmt: skip


@dataclass(frozen=True)
class HtmlBlock:
    """One visible-text block in document order, before section assignment.

    `table_xpath` names the nearest enclosing `<table>` (for a table block,
    the table around it). `link_text_length` counts the characters of `text`
    that come from `<a href>` links, a signal for table-of-contents navigation.
    """

    text: str
    block_type: BlockType
    source_tag: str
    source_xpath: str
    heading_level: int | None
    table_xpath: str | None
    link_text_length: int


@dataclass(frozen=True)
class ExtractedBlocks:
    """A document's blocks, and how many `<table>`s had no visible cell.

    Tables without cells are markup only: they get no table block (any text
    inside them flows into the surrounding blocks) and stay in Bronze.
    """

    blocks: list[HtmlBlock]
    empty_tables: int


@dataclass
class _OpenContainer:
    element: etree._Element
    tag: str
    pieces: list[str] = field(default_factory=list)
    link_pieces: list[str] = field(default_factory=list)


def is_block_container(tag: str) -> bool:
    """Whether an element of this tag owns the text runs written directly in it."""

    return tag in _BLOCK_TYPE_BY_TAG or tag in _OTHER_CONTAINERS


def _table_xpath(element: etree._Element, tree: etree._ElementTree) -> str | None:
    table = next(element.iterancestors("table"), None)
    return None if table is None else tree.getpath(table)


class _BlockWalk:
    """Text runs of the open block containers while walking the DOM in order."""

    def __init__(self, tree: etree._ElementTree) -> None:
        self.tree = tree
        self.blocks: list[HtmlBlock] = []
        self.stack: list[_OpenContainer] = []
        self.open_tables: list[etree._Element] = []
        self.empty_tables = 0
        self.link_depth = 0

    def add(self, text: str | None) -> None:
        if text and self.stack and not self.open_tables:
            self.stack[-1].pieces.append(text)
            if self.link_depth:
                self.stack[-1].link_pieces.append(text)

    def open(self, element: etree._Element, tag: str) -> None:
        if self.stack:
            self._flush(self.stack[-1])
        self.stack.append(_OpenContainer(element, tag))

    def close(self) -> None:
        self._flush(self.stack.pop())

    def open_table(self, table: etree._Element, cells: list[PhysicalCell]) -> None:
        """Emit a table as one block; its content stays out of other blocks."""

        if self.stack and not self.open_tables:
            self._flush(self.stack[-1])
        self.open_tables.append(table)
        self.blocks.append(
            HtmlBlock(
                text="\n".join(cell.text for cell in cells if cell.text),
                block_type="table",
                source_tag="table",
                source_xpath=self.tree.getpath(table),
                heading_level=None,
                table_xpath=_table_xpath(table, self.tree),
                link_text_length=sum(len(cell.link_text) for cell in cells),
            )
        )

    def start(self, element: etree._Element, tag: str) -> bool:
        """Handle an opening element; return True to skip its subtree."""

        if is_hidden(element, tag):
            return True
        if tag == "table":
            cells = physical_cells(element)
            if cells:
                self.open_table(element, cells)
                return False
            self.empty_tables += 1
        if self.open_tables:
            return False
        if is_block_container(tag):
            self.open(element, tag)
        self.link_depth += is_link(element, tag)
        self.add(" " if tag == "br" else element.text)
        return False

    def end(self, element: etree._Element, tag: str) -> None:
        """Handle a closing element, including one whose subtree was skipped."""

        if self.open_tables and element is self.open_tables[-1]:
            self.open_tables.pop()
        elif self.open_tables:
            return
        # skip_subtree() still reports the end of a hidden element: only its
        # tail, which follows it in the parent, is visible.
        elif not is_hidden(element, tag):
            self.link_depth -= is_link(element, tag)
            if is_block_container(tag):
                self.close()
        self.add(element.tail)

    def _flush(self, container: _OpenContainer) -> None:
        """Emit the container's pending text run as a block if it has text."""

        text = normalize_text("".join(container.pieces))
        if text:
            self.blocks.append(
                HtmlBlock(
                    text=text,
                    block_type=_BLOCK_TYPE_BY_TAG.get(container.tag, "other"),
                    source_tag=container.tag,
                    source_xpath=self.tree.getpath(container.element),
                    heading_level=HEADING_TAGS.get(container.tag),
                    table_xpath=_table_xpath(container.element, self.tree),
                    link_text_length=len(
                        normalize_text("".join(container.link_pieces))
                    ),
                )
            )
        container.pieces.clear()
        container.link_pieces.clear()


def extract_blocks(root: etree._Element) -> ExtractedBlocks:
    """Return the visible-text blocks of a parsed HTML document in reading order.

    Every visible text segment outside tables belongs to its nearest block
    container (h1-h6, p, li, pre, blockquote, div or another block element);
    inline elements such as span or ix:nonNumeric add their text to that
    container. A child container splits its parent's text into separate runs,
    so no text is emitted twice and document order is kept. Each `<table>`
    with a visible cell is one `table` block; a nested table is its own block.
    Hidden elements contribute nothing, but the text after them still counts.
    Raises ValueError for a table cell outside any row of its table.
    """

    walk = _BlockWalk(root.getroottree())
    walker = etree.iterwalk(root, events=("start", "end"))
    for event, element in walker:
        tag = element_tag(element)
        if event == "end":
            walk.end(element, tag)
        elif walk.start(element, tag):
            walker.skip_subtree()
    return ExtractedBlocks(walk.blocks, walk.empty_tables)

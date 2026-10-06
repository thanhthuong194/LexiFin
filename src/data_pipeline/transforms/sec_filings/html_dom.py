from __future__ import annotations

import re
import unicodedata

from lxml import etree

# EDGAR wraps the HTML of inline XBRL documents in <XBRL> plus an XML prolog;
# parsing from the <html> tag keeps the document's <head> out of <body>.
_HTML_START = re.compile(rb"<html[\s>]", re.IGNORECASE)
_NON_CONTENT_TAGS = frozenset({"head", "script", "style", "noscript", "template"})
# Inline XBRL metadata: hidden facts and the header holding them are never shown.
_INLINE_XBRL_METADATA = frozenset({"ix:header", "ix:hidden"})
_INLINE_XBRL_NAMESPACE = "http://www.xbrl.org/2013/inlinexbrl"
_HIDDEN_STYLE = re.compile(
    r"(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*hidden)\b"
)
# Zero-width characters that carry no content; ZWJ/ZWNJ are kept because they
# change how some scripts and emoji render.
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u2060\ufeff"))

HEADING_TAGS = {f"h{level}": level for level in range(1, 7)}
# Block-level elements: text on either side of one is not joined into a word.
BLOCK_LEVEL_TAGS = frozenset(
    {
        *HEADING_TAGS, "p", "blockquote", "div", "li", "pre", "td", "th", "body",
        "table", "tr", "ul", "ol", "dl", "dt", "dd", "caption", "center",
        "section", "article", "header", "footer", "address", "figcaption",
    }
)  # fmt: skip


def normalize_text(raw: str) -> str:
    """Normalize visible text: NFC, NBSP to space, drop zero-width, collapse spaces.

    Case, punctuation, currency signs and other characters are kept as is.
    """

    text = unicodedata.normalize("NFC", raw).replace(" ", " ")
    return " ".join(text.translate(_ZERO_WIDTH).split())


def parse_html(html_bytes: bytes) -> etree._Element:
    """Parse document HTML bytes in HTML (recovering) mode without network access.

    Parsing starts at the `<html>` tag when there is one, skipping EDGAR's
    `<XBRL>` wrapper and XML prolog. Comments and processing instructions are
    dropped. Raises ValueError when no document can be built.
    """

    match = _HTML_START.search(html_bytes)
    parser = etree.HTMLParser(
        recover=True, no_network=True, remove_comments=True, remove_pis=True
    )
    try:
        root = etree.fromstring(html_bytes[match.start() if match else 0 :], parser)
    except etree.LxmlError as error:
        raise ValueError(f"HTML could not be parsed: {error}") from error
    if root is None:
        raise ValueError("HTML document is empty")
    return root


def element_tag(element: etree._Element) -> str:
    """Return the lowercase tag, as `prefix:name` for Inline XBRL in either form."""

    tag = str(element.tag).lower()
    if tag.startswith("{"):
        namespace, _, local_name = tag[1:].partition("}")
        return f"ix:{local_name}" if namespace == _INLINE_XBRL_NAMESPACE else local_name
    return tag


def is_hidden(element: etree._Element, tag: str) -> bool:
    """Whether an element and its subtree are never rendered as visible text."""

    if tag in _NON_CONTENT_TAGS or tag in _INLINE_XBRL_METADATA:
        return True
    if element.get("hidden") is not None:
        return True
    if (element.get("aria-hidden") or "").strip().lower() == "true":
        return True
    return bool(_HIDDEN_STYLE.search((element.get("style") or "").lower()))


def is_link(element: etree._Element, tag: str) -> bool:
    """Whether an element is a hyperlink (`<a href>`)."""

    return tag == "a" and element.get("href") is not None


def is_owned_by(element: etree._Element, table: etree._Element) -> bool:
    """Whether `table` is the element's nearest table and nothing between is hidden."""

    if next(element.iterancestors("table"), None) is not table:
        return False
    node: etree._Element | None = element
    while node is not None and node is not table:
        if is_hidden(node, element_tag(node)):
            return False
        node = node.getparent()
    return True


def has_visible_cell(table: etree._Element) -> bool:
    """Whether a table owns a visible `<td>`/`<th>`, so it is represented as a table."""

    return any(is_owned_by(cell, table) for cell in table.iter("td", "th"))


def visible_text(element: etree._Element) -> tuple[str, str]:
    """Return the normalized visible text of an element and of its links.

    Hidden subtrees and nested tables with a visible cell contribute nothing
    (such a table is represented on its own); the text of a nested table
    without cells stays with the element. Block-level children and `<br>`
    separate words.
    """

    pieces: list[str] = []
    link_pieces: list[str] = []
    link_depth = 0
    walker = etree.iterwalk(element, events=("start", "end"))
    for event, node in walker:
        tag = element_tag(node)
        skipped = node is not element and (
            is_hidden(node, tag) or (tag == "table" and has_visible_cell(node))
        )
        if event == "start":
            if skipped:
                walker.skip_subtree()
                continue
            link_depth += is_link(node, tag)
            text = " " if tag in BLOCK_LEVEL_TAGS or tag == "br" else ""
            text += node.text or ""
        else:
            if node is element:
                continue
            if not skipped:
                link_depth -= is_link(node, tag)
            text = (" " if tag in BLOCK_LEVEL_TAGS else "") + (node.tail or "")
        pieces.append(text)
        if link_depth:
            link_pieces.append(text)
    return normalize_text("".join(pieces)), normalize_text("".join(link_pieces))

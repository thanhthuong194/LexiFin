from __future__ import annotations

import io
from collections.abc import Iterable
from dataclasses import dataclass

from lxml import etree

# TODO: evaluate an XBRL processor (e.g. Arelle) before building xbrl_facts,
# taxonomies, linkbases or calculation checks; this module only reads what
# instances, contexts and units need, with the namespace-aware lxml parser.
INLINE_XBRL = "http://www.xbrl.org/2013/inlineXBRL"
XBRLI = "http://www.xbrl.org/2003/instance"
LINK = "http://www.xbrl.org/2003/linkbase"
XLINK = "http://www.w3.org/1999/xlink"
_INLINE_FACTS = ("nonNumeric", "nonFraction", "fraction")
_NUMERIC_INLINE_FACTS = frozenset({"nonFraction", "fraction"})
# EDGAR wraps Inline XBRL documents in <XBRL> and XML documents in <XML>; the
# envelope is SGML around the XML document, not part of it.
_EDGAR_ENVELOPES = (b"XBRL", b"XML")


class XbrlConflict(ValueError):
    """Two representations or definitions of the same XBRL data disagree."""


@dataclass(frozen=True)
class ParsedSource:
    """A parsed XBRL source and every namespace declaration it makes, in order."""

    root: etree._Element
    namespaces: tuple[tuple[str, str], ...]

    @property
    def is_inline(self) -> bool:
        """Whether the document declares the Inline XBRL 1.1 namespace."""

        return any(uri == INLINE_XBRL for _, uri in self.namespaces)

    @property
    def is_instance(self) -> bool:
        """Whether the root element is an XBRL instance (`xbrli:xbrl`)."""

        return bool(self.root.tag == f"{{{XBRLI}}}xbrl")


@dataclass(frozen=True)
class FactReference:
    """One fact occurrence: concept (Clark notation), refs and Inline target."""

    concept: str
    context_ref: str
    unit_ref: str | None
    target_name: str | None


def strip_edgar_envelope(content: bytes) -> bytes:
    """Return the XML document inside EDGAR's `<XBRL>` or `<XML>` envelope."""

    stripped = content.strip()
    for tag in _EDGAR_ENVELOPES:
        start, end = b"<" + tag + b">", b"</" + tag + b">"
        if (
            stripped[: len(start)].upper() == start
            and stripped[-len(end) :].upper() == end
        ):
            return stripped[len(start) : -len(end)].strip()
    return stripped


def parse_xbrl_source(content: bytes) -> ParsedSource:
    """Parse document content as namespace-aware XML, safely.

    DTDs, external entities and network access are disabled; large documents
    are allowed. Raises ValueError when the content is not well-formed XML.
    """

    events = etree.iterparse(
        io.BytesIO(strip_edgar_envelope(content)),
        events=("start-ns",),
        resolve_entities=False,
        load_dtd=False,
        no_network=True,
        huge_tree=True,
    )
    try:
        namespaces = tuple((prefix or "", uri) for _, (prefix, uri) in events)
    except etree.XMLSyntaxError as error:
        raise ValueError(f"not well-formed XML: {error}") from error
    return ParsedSource(events.root, namespaces)


def expand_qname(value: str | None, element: etree._Element) -> str:
    """Expand a QName value to `{namespace}local` with the namespaces in scope at `element`."""

    prefix, _, local = (value or "").strip().rpartition(":")
    uri = element.nsmap.get(prefix or None)
    if not local or uri is None:
        raise ValueError(f"QName {value!r} has no namespace in scope")
    return f"{{{uri}}}{local}"


def namespace_map(declarations: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Merge prefix declarations into one map ordered by prefix (default is "").

    Declarations come in document order, and the first binding of a prefix
    wins: a nested element may rebind it for its own subtree (an
    `<ix:resources xmlns="http://www.xbrl.org/2003/instance">` inside XHTML),
    which QNames already resolve in scope, so the map keeps the outer one.
    """

    mapping: dict[str, str] = {}
    for prefix, uri in declarations:
        mapping.setdefault(prefix, uri)
    return dict(sorted(mapping.items()))


def schema_refs(elements: Iterable[etree._Element]) -> list[str]:
    """Sorted unique `xlink:href` values of the `link:schemaRef`s in `elements`."""

    return sorted(
        {
            (ref.get(f"{{{XLINK}}}href") or "").strip()
            for element in elements
            for ref in element.iter(f"{{{LINK}}}schemaRef")
        }
        - {""}
    )


def _path(element: etree._Element) -> str:
    return str(element.getroottree().getpath(element))


def inline_facts(root: etree._Element) -> list[FactReference]:
    """Every Inline XBRL fact occurrence of a document, in document order.

    `ix:nonNumeric`, `ix:nonFraction` and `ix:fraction` are facts; continuations,
    footnotes, headers, references and resources are not. Raises ValueError
    for an `ix:tuple` (unsupported), a fact without contextRef or with an
    unresolvable name, and a numeric fact without unitRef.
    """

    tuple_element = next(root.iter(f"{{{INLINE_XBRL}}}tuple"), None)
    if tuple_element is not None:
        raise ValueError(f"unsupported_inline_tuple at {_path(tuple_element)}")
    facts = []
    for fact in root.iter(*(f"{{{INLINE_XBRL}}}{name}" for name in _INLINE_FACTS)):
        context_ref, unit_ref = fact.get("contextRef"), fact.get("unitRef")
        if not context_ref:
            raise ValueError(f"Inline fact {_path(fact)} has no contextRef")
        if etree.QName(fact).localname in _NUMERIC_INLINE_FACTS and not unit_ref:
            raise ValueError(f"numeric Inline fact {_path(fact)} has no unitRef")
        facts.append(
            FactReference(
                expand_qname(fact.get("name"), fact),
                context_ref,
                unit_ref or None,
                fact.get("target"),
            )
        )
    return facts


def xml_facts(root: etree._Element) -> list[FactReference]:
    """Every fact of a standalone instance: each element with a contextRef."""

    return [
        FactReference(
            element.tag, element.get("contextRef"), element.get("unitRef"), None
        )
        for element in root.iter()
        if isinstance(element.tag, str) and element.get("contextRef") is not None
    ]

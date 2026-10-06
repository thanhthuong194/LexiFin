from __future__ import annotations

import posixpath
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import unquote, urlsplit

from lxml import etree

from src.data_pipeline.contracts.sec_filings.document_links import (
    DocumentLinkRecord,
    LinkKind,
    ResolutionStatus,
    link_id_for,
)
from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.transforms.sec_filings.html_dom import (
    element_tag,
    is_hidden,
    normalize_text,
    visible_text,
)

_SCHEME_KINDS: dict[str, LinkKind] = {
    "mailto": "email",
    "tel": "telephone",
    "javascript": "javascript",
    "data": "data",
}
# Kinds that name a document of the filing; the others have no target document.
_DOCUMENT_KINDS = frozenset({"fragment", "relative_document", "sec_url"})
_SEC_ARCHIVES_PATH = "/Archives/"
# EDGAR serves a filing's documents from
# /Archives/edgar/data/<cik>/<accession number without dashes>/<filename>.
_ARCHIVES_FOLDER = re.compile(r"/Archives/edgar/data/\d+/(\d{18})/", re.IGNORECASE)


@dataclass(frozen=True)
class HtmlLink:
    """One `<a href>` element of a parsed document, before resolution."""

    href: str
    link_text: str
    title: str | None
    source_xpath: str


@dataclass(frozen=True)
class LinkResolution:
    """A link's kind and the document of the source's filing it targets."""

    link_kind: LinkKind
    resolution_status: ResolutionStatus
    target_document_id: str | None = None
    target_fragment: str | None = None


def extract_links(root: etree._Element) -> list[HtmlLink]:
    """Return every `<a href>` element of a parsed HTML document in DOM order.

    Anchors without `href` are named anchors and are left out; an empty `href`
    is kept. `link_text` is the anchor's normalized visible text, empty when
    the anchor or an ancestor is hidden; `title` is the normalized title
    attribute, or None when it is missing or blank.
    """

    tree = root.getroottree()
    links = []
    for anchor in root.iter("a"):
        href = anchor.get("href")
        if href is None:
            continue
        hidden = any(
            is_hidden(node, element_tag(node))
            for node in (anchor, *anchor.iterancestors())
        )
        links.append(
            HtmlLink(
                href=href,
                link_text="" if hidden else visible_text(anchor)[0],
                title=normalize_text(anchor.get("title") or "") or None,
                source_xpath=tree.getpath(anchor),
            )
        )
    return links


def normalize_href(href: str) -> str:
    """Return `href` in Unicode NFC without surrounding whitespace, else as is."""

    return unicodedata.normalize("NFC", href).strip()


def classify_href(normalized_href: str) -> LinkKind:
    """Classify a normalized href by its syntax alone (no network access).

    Precedence: a leading `#` is a fragment; a reference without scheme or
    host is an SEC URL when its path starts with `/Archives/`, else a relative
    document; an http(s) or protocol-relative URL is an SEC URL when its host
    is sec.gov or a subdomain, else external; then mailto, tel, javascript and
    data. Anything else, including an href that is not a valid URL, is other.
    """

    if normalized_href.startswith("#"):
        return "fragment"
    try:
        url = urlsplit(normalized_href)
    except ValueError:  # e.g. an unbalanced IPv6 bracket in the host
        return "other"
    scheme = url.scheme.lower()
    if not scheme and not url.netloc:
        if url.path.startswith(_SEC_ARCHIVES_PATH):
            return "sec_url"
        return "relative_document"
    if scheme in ("", "http", "https"):
        host = url.hostname or ""
        is_sec = host == "sec.gov" or host.endswith(".sec.gov")
        return "sec_url" if is_sec else "external_url"
    return _SCHEME_KINDS.get(scheme, "other")


def resolve_link(
    normalized_href: str,
    source: DocumentRecord,
    filenames: Mapping[str, Sequence[str]],
) -> LinkResolution:
    """Classify a link and resolve it to a document of the source's filing.

    `filenames` maps the filenames of the source's filing to their document
    IDs. A fragment, or a relative reference with an empty path (`""`,
    `?query`), targets the source itself. Relative and SEC links match the
    percent-decoded basename of their path exactly: one match resolves, none
    is unresolved, several are ambiguous. An SEC URL under another filing's
    archive folder stays unresolved even when its basename matches. Other
    links have no target document.
    """

    kind = classify_href(normalized_href)
    if kind not in _DOCUMENT_KINDS:
        return LinkResolution(
            kind, "unresolved" if kind == "other" else "not_applicable"
        )
    url = urlsplit(normalized_href)
    fragment = url.fragment or None
    if kind == "fragment" or (kind == "relative_document" and not url.path):
        return LinkResolution(kind, "resolved", source.document_id, fragment)
    folder = _ARCHIVES_FOLDER.match(url.path)
    # filing_id is "sec:" + the dashed accession number.
    accession = source.filing_id.removeprefix("sec:").replace("-", "")
    if kind == "sec_url" and folder is not None and folder.group(1) != accession:
        return LinkResolution(kind, "unresolved", None, fragment)
    candidates = filenames.get(posixpath.basename(unquote(url.path)), ())
    if len(candidates) == 1:
        return LinkResolution(kind, "resolved", candidates[0], fragment)
    status: ResolutionStatus = "ambiguous" if candidates else "unresolved"
    return LinkResolution(kind, status, None, fragment)


def index_filenames(
    documents: Iterable[DocumentRecord],
) -> dict[str, dict[str, list[str]]]:
    """Map each filing ID to its documents' filenames and their document IDs."""

    index: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for document in documents:
        if document.filename is not None:
            index[document.filing_id][document.filename].append(document.document_id)
    return {filing_id: dict(names) for filing_id, names in index.items()}


def build_document_links(
    source: DocumentRecord,
    links: Sequence[HtmlLink],
    filenames: Mapping[str, Sequence[str]],
    processed_at: datetime,
) -> list[DocumentLinkRecord]:
    """Build the link records of one source document, ordinals in `links` order.

    `filenames` maps the filenames of the source's filing to document IDs.
    """

    records = []
    for ordinal, link in enumerate(links, start=1):
        normalized = normalize_href(link.href)
        resolution = resolve_link(normalized, source, filenames)
        records.append(
            DocumentLinkRecord.model_validate(
                {
                    "link_id": link_id_for(source.document_id, ordinal),
                    "filing_id": source.filing_id,
                    "source_document_id": source.document_id,
                    "target_document_id": resolution.target_document_id,
                    "filing_date": source.filing_date,
                    "link_ordinal": ordinal,
                    "document_link_count": len(links),
                    "href": link.href,
                    "normalized_href": normalized,
                    "link_kind": resolution.link_kind,
                    "resolution_status": resolution.resolution_status,
                    "target_fragment": resolution.target_fragment,
                    "link_text": link.link_text,
                    "title": link.title,
                    "source_xpath": link.source_xpath,
                    "source_document_sha256": source.content_sha256,
                    "processed_at": processed_at,
                }
            )
        )
    return records

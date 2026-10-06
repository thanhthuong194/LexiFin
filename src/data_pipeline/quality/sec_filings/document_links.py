from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.data_pipeline.contracts.sec_filings.document_links import DocumentLinkRecord
from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.quality.sec_filings.narrative import declared_count_problem


def _target_problem(
    source: DocumentRecord,
    links: Sequence[DocumentLinkRecord],
    documents: Mapping[str, DocumentRecord],
) -> str | None:
    """Every resolved target must be a stored document of the source's filing."""

    for link in links:
        if link.target_document_id is None:
            continue
        target = documents.get(link.target_document_id)
        if target is None:
            return f"link {link.link_ordinal} targets a missing document"
        if target.filing_id != source.filing_id:
            return f"link {link.link_ordinal} targets a document of another filing"
    return None


def links_problem(
    source: DocumentRecord,
    links: Sequence[DocumentLinkRecord],
    documents: Mapping[str, DocumentRecord],
) -> str | None:
    """Return why a source document's links are not one complete group, or None.

    Checks lineage and source hash against the source document, declared
    count against ordinals 1..N (so IDs are unique), unique source XPaths, and
    that every target is a document of the source's filing. `documents` maps
    every document ID to its row. Row invariants are enforced by
    DocumentLinkRecord.
    """

    lineage = (
        source.document_id,
        source.filing_id,
        source.filing_date,
        source.content_sha256,
    )
    for link in links:
        if (
            link.source_document_id,
            link.filing_id,
            link.filing_date,
            link.source_document_sha256,
        ) != lineage:
            return f"link {link.link_ordinal} does not carry its source's lineage"
    problem = declared_count_problem(
        [link.link_ordinal for link in links],
        {link.document_link_count for link in links},
        "links",
    )
    if problem is None and len({link.source_xpath for link in links}) != len(links):
        problem = "links share a source_xpath"
    return problem or _target_problem(source, links, documents)

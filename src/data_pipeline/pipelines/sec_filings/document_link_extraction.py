from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter_ns

from pydantic import ValidationError

from src.data_pipeline.contracts.sec_filings.document_links import DocumentLinkRecord
from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.storage.sec_filings.documents import read_document_content
from src.data_pipeline.transforms.sec_filings.document_links import (
    HtmlLink,
    build_document_links,
    extract_links,
    index_filenames,
)
from src.data_pipeline.transforms.sec_filings.html_dom import parse_html
from src.utils.logger import get_logger
from src.utils.timer import timed_operation

logger = get_logger(__name__)


@dataclass
class LinkExtraction:
    """Link records built from Bronze, keyed by source document ID.

    `failures` counts sources that could not be read, parsed or turned into
    valid records; `without_links` those parsed without any `<a href>`.
    """

    built: dict[str, list[DocumentLinkRecord]] = field(default_factory=dict)
    failures: int = 0
    without_links: int = 0


def source_context(source: DocumentRecord) -> dict[str, str]:
    """Identify a source document in logs."""

    return {
        "filing_id": source.filing_id,
        "source_document_id": source.document_id,
        "source_path": source.source_path,
    }


def _extract(
    candidates: Sequence[DocumentRecord], bronze: SecBronze, result: LinkExtraction
) -> dict[str, list[HtmlLink]]:
    """Read, verify and parse each candidate once, keeping only its links.

    Phase durations are summed over documents and logged once, so the loop
    does not log per document.
    """

    extracted: dict[str, list[HtmlLink]] = {}
    elapsed_ns: Counter[str] = Counter()
    verified_bytes = 0
    for source in candidates:
        try:
            started = perf_counter_ns()
            content = read_document_content(source, bronze)
            verified_bytes += len(content)
            read_at = perf_counter_ns()
            root = parse_html(content)
            parsed_at = perf_counter_ns()
            extracted[source.document_id] = extract_links(root)
        except (OSError, ValueError) as error:
            result.failures += 1
            logger.error(
                "document_links_source_failed",
                error=str(error),
                **source_context(source),
            )
            continue
        elapsed_ns.update(
            read=read_at - started,
            parse_html=parsed_at - read_at,
            extract=perf_counter_ns() - parsed_at,
        )
    logger.info(
        "document_link_content_verified",
        documents=len(extracted),
        bytes=verified_bytes,
    )
    logger.info(
        "document_links_extracted",
        documents=len(extracted),
        links=sum(len(links) for links in extracted.values()),
        **{f"{phase}_ms": round(ns / 1_000_000, 3) for phase, ns in elapsed_ns.items()},
    )
    return extracted


def _log_unresolved(records: Sequence[DocumentLinkRecord]) -> None:
    """Log each ambiguous link and, per document, the links left unresolved."""

    unresolved = [r for r in records if r.resolution_status == "unresolved"]
    if unresolved:
        logger.info(
            "document_link_unresolved",
            filing_id=unresolved[0].filing_id,
            source_document_id=unresolved[0].source_document_id,
            links=len(unresolved),
            link_kinds=dict(Counter(r.link_kind for r in unresolved)),
            first_link_id=unresolved[0].link_id,
        )
    for record in records:
        if record.resolution_status == "ambiguous":
            logger.warning(
                "document_link_ambiguous",
                filing_id=record.filing_id,
                source_document_id=record.source_document_id,
                link_id=record.link_id,
                link_ordinal=record.link_ordinal,
                link_kind=record.link_kind,
                normalized_href=record.normalized_href,
            )


def _build(
    candidates: Sequence[DocumentRecord],
    extracted: Mapping[str, list[HtmlLink]],
    documents: Mapping[str, DocumentRecord],
    processed_at: datetime,
    result: LinkExtraction,
) -> None:
    """Resolve every extracted group into records; documents without links get none."""

    filenames = index_filenames(documents.values())
    for source in candidates:
        links = extracted.get(source.document_id)
        if links is None:
            continue
        if not links:
            result.without_links += 1
            continue
        try:
            records = build_document_links(
                source, links, filenames.get(source.filing_id, {}), processed_at
            )
        except ValidationError as error:
            result.failures += 1
            logger.error(
                "document_links_invalid",
                error=f"invalid link: {error.errors()[0]['msg']}",
                **source_context(source),
            )
            continue
        result.built[source.document_id] = records
        _log_unresolved(records)
    logger.info("document_without_links", documents=result.without_links)


def extract_document_links(
    candidates: Sequence[DocumentRecord],
    documents: Mapping[str, DocumentRecord],
    bronze: SecBronze,
    processed_at: datetime,
) -> LinkExtraction:
    """Build the link records of every candidate HTML document.

    Each source is read by offsets, verified and parsed once
    (`document_links.extract`); its links are then resolved against the
    documents of its filing (`document_links.resolve`). `documents` maps every
    document ID to its row.
    """

    result = LinkExtraction()
    with timed_operation("document_links.extract", documents=len(candidates)):
        extracted = _extract(candidates, bronze, result)
    with timed_operation("document_links.resolve", documents=len(extracted)):
        _build(candidates, extracted, documents, processed_at, result)
    return result

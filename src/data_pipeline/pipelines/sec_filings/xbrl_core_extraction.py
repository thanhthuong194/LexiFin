from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter_ns

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.filings import FilingRecord
from src.data_pipeline.contracts.sec_filings.xbrl_core import (
    XbrlContextRecord,
    XbrlInstanceRecord,
    XbrlUnitRecord,
)
from src.data_pipeline.storage.sec_filings.documents import read_document_content
from src.data_pipeline.transforms.sec_filings.xbrl_core import (
    LogicalInstance,
    SourceDocument,
    build_logical_instances,
    context_records,
    instance_record,
    unit_records,
)
from src.data_pipeline.transforms.sec_filings.xbrl_sources import (
    INLINE_XBRL,
    XbrlConflict,
    parse_xbrl_source,
)
from src.utils.logger import get_logger
from src.utils.timer import timed_operation

logger = get_logger(__name__)


@dataclass
class XbrlExtraction:
    """Logical instances discovered per filing, and the filings that failed.

    `conflicts` are filings whose representations or definitions disagree;
    `failures` unreadable or invalid sources; `without_xbrl` filings parsed
    without any instance (they are parsed again next run).
    """

    discovered: dict[str, list[LogicalInstance]] = field(default_factory=dict)
    conflicts: int = 0
    failures: int = 0
    without_xbrl: int = 0


@dataclass(frozen=True)
class BuiltFiling:
    """Expected rows of one filing: instances, and contexts and units by instance."""

    instances: list[XbrlInstanceRecord]
    contexts: dict[str, list[XbrlContextRecord]]
    units: dict[str, list[XbrlUnitRecord]]


def _sources(
    filing: FilingRecord, documents: Sequence[DocumentRecord], data_dir: Path
) -> tuple[list[SourceDocument], list[SourceDocument], int]:
    """Read, verify and parse a filing's Inline and XML instance candidates.

    An HTML document is a candidate only if its bytes name the Inline XBRL
    namespace, and is Inline only if it declares it; XML instances are the
    documents classified `xbrl_instance`. Returns them and the bytes read.
    """

    inline, xml, size = [], [], 0
    for document in documents:
        if document.content_sha256 is None:
            continue
        is_html = document.content_kind == "html"
        if not is_html and document.role != "xbrl_instance":
            continue
        content = read_document_content(document, data_dir)
        size += len(content)
        if is_html and INLINE_XBRL.encode() not in content:
            continue
        try:
            parsed = parse_xbrl_source(content)
        except ValueError as error:
            raise ValueError(f"{document.document_id}: {error}") from error
        if not is_html:
            xml.append(SourceDocument(document, parsed))
        elif parsed.is_inline:
            inline.append(SourceDocument(document, parsed))
    logger.info(
        "xbrl_candidate_documents_loaded",
        filing_id=filing.filing_id,
        inline_documents=[s.document.document_id for s in inline],
        xml_documents=[s.document.document_id for s in xml],
    )
    return inline, xml, size


def _log_discovered(instances: Sequence[LogicalInstance]) -> None:
    for instance in instances:
        logger.info(
            "xbrl_instance_discovered",
            filing_id=instance.filing_id,
            instance_id=instance.instance_id,
            target_name=instance.target_name,
            is_inline=instance.is_inline,
            facts=len(instance.facts),
        )
        merged = [d.document_id for d in instance.sources if d.content_kind != "html"]
        if instance.is_inline and merged:
            logger.info(
                "xbrl_sources_merged",
                filing_id=instance.filing_id,
                instance_id=instance.instance_id,
                document_ids=merged,
            )


def extract_filings(
    filings: Sequence[FilingRecord],
    documents: Mapping[str, list[DocumentRecord]],
    data_dir: Path,
) -> XbrlExtraction:
    """Discover each filing's logical instances, parsing every source once.

    Reading, checksum and parsing time is summed over filings and logged once.
    """

    extraction = XbrlExtraction()
    read_ns = verified_bytes = 0
    for filing in filings:
        context = {"filing_id": filing.filing_id}
        try:
            started = perf_counter_ns()
            inline, xml, size = _sources(filing, documents[filing.filing_id], data_dir)
            read_ns += perf_counter_ns() - started
            verified_bytes += size
            instances = build_logical_instances(filing.filing_id, inline, xml)
        except XbrlConflict as error:
            extraction.conflicts += 1
            logger.error("xbrl_instance_conflict", reason=str(error), **context)
            continue
        except (OSError, ValueError) as error:
            extraction.failures += 1
            logger.error("xbrl_core_filing_failed", error=str(error), **context)
            continue
        if not instances:
            extraction.without_xbrl += 1
            logger.info("xbrl_no_instance_found", **context)
        _log_discovered(instances)
        extraction.discovered[filing.filing_id] = instances
    logger.info(
        "xbrl_document_content_verified",
        filings=len(filings),
        bytes=verified_bytes,
        read_and_parse_ms=round(read_ns / 1_000_000, 3),
    )
    return extraction


def build_rows(
    discovered: Mapping[str, list[LogicalInstance]], processed_at: datetime
) -> dict[str, BuiltFiling]:
    built = {filing_id: BuiltFiling([], {}, {}) for filing_id in discovered}
    with timed_operation("xbrl_core.build_instances"):
        for filing_id, instances in discovered.items():
            built[filing_id].instances.extend(
                instance_record(i, processed_at) for i in instances
            )
    with timed_operation("xbrl_core.build_contexts"):
        for filing_id, instances in discovered.items():
            for i in instances:
                built[filing_id].contexts[i.instance_id] = context_records(
                    i, processed_at
                )
    with timed_operation("xbrl_core.build_units"):
        for filing_id, instances in discovered.items():
            for i in instances:
                built[filing_id].units[i.instance_id] = unit_records(i, processed_at)
    return built

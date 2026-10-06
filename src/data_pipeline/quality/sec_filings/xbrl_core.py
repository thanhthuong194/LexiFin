from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from lxml import etree

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.xbrl_core import (
    XbrlContextRecord,
    XbrlInstanceRecord,
    XbrlUnitRecord,
    source_set_sha256,
)


def instance_problem(
    instance: XbrlInstanceRecord, documents: Mapping[str, DocumentRecord]
) -> str | None:
    """Return why an instance does not fit the current document inventory, or None.

    Its sources must be documents of its filing, with content, in document
    order; `source_set_sha256` must match their current content hashes.
    `documents` maps every document ID to its row.
    """

    sources = [
        documents.get(document_id) for document_id in instance.source_document_ids
    ]
    if None in sources:
        return "a source document is not in sec_filings.documents"
    present = [source for source in sources if source is not None]
    if any(source.filing_id != instance.filing_id for source in present):
        return "a source document belongs to another filing"
    order = [(source.document_ordinal, source.document_id) for source in present]
    if order != sorted(order):
        return "source documents are not in document order"
    hashes = [(source.document_id, source.content_sha256 or "") for source in present]
    current = source_set_sha256(hashes)
    if current != instance.source_set_sha256:
        return f"source_set_sha256 is {instance.source_set_sha256}, documents give {current}"
    return None


def _unique(refs: Sequence[str], kind: str) -> str | None:
    duplicates = sorted(ref for ref, count in Counter(refs).items() if count > 1)
    return f"{kind} refs repeat: {duplicates[:5]}" if duplicates else None


def contexts_problem(
    instance: XbrlInstanceRecord, contexts: Sequence[XbrlContextRecord]
) -> str | None:
    """A complete context group: `context_count` unique refs of this instance.

    Segment and scenario XML must parse; keys, periods and hashes are
    enforced by XbrlContextRecord.
    """

    if len(contexts) != instance.context_count:
        return f"{len(contexts)} contexts, but the instance declares {instance.context_count}"
    if any(context.instance_id != instance.instance_id for context in contexts):
        return "contexts belong to another instance"
    for context in contexts:
        for xml in (context.segment_xml, context.scenario_xml):
            try:
                if xml is not None:
                    etree.fromstring(xml.encode("utf-8"))
            except etree.XMLSyntaxError as error:
                return f"context {context.context_ref!r} XML does not parse: {error}"
    return _unique([context.context_ref for context in contexts], "context")


def units_problem(
    instance: XbrlInstanceRecord, units: Sequence[XbrlUnitRecord]
) -> str | None:
    """A complete unit group: `unit_count` unique refs of this instance."""

    if len(units) != instance.unit_count:
        return f"{len(units)} units, but the instance declares {instance.unit_count}"
    if any(unit.instance_id != instance.instance_id for unit in units):
        return "units belong to another instance"
    return _unique([unit.unit_ref for unit in units], "unit")

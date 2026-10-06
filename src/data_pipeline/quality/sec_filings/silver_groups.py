from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from operator import attrgetter
from pathlib import Path
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, ValidationError

from src.data_pipeline.contracts.sec_filings.content_blocks import ContentBlockRecord
from src.data_pipeline.contracts.sec_filings.document_links import DocumentLinkRecord
from src.data_pipeline.contracts.sec_filings.document_table_cells import (
    DocumentTableCellRecord,
)
from src.data_pipeline.contracts.sec_filings.document_tables import DocumentTableRecord
from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.filing_entities import FilingEntityRecord
from src.data_pipeline.contracts.sec_filings.filings import FilingRecord
from src.data_pipeline.contracts.sec_filings.sections import SectionRecord
from src.data_pipeline.quality.sec_filings.document_links import links_problem
from src.data_pipeline.quality.sec_filings.document_tables import (
    cells_problem,
    tables_problem,
)
from src.data_pipeline.quality.sec_filings.documents import documents_problem
from src.data_pipeline.quality.sec_filings.narrative import (
    declared_count_problem,
    narrative_problem,
    sections_problem,
)
from src.data_pipeline.quality.sec_filings.silver_report import SilverAuditReport
from src.data_pipeline.storage.bronze import resolve_relative_path
from src.data_pipeline.storage.sec_filings.content_blocks import CONTENT_BLOCKS_TABLE
from src.data_pipeline.storage.sec_filings.document_links import DOCUMENT_LINKS_TABLE
from src.data_pipeline.storage.sec_filings.document_table_cells import (
    DOCUMENT_TABLE_CELLS_TABLE,
)
from src.data_pipeline.storage.sec_filings.document_tables import DOCUMENT_TABLES_TABLE
from src.data_pipeline.storage.sec_filings.documents import DOCUMENTS_TABLE
from src.data_pipeline.storage.sec_filings.filing_entities import FILING_ENTITIES_TABLE
from src.data_pipeline.storage.sec_filings.filings import FILINGS_TABLE
from src.data_pipeline.storage.sec_filings.sections import SECTIONS_TABLE

_document_id = attrgetter("document_id")


@dataclass(frozen=True)
class ParentRecords:
    """Validated filings, documents by ID and entities by filing ID."""

    filings: list[FilingRecord]
    documents: dict[str, DocumentRecord]
    entities: dict[str, list[FilingEntityRecord]]


def _records[Record: BaseModel](
    report: SilverAuditReport,
    table: str,
    rows: Iterable[dict[str, Any]],
    model: type[Record],
    key: str,
) -> list[Record]:
    """Validate rows against their contract; report every invalid row."""

    records = []
    for row in rows:
        try:
            records.append(model.model_validate(row))
        except ValidationError as error:
            first = error.errors()[0]
            location = ".".join(str(part) for part in first["loc"]) or "row"
            report.add(
                "error",
                "row_contract",
                table,
                f"{location}: {first['msg']}",
                row.get(key),
            )
    return records


def _load[Record: BaseModel](
    report: SilverAuditReport,
    data: Mapping[str, pa.Table],
    table: str,
    model: type[Record],
    key: str,
) -> list[Record]:
    return _records(report, table, data[table].to_pylist(), model, key)


def _by[Record: BaseModel](
    records: Iterable[Record], key: Callable[[Record], str]
) -> dict[str, list[Record]]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    return dict(groups)


def _sorted_groups(
    table: pa.Table, key: str
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Yield each key's rows in key order, converting one group to Python at a time."""

    if table.num_rows == 0:
        return
    ordered = table.sort_by(key)
    start = 0
    for value, run in groupby(ordered.column(key).to_pylist()):
        length = sum(1 for _ in run)
        yield value, ordered.slice(start, length).to_pylist()
        start += length


def _check_filings(
    parents: ParentRecords, data_dir: Path, report: SilverAuditReport
) -> None:
    """Paths, document inventory, primary form and entity group of every filing."""

    report.ran("filings.groups")
    by_filing = _by(parents.documents.values(), attrgetter("filing_id"))
    for filing in parents.filings:
        for path in (filing.full_submission_path, filing.primary_document_path):
            try:
                if path is not None:
                    resolve_relative_path(path, data_dir)
            except ValueError as error:
                report.add(
                    "error", "source_path", FILINGS_TABLE, str(error), filing.filing_id
                )
        documents = by_filing.get(filing.filing_id, [])
        problem = documents_problem(filing, documents) if documents else "no documents"
        if problem is None:
            primary = next(d for d in documents if d.role == "primary_document")
            if primary.document_type != filing.form:
                problem = f"primary document type {primary.document_type!r} is not {filing.form!r}"
        if problem is not None:
            report.group_violation(DOCUMENTS_TABLE, filing.filing_id, problem)
        declared = filing.public_document_count
        if declared is not None and declared != len(documents):
            report.add(
                "warning",
                "public_document_count",
                DOCUMENTS_TABLE,
                f"{len(documents)} documents; header PUBLIC DOCUMENT COUNT is {declared}",
                filing.filing_id,
            )
        entities = parents.entities.get(filing.filing_id, [])
        ordinals = [entity.entity_ordinal for entity in entities]
        problem = (
            declared_count_problem(ordinals, {len(ordinals)}, "entities")
            if entities
            else "no entities"
        )
        if problem is not None:
            report.group_violation(FILING_ENTITIES_TABLE, filing.filing_id, problem)


def _narrative_warnings(
    sections: Sequence[SectionRecord], report: SilverAuditReport
) -> None:
    """Root-only documents and repeated codes under one parent are warnings."""

    document_id = sections[0].document_id
    if len(sections) == 1:
        report.add(
            "warning",
            "root_only_document",
            SECTIONS_TABLE,
            "no section detected",
            document_id,
        )
    codes = {section.section_id: section.section_code for section in sections}
    seen = Counter(
        (codes.get(section.parent_section_id or ""), section.section_code)
        for section in sections[1:]
    )
    repeated = sum(count - 1 for count in seen.values())
    if repeated:
        report.add(
            "warning",
            "duplicate_section_code",
            SECTIONS_TABLE,
            f"{repeated} sections repeat a code under the same parent",
            document_id,
        )


def _tables_problem(
    parent: DocumentRecord,
    table_blocks: Sequence[ContentBlockRecord],
    tables: Sequence[DocumentTableRecord],
) -> str | None:
    """Each table block has exactly one table and each table one table block.

    A document without table blocks and tables has nothing to check.
    """

    if not table_blocks and not tables:
        return None
    block_ids = {block.block_id for block in table_blocks}
    table_block_ids = {table.block_id for table in tables}
    if table_block_ids - block_ids:
        return f"{len(table_block_ids - block_ids)} tables belong to no table block"
    if block_ids - table_block_ids:
        return f"{len(block_ids - table_block_ids)} table blocks have no document_table"
    return tables_problem(parent, table_blocks, tables)


def _check_primary_documents(
    data: Mapping[str, pa.Table],
    documents: Mapping[str, DocumentRecord],
    report: SilverAuditReport,
) -> dict[str, tuple[list[ContentBlockRecord], list[DocumentTableRecord]]]:
    """Sections, blocks and tables of every primary HTML document.

    Returns, for documents whose tables are complete, their table blocks and
    tables, against which their cells are then checked.
    """

    report.ran("narrative.groups")
    report.ran("tables.groups")
    sections = _load(report, data, SECTIONS_TABLE, SectionRecord, "section_id")
    blocks = _load(report, data, CONTENT_BLOCKS_TABLE, ContentBlockRecord, "block_id")
    tables = _load(report, data, DOCUMENT_TABLES_TABLE, DocumentTableRecord, "table_id")
    groups: dict[str, dict[str, list[Any]]] = {
        SECTIONS_TABLE: _by(sections, _document_id),
        CONTENT_BLOCKS_TABLE: _by(blocks, _document_id),
        DOCUMENT_TABLES_TABLE: _by(tables, _document_id),
    }
    primaries = {
        d.document_id: d
        for d in documents.values()
        if d.role == "primary_document" and d.content_kind == "html"
    }
    for table, by_document in groups.items():
        for document_id in sorted(by_document.keys() - primaries.keys()):
            if document_id in documents:  # rows of unknown documents are orphans
                report.group_violation(
                    table, document_id, "document is not primary HTML"
                )

    complete: dict[str, tuple[list[ContentBlockRecord], list[DocumentTableRecord]]] = {}
    for document_id, parent in sorted(primaries.items()):
        sections = groups[SECTIONS_TABLE].get(document_id, [])
        blocks = groups[CONTENT_BLOCKS_TABLE].get(document_id, [])
        problem = sections_problem(parent, sections)
        if problem is not None:
            report.group_violation(SECTIONS_TABLE, document_id, problem)
            continue
        problem = narrative_problem(parent, sections, blocks)
        if problem is not None:
            report.group_violation(CONTENT_BLOCKS_TABLE, document_id, problem)
            continue
        _narrative_warnings(sorted(sections, key=lambda s: s.section_ordinal), report)
        table_blocks = sorted(
            (block for block in blocks if block.block_type == "table"),
            key=lambda block: block.block_ordinal,
        )
        document_tables = groups[DOCUMENT_TABLES_TABLE].get(document_id, [])
        problem = _tables_problem(parent, table_blocks, document_tables)
        if problem is not None:
            report.group_violation(DOCUMENT_TABLES_TABLE, document_id, problem)
        elif document_tables:
            complete[document_id] = (table_blocks, document_tables)
    return complete


def _check_cells(
    cells: pa.Table,
    complete: Mapping[str, tuple[list[ContentBlockRecord], list[DocumentTableRecord]]],
    report: SilverAuditReport,
) -> None:
    """Every complete table group needs exactly its cells; one document at a time."""

    report.ran("cells.groups")
    seen = set()
    for document_id, rows in _sorted_groups(cells, "document_id"):
        seen.add(document_id)
        records = _records(
            report, DOCUMENT_TABLE_CELLS_TABLE, rows, DocumentTableCellRecord, "cell_id"
        )
        if document_id in complete:
            problem = cells_problem(*complete[document_id], records)
            if problem is not None:
                report.group_violation(DOCUMENT_TABLE_CELLS_TABLE, document_id, problem)
    for document_id in sorted(complete.keys() - seen):
        report.group_violation(
            DOCUMENT_TABLE_CELLS_TABLE, document_id, "tables have no cells"
        )


def _check_links(
    links: pa.Table, documents: Mapping[str, DocumentRecord], report: SilverAuditReport
) -> None:
    """Every stored link group is complete and its targets stay in its filing."""

    report.ran("links.groups")
    for source_id, rows in _sorted_groups(links, "source_document_id"):
        records = _records(
            report, DOCUMENT_LINKS_TABLE, rows, DocumentLinkRecord, "link_id"
        )
        source = documents.get(source_id)
        if source is None:
            continue  # an orphan, reported by the reference check
        problem = (
            "source document is not HTML"
            if source.content_kind != "html"
            else links_problem(source, records, documents)
        )
        if problem is not None:
            report.group_violation(DOCUMENT_LINKS_TABLE, source_id, problem)


def check_groups(
    data: Mapping[str, pa.Table], data_dir: Path, report: SilverAuditReport
) -> ParentRecords:
    """Validate every row against its contract and every group's completeness.

    Reuses the stage checks: documents per filing, sections and blocks per
    primary HTML document, a table per table block and its cells, and links
    per source document. Returns the validated parent records.
    """

    documents = _load(report, data, DOCUMENTS_TABLE, DocumentRecord, "document_id")
    entities = _load(
        report, data, FILING_ENTITIES_TABLE, FilingEntityRecord, "filing_entity_id"
    )
    parents = ParentRecords(
        filings=_load(report, data, FILINGS_TABLE, FilingRecord, "filing_id"),
        documents={document.document_id: document for document in documents},
        entities=_by(entities, attrgetter("filing_id")),
    )
    _check_filings(parents, data_dir, report)
    complete = _check_primary_documents(data, parents.documents, report)
    _check_cells(data[DOCUMENT_TABLE_CELLS_TABLE], complete, report)
    _check_links(data[DOCUMENT_LINKS_TABLE], parents.documents, report)
    return parents

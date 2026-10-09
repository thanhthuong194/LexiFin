from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic import ValidationError
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.content_blocks import ContentBlockRecord
from src.data_pipeline.contracts.sec_filings.document_table_cells import (
    DocumentTableCellRecord,
)
from src.data_pipeline.contracts.sec_filings.document_tables import DocumentTableRecord
from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.quality.sec_filings.document_tables import (
    cells_problem,
    tables_problem,
)
from src.data_pipeline.quality.sec_filings.narrative import (
    declared_count_problem,
    stored_group_problem,
)
from src.data_pipeline.storage.iceberg import (
    GroupSummary,
    filing_dates_filter,
    read_groups_in,
)

StoredRows = Mapping[str, Sequence[dict[str, Any]]]


@dataclass
class TablePlan:
    """One batch of documents to parse, with what building them needs.

    `blocks` holds each document's `table` blocks in block order; `resumed`
    the exact stored tables of documents whose cells are appended now.
    """

    parents: Sequence[DocumentRecord]
    blocks: Mapping[str, list[ContentBlockRecord]]
    resumed: dict[str, list[DocumentTableRecord]] = field(default_factory=dict)

    @property
    def candidates(self) -> Sequence[DocumentRecord]:
        """Documents to parse, in parent order."""

        return self.parents


def document_context(parent: DocumentRecord) -> dict[str, str]:
    """Identify a document in logs."""

    return {
        "filing_id": parent.filing_id,
        "document_id": parent.document_id,
        "source_path": parent.source_path,
    }


def stored_state(
    parent: DocumentRecord,
    blocks: Sequence[ContentBlockRecord],
    table_rows: Sequence[dict[str, Any]],
    cell_rows: Sequence[dict[str, Any]] | None,
) -> tuple[list[DocumentTableRecord], str | None]:
    """Validate a document's stored rows; return its tables and any problem.

    With `cell_rows` None only the tables are judged.
    """

    try:
        tables = [DocumentTableRecord.model_validate(row) for row in table_rows]
        cells = [DocumentTableCellRecord.model_validate(row) for row in cell_rows or ()]
    except ValidationError as error:
        return [], f"stored row is invalid: {error.errors()[0]['msg']}"
    problem = tables_problem(parent, blocks, tables)
    if problem is None and cell_rows is not None:
        problem = cells_problem(blocks, tables, cells)
    return tables, problem


@dataclass(frozen=True)
class BlockHead:
    """A stored `table` block's place, without its text."""

    block_id: str
    block_ordinal: int
    source_xpath: str
    lineage: tuple[str, date, str]


@dataclass(frozen=True)
class TableHead:
    """A stored table's identity and place, without its content columns."""

    table_id: str
    table_ordinal: int
    document_table_count: int
    block_id: str
    source_xpath: str
    cell_count: int
    lineage: tuple[str, date, str]


@dataclass
class TableClassification:
    """Every primary document sorted before any parsing.

    `new` documents have table blocks but nothing stored; `resumed` ones have
    complete stored tables and no cells. Conflicts hold (context, reason).
    """

    new: list[DocumentRecord] = field(default_factory=list)
    resumed: list[DocumentRecord] = field(default_factory=list)
    without_tables: int = 0
    skipped: int = 0
    conflicts: list[tuple[dict[str, str], str]] = field(default_factory=list)


def _lineage(parent: DocumentRecord) -> tuple[str, date, str]:
    return (parent.filing_id, parent.filing_date, parent.content_sha256 or "")


def _tables_problem(
    parent: DocumentRecord, blocks: Sequence[BlockHead], tables: Sequence[TableHead]
) -> str | None:
    """Stored tables: ordinals 1..N of their count, the parent's lineage, and
    exactly one per table block, in block order."""

    ordered = sorted(tables, key=lambda table: table.table_ordinal)
    problem = declared_count_problem(
        [t.table_ordinal for t in ordered],
        {t.document_table_count for t in ordered},
        "tables",
    )
    if problem is not None:
        return problem
    if any(table.lineage != _lineage(parent) for table in ordered):
        return "tables do not carry their parent's lineage or source hash"
    places = [(t.block_id, t.source_xpath) for t in ordered]
    if places != [(b.block_id, b.source_xpath) for b in blocks]:
        return f"{len(tables)} tables do not match the {len(blocks)} table blocks"
    return None


def _cells_problem(
    parent: DocumentRecord,
    tables: Sequence[TableHead],
    cells: Mapping[str, GroupSummary],
) -> str | None:
    """Every stored table has its complete cells, carrying the parent's lineage."""

    lineage = dict(
        zip(
            ("filing_id", "filing_date", "source_document_sha256"),
            _lineage(parent),
            strict=True,
        )
    )
    lineage["document_id"] = parent.document_id
    for table in tables:
        summary = cells.get(table.table_id)
        if summary is None:
            return "only some tables have cells"
        problem = stored_group_problem(summary, lineage, "cells")
        if problem is None and summary.rows != table.cell_count:
            problem = f"{summary.rows} cells, but the table declares {table.cell_count}"
        if problem is not None:
            return f"{table.table_id}: {problem}"
    return None


def _classify_document(
    result: TableClassification,
    parent: DocumentRecord,
    blocks: Sequence[BlockHead],
    tables: Sequence[TableHead] | None,
    cells: Mapping[str, GroupSummary],
) -> str | None:
    """Place one document; return a conflict reason, if any."""

    if any(block.lineage != _lineage(parent) for block in blocks):
        return "table blocks and documents disagree on the source hash"
    if not blocks:
        if tables:
            return "table rows exist but the document has no table blocks"
        result.without_tables += 1
        return None
    if tables is None:
        result.new.append(parent)
        return None
    problem = _tables_problem(parent, blocks, tables)
    if problem is None and not any(table.table_id in cells for table in tables):
        result.resumed.append(parent)
    elif problem is None:
        problem = _cells_problem(parent, tables, cells)
        result.skipped += problem is None
    return problem


def classify_documents(
    parents: Sequence[DocumentRecord],
    blocks: Mapping[str, list[BlockHead]],
    tables: Mapping[str, list[TableHead]],
    cells: Mapping[str, GroupSummary],
) -> TableClassification:
    """Classify every document against both tables before any parsing.

    Stored tables are judged by their heads and cells by per-table
    summaries, never by whole rows; the integrity audit checks the rows.
    Complete groups are skipped without reading Bronze; partial, orphaned
    or inconsistent groups, and cells without tables, are conflicts.
    """

    result = TableClassification()
    known = {parent.document_id for parent in parents}
    cell_tables: dict[str, set[str]] = defaultdict(set)
    for table_id, summary in cells.items():
        cell_tables[str(summary.value("document_id"))].add(table_id)
    for document_id in sorted((set(tables) | set(cell_tables)) - known):
        result.conflicts.append(
            ({"document_id": document_id}, "rows have no primary HTML parent")
        )
    for parent in parents:
        stored = tables.get(parent.document_id)
        stored_ids = {table.table_id for table in stored or ()}
        reason = (
            "cells belong to tables that are not stored"
            if cell_tables.get(parent.document_id, set()) - stored_ids
            else _classify_document(
                result,
                parent,
                blocks.get(parent.document_id, []),
                stored,
                cells,
            )
        )
        if reason is not None:
            result.conflicts.append((document_context(parent), reason))
    return result


def table_difference(
    stored: Sequence[DocumentTableRecord], expected: Sequence[DocumentTableRecord]
) -> str | None:
    """Compare tables on every field except `processed_at`."""

    old = {t.table_ordinal: t.model_dump(exclude={"processed_at"}) for t in stored}
    new = {t.table_ordinal: t.model_dump(exclude={"processed_at"}) for t in expected}
    if old.keys() != new.keys():
        return f"stored {len(old)} tables, but the document now yields {len(new)}"
    changed = sorted(
        {
            name
            for o, row in new.items()
            for name, value in row.items()
            if old[o][name] != value
        }
    )
    return f"stored tables differ in {changed}" if changed else None


def verify_written(
    tables: tuple[Table, Table | None],
    plan: TablePlan,
    built_tables: Mapping[str, Sequence[DocumentTableRecord]],
) -> None:
    """Re-read the plan's groups: each group built now is stored exactly.

    With no cells table only tables are read. Raises ValueError.
    """

    tables_table, cells_table = tables
    parents = plan.candidates
    ids = {parent.document_id for parent in parents}
    files = filing_dates_filter({parent.filing_date for parent in parents})
    stored_tables = read_groups_in(tables_table, "document_id", ids, files=files)
    stored_cells = (
        {}
        if cells_table is None
        else read_groups_in(cells_table, "document_id", ids, files=files)
    )
    for parent in parents:
        expected = built_tables.get(parent.document_id)
        if expected is None:
            continue
        stored, problem = stored_state(
            parent,
            plan.blocks[parent.document_id],
            stored_tables.get(parent.document_id, ()),
            None if cells_table is None else stored_cells.get(parent.document_id, []),
        )
        problem = problem or table_difference(stored, expected)
        if problem is not None:
            raise ValueError(f"{parent.document_id} after commit: {problem}")

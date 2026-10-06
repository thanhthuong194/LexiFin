from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
from src.data_pipeline.storage.iceberg import group_rows_by

StoredRows = Mapping[str, Sequence[dict[str, Any]]]


@dataclass
class TablePlan:
    """How each primary document relates to the stored table groups.

    `new` documents have table blocks but nothing stored; `resumed` ones have
    exact stored tables and no cells. Conflicts hold (context, reason) pairs.
    """

    parents: Sequence[DocumentRecord]
    blocks: Mapping[str, list[ContentBlockRecord]]
    new: list[DocumentRecord] = field(default_factory=list)
    resumed: dict[str, list[DocumentTableRecord]] = field(default_factory=dict)
    without_tables: int = 0
    skipped: int = 0
    conflicts: list[tuple[dict[str, str], str]] = field(default_factory=list)

    @property
    def candidates(self) -> list[DocumentRecord]:
        """Documents to parse in this run, in parent order."""

        new = {parent.document_id for parent in self.new}
        return [
            parent
            for parent in self.parents
            if parent.document_id in new or parent.document_id in self.resumed
        ]


def document_context(parent: DocumentRecord) -> dict[str, str]:
    """Identify a document in logs."""

    return {
        "filing_id": parent.filing_id,
        "document_id": parent.document_id,
        "source_path": parent.source_path,
    }


def _stored_state(
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


def _classify_document(
    plan: TablePlan,
    parent: DocumentRecord,
    table_rows: Sequence[dict[str, Any]] | None,
    cell_rows: Sequence[dict[str, Any]] | None,
) -> str | None:
    """Place one document in the plan; return a conflict reason, if any."""

    blocks = plan.blocks.get(parent.document_id, [])
    if any(block.source_document_sha256 != parent.content_sha256 for block in blocks):
        return "table blocks and documents disagree on the source hash"
    if not blocks:
        if table_rows or cell_rows:
            return "table rows exist but the document has no table blocks"
        plan.without_tables += 1
    elif table_rows is None:
        if cell_rows:
            return "cells exist without tables"
        plan.new.append(parent)
    else:
        tables, problem = _stored_state(parent, blocks, table_rows, cell_rows)
        if problem is not None:
            return problem
        if cell_rows is None:
            plan.resumed[parent.document_id] = tables
        else:
            plan.skipped += 1
    return None


def classify_documents(
    parents: Sequence[DocumentRecord],
    blocks: Mapping[str, list[ContentBlockRecord]],
    stored_tables: StoredRows,
    stored_cells: StoredRows,
) -> TablePlan:
    """Classify every document against both tables before any parsing.

    Complete groups are skipped without reading Bronze; partial, orphaned or
    inconsistent groups, and cells without tables, are conflicts.
    """

    plan = TablePlan(parents, blocks)
    known = {parent.document_id for parent in parents}
    for document_id in sorted((set(stored_tables) | set(stored_cells)) - known):
        plan.conflicts.append(
            ({"document_id": document_id}, "rows have no primary HTML parent")
        )
    for parent in parents:
        reason = _classify_document(
            plan,
            parent,
            stored_tables.get(parent.document_id),
            stored_cells.get(parent.document_id),
        )
        if reason is not None:
            plan.conflicts.append((document_context(parent), reason))
    return plan


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
    tables_table: Table,
    cells_table: Table | None,
    plan: TablePlan,
    built_tables: Mapping[str, Sequence[DocumentTableRecord]],
) -> None:
    """Re-read the tables (and cells): no orphans, and each built group stored exactly.

    With `cells_table` None only tables are read. Raises ValueError.
    """

    known = {parent.document_id for parent in plan.parents}
    stored_tables = group_rows_by(tables_table, "document_id")
    stored_cells = (
        {} if cells_table is None else group_rows_by(cells_table, "document_id")
    )
    orphans = (set(stored_tables) | set(stored_cells)) - known
    if orphans:
        raise ValueError(f"Table rows without a parent document: {sorted(orphans)[:5]}")
    for parent in plan.candidates:
        expected = built_tables.get(parent.document_id)
        if expected is None:
            continue
        tables, problem = _stored_state(
            parent,
            plan.blocks[parent.document_id],
            stored_tables.get(parent.document_id, ()),
            None if cells_table is None else stored_cells.get(parent.document_id, []),
        )
        problem = problem or table_difference(tables, expected)
        if problem is not None:
            raise ValueError(f"{parent.document_id} after commit: {problem}")

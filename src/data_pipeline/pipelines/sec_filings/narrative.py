from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import ValidationError
from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.content_blocks import (
    CONTENT_BLOCKS_ARROW_SCHEMA,
    ContentBlockRecord,
)
from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.sections import (
    SECTIONS_ARROW_SCHEMA,
    SectionRecord,
)
from src.data_pipeline.quality.sec_filings.narrative import (
    narrative_problem,
    sections_problem,
)
from src.data_pipeline.storage.iceberg import (
    append_rows,
    current_snapshot_id,
    group_rows_by,
    open_silver_catalog,
    verify_append,
)
from src.data_pipeline.storage.sec_filings.content_blocks import (
    CONTENT_BLOCKS_TABLE,
    count_table_cell_blocks,
    open_content_blocks_table,
)
from src.data_pipeline.storage.sec_filings.documents import (
    open_documents_table,
    read_document_content,
    read_primary_html_documents,
)
from src.data_pipeline.storage.sec_filings.sections import (
    SECTIONS_TABLE,
    open_sections_table,
)
from src.data_pipeline.transforms.sec_filings.html_blocks import extract_blocks
from src.data_pipeline.transforms.sec_filings.html_dom import parse_html
from src.data_pipeline.transforms.sec_filings.narrative import (
    DocumentNarrative,
    build_document_narrative,
    detect_section_headings,
)
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)

StoredRows = Mapping[str, Sequence[dict[str, Any]]]


@dataclass
class NarrativeStageSummary:
    """Outcome of one `sec_filings.sections` + `sec_filings.content_blocks` run.

    Each considered primary HTML document is counted in exactly one of
    inserted (both groups new), skipped (both complete), resumed (exact stored
    sections, blocks appended now), conflicted (invalid or changed stored
    state) or failed (unreadable source, parse or validation error); a
    rejected run counts would-be inserts and resumes in neither. Block metrics
    describe the documents built in this run. `legacy_table_cell_blocks`
    counts stored rows from the retired one-block-per-cell table layout; any
    rejects the run until both narrative tables are rebuilt.
    """

    status: Literal["completed", "rejected"] = "rejected"
    documents_considered: int = 0
    legacy_table_cell_blocks: int = 0
    documents_inserted: int = 0
    documents_skipped: int = 0
    documents_resumed: int = 0
    documents_conflicted: int = 0
    documents_failed: int = 0
    sections_inserted: int = 0
    blocks_inserted: int = 0
    heading_blocks: int = 0
    table_blocks: int = 0
    extracted_characters: int = 0
    root_only_documents: int = 0
    duplicate_section_codes: int = 0
    duration_seconds: float = 0.0
    sections_snapshot_before: int | None = None
    sections_snapshot_after: int | None = None
    content_blocks_snapshot_before: int | None = None
    content_blocks_snapshot_after: int | None = None


@dataclass
class _Plan:
    """All parent documents, and those to parse: new and resumed ones."""

    parents: Sequence[DocumentRecord]
    new: list[DocumentRecord] = field(default_factory=list)
    resumed: list[tuple[DocumentRecord, list[SectionRecord]]] = field(
        default_factory=list
    )

    @property
    def candidates(self) -> list[DocumentRecord]:
        return self.new + [parent for parent, _ in self.resumed]


def _conflict(summary: NarrativeStageSummary, reason: str, **context: object) -> None:
    summary.documents_conflicted += 1
    logger.error("document_narrative_conflict", reason=reason, **context)


def _stored_state(
    parent: DocumentRecord,
    section_rows: Sequence[dict[str, Any]],
    block_rows: Sequence[dict[str, Any]] | None,
) -> tuple[list[SectionRecord], str | None]:
    """Validate a document's stored rows; return its sections and any problem."""

    try:
        sections = [SectionRecord.model_validate(row) for row in section_rows]
        blocks = [ContentBlockRecord.model_validate(row) for row in block_rows or ()]
    except ValidationError as error:
        return [], f"stored row is invalid: {error.errors()[0]['msg']}"
    problem = sections_problem(parent, sections)
    if problem is None and block_rows is not None:
        problem = narrative_problem(parent, sections, blocks)
    return sections, problem


def _classify(
    parents: Sequence[DocumentRecord],
    stored_sections: StoredRows,
    stored_blocks: StoredRows,
    summary: NarrativeStageSummary,
) -> _Plan:
    """Sort documents into new, skipped, resumable and conflicting before parsing."""

    plan = _Plan(parents)
    known = {parent.document_id for parent in parents}
    for document_id in sorted((set(stored_sections) | set(stored_blocks)) - known):
        _conflict(summary, "rows have no primary HTML parent", document_id=document_id)
    for parent in parents:
        context = {
            "filing_id": parent.filing_id,
            "document_id": parent.document_id,
            "source_path": parent.source_path,
        }
        section_rows = stored_sections.get(parent.document_id)
        block_rows = stored_blocks.get(parent.document_id)
        if section_rows is None:
            if block_rows is None:
                plan.new.append(parent)
            else:
                _conflict(summary, "content blocks exist without sections", **context)
            continue
        sections, problem = _stored_state(parent, section_rows, block_rows)
        if problem is not None:
            _conflict(summary, problem, **context)
        elif block_rows is None:
            plan.resumed.append((parent, sections))
        else:
            summary.documents_skipped += 1
    logger.info("document_narrative_skipped", documents=summary.documents_skipped)
    return plan


def _record_metrics(
    narrative: DocumentNarrative, summary: NarrativeStageSummary
) -> None:
    blocks, sections = narrative.blocks, narrative.sections
    heading_blocks = sum(block.block_type == "heading" for block in blocks)
    table_blocks = sum(block.block_type == "table" for block in blocks)
    characters = sum(block.text_length for block in blocks)
    summary.heading_blocks += heading_blocks
    summary.table_blocks += table_blocks
    summary.extracted_characters += characters
    logger.info(
        "html_blocks_extracted",
        blocks=len(blocks),
        heading_blocks=heading_blocks,
        table_blocks=table_blocks,
        characters=characters,
    )
    logger.info(
        "document_sections_detected",
        sections=len(sections),
        section_codes=[section.section_code for section in sections[1:]],
    )
    if len(sections) == 1:
        summary.root_only_documents += 1
        logger.warning("root_only_document")
    if narrative.duplicate_section_codes:
        summary.duplicate_section_codes += narrative.duplicate_section_codes
        logger.warning(
            "duplicate_section_codes", count=narrative.duplicate_section_codes
        )


def _build_candidates(
    candidates: Sequence[DocumentRecord],
    data_dir: Path,
    processed_at: datetime,
    summary: NarrativeStageSummary,
) -> dict[str, DocumentNarrative]:
    """Parse each candidate's verified HTML once and build both record groups."""

    built: dict[str, DocumentNarrative] = {}
    verified_bytes = verified_documents = 0
    for parent in candidates:
        with log_context(
            filing_id=parent.filing_id,
            document_id=parent.document_id,
            form=parent.document_type,
        ):
            try:
                content = read_document_content(parent, data_dir)
                verified_documents += 1
                verified_bytes += len(content)
                blocks = extract_blocks(parse_html(content)).blocks
                headings = detect_section_headings(parent.document_type, blocks)
                narrative = build_document_narrative(
                    parent, blocks, headings, processed_at
                )
            except (OSError, ValueError) as error:
                summary.documents_failed += 1
                logger.error(
                    "document_narrative_invalid",
                    source_path=parent.source_path,
                    error=str(error),
                )
                continue
            _record_metrics(narrative, summary)
            built[parent.document_id] = narrative
    logger.info(
        "primary_document_content_verified",
        documents=verified_documents,
        bytes=verified_bytes,
    )
    return built


def _section_difference(
    stored: Sequence[SectionRecord], expected: Sequence[SectionRecord]
) -> str | None:
    """Compare sections on every field except `processed_at`."""

    old = {s.section_ordinal: s.model_dump(exclude={"processed_at"}) for s in stored}
    new = {s.section_ordinal: s.model_dump(exclude={"processed_at"}) for s in expected}
    if old.keys() != new.keys():
        return f"stored {len(old)} sections, but the document now yields {len(new)}"
    for ordinal, fields in new.items():
        changed = sorted(
            name for name, value in fields.items() if old[ordinal][name] != value
        )
        if changed:
            return f"section {ordinal} differs in {changed}"
    return None


def _validate_built(
    plan: _Plan, built: Mapping[str, DocumentNarrative], summary: NarrativeStageSummary
) -> None:
    """Check every built group; resumed sections must equal the stored ones."""

    stored_by_id = {parent.document_id: sections for parent, sections in plan.resumed}
    for parent in plan.candidates:
        narrative = built.get(parent.document_id)
        if narrative is None:
            continue
        context = {
            "filing_id": parent.filing_id,
            "document_id": parent.document_id,
            "source_path": parent.source_path,
        }
        problem = sections_problem(parent, narrative.sections) or narrative_problem(
            parent, narrative.sections, narrative.blocks
        )
        if problem is not None:
            summary.documents_failed += 1
            logger.error("document_narrative_invalid", error=problem, **context)
        elif parent.document_id in stored_by_id:
            difference = _section_difference(
                stored_by_id[parent.document_id], narrative.sections
            )
            if difference is not None:
                _conflict(summary, difference, **context)


def _verify_written(
    sections_table: Table,
    blocks_table: Table | None,
    parents: Sequence[DocumentRecord],
    built: Mapping[str, DocumentNarrative],
) -> None:
    """Re-read both tables: no rows outside `parents`, and every group built in
    this run is stored exactly. With `blocks_table` None only sections are read.
    """

    known = {parent.document_id for parent in parents}
    stored_sections = group_rows_by(sections_table, "document_id")
    stored_blocks = (
        {} if blocks_table is None else group_rows_by(blocks_table, "document_id")
    )
    orphans = (set(stored_sections) | set(stored_blocks)) - known
    if orphans:
        raise ValueError(
            f"Narrative rows without a parent document: {sorted(orphans)[:5]}"
        )
    for parent in parents:
        narrative = built.get(parent.document_id)
        if narrative is None:
            continue
        sections, problem = _stored_state(
            parent,
            stored_sections.get(parent.document_id, ()),
            None if blocks_table is None else stored_blocks.get(parent.document_id, []),
        )
        problem = problem or _section_difference(sections, narrative.sections)
        if problem is not None:
            raise ValueError(f"{parent.document_id} after commit: {problem}")


def _write(
    tables: tuple[Table, Table],
    plan: _Plan,
    built: Mapping[str, DocumentNarrative],
    stored_rows: tuple[int, int],
    summary: NarrativeStageSummary,
) -> None:
    """Append new sections, verify them, then append and verify the blocks."""

    sections_table, blocks_table = tables
    new_sections = sorted(
        (
            section
            for parent in plan.new
            for section in built[parent.document_id].sections
        ),
        key=lambda s: (s.filing_date, s.document_id, s.section_ordinal),
    )
    new_blocks = sorted(
        (
            block
            for parent in plan.candidates
            for block in built[parent.document_id].blocks
        ),
        key=lambda b: (b.filing_date, b.document_id, b.block_ordinal),
    )
    if new_sections:
        with timed_operation("narrative.write_sections", records=len(new_sections)):
            append_rows(sections_table, new_sections, SECTIONS_ARROW_SCHEMA)
    with timed_operation("narrative.verify", table=SECTIONS_TABLE):
        summary.sections_snapshot_after = verify_append(
            sections_table,
            key_column="section_id",
            expected_rows=stored_rows[0] + len(new_sections),
            inserted_keys=[section.section_id for section in new_sections],
            snapshot_before=summary.sections_snapshot_before,
        )
        _verify_written(sections_table, None, plan.parents, built)
    if new_blocks:
        with timed_operation("narrative.write_blocks", records=len(new_blocks)):
            append_rows(blocks_table, new_blocks, CONTENT_BLOCKS_ARROW_SCHEMA)
    with timed_operation("narrative.verify", table=CONTENT_BLOCKS_TABLE):
        summary.content_blocks_snapshot_after = verify_append(
            blocks_table,
            key_column="block_id",
            expected_rows=stored_rows[1] + len(new_blocks),
            inserted_keys=[block.block_id for block in new_blocks],
            snapshot_before=summary.content_blocks_snapshot_before,
        )
        _verify_written(sections_table, blocks_table, plan.parents, built)
    summary.documents_inserted, summary.documents_resumed = (
        len(plan.new),
        len(plan.resumed),
    )
    summary.sections_inserted, summary.blocks_inserted = (
        len(new_sections),
        len(new_blocks),
    )
    logger.info(
        "document_narrative_inserted",
        documents=summary.documents_inserted,
        sections=summary.sections_inserted,
    )
    logger.info("document_narrative_resumed", documents=summary.documents_resumed)


def _open_tables(catalog: Catalog, data_dir: Path) -> tuple[Table, Table]:
    sections_table, created = open_sections_table(catalog, data_dir)
    logger.info(
        "sections_table_created" if created else "sections_table_loaded",
        table=SECTIONS_TABLE,
        location=sections_table.location(),
    )
    blocks_table, created = open_content_blocks_table(catalog, data_dir)
    logger.info(
        "content_blocks_table_created" if created else "content_blocks_table_loaded",
        table=CONTENT_BLOCKS_TABLE,
        location=blocks_table.location(),
    )
    return sections_table, blocks_table


def _run_narrative(
    catalog: Catalog,
    data_dir: Path,
    processed_at: datetime,
    summary: NarrativeStageSummary,
) -> None:
    documents_table, _ = open_documents_table(catalog, data_dir)
    sections_table, blocks_table = _open_tables(catalog, data_dir)
    summary.sections_snapshot_before = summary.sections_snapshot_after = (
        current_snapshot_id(sections_table)
    )
    summary.content_blocks_snapshot_before = summary.content_blocks_snapshot_after = (
        current_snapshot_id(blocks_table)
    )
    summary.legacy_table_cell_blocks = count_table_cell_blocks(blocks_table)
    if summary.legacy_table_cell_blocks:
        # Rebuilding would change block ordinals, IDs and section ranges of
        # stored rows, so it must be a controlled operation, not a side effect.
        logger.error(
            "narrative_table_semantics_incompatible",
            table_cell_blocks=summary.legacy_table_cell_blocks,
            required_action="controlled rebuild of sec_filings.sections and "
            "sec_filings.content_blocks",
        )
        return
    with timed_operation("narrative.read"):
        parents = sorted(
            read_primary_html_documents(documents_table),
            key=lambda parent: (parent.filing_date, parent.document_id),
        )
        stored_sections = group_rows_by(sections_table, "document_id")
        stored_blocks = group_rows_by(blocks_table, "document_id")
    summary.documents_considered = len(parents)
    logger.info("primary_documents_loaded", documents=len(parents))

    with timed_operation("narrative.validate", phase="existing_state"):
        plan = _classify(parents, stored_sections, stored_blocks, summary)
    if summary.documents_conflicted:
        return
    with timed_operation("narrative.parse_html", documents=len(plan.candidates)):
        built = _build_candidates(plan.candidates, data_dir, processed_at, summary)
    with timed_operation("narrative.validate", phase="built_records"):
        _validate_built(plan, built, summary)
    if summary.documents_failed or summary.documents_conflicted:
        return

    stored_rows = (
        sum(len(rows) for rows in stored_sections.values()),
        sum(len(rows) for rows in stored_blocks.values()),
    )
    _write((sections_table, blocks_table), plan, built, stored_rows, summary)
    summary.status = "completed"


def run_narrative_stage(
    data_dir: Path, processed_at: datetime
) -> NarrativeStageSummary:
    """Build `sec_filings.sections` and `sec_filings.content_blocks` together.

    Every primary HTML document of `sec_filings.documents` is classified
    against both tables before any parsing; documents to build are parsed
    once and validated as a batch. Sections are appended first and verified,
    then blocks, so a failed blocks commit is resumed by the next run. Any
    conflict or invalid document rejects the batch before the first write;
    unexpected errors are logged and re-raised.
    """

    started = perf_counter()
    summary = NarrativeStageSummary()
    with log_context(source="sec_filings"):
        logger.info("narrative_stage_started")
        try:
            with (
                timed_operation("narrative.total"),
                open_silver_catalog(data_dir) as catalog,
            ):
                _run_narrative(catalog, data_dir, processed_at, summary)
        except Exception as error:
            logger.error(
                "narrative_stage_failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise

        summary.duration_seconds = round(perf_counter() - started, 3)
        if summary.status == "completed":
            logger.info("narrative_stage_completed", **asdict(summary))
        else:
            logger.error("narrative_stage_failed", **asdict(summary))
    return summary

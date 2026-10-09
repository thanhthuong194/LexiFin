from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import batched
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
    stored_group_problem,
)
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.storage.iceberg import (
    GroupSummary,
    append_rows,
    current_snapshot_id,
    filing_dates_filter,
    open_silver_catalog,
    read_groups_in,
    snapshot_total,
    summarize_groups,
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

# Documents parsed, validated and committed together: memory holds one
# batch's sections and blocks, and each batch is one commit per table.
DOCUMENTS_PER_COMMIT = 200
# Columns every row of a document's groups repeats from the document.
_LINEAGE_COLUMNS = ("filing_id", "filing_date", "source_document_sha256")

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
    """All parent documents, and those to parse: new and resumed ones.

    Resumed documents have complete stored sections and no blocks.
    """

    parents: Sequence[DocumentRecord]
    new: list[DocumentRecord] = field(default_factory=list)
    resumed: list[DocumentRecord] = field(default_factory=list)

    @property
    def candidates(self) -> list[DocumentRecord]:
        """Documents to parse in this run, in parent order."""

        resumed = {parent.document_id for parent in self.resumed}
        new = {parent.document_id for parent in self.new}
        return [
            parent
            for parent in self.parents
            if parent.document_id in new or parent.document_id in resumed
        ]


@dataclass(frozen=True)
class _Run:
    """What every batch of one run shares."""

    sections_table: Table
    blocks_table: Table
    bronze: SecBronze
    processed_at: datetime
    summary: NarrativeStageSummary


def _context(parent: DocumentRecord) -> dict[str, str]:
    return {
        "filing_id": parent.filing_id,
        "document_id": parent.document_id,
        "source_path": parent.source_path,
    }


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


def _summary_problem(
    parent: DocumentRecord, sections: GroupSummary, blocks: GroupSummary | None
) -> str | None:
    lineage = {
        "filing_id": parent.filing_id,
        "filing_date": parent.filing_date,
        "source_document_sha256": parent.content_sha256,
    }
    return stored_group_problem(sections, lineage, "sections") or (
        None
        if blocks is None
        else stored_group_problem(blocks, lineage, "content blocks")
    )


def _classify(
    parents: Sequence[DocumentRecord],
    stored_sections: Mapping[str, GroupSummary],
    stored_blocks: Mapping[str, GroupSummary],
    summary: NarrativeStageSummary,
) -> _Plan:
    """Sort documents into new, skipped, resumable and conflicting before parsing.

    Stored groups are judged by their summaries (complete ordinals, the
    parent's lineage); the integrity audit checks their rows.
    """

    plan = _Plan(parents)
    known = {parent.document_id for parent in parents}
    for document_id in sorted((set(stored_sections) | set(stored_blocks)) - known):
        _conflict(summary, "rows have no primary HTML parent", document_id=document_id)
    for parent in parents:
        sections = stored_sections.get(parent.document_id)
        blocks = stored_blocks.get(parent.document_id)
        if sections is None:
            if blocks is None:
                plan.new.append(parent)
            else:
                _conflict(
                    summary, "content blocks exist without sections", **_context(parent)
                )
            continue
        problem = _summary_problem(parent, sections, blocks)
        if problem is not None:
            _conflict(summary, problem, **_context(parent))
        elif blocks is None:
            plan.resumed.append(parent)
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
    bronze: SecBronze,
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
                content = read_document_content(parent, bronze)
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
    batch: Sequence[DocumentRecord],
    built: Mapping[str, DocumentNarrative],
    stored_sections: Mapping[str, Sequence[SectionRecord]],
    summary: NarrativeStageSummary,
) -> None:
    """Check every built group; resumed sections must equal the stored ones."""

    for parent in batch:
        narrative = built.get(parent.document_id)
        if narrative is None:
            continue
        problem = sections_problem(parent, narrative.sections) or narrative_problem(
            parent, narrative.sections, narrative.blocks
        )
        if problem is not None:
            summary.documents_failed += 1
            logger.error(
                "document_narrative_invalid", error=problem, **_context(parent)
            )
        elif parent.document_id in stored_sections:
            difference = _section_difference(
                stored_sections[parent.document_id], narrative.sections
            )
            if difference is not None:
                _conflict(summary, difference, **_context(parent))


def _read_groups(table: Table, parents: Sequence[DocumentRecord]) -> StoredRows:
    """The stored rows of the given documents, reading only their partitions."""

    return read_groups_in(
        table,
        "document_id",
        {parent.document_id for parent in parents},
        files=filing_dates_filter({parent.filing_date for parent in parents}),
    )


def _read_resumed(
    run: _Run, batch: Sequence[DocumentRecord], resumed: set[str]
) -> dict[str, list[SectionRecord]]:
    """The stored sections of the batch's resumed documents; problems conflict."""

    parents = [parent for parent in batch if parent.document_id in resumed]
    rows = _read_groups(run.sections_table, parents)
    stored = {}
    for parent in parents:
        sections, problem = _stored_state(
            parent, rows.get(parent.document_id, ()), None
        )
        if problem is not None:
            _conflict(run.summary, problem, **_context(parent))
        stored[parent.document_id] = sections
    return stored


def _verify_written(
    run: _Run,
    batch: Sequence[DocumentRecord],
    built: Mapping[str, DocumentNarrative],
    *,
    with_blocks: bool,
) -> None:
    """Re-read the batch's groups: every group built now is stored exactly.

    Without blocks only sections are read. Raises ValueError.
    """

    sections = _read_groups(run.sections_table, batch)
    blocks = _read_groups(run.blocks_table, batch) if with_blocks else {}
    for parent in batch:
        narrative = built[parent.document_id]
        stored, problem = _stored_state(
            parent,
            sections.get(parent.document_id, ()),
            blocks.get(parent.document_id, []) if with_blocks else None,
        )
        problem = problem or _section_difference(stored, narrative.sections)
        if problem is not None:
            raise ValueError(f"{parent.document_id} after commit: {problem}")


def _write(
    run: _Run,
    batch: Sequence[DocumentRecord],
    built: Mapping[str, DocumentNarrative],
    resumed: set[str],
) -> None:
    """Append the batch's new sections, verify them, then its blocks."""

    summary = run.summary
    new_sections = [
        section
        for parent in batch
        if parent.document_id not in resumed
        for section in built[parent.document_id].sections
    ]
    new_blocks = [block for p in batch for block in built[p.document_id].blocks]
    rows = (
        snapshot_total(run.sections_table, "total-records"),
        snapshot_total(run.blocks_table, "total-records"),
    )
    if new_sections:
        with timed_operation("narrative.write_sections", records=len(new_sections)):
            append_rows(run.sections_table, new_sections, SECTIONS_ARROW_SCHEMA)
    with timed_operation("narrative.verify", table=SECTIONS_TABLE):
        summary.sections_snapshot_after = verify_append(
            run.sections_table,
            key_column="section_id",
            expected_rows=rows[0] + len(new_sections),
            inserted_keys=[section.section_id for section in new_sections],
            snapshot_before=summary.sections_snapshot_after,
        )
        _verify_written(run, batch, built, with_blocks=False)
    if new_blocks:
        with timed_operation("narrative.write_blocks", records=len(new_blocks)):
            append_rows(run.blocks_table, new_blocks, CONTENT_BLOCKS_ARROW_SCHEMA)
    with timed_operation("narrative.verify", table=CONTENT_BLOCKS_TABLE):
        summary.content_blocks_snapshot_after = verify_append(
            run.blocks_table,
            key_column="block_id",
            expected_rows=rows[1] + len(new_blocks),
            inserted_keys=[block.block_id for block in new_blocks],
            snapshot_before=summary.content_blocks_snapshot_after,
        )
        _verify_written(run, batch, built, with_blocks=True)
    summary.sections_inserted += len(new_sections)
    summary.blocks_inserted += len(new_blocks)


def _run_batch(run: _Run, batch: Sequence[DocumentRecord], resumed: set[str]) -> bool:
    """Parse, validate and commit one batch; return whether it was written.

    A conflicting or invalid document rejects the batch before its first write.
    """

    summary = run.summary
    stored_sections = _read_resumed(run, batch, resumed)
    if summary.documents_conflicted:
        return False
    with timed_operation("narrative.parse_html", documents=len(batch)):
        built = _build_candidates(batch, run.bronze, run.processed_at, summary)
    with timed_operation("narrative.validate", phase="built_records"):
        _validate_built(batch, built, stored_sections, summary)
    if summary.documents_failed or summary.documents_conflicted:
        return False
    _write(run, batch, built, resumed)
    resumed_now = sum(parent.document_id in resumed for parent in batch)
    summary.documents_inserted += len(batch) - resumed_now
    summary.documents_resumed += resumed_now
    logger.info(
        "document_narrative_committed",
        documents=len(batch),
        resumed=resumed_now,
        sections=summary.sections_inserted,
        blocks=summary.blocks_inserted,
    )
    return True


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
    run_inputs: tuple[SecBronze, datetime],
    summary: NarrativeStageSummary,
) -> None:
    bronze, processed_at = run_inputs
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
        stored_sections = summarize_groups(
            sections_table,
            "document_id",
            "section_ordinal",
            declared="document_section_count",
            uniform=_LINEAGE_COLUMNS,
        )
        stored_blocks = summarize_groups(
            blocks_table,
            "document_id",
            "block_ordinal",
            declared="document_block_count",
            uniform=_LINEAGE_COLUMNS,
        )
    summary.documents_considered = len(parents)
    logger.info("primary_documents_loaded", documents=len(parents))

    with timed_operation("narrative.validate", phase="existing_state"):
        plan = _classify(parents, stored_sections, stored_blocks, summary)
    if summary.documents_conflicted:
        return
    run = _Run(sections_table, blocks_table, bronze, processed_at, summary)
    resumed = {parent.document_id for parent in plan.resumed}
    for batch in batched(plan.candidates, DOCUMENTS_PER_COMMIT, strict=False):
        if not _run_batch(run, batch, resumed):
            return
    summary.status = "completed"


def run_narrative_stage(
    data_dir: Path, processed_at: datetime, bronze: SecBronze | None = None
) -> NarrativeStageSummary:
    """Build `sec_filings.sections` and `sec_filings.content_blocks` together.

    Every primary HTML document of `sec_filings.documents` is classified
    against summaries of both tables before any parsing; any conflict
    rejects the run before the first write. Documents to build are parsed,
    validated and committed in batches of DOCUMENTS_PER_COMMIT: sections
    first, verified, then blocks, so a failed blocks commit is resumed by the
    next run. An invalid document rejects its batch before that batch's first
    write; earlier batches stay committed and are skipped by the next run.
    Unexpected errors are logged and re-raised.
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
                _run_narrative(
                    catalog,
                    data_dir,
                    (bronze or SecBronze.under(data_dir), processed_at),
                    summary,
                )
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

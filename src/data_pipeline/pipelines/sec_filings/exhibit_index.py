from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import ValidationError
from pyiceberg.catalog import Catalog

from src.data_pipeline.contracts.sec_filings.exhibits import (
    EXHIBITS_ARROW_SCHEMA,
    ExhibitRecord,
)
from src.data_pipeline.contracts.sec_filings.filings import FilingRecord
from src.data_pipeline.storage.iceberg import (
    append_rows,
    current_snapshot_id,
    group_rows_by,
    open_silver_catalog,
    read_rows_in,
    verify_append,
    verify_written_groups,
)
from src.data_pipeline.storage.sec_filings.content_blocks import (
    open_content_blocks_table,
)
from src.data_pipeline.storage.sec_filings.document_links import (
    open_document_links_table,
)
from src.data_pipeline.storage.sec_filings.document_table_cells import (
    open_document_table_cells_table,
)
from src.data_pipeline.storage.sec_filings.document_tables import (
    open_document_tables_table,
)
from src.data_pipeline.storage.sec_filings.exhibits import (
    EXHIBITS_TABLE,
    open_exhibits_table,
)
from src.data_pipeline.storage.sec_filings.filings import (
    open_filings_table,
    read_filings,
)
from src.data_pipeline.storage.sec_filings.sections import open_sections_table
from src.data_pipeline.transforms.sec_filings.exhibits import (
    CellLink,
    ExhibitEntry,
    GridCell,
    exhibit_records,
    is_exhibit_region,
    table_entries,
)
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)


@dataclass
class ExhibitIndexStageSummary:
    """Outcome of one `exhibits` run.

    A filing with Exhibit Index entries is inserted, skipped (stored exactly)
    or conflicted (stored otherwise). A filing whose exhibit regions hold no
    table with an exhibit header gets no entry; rows of exhibit tables that
    hold no exhibit number are rejected rows. A rejected run writes nothing.
    """

    status: Literal["completed", "rejected"] = "rejected"
    filings_considered: int = 0
    filings_with_exhibit_index: int = 0
    filings_inserted: int = 0
    filings_skipped: int = 0
    filings_conflicted: int = 0
    exhibit_tables: int = 0
    filings_without_exhibit_table: int = 0
    rejected_rows: int = 0
    entries_inserted: int = 0
    entries_with_href: int = 0
    entries_resolved: int = 0
    conflicting_links: int = 0
    snapshot_before: int | None = None
    snapshot_after: int | None = None
    duration_seconds: float = 0.0


@dataclass
class _Region:
    """An exhibit region section and the entries of its exhibit tables."""

    section: dict[str, Any]
    entries: list[ExhibitEntry] = field(default_factory=list)
    exhibit_tables: int = 0
    rejected_rows: int = 0


def _regions(
    catalog: Catalog, data_dir: Path, forms: dict[str, str]
) -> dict[str, _Region]:
    """Exhibit region sections, by section_id."""

    rows = (
        open_sections_table(catalog, data_dir)[0]
        .scan(
            selected_fields=(
                "section_id",
                "filing_id",
                "document_id",
                "section_kind",
                "section_code",
                "parent_section_id",
            )
        )
        .to_arrow()
        .to_pylist()
    )
    codes = {row["section_id"]: row["section_code"] for row in rows}
    return {
        row["section_id"]: _Region(row)
        for row in rows
        if is_exhibit_region(
            forms[row["filing_id"]],
            row["section_kind"],
            row["section_code"],
            codes.get(row["parent_section_id"] or ""),
        )
    }


def _read_tables(
    catalog: Catalog, data_dir: Path, regions: dict[str, _Region]
) -> list[dict[str, Any]]:
    """The stored tables inside exhibit regions, in document and table order."""

    section_of = {
        row["block_id"]: row["section_id"]
        for row in read_rows_in(
            open_content_blocks_table(catalog, data_dir)[0],
            "section_id",
            regions,
            columns=("block_id", "block_type"),
        )
        if row["block_type"] == "table"
    }
    tables = read_rows_in(
        open_document_tables_table(catalog, data_dir)[0],
        "block_id",
        section_of,
        columns=("table_id", "document_id", "table_ordinal"),
    )
    for table in tables:
        table["section_id"] = section_of[table["block_id"]]
    return sorted(tables, key=lambda t: (t["document_id"], t["table_ordinal"]))


def _extract(catalog: Catalog, data_dir: Path, regions: dict[str, _Region]) -> None:
    """Fill each region with the entries of its exhibit tables."""

    tables = _read_tables(catalog, data_dir, regions)
    cells: dict[str, list[GridCell]] = defaultdict(list)
    for row in read_rows_in(
        open_document_table_cells_table(catalog, data_dir)[0],
        "table_id",
        {table["table_id"] for table in tables},
        columns=("row_ordinal", "column_ordinal", "colspan", "text", "source_xpath"),
    ):
        cells[row.pop("table_id")].append(GridCell(**row))
    links: dict[str, list[CellLink]] = defaultdict(list)
    for row in read_rows_in(
        open_document_links_table(catalog, data_dir)[0],
        "source_document_id",
        {table["document_id"] for table in tables},
        columns=("source_xpath", "href", "target_document_id"),
    ):
        links[row.pop("source_document_id")].append(CellLink(**row))
    for table in tables:
        region = regions[table["section_id"]]
        found = table_entries(cells[table["table_id"]], links[table["document_id"]])
        if found is not None:
            region.exhibit_tables += 1
            region.rejected_rows += found.rejected_rows
            region.entries.extend(found.entries)


def _build(
    regions: dict[str, _Region],
    filings: dict[str, FilingRecord],
    processed_at: datetime,
    summary: ExhibitIndexStageSummary,
) -> dict[str, list[ExhibitRecord]]:
    """Each filing's entries, in document order, from its exhibit regions."""

    by_filing: dict[str, list[_Region]] = defaultdict(list)
    for region in sorted(regions.values(), key=lambda r: r.section["section_id"]):
        by_filing[region.section["filing_id"]].append(region)
    built: dict[str, list[ExhibitRecord]] = {}
    for filing_id, group in sorted(by_filing.items()):
        summary.exhibit_tables += sum(r.exhibit_tables for r in group)
        summary.rejected_rows += sum(r.rejected_rows for r in group)
        # An exhibits item often only introduces the EXHIBIT INDEX section
        # that holds the table, so only a filing without any is reported.
        if not any(r.exhibit_tables for r in group):
            summary.filings_without_exhibit_table += 1
            logger.warning(
                "exhibit_index_not_parsed",
                filing_id=filing_id,
                sections=[r.section["section_code"] for r in group],
            )
        entries = [entry for region in group for entry in region.entries]
        summary.conflicting_links += sum(e.has_conflicting_links for e in entries)
        if entries:
            built[filing_id] = exhibit_records(
                filing_id,
                filings[filing_id].filing_date,
                group[0].section["document_id"],
                entries,
                processed_at,
            )
    if summary.conflicting_links:
        logger.warning(
            "exhibit_entry_links_conflict", entries=summary.conflicting_links
        )
    return built


def _compare(
    built: dict[str, list[ExhibitRecord]],
    stored: dict[str, list[dict[str, Any]]],
    summary: ExhibitIndexStageSummary,
) -> list[ExhibitRecord]:
    """New entries of the filings not stored yet; stored groups must equal the build."""

    new: list[ExhibitRecord] = []
    for filing_id in sorted(built.keys() | stored.keys()):
        expected = [
            r.model_dump(exclude={"processed_at"}) for r in built.get(filing_id, [])
        ]
        rows = stored.get(filing_id, [])
        if not rows:
            new.extend(built[filing_id])
            continue
        try:
            found = [
                r.model_dump(exclude={"processed_at"})
                for r in sorted(
                    (ExhibitRecord.model_validate(row) for row in rows),
                    key=lambda r: r.exhibit_ordinal,
                )
            ]
        except ValidationError as error:
            reason = f"stored entry is invalid: {error.errors()[0]['msg']}"
        else:
            if found == expected:
                summary.filings_skipped += 1
                continue
            reason = "stored entries differ from the Exhibit Index"
        summary.filings_conflicted += 1
        logger.error(
            "exhibit_index_filing_conflict", filing_id=filing_id, reason=reason
        )
    return new


def _run_exhibits(
    catalog: Catalog,
    data_dir: Path,
    processed_at: datetime,
    summary: ExhibitIndexStageSummary,
) -> None:
    table, created = open_exhibits_table(catalog, data_dir)
    logger.info(
        "exhibits_table_created" if created else "exhibits_table_loaded",
        location=table.location(),
    )
    summary.snapshot_before = summary.snapshot_after = current_snapshot_id(table)
    with timed_operation("exhibit_index.read_parents"):
        filings = {
            f.filing_id: f
            for f in read_filings(open_filings_table(catalog, data_dir)[0])
        }
        regions = _regions(catalog, data_dir, {i: f.form for i, f in filings.items()})
        stored = group_rows_by(table, "filing_id")
    with timed_operation("exhibit_index.extract", regions=len(regions)):
        _extract(catalog, data_dir, regions)
        built = _build(regions, filings, processed_at, summary)
    summary.filings_considered = len(filings)
    summary.filings_with_exhibit_index = len(built)
    with timed_operation("exhibit_index.validate"):
        new = _compare(built, stored, summary)
    if summary.filings_conflicted:
        return
    if new:
        with timed_operation("exhibit_index.write", records=len(new)):
            append_rows(table, new, EXHIBITS_ARROW_SCHEMA)
    with timed_operation("exhibit_index.verify"):
        summary.snapshot_after = verify_append(
            table,
            key_column="exhibit_id",
            expected_rows=sum(map(len, stored.values())) + len(new),
            inserted_keys=[record.exhibit_id for record in new],
            snapshot_before=summary.snapshot_before,
        )
        verify_written_groups(
            table, new, key_column="exhibit_id", group_column="filing_id"
        )
    summary.filings_inserted = len({record.filing_id for record in new})
    summary.entries_inserted = len(new)
    summary.entries_with_href = sum(record.href is not None for record in new)
    summary.entries_resolved = sum(
        record.resolved_document_id is not None for record in new
    )
    summary.status = "completed"


def run_exhibit_index_stage(
    data_dir: Path, processed_at: datetime
) -> ExhibitIndexStageSummary:
    """Build `sec_filings.exhibits` from the stored Exhibit Index tables.

    Reads the exhibit regions of the stored sections and their tables,
    cells and links; no Bronze is read. Every run rebuilds each filing's
    entries: filings not stored yet are appended, stored filings must match
    exactly. Any conflict rejects the batch; unexpected errors are logged
    and raised.
    """

    started = perf_counter()
    summary = ExhibitIndexStageSummary()
    with log_context(source="sec_filings", table=EXHIBITS_TABLE):
        logger.info("exhibit_index_stage_started")
        try:
            with (
                timed_operation("exhibit_index.total"),
                open_silver_catalog(data_dir) as catalog,
            ):
                _run_exhibits(catalog, data_dir, processed_at, summary)
        except Exception as error:
            logger.error(
                "exhibit_index_stage_failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise

        summary.duration_seconds = round(perf_counter() - started, 3)
        if summary.status == "completed":
            logger.info("exhibit_index_stage_completed", **asdict(summary))
        else:
            logger.error("exhibit_index_stage_failed", **asdict(summary))
    return summary

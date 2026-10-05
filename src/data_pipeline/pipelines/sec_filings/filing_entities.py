from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.filing_entities import (
    FILING_ENTITIES_ARROW_SCHEMA,
    FilingEntityRecord,
)
from src.data_pipeline.contracts.sec_filings.filings import FilingRecord
from src.data_pipeline.storage.bronze import resolve_relative_path
from src.data_pipeline.storage.iceberg import (
    append_rows,
    current_snapshot_id,
    open_silver_catalog,
    verify_append,
)
from src.data_pipeline.storage.sec_filings.filing_entities import (
    FILING_ENTITIES_TABLE,
    open_filing_entities_table,
    read_filing_entity_groups,
)
from src.data_pipeline.storage.sec_filings.filings import (
    open_filings_table,
    read_filings,
)
from src.data_pipeline.transforms.sec_filings.filing_entities import (
    build_filing_entity_records,
    check_entity_offsets,
)
from src.data_pipeline.transforms.sec_filings.filings import (
    RawSecHeader,
    read_raw_sec_header,
)
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)

EntityGroups = Mapping[str, Sequence[FilingEntityRecord]]


@dataclass
class FilingEntitiesStageSummary:
    """Outcome of one `sec_filings.filing_entities` stage run.

    `filings_*` count parent filings and `entities_*` count entity rows, except
    `entities_failed`: the parent filings whose SEC header could not be read or
    parsed, each reported at its first error. `rejected` means nothing was
    written.
    """

    status: Literal["completed", "rejected"] = "rejected"
    filings_considered: int = 0
    filings_inserted: int = 0
    filings_skipped: int = 0
    filings_conflicted: int = 0
    entities_inserted: int = 0
    entities_skipped: int = 0
    entities_failed: int = 0
    duration_seconds: float = 0.0
    snapshot_before: int | None = None
    snapshot_after: int | None = None


def _read_header(parent: FilingRecord, data_dir: Path) -> RawSecHeader:
    """Read only the SEC header of a parent's full submission.

    The filings stage of the same run has just verified the file's SHA-256, so
    the whole submission is not hashed again here.
    """

    path = resolve_relative_path(parent.full_submission_path, data_dir)
    if path.stat().st_size == 0:
        raise ValueError("Full submission is empty")
    return read_raw_sec_header(path)


def _parse_headers(
    parents: Sequence[FilingRecord],
    data_dir: Path,
    processed_at: datetime,
    summary: FilingEntitiesStageSummary,
) -> dict[str, list[FilingEntityRecord]]:
    """Build and offset-check the entity group of every parent filing."""

    parsed: dict[str, list[FilingEntityRecord]] = {}
    for parent in parents:
        try:
            header = _read_header(parent, data_dir)
            records = build_filing_entity_records(parent, header, processed_at)
            check_entity_offsets(records, header)
        except (OSError, ValueError) as error:
            summary.entities_failed += 1
            logger.error(
                "filing_entities_invalid",
                filing_id=parent.filing_id,
                source_path=parent.full_submission_path,
                error=str(error),
            )
            continue
        parsed[parent.filing_id] = records
    logger.info(
        "filing_entity_blocks_parsed",
        filings=len(parsed),
        entities=sum(len(records) for records in parsed.values()),
    )
    return parsed


def _group_conflict(
    stored: Sequence[FilingEntityRecord], parsed: Sequence[FilingEntityRecord]
) -> dict[str, object] | None:
    """Describe how a stored entity group differs from the header parsed now.

    Rows are compared on every field except `processed_at`, which changes with
    each run. Returns None when the stored group is exactly the parsed one.
    """

    stored_by_id = {record.filing_entity_id: record for record in stored}
    if len(stored_by_id) != len(stored):
        return {"reason": "stored group has duplicate filing_entity_id values"}
    parsed_ids = {record.filing_entity_id for record in parsed}
    if set(stored_by_id) != parsed_ids:
        return {
            "reason": "stored group does not match the entity blocks in the header",
            "stored_entities": len(stored),
            "parsed_entities": len(parsed),
            "missing_ids": sorted(parsed_ids - set(stored_by_id)),
            "unexpected_ids": sorted(set(stored_by_id) - parsed_ids),
        }
    for record in parsed:
        old = stored_by_id[record.filing_entity_id].model_dump(exclude={"processed_at"})
        new = record.model_dump(exclude={"processed_at"})
        changed = sorted(field for field in new if old[field] != new[field])
        if changed:
            return {
                "reason": "stored entity differs from the header",
                "filing_entity_id": record.filing_entity_id,
                "entity_ordinal": record.entity_ordinal,
                "entity_role": record.entity_role,
                "cik": record.cik,
                "changed_fields": changed,
                "stored_entity_block_sha256": old["entity_block_sha256"],
                "parsed_entity_block_sha256": new["entity_block_sha256"],
            }
    return None


def _plan_inserts(
    parent_ids: set[str],
    parsed: EntityGroups,
    stored: EntityGroups,
    summary: FilingEntitiesStageSummary,
) -> list[FilingEntityRecord]:
    """Skip groups already stored exactly, flag conflicts, return new rows."""

    for filing_id in sorted(set(stored) - parent_ids):
        summary.filings_conflicted += 1
        logger.error(
            "filing_entities_conflict",
            filing_id=filing_id,
            reason="entities have no parent in sec_filings.filings",
        )
    new: list[FilingEntityRecord] = []
    for filing_id, records in parsed.items():
        existing = stored.get(filing_id)
        if existing is None:
            new.extend(records)
            continue
        conflict = _group_conflict(existing, records)
        if conflict is None:
            summary.filings_skipped += 1
            summary.entities_skipped += len(existing)
        else:
            summary.filings_conflicted += 1
            logger.error(
                "filing_entities_conflict",
                filing_id=filing_id,
                source_path=records[0].source_path,
                **conflict,
            )
    logger.info(
        "filing_entities_skipped",
        filings=summary.filings_skipped,
        entities=summary.entities_skipped,
    )
    return new


def _verify_stored_groups(table: Table, parsed: EntityGroups) -> None:
    """Re-read the table: it must hold exactly the parsed group of every parent."""

    stored = read_filing_entity_groups(table)
    if set(stored) != set(parsed):
        raise ValueError(
            f"Stored entity groups for {len(stored)} filings; expected {len(parsed)}"
        )
    for filing_id, records in parsed.items():
        conflict = _group_conflict(stored[filing_id], records)
        if conflict is not None:
            raise ValueError(f"Entities of {filing_id} after commit: {conflict}")


def _run_filing_entities(
    catalog: Catalog,
    data_dir: Path,
    processed_at: datetime,
    summary: FilingEntitiesStageSummary,
) -> None:
    filings_table, _ = open_filings_table(catalog, data_dir)
    parents = sorted(
        read_filings(filings_table), key=lambda p: (p.filing_date, p.filing_id)
    )
    summary.filings_considered = len(parents)

    table, created = open_filing_entities_table(catalog, data_dir)
    logger.info(
        "filing_entities_table_created" if created else "filing_entities_table_loaded",
        location=table.location(),
    )
    summary.snapshot_before = summary.snapshot_after = current_snapshot_id(table)
    stored = read_filing_entity_groups(table)

    with timed_operation("filing_entities.parse", filings=len(parents)):
        parsed = _parse_headers(parents, data_dir, processed_at, summary)
    with timed_operation("filing_entities.validate"):
        new = _plan_inserts(
            {parent.filing_id for parent in parents}, parsed, stored, summary
        )
    if summary.entities_failed or summary.filings_conflicted:
        return

    new.sort(key=lambda r: (r.filing_date, r.filing_id, r.entity_ordinal))
    if new:
        with timed_operation("filing_entities.write", records=len(new)):
            append_rows(table, new, FILING_ENTITIES_ARROW_SCHEMA)
    summary.filings_inserted = len({record.filing_id for record in new})
    summary.entities_inserted = len(new)
    logger.info(
        "filing_entities_inserted",
        filings=summary.filings_inserted,
        entities=summary.entities_inserted,
    )

    with timed_operation("filing_entities.verify"):
        summary.snapshot_after = verify_append(
            table,
            key_column="filing_entity_id",
            expected_rows=sum(len(group) for group in stored.values()) + len(new),
            inserted_keys=[record.filing_entity_id for record in new],
            snapshot_before=summary.snapshot_before,
        )
        _verify_stored_groups(table, parsed)
    summary.status = "completed"


def run_filing_entities_stage(
    data_dir: Path, processed_at: datetime
) -> FilingEntitiesStageSummary:
    """Record the entity-role blocks of every filing in `sec_filings.filings`.

    Every parent's SEC header is parsed (headers are small) and compared with
    the stored group: identical groups are skipped, new groups are appended in
    one commit, and any invalid header or differing stored group rejects the
    batch without writing. Unexpected errors are logged and re-raised.
    """

    started = perf_counter()
    summary = FilingEntitiesStageSummary()
    with log_context(source="sec_filings", table=FILING_ENTITIES_TABLE):
        logger.info("filing_entities_stage_started")
        try:
            with (
                timed_operation("filing_entities.total"),
                open_silver_catalog(data_dir) as catalog,
            ):
                _run_filing_entities(catalog, data_dir, processed_at, summary)
        except Exception as error:
            logger.error(
                "filing_entities_stage_failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise

        summary.duration_seconds = round(perf_counter() - started, 3)
        if summary.status == "completed":
            logger.info("filing_entities_stage_completed", **asdict(summary))
        else:
            logger.error("filing_entities_stage_failed", **asdict(summary))
    return summary

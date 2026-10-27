from __future__ import annotations

import tempfile
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import get_args

from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import NoSuchNamespaceError, NoSuchTableError
from pyiceberg.table import Table

from src.data_pipeline.contracts.ingestion import FormType
from src.data_pipeline.quality.sec_filings.silver_audit_bronze import (
    check_bronze_files,
)
from src.data_pipeline.quality.sec_filings.silver_audit_progress import (
    AuditProgress,
    NoProgress,
    audit_fingerprint,
    discard_scratch,
    prepare_scratch,
)
from src.data_pipeline.quality.sec_filings.silver_audit_report import (
    FilingSample,
    SilverAuditReport,
    TableMetadata,
)
from src.data_pipeline.quality.sec_filings.silver_audit_samples import (
    SampleRows,
    check_sample,
)
from src.data_pipeline.quality.sec_filings.silver_audit_scan import (
    RECORD_MODELS,
    check_data_files,
    contract_problem,
    scan_table,
)
from src.data_pipeline.quality.sec_filings.silver_audit_sql import (
    ScratchDatabase,
    deriver,
    run_sql_checks,
)
from src.data_pipeline.quality.sec_filings.silver_model import SILVER_TABLES
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.storage.iceberg import (
    current_snapshot_id,
    open_silver_catalog_read_only,
    silver_catalog_paths,
    silver_table_location,
    snapshot_total,
)
from src.data_pipeline.storage.sec_filings.documents import DOCUMENTS_TABLE
from src.data_pipeline.storage.sec_filings.filings import FILINGS_TABLE
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)

NAMESPACE = "sec_filings"
# The documents table's one name, and the singular form that must not appear.
CANONICAL_DOCUMENTS_TABLE = "sec_filings.documents"
LEGACY_DOCUMENTS_TABLE = "sec_filings.document"
FORMS: tuple[str, ...] = get_args(FormType)


def _file_states(data_dir: Path) -> dict[str, tuple[int, int]]:
    """Size and modification time of every catalog and warehouse file."""

    catalog_db, warehouse = silver_catalog_paths(data_dir)
    states = {}
    for path in sorted([*catalog_db.parent.rglob("*"), *warehouse.rglob("*")]):
        if path.is_file():
            stat = path.stat()
            states[str(path.relative_to(data_dir))] = (stat.st_size, stat.st_mtime_ns)
    return states


def _metadata(table: Table, identifier: str) -> TableMetadata:
    schema, spec = table.schema(), table.spec()
    partition = tuple(
        f"{field.transform}({schema.find_column_name(field.source_id)}) as {field.name}"
        for field in spec.fields
    )
    return TableMetadata(
        identifier=identifier,
        location=table.location(),
        metadata_location=table.metadata_location,
        snapshot_id=current_snapshot_id(table),
        snapshot_count=len(table.snapshots()),
        schema_id=schema.schema_id,
        fields=tuple(
            f"{f.name} {f.field_type} {'required' if f.required else 'optional'}"
            for f in schema.fields
        ),
        partition_spec=partition or ("unpartitioned",),
        sort_order="unsorted"
        if table.sort_order().is_unsorted
        else str(table.sort_order()),
        row_count=snapshot_total(table, "total-records"),
        data_file_count=snapshot_total(table, "total-data-files"),
        data_file_bytes=snapshot_total(table, "total-files-size"),
    )


def _inventory(catalog: Catalog, report: SilverAuditReport) -> dict[str, Table]:
    """Load every table of the namespace; compare them with the code's tables."""

    try:
        present = sorted(".".join(name) for name in catalog.list_tables(NAMESPACE))
    except NoSuchNamespaceError:
        present = []
    tables = {identifier: catalog.load_table(identifier) for identifier in present}
    for identifier, table in tables.items():
        report.tables[identifier] = _metadata(table, identifier)
        logger.info("silver_integrity_table", **asdict(report.tables[identifier]))
    defined = [table.identifier for table in SILVER_TABLES]
    with report.check(
        "catalog_has_defined_tables",
        None,
        f"every table the code defines ({len(defined)}) is in the catalog",
    ) as run:
        run.rows_checked = len(defined)
        for identifier in defined:
            if identifier not in tables:
                run.found(
                    identifier, f"{identifier} is defined in code, not in the catalog"
                )
    with report.check(
        "catalog_tables_managed",
        None,
        f"every table of namespace {NAMESPACE} is defined in code",
        severity="WARN",
    ) as run:
        run.rows_checked = len(present)
        for identifier in sorted(set(present) - set(defined)):
            run.found(identifier, f"{identifier} is in the catalog, not in code")
    with report.check(
        "documents_identifier",
        CANONICAL_DOCUMENTS_TABLE,
        f"code and catalog name the documents table {CANONICAL_DOCUMENTS_TABLE}, "
        f"never {LEGACY_DOCUMENTS_TABLE}",
    ) as run:
        run.rows_checked = len(present) + len(defined)
        if LEGACY_DOCUMENTS_TABLE in present:
            run.found("catalog", f"namespace {NAMESPACE} has {LEGACY_DOCUMENTS_TABLE}")
        if DOCUMENTS_TABLE != CANONICAL_DOCUMENTS_TABLE:
            run.found("code", f"storage DOCUMENTS_TABLE is {DOCUMENTS_TABLE}")
    logger.info("silver_integrity_inventory", tables=present, defined=len(defined))
    return tables


def _check_definitions(
    tables: Mapping[str, Table], data_dir: Path, report: SilverAuditReport
) -> bool:
    """Contract, schema, partition spec, location and data files of each table.

    Returns whether every defined table exists and matches its definition,
    which the row checks need.
    """

    valid = True
    for audited in SILVER_TABLES:
        identifier, spec = audited.identifier, audited.spec
        model = RECORD_MODELS[identifier]
        with report.check(
            "record_contract",
            identifier,
            "the record contract matches the Arrow schema",
        ) as run:
            run.rows_checked = len(model.model_fields)
            problem = contract_problem(model, spec.arrow_schema)
            if problem is not None:
                run.found(model.__name__, problem)
        table = tables.get(identifier)
        if table is None:
            valid = False
            continue
        with report.check(
            "schema_and_partition",
            identifier,
            "the Iceberg schema (field IDs, names, types, nullability) and partition "
            "spec equal the contract",
        ) as run:
            run.rows_checked = len(spec.arrow_schema)
            problem = spec.definition_problem(table)
            if problem is not None:
                valid = False
                run.found(identifier, problem)
        with report.check(
            "location",
            identifier,
            "the table lives at <data_dir>/silver/<namespace>/<table>",
        ) as run:
            run.rows_checked = 1
            expected = silver_table_location(data_dir, identifier)
            if Path(table.location()) != expected:
                run.found(identifier, f"{table.location()} is not {expected}")
        valid = check_data_files(table, identifier, report) and valid
    return valid


def _check_samples(
    samples: SampleRows,
    scratch: ScratchDatabase,
    bronze: SecBronze,
    report: SilverAuditReport,
) -> None:
    """Trace every chosen filing back to Bronze; warn about forms without one."""

    with report.check(
        "sample_forms",
        FILINGS_TABLE,
        f"{samples.per_form} filing(s) of each form {', '.join(FORMS)} can be sampled",
        severity="WARN",
    ) as run:
        run.rows_checked = len(FORMS)
        if samples.per_form == 0:
            run.skip("--sample-per-form is 0")
        for form, filings in samples.chosen.items():
            if samples.per_form and not filings:
                run.found(form, f"no {form} filing in Silver")
    with timed_operation("silver_audit.samples"):
        for form, filings in samples.chosen.items():
            for row in filings:
                sample = FilingSample(form, row["filing_id"])
                report.samples.append(sample)
                with log_context(filing_id=sample.filing_id, form=form):
                    check_sample(samples, sample, bronze, report, scratch.local_names)


@dataclass(frozen=True)
class AuditOptions:
    """How much of the audit reads: rows per batch, filings traced per form,
    and whether every filing's Bronze bytes are re-hashed.

    With a `work_dir`, the scratch database lives there and records each
    completed table scan and set-based check, so an audit stopped by a crash
    or power loss resumes where it stopped when it runs again on the same
    snapshots with the same settings and code; the database is removed once
    an audit completes. Without one, it lives in a temporary directory.
    """

    samples_per_form: int = 1
    batch_size: int = 10_000
    verify_source_hashes: bool = False
    # The SEC Bronze directory; None means <data_dir>/bronze/sec_filings.
    bronze_dir: Path | None = None
    work_dir: Path | None = None
    # Processes validating data files against their record contracts; the
    # report does not depend on it.
    workers: int = 1


@contextmanager
def _executor(workers: int) -> Iterator[ProcessPoolExecutor | None]:
    """Worker processes for the row scans, or None to scan in this process."""

    if workers == 1:
        yield None
        return
    # Spawned workers start clean instead of copying this process's memory.
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as executor:
        yield executor


@contextmanager
def _scratch(
    options: AuditOptions, report: SilverAuditReport
) -> Iterator[tuple[ScratchDatabase, AuditProgress | NoProgress]]:
    """The scratch database and the progress it holds."""

    if options.work_dir is None:
        with (
            tempfile.TemporaryDirectory(prefix="silver_audit_") as scratch_dir,
            closing(ScratchDatabase(Path(scratch_dir) / "audit.duckdb")) as scratch,
        ):
            yield scratch, NoProgress()
            logger.info(
                "silver_integrity_scratch_database",
                size_bytes=(Path(scratch_dir) / "audit.duckdb").stat().st_size,
            )
        return
    fingerprint = audit_fingerprint(
        report,
        {
            "samples_per_form": options.samples_per_form,
            "batch_size": options.batch_size,
        },
    )
    path = prepare_scratch(options.work_dir.expanduser().resolve(), fingerprint)
    with closing(ScratchDatabase(path)) as scratch:
        progress = AuditProgress(scratch.connection, report, fingerprint)
        yield scratch, progress
        logger.info("silver_integrity_scratch_database", size_bytes=path.stat().st_size)
    discard_scratch(path)


def _check_rows(
    tables: Mapping[str, Table],
    data_dir: Path,
    report: SilverAuditReport,
    options: AuditOptions,
) -> None:
    """Stream every table once: row contracts, then the scratch database and
    the sampled filings' rows. Run the set-based and Bronze checks on the
    scratch database, then trace the sampled filings to Bronze."""

    bronze = (
        SecBronze.under(data_dir)
        if options.bronze_dir is None
        else SecBronze(options.bronze_dir.expanduser().resolve())
    )
    batch_size = options.batch_size
    samples = SampleRows(FORMS, options.samples_per_form)
    with _scratch(options, report) as (scratch, progress):
        with (
            _executor(options.workers) as executor,
            timed_operation("silver_audit.scan", batch_size=batch_size),
        ):
            for audited in SILVER_TABLES:
                identifier = audited.identifier
                step = f"scan:{identifier}"
                done, state = progress.replay(step)
                if done:  # the scanned rows, and the samples kept up to then
                    rows, samples = state
                    scratch.row_counts[identifier] = rows
                    report.rows_scanned += rows
                    continue
                scratch.reset(identifier)
                scan = scan_table(
                    tables[identifier],
                    identifier,
                    batch_size,
                    report,
                    keep=samples.keep(identifier),
                    derive=deriver(identifier),
                    executor=executor,
                )
                scratch.load(identifier, scan.paths, scan.rows, scan.derived)
                samples.add(identifier, scan.kept)
                report.rows_scanned += scan.rows
                progress.complete(step, (scan.rows, samples))
        if progress.replayed:
            logger.info("silver_integrity_resumed", steps_replayed=progress.replayed)
        with timed_operation("silver_audit.integrity"):
            run_sql_checks(scratch, report, progress)
        with timed_operation("silver_audit.bronze"):
            check_bronze_files(
                scratch,
                bronze,
                report,
                verify_hashes=options.verify_source_hashes,
            )
        _check_samples(samples, scratch, bronze, report)


def _verify_read_only(
    data_dir: Path,
    files_before: Mapping[str, tuple[int, int]],
    report: SilverAuditReport,
) -> None:
    """Reload every table: its metadata, snapshots and row counts, and the
    catalog and warehouse files, must be as the audit found them."""

    with report.check(
        "read_only",
        None,
        "no table, snapshot, row count, metadata file, data file or catalog "
        "file changed during the audit",
    ) as run:
        with open_silver_catalog_read_only(data_dir) as catalog:
            for identifier, before in report.tables.items():
                try:
                    after: TableMetadata | None = _metadata(
                        catalog.load_table(identifier), identifier
                    )
                except NoSuchTableError:
                    after = None
                if after != before:
                    change = (
                        "dropped"
                        if after is None
                        else f"snapshot {before.snapshot_id} -> {after.snapshot_id}, "
                        f"rows {before.row_count} -> {after.row_count}"
                    )
                    report.snapshot_changes[identifier] = change
                    run.found(identifier, f"{identifier}: {change}")
        run.rows_checked = len(report.tables)
        files_after = _file_states(data_dir)
        for path in sorted(files_before.keys() | files_after.keys()):
            if files_before.get(path) != files_after.get(path):
                run.found(path, f"{path} was added, removed or modified")


def audit_silver(
    data_dir: Path, options: AuditOptions | None = None
) -> SilverAuditReport:
    """Audit the SEC Silver layer of `data_dir` without writing anything.

    Inventories the catalog against the tables the code defines; compares
    each table's contract, schema, partition spec, location and data files;
    streams every row through its record contract into a temporary SQLite
    database (`options.batch_size` rows at a time) for key, reference,
    lineage, group, hierarchy and semantic checks; checks every filing's
    Bronze files (re-hashed with `options.verify_source_hashes`); traces the
    `options.samples_per_form` latest filings of each form back to Bronze;
    finally checks that no snapshot, row count or catalog file changed. The
    catalog is opened read-only. A missing catalog or an unexpected error is
    logged as `silver_integrity_failed` and raised; findings are in the report.
    """

    options = options or AuditOptions()
    data_dir = data_dir.expanduser().resolve()
    started = perf_counter()
    report = SilverAuditReport(data_dir=str(data_dir), started_at=datetime.now(UTC))
    with log_context(source="sec_filings", audit="silver_audit"):
        logger.info(
            "silver_integrity_started",
            data_dir=str(data_dir),
            **asdict(options),
        )
        try:
            with (
                timed_operation("silver_audit.total"),
                open_silver_catalog_read_only(data_dir) as catalog,
            ):
                files_before = _file_states(data_dir)
                tables = _inventory(catalog, report)
                if _check_definitions(tables, data_dir, report):
                    _check_rows(tables, data_dir, report, options)
                else:
                    with report.check(
                        "row_checks",
                        None,
                        "row, integrity and sample checks",
                    ) as run:
                        run.skip(
                            "they need every defined table, matching its definition"
                        )
            _verify_read_only(data_dir, files_before, report)
        except Exception as error:
            logger.error(
                "silver_integrity_failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        report.finished_at = datetime.now(UTC)
        report.duration_seconds = round(perf_counter() - started, 3)
        log = logger.error if report.status == "FAIL" else logger.info
        log("silver_integrity_completed", **report.summary())
    return report

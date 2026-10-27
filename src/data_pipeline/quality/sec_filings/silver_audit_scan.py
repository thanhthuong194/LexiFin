from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Executor
from dataclasses import dataclass, field
from datetime import date
from functools import partial
from types import NoneType
from typing import Any, Literal, get_args, get_origin
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import AwareDatetime, BaseModel, NaiveDatetime, ValidationError
from pyiceberg.io.pyarrow import pyarrow_to_schema
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.content_blocks import ContentBlockRecord
from src.data_pipeline.contracts.sec_filings.document_links import DocumentLinkRecord
from src.data_pipeline.contracts.sec_filings.document_table_cells import (
    DocumentTableCellRecord,
)
from src.data_pipeline.contracts.sec_filings.document_tables import DocumentTableRecord
from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.exhibits import ExhibitRecord
from src.data_pipeline.contracts.sec_filings.filing_entities import FilingEntityRecord
from src.data_pipeline.contracts.sec_filings.filings import FilingRecord
from src.data_pipeline.contracts.sec_filings.sections import SectionRecord
from src.data_pipeline.contracts.sec_filings.xbrl_concepts import XbrlConceptRecord
from src.data_pipeline.contracts.sec_filings.xbrl_core import (
    XbrlContextRecord,
    XbrlInstanceRecord,
    XbrlUnitRecord,
)
from src.data_pipeline.contracts.sec_filings.xbrl_dimensions import (
    XbrlDimensionRecord,
)
from src.data_pipeline.contracts.sec_filings.xbrl_fact_footnotes import (
    XbrlFactFootnoteRecord,
)
from src.data_pipeline.contracts.sec_filings.xbrl_facts import XbrlFactRecord
from src.data_pipeline.contracts.sec_filings.xbrl_labels import XbrlLabelRecord
from src.data_pipeline.contracts.sec_filings.xbrl_networks import (
    XbrlNetworkRecord,
    XbrlRelationshipRecord,
)
from src.data_pipeline.quality.sec_filings.silver_audit_report import (
    SAMPLE_IDS,
    SilverAuditReport,
)
from src.data_pipeline.storage.iceberg import snapshot_total
from src.data_pipeline.storage.sec_filings.content_blocks import CONTENT_BLOCKS_TABLE
from src.data_pipeline.storage.sec_filings.document_links import DOCUMENT_LINKS_TABLE
from src.data_pipeline.storage.sec_filings.document_table_cells import (
    DOCUMENT_TABLE_CELLS_TABLE,
)
from src.data_pipeline.storage.sec_filings.document_tables import DOCUMENT_TABLES_TABLE
from src.data_pipeline.storage.sec_filings.documents import DOCUMENTS_TABLE
from src.data_pipeline.storage.sec_filings.exhibits import EXHIBITS_TABLE
from src.data_pipeline.storage.sec_filings.filing_entities import FILING_ENTITIES_TABLE
from src.data_pipeline.storage.sec_filings.filings import FILINGS_TABLE
from src.data_pipeline.storage.sec_filings.sections import SECTIONS_TABLE
from src.data_pipeline.storage.sec_filings.xbrl_core import (
    XBRL_CONTEXTS_TABLE,
    XBRL_INSTANCES_TABLE,
    XBRL_UNITS_TABLE,
)
from src.data_pipeline.storage.sec_filings.xbrl_enrichment import (
    XBRL_DIMENSIONS_TABLE,
    XBRL_FACT_FOOTNOTES_TABLE,
)
from src.data_pipeline.storage.sec_filings.xbrl_facts import (
    XBRL_CONCEPTS_TABLE,
    XBRL_FACTS_TABLE,
)
from src.data_pipeline.storage.sec_filings.xbrl_linkbases import (
    XBRL_LABELS_TABLE,
    XBRL_NETWORKS_TABLE,
    XBRL_RELATIONSHIPS_TABLE,
)

# The record contract of every Silver table.
RECORD_MODELS: dict[str, type[BaseModel]] = {
    FILINGS_TABLE: FilingRecord,
    DOCUMENTS_TABLE: DocumentRecord,
    FILING_ENTITIES_TABLE: FilingEntityRecord,
    SECTIONS_TABLE: SectionRecord,
    CONTENT_BLOCKS_TABLE: ContentBlockRecord,
    DOCUMENT_TABLES_TABLE: DocumentTableRecord,
    DOCUMENT_TABLE_CELLS_TABLE: DocumentTableCellRecord,
    DOCUMENT_LINKS_TABLE: DocumentLinkRecord,
    EXHIBITS_TABLE: ExhibitRecord,
    XBRL_INSTANCES_TABLE: XbrlInstanceRecord,
    XBRL_CONTEXTS_TABLE: XbrlContextRecord,
    XBRL_UNITS_TABLE: XbrlUnitRecord,
    XBRL_CONCEPTS_TABLE: XbrlConceptRecord,
    XBRL_FACTS_TABLE: XbrlFactRecord,
    XBRL_DIMENSIONS_TABLE: XbrlDimensionRecord,
    XBRL_FACT_FOOTNOTES_TABLE: XbrlFactFootnoteRecord,
    XBRL_LABELS_TABLE: XbrlLabelRecord,
    XBRL_NETWORKS_TABLE: XbrlNetworkRecord,
    XBRL_RELATIONSHIPS_TABLE: XbrlRelationshipRecord,
}
# The Arrow types a record field's Python type may be stored as.
_ARROW_TYPES: dict[object, Callable[[pa.DataType], bool]] = {
    str: pa.types.is_string,
    bool: pa.types.is_boolean,
    int: pa.types.is_integer,
    date: pa.types.is_date32,
    AwareDatetime: lambda arrow: pa.types.is_timestamp(arrow) and arrow.tz is not None,
    NaiveDatetime: lambda arrow: pa.types.is_timestamp(arrow) and arrow.tz is None,
    list: pa.types.is_list,
    dict: pa.types.is_map,
}


def contract_row(identifier: str, row: dict[str, Any]) -> dict[str, Any]:
    """A stored row as its record contract takes it."""

    if identifier == XBRL_INSTANCES_TABLE:  # Arrow gives a map as (key, value) pairs
        return {**row, "namespace_map": dict(row["namespace_map"])}
    return row


def contract_problem(model: type[BaseModel], schema: pa.Schema) -> str | None:
    """How a record contract and its Arrow schema disagree, or None.

    Fields must have the same names in the same order, the same nullability
    (Optional in the record, nullable in Arrow) and compatible types.
    """

    fields = model.model_fields
    if list(fields) != schema.names:
        return f"record fields {list(fields)} are not the Arrow columns {schema.names}"
    for name, info in fields.items():
        arguments = get_args(info.annotation)
        optional = NoneType in arguments
        python_type = (
            next(a for a in arguments if a is not NoneType)
            if optional
            else info.annotation
        )
        origin = get_origin(python_type) or python_type
        arrow = schema.field(name)
        if optional != arrow.nullable:
            return f"{name}: optional in the record is {optional}, nullable in Arrow is {arrow.nullable}"
        fits = _ARROW_TYPES.get(str if origin is Literal else origin)
        if fits is None or not fits(arrow.type):
            return f"{name}: record type {python_type} does not fit Arrow {arrow.type}"
    return None


def _file_schema_problem(table: Table, path: str) -> str | None:
    with table.io.new_input(path).open() as stream:
        schema = pq.read_schema(stream)
    if pyarrow_to_schema(schema).as_struct() != table.schema().as_struct():
        return f"{path}: Parquet columns or field IDs differ from the table schema"
    return None


def check_data_files(table: Table, identifier: str, report: SilverAuditReport) -> bool:
    """Each live data file: inside the table, no deletes, the table's schema.

    Reads Parquet footers only; file and record counts must match the
    snapshot summary. Returns whether every file passed, which reading the
    files directly (`table_batches`) needs.
    """

    location = table.location().rstrip("/") + "/"
    with report.check(
        "data_files",
        identifier,
        "live data files lie in the table location, have no delete files, carry "
        "the table schema and field IDs, and add up to the snapshot summary",
    ) as run:
        files = 0
        for task in table.scan().plan_files():
            files += 1
            run.rows_checked += task.file.record_count
            path = task.file.file_path
            problem = (
                f"{path}: has delete files"
                if task.delete_files
                else f"{path}: outside {location}"
                if not path.startswith(location)
                else _file_schema_problem(table, path)
            )
            if problem is not None:
                run.found(path, problem)
        summary = (
            snapshot_total(table, "total-data-files"),
            snapshot_total(table, "total-records"),
        )
        if (files, run.rows_checked) != summary:
            run.found(
                identifier,
                f"{files} files with {run.rows_checked} records; the snapshot "
                f"summary says {summary[0]} files with {summary[1]} records",
            )
    return run.failed_count == 0


def local_path(path: str) -> str:
    """A data file's path on this machine (Silver warehouses are local)."""

    return urlsplit(path).path if path.startswith("file:") else path


@dataclass(frozen=True)
class Keep:
    """The rows of a scan to keep for the sampled filings: those whose
    `column` holds one of `values`, or every row when `column` is None."""

    column: str | None
    values: frozenset[str] = frozenset()


@dataclass
class FileScan:
    """What validating one data file found.

    `failures` holds the first SAMPLE_IDS failing rows (identifier, problem)
    of `failed`; `kept` the rows `Keep` selects; `derived` the derived
    column's value of every row, in file order, for a table that has one.
    """

    rows: int = 0
    failed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    kept: list[dict[str, Any]] = field(default_factory=list)
    derived: list[object] = field(default_factory=list)


def scan_file(
    identifier: str,
    path: str,
    batch_size: int,
    keep: Keep | None,
    derive: Callable[[Mapping[str, Any]], object] | None,
) -> FileScan:
    """Validate every row of one data file against its record contract.

    Runs in a worker process or in the caller; reads `batch_size` rows at a
    time on this thread and returns memory to the allocator afterwards.
    """

    model = RECORD_MODELS[identifier]
    key = next(iter(model.model_fields))
    result = FileScan()
    parquet = pq.ParquetFile(local_path(path))
    for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False):
        rows = batch.to_pylist()
        for row in rows:
            try:
                model.model_validate(contract_row(identifier, row))
            except ValidationError as error:
                result.failed += 1
                if len(result.failures) < SAMPLE_IDS:
                    first = error.errors()[0]
                    location = ".".join(str(part) for part in first["loc"]) or "row"
                    result.failures.append(
                        (str(row[key]), f"{row[key]}: {location}: {first['msg']}")
                    )
        if keep is not None:
            result.kept += [
                row
                for row in rows
                if keep.column is None or row[keep.column] in keep.values
            ]
        if derive is not None:
            result.derived += map(derive, rows)
        result.rows += len(rows)
    parquet.close()
    pa.default_memory_pool().release_unused()
    return result


@dataclass
class TableScan:
    """A table's rows: how many, the data files read, the rows kept for the
    sampled filings and the derived column's values by data file."""

    rows: int
    paths: list[str]
    kept: list[dict[str, Any]]
    derived: dict[str, list[object]] | None


def scan_table(  # noqa: PLR0913 - the scan's inputs and its executor
    table: Table,
    identifier: str,
    batch_size: int,
    report: SilverAuditReport,
    *,
    keep: Keep | None,
    derive: Callable[[Mapping[str, Any]], object] | None = None,
    executor: Executor | None = None,
) -> TableScan:
    """Validate every row against its contract, one data file per task.

    Files are validated by `executor` (in this process without one) and
    their findings merged in file order, so the report does not depend on
    the number of workers. The rows read must equal the snapshot's
    total-records.
    """

    paths = [task.file.file_path for task in table.scan().plan_files()]
    task = partial(
        scan_file, identifier, batch_size=batch_size, keep=keep, derive=derive
    )
    results = map(task, paths) if executor is None else executor.map(task, paths)
    scan = TableScan(0, paths, [], None if derive is None else {})
    with report.check(
        "row_contract",
        identifier,
        "every row validates against its record contract: required values, "
        "types, enums, ranges, SHA-256 formats, derived IDs and row hashes",
    ) as run:
        for path, result in zip(paths, results, strict=True):
            for identifier_value, problem in result.failures:
                run.found(identifier_value, problem)
            run.failed_count += result.failed - len(result.failures)
            run.rows_checked += result.rows
            scan.kept += result.kept
            if scan.derived is not None:
                scan.derived[path] = result.derived
        expected = snapshot_total(table, "total-records")
        if run.rows_checked != expected:
            run.found(
                identifier,
                f"read {run.rows_checked} rows; the snapshot summary says {expected}",
            )
    scan.rows = run.rows_checked
    return scan

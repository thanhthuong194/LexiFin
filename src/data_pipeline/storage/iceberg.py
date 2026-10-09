from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from itertools import count
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel
from pyiceberg.catalog import Catalog
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.expressions import AlwaysTrue, BooleanExpression, In
from pyiceberg.io.pyarrow import pyarrow_to_schema
from pyiceberg.manifest import ManifestEntryStatus
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.table.name_mapping import MappedField, NameMapping
from pyiceberg.transforms import Transform
from pyiceberg.types import NestedField

SILVER_CATALOG_NAME = "lexifin_silver"
# Iceberg reserves partition field IDs from 1000 upwards.
_FIRST_PARTITION_FIELD_ID = 1000


@dataclass(frozen=True)
class PartitionColumn:
    """A hidden partition: `transform(source_column)` stored as `name`."""

    source_column: str
    transform: Transform[Any, Any]
    name: str


def silver_catalog_paths(data_dir: Path) -> tuple[Path, Path]:
    """Return the Silver catalog database and default warehouse under `data_dir`."""

    return data_dir / "_catalog" / "silver.db", data_dir / "silver"


def open_silver_catalog(data_dir: Path) -> SqlCatalog:
    """Open (creating if needed) the local SQLite catalog of the Silver layer.

    The catalog database lives at `<data_dir>/_catalog/silver.db` and the
    default warehouse at `<data_dir>/silver`. Use it as a context manager so
    its database connections are closed; tables need it open to commit.
    """

    database_path, warehouse_path = silver_catalog_paths(data_dir)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    warehouse_path.mkdir(parents=True, exist_ok=True)
    return SqlCatalog(
        SILVER_CATALOG_NAME,
        uri=f"sqlite:///{database_path}",
        warehouse=str(warehouse_path),
    )


def open_silver_catalog_read_only(data_dir: Path) -> SqlCatalog:
    """Open the existing Silver catalog over a read-only SQLite connection.

    Nothing is created: a missing catalog database or warehouse raises
    FileNotFoundError, and SQLite refuses any write made through the catalog
    (table commits included). Use it as a context manager, like
    `open_silver_catalog`.
    """

    database_path, warehouse_path = silver_catalog_paths(data_dir)
    for path in (database_path, warehouse_path):
        if not path.exists():
            raise FileNotFoundError(f"Silver catalog path is missing: {path}")
    return SqlCatalog(
        SILVER_CATALOG_NAME,
        uri=f"sqlite:///file:{quote(str(database_path))}?mode=ro&uri=true",
        warehouse=str(warehouse_path),
    )


def silver_table_location(data_dir: Path, identifier: str) -> Path:
    """Return `<data_dir>/silver/<namespace>/<table>` for a Silver table."""

    namespace, table_name = identifier.rsplit(".", maxsplit=1)
    return data_dir / "silver" / namespace / table_name


def _iceberg_schema(arrow_schema: pa.Schema) -> Schema:
    """Convert an Arrow schema to Iceberg, numbering fields in column order.

    Top-level fields get IDs 1..N, so flat tables keep the IDs they were
    created with; list elements and map keys and values follow from N + 1,
    also in column order.
    """

    nested_ids = count(len(arrow_schema) + 1)

    def nested(arrow_type: pa.DataType) -> list[MappedField]:
        if pa.types.is_map(arrow_type):
            parts = [("key", arrow_type.key_type), ("value", arrow_type.item_type)]
        elif pa.types.is_list(arrow_type):
            parts = [("element", arrow_type.value_type)]
        else:
            return []
        return [
            MappedField.model_validate(
                {"field-id": next(nested_ids), "names": [name], "fields": nested(type_)}
            )
            for name, type_ in parts
        ]

    name_mapping = NameMapping(
        [
            MappedField.model_validate(
                {
                    "field-id": field_id,
                    "names": [field.name],
                    "fields": nested(field.type),
                }
            )
            for field_id, field in enumerate(arrow_schema, start=1)
        ]
    )
    return pyarrow_to_schema(arrow_schema, name_mapping=name_mapping)


def _partition_spec(
    schema: Schema, columns: Sequence[PartitionColumn]
) -> PartitionSpec:
    return PartitionSpec(
        *(
            PartitionField(
                source_id=schema.find_field(column.source_column).field_id,
                field_id=_FIRST_PARTITION_FIELD_ID + position,
                transform=column.transform,
                name=column.name,
            )
            for position, column in enumerate(columns)
        )
    )


def create_or_load_table(
    catalog: Catalog,
    identifier: str,
    *,
    arrow_schema: pa.Schema,
    location: Path,
    partition_by: Sequence[PartitionColumn],
) -> tuple[Table, bool]:
    """Load a table, or create it with the given schema and hidden partitioning.

    Returns the table and whether it was created. An existing table must have
    exactly the expected schema and partition spec; schema or partition
    evolution is not handled here, so a mismatch raises ValueError.
    """

    schema = _iceberg_schema(arrow_schema)
    spec = _partition_spec(schema, partition_by)
    namespace = identifier.rsplit(".", maxsplit=1)[0]
    catalog.create_namespace_if_not_exists(namespace)
    try:
        table = catalog.load_table(identifier)
    except NoSuchTableError:
        table = catalog.create_table(
            identifier,
            schema=schema,
            location=str(location),
            partition_spec=spec,
        )
        return table, True

    problem = table_definition_problem(table, arrow_schema, partition_by)
    if problem is not None:
        raise ValueError(f"Iceberg table {identifier} {problem}")
    return table, False


def _describe_field(field: NestedField) -> str:
    nullability = "required" if field.required else "optional"
    return f"{field.field_id}:{field.name} {field.field_type} {nullability}"


def table_definition_problem(
    table: Table, arrow_schema: pa.Schema, partition_by: Sequence[PartitionColumn]
) -> str | None:
    """Describe how a table's schema or partition spec differs from its contract.

    The schema must equal the contract's field for field (ID in column order,
    name, type, nullability); the partition fields must be exactly the
    contract's. Returns None when both match.
    """

    schema = _iceberg_schema(arrow_schema)
    if table.schema() != schema:
        expected = [_describe_field(field) for field in schema.fields]
        actual = [_describe_field(field) for field in table.schema().fields]
        if expected == actual:
            return "schema differs from contract in its identifier fields"
        position = next(
            (
                i
                for i, pair in enumerate(zip(expected, actual, strict=False))
                if pair[0] != pair[1]
            ),
            min(len(expected), len(actual)),
        )
        return (
            f"schema differs from contract at column {position + 1}: expected "
            f"{expected[position] if position < len(expected) else 'no column'}, "
            f"found {actual[position] if position < len(actual) else 'no column'}"
        )
    spec = _partition_spec(schema, partition_by)
    if table.spec().fields != spec.fields:
        return f"partition spec differs: expected {spec}, found {table.spec()}"
    return None


@dataclass(frozen=True)
class SilverTableSpec:
    """A Silver table's identifier, contract schema and hidden partitioning.

    The one definition that its stage creates the table from and that the
    audit and the pipeline registry check it against.
    """

    identifier: str
    arrow_schema: pa.Schema
    partition_by: tuple[PartitionColumn, ...] = ()

    def open(self, catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
        """Load or create the table under `data_dir`; return whether it was created."""

        return create_or_load_table(
            catalog,
            self.identifier,
            arrow_schema=self.arrow_schema,
            location=silver_table_location(data_dir, self.identifier),
            partition_by=self.partition_by,
        )

    def definition_problem(self, table: Table) -> str | None:
        """How a loaded table differs from this definition, or None."""

        return table_definition_problem(table, self.arrow_schema, self.partition_by)


def current_snapshot_id(table: Table) -> int | None:
    """Return the table's current snapshot ID, or None before the first write."""

    snapshot = table.current_snapshot()
    return snapshot.snapshot_id if snapshot is not None else None


# Which data files a read covers: a filter that prunes them, or their paths.
DataFiles = BooleanExpression | Collection[str]
# A files filter that keeps every data file.
_EVERY_FILE = AlwaysTrue()
# Rows decoded at a time by the streaming readers below.
BATCH_ROWS = 10_000
# Rows per batch when only a few narrow columns are aggregated.
SUMMARY_BATCH_ROWS = 100_000


def snapshot_total(table: Table, key: str) -> int:
    """A total of the current snapshot's summary (`total-records`...); 0 before any write."""

    snapshot = table.current_snapshot()
    if snapshot is None or snapshot.summary is None:
        return 0
    value = snapshot.summary[key]
    if value is None:
        raise ValueError(f"{table.name()} snapshot summary has no {key}")
    return int(value)


def values_filter(column: str, values: Collection[str]) -> BooleanExpression:
    """A files filter keeping files whose `column` may hold one of `values`."""

    # PyIceberg predicates take (term, literals) at runtime; mypy only sees
    # the fields pydantic generates for them.
    return In(column, set(values))  # type: ignore[call-arg, arg-type]


def filing_dates_filter(dates: Collection[date]) -> BooleanExpression:
    """A files filter keeping the partitions of the given filing dates."""

    return values_filter("filing_date", {d.isoformat() for d in dates})


def added_data_files(table: Table) -> list[str]:
    """Paths of the data files the current snapshot added (none before a write)."""

    snapshot = table.current_snapshot()
    if snapshot is None:
        return []
    return [
        entry.data_file.file_path
        for manifest in snapshot.manifests(table.io)
        if manifest.added_snapshot_id == snapshot.snapshot_id
        for entry in manifest.fetch_manifest_entry(table.io)
        if entry.status == ManifestEntryStatus.ADDED
    ]


def scan_batches(
    table: Table,
    columns: Sequence[str] | None = None,
    *,
    files: DataFiles = _EVERY_FILE,
    batch_size: int = BATCH_ROWS,
) -> Iterator[pa.RecordBatch]:
    """The `columns` of the current snapshot's rows, at most `batch_size` at a time.

    `files` is a filter that only prunes data files (by partition and column
    bounds; rows of a kept file are all returned, so callers filter rows
    themselves), or the exact paths to read, such as the files an append
    just added (`added_data_files`). Data files are read one by one
    with Parquet `iter_batches`, so memory holds one batch and the pages it
    is decoded from, never a whole table: PyIceberg 0.12's
    `to_arrow_batch_reader` reads every file of a scan concurrently and keeps
    each file's batches until they are consumed. Decoding stays on this
    thread and Arrow's pool returns freed memory after each file; with
    decoder threads the allocator kept what each thread had freed, and RSS
    grew with every file read. Tables are append-only, so a data file with
    delete files raises ValueError.
    """

    pool = pa.default_memory_pool()
    selected = None if columns is None else list(dict.fromkeys(columns))
    if isinstance(files, BooleanExpression):
        tasks = list(table.scan(row_filter=files).plan_files())
        if any(task.delete_files for task in tasks):
            raise ValueError(f"{table.name()} has delete files")
        files = [task.file.file_path for task in tasks]
    for path in files:
        with table.io.new_input(path).open() as stream:
            parquet = pq.ParquetFile(stream)
            yield from parquet.iter_batches(
                batch_size=batch_size, columns=selected, use_threads=False
            )
        pool.release_unused()


def iter_rows(
    table: Table, columns: Sequence[str] | None = None
) -> Iterator[dict[str, Any]]:
    """The current snapshot's rows as plain dicts, one batch in memory at a time."""

    for batch in scan_batches(table, columns):
        yield from batch.to_pylist()


def count_groups(table: Table, group: str) -> Counter[str]:
    """Rows per value of `group`, read one narrow column at a time."""

    counts: Counter[str] = Counter()
    for batch in scan_batches(table, (group,), batch_size=SUMMARY_BATCH_ROWS):
        counts.update(batch[group].to_pylist())
    return counts


@dataclass(slots=True)
class GroupSummary:
    """Aggregates of one group's stored rows, enough to judge it without them.

    The group is complete when its ordinals are exactly 1..N (count, bounds,
    and the sums of ordinals and of their squares agree; the integrity audit
    checks the exact uniqueness) and, with a declared-count column, every
    row declares N. `values` holds the distinct values of the columns that
    must be uniform within the group.
    """

    rows: int = 0
    ordinal_min: int | None = None
    ordinal_max: int | None = None
    ordinal_sum: int = 0
    ordinal_square_sum: int = 0
    declared: set[int] = field(default_factory=set)
    values: dict[str, set[Any]] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Ordinals exactly 1..N and, when declared, a declared count of N."""

        n = self.rows
        return (
            n > 0
            and (self.ordinal_min, self.ordinal_max) == (1, n)
            and self.ordinal_sum == n * (n + 1) // 2
            and self.ordinal_square_sum == n * (n + 1) * (2 * n + 1) // 6
            and self.declared <= {n}
        )

    def value(self, column: str) -> Any:
        """The one value `column` holds in the group, or None when it varies."""

        values = self.values.get(column, set())
        return next(iter(values)) if len(values) == 1 else None


def summarize_groups(
    table: Table,
    group: str,
    ordinal: str,
    *,
    declared: str | None = None,
    uniform: Sequence[str] = (),
) -> dict[str, GroupSummary]:
    """Summarize every group of `table` by `group` in one streaming pass.

    Reads only the named columns and aggregates each batch in Arrow, so
    memory grows with the number of groups, not of rows.
    """

    summaries: dict[str, GroupSummary] = {}
    columns = [group, ordinal, *([declared] if declared else []), *uniform]
    for batch in scan_batches(table, columns, batch_size=SUMMARY_BATCH_ROWS):
        rows = pa.Table.from_batches([batch])
        ordinals = rows[ordinal].cast(pa.int64())
        rows = rows.append_column("_square", pc.multiply(ordinals, ordinals))
        aggregates = [
            ([], "count_all"),
            (ordinal, "min"),
            (ordinal, "max"),
            (ordinal, "sum"),
            ("_square", "sum"),
            *(
                (column, "distinct")
                for column in [*([declared] if declared else []), *uniform]
            ),
        ]
        for row in rows.group_by(group).aggregate(aggregates).to_pylist():
            summary = summaries.setdefault(row[group], GroupSummary())
            _merge(summary, row, ordinal, declared, uniform)
    return summaries


def _merge(
    summary: GroupSummary,
    row: dict[str, Any],
    ordinal: str,
    declared: str | None,
    uniform: Sequence[str],
) -> None:
    low, high = row[f"{ordinal}_min"], row[f"{ordinal}_max"]
    summary.rows += row["count_all"]
    summary.ordinal_min = (
        low if summary.ordinal_min is None else min(summary.ordinal_min, low)
    )
    summary.ordinal_max = (
        high if summary.ordinal_max is None else max(summary.ordinal_max, high)
    )
    summary.ordinal_sum += row[f"{ordinal}_sum"]
    summary.ordinal_square_sum += row["_square_sum"]
    if declared:
        summary.declared.update(row[f"{declared}_distinct"])
    for column in uniform:
        summary.values.setdefault(column, set()).update(row[f"{column}_distinct"])


def append_rows(
    table: Table, rows: Sequence[BaseModel], arrow_schema: pa.Schema
) -> None:
    """Append validated rows, in the given order, as one Iceberg commit.

    Rows are converted with the exact Arrow schema, so a value that does not
    fit the contract fails before anything is committed.
    """

    table.append(
        pa.Table.from_pylist([row.model_dump() for row in rows], schema=arrow_schema)
    )


def verify_append(
    table: Table,
    *,
    key_column: str,
    expected_rows: int,
    inserted_keys: Collection[str],
    snapshot_before: int | None,
) -> int | None:
    """Re-read a table after an append (or a skipped one) and return its snapshot.

    Checks that a new snapshot exists exactly when rows were inserted, that
    the snapshot's row count is as expected, and that the files the append
    added hold exactly the inserted keys, once each. Only those files are
    read, so the cost follows the append, not the table; uniqueness against
    older rows is the integrity audit's job. Raises ValueError otherwise.
    """

    table.refresh()
    snapshot_after = current_snapshot_id(table)
    if (snapshot_after != snapshot_before) != bool(inserted_keys):
        raise ValueError(
            f"Unexpected snapshot change {snapshot_before} -> {snapshot_after} "
            f"for {len(inserted_keys)} inserted rows"
        )
    rows = snapshot_total(table, "total-records")
    if rows != expected_rows:
        raise ValueError(f"Table has {rows} rows; expected {expected_rows}")
    found: Counter[str] = Counter()
    if inserted_keys:
        for batch in scan_batches(table, (key_column,), files=added_data_files(table)):
            found.update(batch[key_column].to_pylist())
    missing = set(inserted_keys) - found.keys()
    if missing:
        raise ValueError(f"Inserted rows missing after commit: {sorted(missing)[:5]}")
    unexpected = sorted(found.keys() - set(inserted_keys))
    repeated = sorted(key for key, count in found.items() if count > 1)
    if unexpected or repeated:
        raise ValueError(
            f"Appended {key_column} values differ from the inserted rows: "
            f"{(unexpected or repeated)[:5]}"
        )
    return snapshot_after


def verify_written_groups(
    table: Table, written: Sequence[BaseModel], *, key_column: str, group_column: str
) -> None:
    """Check that every group (by `group_column`) written now is stored exactly.

    Run right after `verify_append` of the same append: only the files it
    added are read, and rows are compared by `key_column`, with
    `processed_at` left out. Raises ValueError on any difference.
    """

    expected: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in written:
        values = row.model_dump(exclude={"processed_at"})
        expected[values[group_column]][values[key_column]] = values
    found: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for stored in read_rows_in(
        table,
        group_column,
        expected,
        columns=table.schema().column_names,
        files=added_data_files(table) if expected else [],
    ):
        stored.pop("processed_at", None)
        found[stored[group_column]][stored[key_column]] = stored
    for group, rows in expected.items():
        if found.get(group, {}) != rows:
            raise ValueError(f"{table.name()} rows of {group} differ after commit")


def read_rows_in(
    table: Table,
    column: str,
    values: Collection[str],
    *,
    columns: Sequence[str],
    files: DataFiles = _EVERY_FILE,
) -> list[dict[str, Any]]:
    """The `columns` (and `column`) of the stored rows whose `column` is in `values`.

    Streams the data `files` (see `scan_batches`) and keeps only matching
    rows, so memory follows the result.
    """

    if not values:
        return []
    wanted = pa.array(list(values), type=pa.string())
    found: list[dict[str, Any]] = []
    for batch in scan_batches(table, (column, *columns), files=files):
        matching = batch.filter(pc.is_in(batch[column], value_set=wanted))
        if matching.num_rows:
            found.extend(matching.to_pylist())
    return found


def read_groups_in(
    table: Table,
    column: str,
    values: Collection[str],
    *,
    files: DataFiles = _EVERY_FILE,
) -> dict[str, list[dict[str, Any]]]:
    """Every column of the stored rows whose `column` is in `values`, grouped by it.

    Rows are plain dicts, not validated, so callers can judge invalid rows.
    """

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_rows_in(
        table,
        column,
        values,
        columns=table.schema().column_names,
        files=files,
    ):
        groups[row[column]].append(row)
    return dict(groups)


def verify_written_values(
    table: Table,
    written: Sequence[BaseModel],
    *,
    key_column: str,
    columns: Sequence[str],
) -> None:
    """Check that every row written now is stored with the same `columns` values.

    Run right after `verify_append` of the same append: only the files it
    added are read, and only `key_column` and `columns`; choose columns,
    such as content hashes, that pin down the rest of the row.
    """

    stored = {
        row.pop(key_column): row
        for row in read_rows_in(
            table,
            key_column,
            [getattr(row, key_column) for row in written],
            columns=columns,
            files=added_data_files(table) if written else [],
        )
    }
    for row in written:
        key = getattr(row, key_column)
        found = stored.get(key, {})
        if any(found.get(column) != getattr(row, column) for column in columns):
            raise ValueError(f"{table.name()} row {key} differs after commit")

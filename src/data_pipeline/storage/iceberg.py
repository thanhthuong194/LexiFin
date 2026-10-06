from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any

import pyarrow as pa
from pydantic import BaseModel
from pyiceberg.catalog import Catalog
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.io.pyarrow import pyarrow_to_schema
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


def current_snapshot_id(table: Table) -> int | None:
    """Return the table's current snapshot ID, or None before the first write."""

    snapshot = table.current_snapshot()
    return snapshot.snapshot_id if snapshot is not None else None


def group_rows_by(table: Table, column: str) -> dict[str, list[dict[str, Any]]]:
    """Return the current snapshot's rows as plain dicts grouped by `column`.

    Rows are not validated here, so callers can judge invalid stored rows.
    """

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in table.scan().to_arrow().to_pylist():
        groups.setdefault(row[column], []).append(row)
    return groups


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

    Checks that a new snapshot exists exactly when rows were inserted, that the
    row count is as expected, that `key_column` is unique and that every
    inserted key is present. Raises ValueError otherwise.
    """

    table.refresh()
    snapshot_after = current_snapshot_id(table)
    if (snapshot_after != snapshot_before) != bool(inserted_keys):
        raise ValueError(
            f"Unexpected snapshot change {snapshot_before} -> {snapshot_after} "
            f"for {len(inserted_keys)} inserted rows"
        )
    keys = table.scan(selected_fields=(key_column,)).to_arrow()[key_column].to_pylist()
    if len(keys) != expected_rows:
        raise ValueError(f"Table has {len(keys)} rows; expected {expected_rows}")
    if len(set(keys)) != len(keys):
        raise ValueError(f"Table contains duplicate {key_column} values")
    missing = set(inserted_keys) - set(keys)
    if missing:
        raise ValueError(f"Inserted rows missing after commit: {sorted(missing)[:5]}")
    return snapshot_after


def verify_written_groups(
    table: Table, written: Sequence[BaseModel], *, key_column: str, group_column: str
) -> None:
    """Check that every group (by `group_column`) written now is stored exactly.

    Run after `verify_append`; rows are compared by `key_column`, with
    `processed_at` left out. Raises ValueError on any difference.
    """

    expected: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in written:
        values = row.model_dump(exclude={"processed_at"})
        expected[values[group_column]][values[key_column]] = values
    groups = group_rows_by(table, group_column) if expected else {}
    for group, rows in expected.items():
        found = {
            row[key_column]: {k: v for k, v in row.items() if k != "processed_at"}
            for row in groups.get(group, [])
        }
        if found != rows:
            raise ValueError(f"{table.name()} rows of {group} differ after commit")

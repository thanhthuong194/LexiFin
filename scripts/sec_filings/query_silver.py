from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import duckdb
from pyiceberg.catalog import Catalog

from src.data_pipeline.quality.sec_filings.silver_model import SILVER_TABLES
from src.data_pipeline.storage.iceberg import open_silver_catalog_read_only

FETCH_ROWS = 10_000


def _param(text: str) -> tuple[str, str]:
    name, separator, value = text.partition("=")
    if not separator or not name.isidentifier():
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    return name, value


def parse_args() -> argparse.Namespace:
    """Parse the SEC Silver query CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a read-only DuckDB SQL query over the sec_filings Iceberg tables; "
            "each table is a view named without its namespace (filings, xbrl_facts...)"
        )
    )
    parser.add_argument("query", type=Path, help="SQL file, e.g. queries/item_1a.sql")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data",
        help="Silver root: catalog and tables (default: <repository_root>/data)",
    )
    parser.add_argument(
        "--param",
        type=_param,
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Value of the query's $NAME parameter, as text (repeatable)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Write every row as CSV to stdout instead of a truncated table",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=40,
        help="Rows shown in the table output (default: 40)",
    )
    parser.add_argument(
        "--memory-limit",
        default="2GB",
        help="DuckDB memory limit; larger work spills to a temporary directory "
        "(default: 2GB)",
    )
    return parser.parse_args()


def _register_tables(con: duckdb.DuckDBPyConnection, catalog: Catalog) -> None:
    """Expose each existing Silver table as a view over its current data files.

    Views read nothing until a query does, and DuckDB then reads only the
    columns and row groups the query needs. Silver tables are append-only;
    a table with delete files would make its view wrong, so it raises.
    """

    for silver_table in SILVER_TABLES:
        if not catalog.table_exists(silver_table.identifier):
            continue
        table = catalog.load_table(silver_table.identifier)
        name = silver_table.identifier.split(".", maxsplit=1)[1]
        tasks = list(table.scan().plan_files())
        if any(task.delete_files for task in tasks):
            raise ValueError(f"{silver_table.identifier} has delete files")
        files = [task.file.file_path for task in tasks]
        relation = (
            con.read_parquet(files)
            if files
            else con.from_arrow(table.schema().as_arrow().empty_table())
        )
        relation.create_view(name)


def _write_csv(relation: duckdb.DuckDBPyRelation) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(relation.columns)
    while rows := relation.fetchmany(FETCH_ROWS):
        writer.writerows(rows)


def main() -> int:
    """Exit 0 after printing the result, 1 when the query cannot run."""

    args = parse_args()
    sql = args.query.read_text(encoding="utf-8")
    with (
        tempfile.TemporaryDirectory(prefix="query_silver_") as spill_dir,
        open_silver_catalog_read_only(args.data_dir) as catalog,
        duckdb.connect(
            config={"memory_limit": args.memory_limit, "temp_directory": spill_dir}
        ) as con,
    ):
        try:
            _register_tables(con, catalog)
            relation = con.sql(sql, params=dict(args.param))
            if args.csv:
                _write_csv(relation)
            else:
                relation.show(max_rows=args.max_rows)
        except duckdb.Error as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

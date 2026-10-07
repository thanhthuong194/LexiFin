from __future__ import annotations

from pathlib import Path

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.content_blocks import (
    CONTENT_BLOCKS_ARROW_SCHEMA,
    ContentBlockRecord,
)
from src.data_pipeline.storage.iceberg import (
    SilverTableSpec,
)
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

CONTENT_BLOCKS_TABLE = "sec_filings.content_blocks"


CONTENT_BLOCKS_SPEC = SilverTableSpec(
    CONTENT_BLOCKS_TABLE, CONTENT_BLOCKS_ARROW_SCHEMA, FILING_YEAR_PARTITION
)


def open_content_blocks_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create `sec_filings.content_blocks`, returning whether it was created."""

    return CONTENT_BLOCKS_SPEC.open(catalog, data_dir)


def count_table_cell_blocks(table: Table) -> int:
    """Count rows stored with the retired `table_cell` type (one block per td/th).

    Such rows predate atomic `table` blocks; their ordinals, IDs and section
    ranges differ from what the current parser builds.
    """

    rows = table.scan(
        row_filter="block_type == 'table_cell'", selected_fields=("block_id",)
    )
    return len(rows.to_arrow())


def read_table_blocks(table: Table) -> list[ContentBlockRecord]:
    """Return the `table` blocks of the current snapshot, validated."""

    rows = table.scan(row_filter="block_type == 'table'").to_arrow()
    return [ContentBlockRecord.model_validate(row) for row in rows.to_pylist()]

from __future__ import annotations

from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.content_blocks import (
    CONTENT_BLOCKS_ARROW_SCHEMA,
)
from src.data_pipeline.storage.iceberg import (
    SilverTableSpec,
    iter_rows,
)
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

CONTENT_BLOCKS_TABLE = "sec_filings.content_blocks"


CONTENT_BLOCKS_SPEC = SilverTableSpec(
    CONTENT_BLOCKS_TABLE, CONTENT_BLOCKS_ARROW_SCHEMA, FILING_YEAR_PARTITION
)


def count_table_cell_blocks(table: Table) -> int:
    """Count rows stored with the retired `table_cell` type (one block per td/th).

    Such rows predate atomic `table` blocks; their ordinals, IDs and section
    ranges differ from what the current parser builds.
    """

    return sum(
        row["block_type"] == "table_cell" for row in iter_rows(table, ("block_type",))
    )

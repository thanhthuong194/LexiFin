from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.data_pipeline.pipelines.sec_filings import document_tables

pytestmark = pytest.mark.unit


def _batches(table_blocks: list[int]) -> list[list[int]]:
    """Batches, as document indexes, of documents with these table counts."""

    parents: list[Any] = [
        SimpleNamespace(document_id=str(i)) for i in range(len(table_blocks))
    ]
    blocks: dict[str, Any] = {str(i): [None] * n for i, n in enumerate(table_blocks)}
    return [
        [int(parent.document_id) for parent in batch]
        for batch in document_tables._batches(parents, blocks)
    ]


def test_batches_stop_before_the_table_block_limit() -> None:
    limit = document_tables.TABLE_BLOCKS_PER_COMMIT
    third = limit // 3

    assert _batches([third, third, third, third, 0]) == [[0, 1, 2], [3, 4]]
    # A document over the limit is a batch of its own.
    assert _batches([1, limit + 1, 1]) == [[0], [1], [2]]


def test_batches_hold_at_most_the_document_limit() -> None:
    documents = document_tables.DOCUMENTS_PER_COMMIT

    assert [len(batch) for batch in _batches([0] * (documents + 1))] == [documents, 1]
    assert _batches([]) == []

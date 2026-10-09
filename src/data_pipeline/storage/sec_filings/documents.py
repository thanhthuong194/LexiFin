from __future__ import annotations

from pathlib import Path

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.documents import (
    DOCUMENTS_ARROW_SCHEMA,
    DocumentRecord,
)
from src.data_pipeline.storage.bronze import SecBronze, sha256_bytes
from src.data_pipeline.storage.iceberg import (
    SilverTableSpec,
    iter_rows,
)
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

DOCUMENTS_TABLE = "sec_filings.documents"


DOCUMENTS_SPEC = SilverTableSpec(
    DOCUMENTS_TABLE, DOCUMENTS_ARROW_SCHEMA, FILING_YEAR_PARTITION
)


def open_documents_table(catalog: Catalog, data_dir: Path) -> tuple[Table, bool]:
    """Load or create `sec_filings.documents`, returning whether it was created."""

    return DOCUMENTS_SPEC.open(catalog, data_dir)


def read_primary_html_documents(table: Table) -> list[DocumentRecord]:
    """Return the primary HTML documents of the current snapshot, validated."""

    return [
        DocumentRecord.model_validate(row)
        for row in iter_rows(table)
        if (row["role"], row["content_kind"]) == ("primary_document", "html")
    ]


def read_document_content(parent: DocumentRecord, bronze: SecBronze) -> bytes:
    """Read a parent document's exact `<TEXT>` slice by seek/read and verify it.

    The source path must stay inside the Bronze root; offsets must lie within the
    file, and the slice must match the recorded size and SHA-256. Raises
    OSError or ValueError otherwise.
    """

    start, end = parent.content_start_byte, parent.content_end_byte
    if start is None or end is None:
        raise ValueError("Parent document has no <TEXT> content offsets")
    path = bronze.path(parent.source_path)
    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError("Source submission is empty")
    if not 0 <= start < end <= file_size:
        raise ValueError(f"Content offsets {start}..{end} outside {file_size} bytes")
    with path.open("rb") as source_file:
        source_file.seek(start)
        content = source_file.read(end - start)
    if len(content) != parent.content_size_bytes:
        raise ValueError(
            f"Content slice has {len(content)} bytes, not the recorded size"
        )
    if sha256_bytes(content) != parent.content_sha256:
        raise ValueError("Content SHA-256 differs from sec_filings.documents")
    return content

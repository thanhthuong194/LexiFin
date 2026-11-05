from __future__ import annotations

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


# Exhibits whose text the narrative and tables stages read besides the primary
# document: EX-99 (press releases, investor material) carries an 8-K's content.
NARRATIVE_EXHIBIT_TYPE = "EX-99"
# The same rule over a `documents` row aliased `d`, for the integrity audit.
NARRATIVE_DOCUMENT_SQL = (
    "(d.content_kind = 'html' AND (d.role = 'primary_document' OR "
    f"(d.role = 'exhibit' AND upper(d.document_type) LIKE '{NARRATIVE_EXHIBIT_TYPE}%')))"
)


def is_narrative_document(role: str, content_kind: str, document_type: str) -> bool:
    """Whether a document's HTML text belongs in sections and content blocks."""

    return content_kind == "html" and (
        role == "primary_document"
        or (
            role == "exhibit"
            and document_type.upper().startswith(NARRATIVE_EXHIBIT_TYPE)
        )
    )


def read_narrative_documents(table: Table) -> list[DocumentRecord]:
    """Return the narrative documents of the current snapshot, validated."""

    return [
        DocumentRecord.model_validate(row)
        for row in iter_rows(table)
        if is_narrative_document(row["role"], row["content_kind"], row["document_type"])
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

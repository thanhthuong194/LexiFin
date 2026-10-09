from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.data_pipeline.contracts.sec_filings.documents import document_id_for
from src.data_pipeline.contracts.sec_filings.filings import FilingRecord


class DocumentGroupMember(Protocol):
    """The document fields a filing's inventory is judged by."""

    @property
    def document_id(self) -> str: ...

    @property
    def document_ordinal(self) -> int: ...

    @property
    def role(self) -> str: ...

    @property
    def source_submission_sha256(self) -> str: ...


def documents_problem(
    parent: FilingRecord, group: Sequence[DocumentGroupMember]
) -> str | None:
    """Return why a filing's documents are not one complete inventory, or None."""

    ordinals = sorted(member.document_ordinal for member in group)
    if ordinals != list(range(1, len(group) + 1)):
        return "document ordinals are not exactly 1..N"
    if any(
        member.document_id != document_id_for(parent.filing_id, member.document_ordinal)
        for member in group
    ):
        return "document_id does not match filing_id and ordinal"
    if {member.source_submission_sha256 for member in group} != {
        parent.full_submission_sha256
    }:
        return "source submission SHA-256 differs from sec_filings.filings"
    primaries = sum(member.role == "primary_document" for member in group)
    if primaries != 1:
        return f"expected one primary_document, found {primaries}"
    return None

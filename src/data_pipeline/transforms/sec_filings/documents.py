from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath

from src.data_pipeline.contracts.sec_filings.documents import (
    ContentKind,
    DocumentRecord,
    DocumentRole,
    document_id_for,
)
from src.data_pipeline.contracts.sec_filings.filings import FilingRecord
from src.data_pipeline.storage.bronze import sha256_bytes

DOCUMENT_START_TAG = b"<DOCUMENT>"
DOCUMENT_END_TAG = b"</DOCUMENT>"
TEXT_START_TAG = b"<TEXT>"
TEXT_END_TAG = b"</TEXT>"
_BLOCK_HEADER_TAGS = frozenset({"TYPE", "SEQUENCE", "FILENAME", "DESCRIPTION"})

# Enough content to see past EDGAR's <XBRL>/<XML> wrapper and the XML prolog.
_XML_SNIFF_BYTES = 4096
_XML_START_TAG = re.compile(rb"<(?![?!/])([A-Za-z_][\w.:-]*)")
_EDGAR_CONTENT_WRAPPERS = frozenset({b"XBRL", b"XML"})

_XBRL_ROLE_BY_TYPE: dict[str, DocumentRole] = {
    "EX-101.SCH": "xbrl_schema",
    "EX-101.CAL": "xbrl_calculation",
    "EX-101.DEF": "xbrl_definition",
    "EX-101.LAB": "xbrl_label",
    "EX-101.PRE": "xbrl_presentation",
    "EX-101.INS": "xbrl_instance",
}
# EDGAR files the documents it renders itself (R*.htm pages, FilingSummary.xml,
# MetaLinks.json, report.css, Show.js, Financial_Report.xlsx, the XBRL zip)
# under these TYPE values; filer-submitted documents use form, EX-* or GRAPHIC.
_SEC_GENERATED_TYPES = frozenset({"XML", "JSON", "ZIP", "EXCEL"})
_UNOFFICIAL_COPY_SUFFIX = ".pdf"

_CONTENT_KIND_BY_EXTENSION: dict[str, ContentKind] = {
    ".htm": "html",
    ".html": "html",
    ".xml": "xml",
    ".xsd": "xml",
    ".json": "json",
    ".txt": "text",
    ".css": "text",
    ".js": "text",
    ".jpg": "graphic",
    ".jpeg": "graphic",
    ".png": "graphic",
    ".gif": "graphic",
    ".svg": "graphic",
    ".zip": "binary",
    ".pdf": "binary",
    ".xlsx": "binary",
}


@dataclass(frozen=True)
class SubmissionDocument:
    """Header fields and byte spans of one `<DOCUMENT>` block in raw bytes.

    Spans follow DocumentRecord: zero-based, start inclusive, end exclusive.
    """

    ordinal: int
    document_type: str
    sequence: str | None
    filename: str | None
    description: str | None
    document_start: int
    document_end: int
    content_start: int | None
    content_end: int | None


def _is_line_start(raw: bytes, position: int) -> bool:
    return position == 0 or raw[position - 1] == ord("\n")


def _find_line_tag(raw: bytes, tag: bytes, start: int, end: int) -> int:
    """Return the first `tag` beginning a line within raw[start:end], or -1."""

    position = raw.find(tag, start, end)
    while position != -1 and not _is_line_start(raw, position):
        position = raw.find(tag, position + 1, end)
    return position


def _rfind_line_tag(raw: bytes, tag: bytes, start: int, end: int) -> int:
    """Return the last `tag` beginning a line within raw[start:end], or -1."""

    position = raw.rfind(tag, start, end)
    while position != -1 and not _is_line_start(raw, position):
        position = raw.rfind(tag, start, position)
    return position


def _block_header_fields(header: bytes, ordinal: int) -> dict[str, str]:
    """Parse `<TYPE>value` style lines; empty values count as absent."""

    fields: dict[str, str] = {}
    for raw_line in header.splitlines():
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("<"):
            continue
        name, _, value = line[1:].partition(">")
        value = value.strip()
        if name not in _BLOCK_HEADER_TAGS or not value:
            continue
        if name in fields:
            raise ValueError(f"Document {ordinal}: repeated <{name}>")
        fields[name] = value
    return fields


def _parse_block(
    raw: bytes, ordinal: int, start: int, end_tag: int
) -> SubmissionDocument:
    text_tag = _find_line_tag(raw, TEXT_START_TAG, start, end_tag)
    header_end = end_tag if text_tag == -1 else text_tag
    fields = _block_header_fields(
        raw[start + len(DOCUMENT_START_TAG) : header_end], ordinal
    )
    if "TYPE" not in fields:
        raise ValueError(f"Document {ordinal}: missing <TYPE>")

    content_start = content_end = None
    if text_tag != -1:
        content_start = text_tag + len(TEXT_START_TAG)
        content_end = raw.rfind(TEXT_END_TAG, content_start, end_tag)
        if content_end == -1:
            raise ValueError(f"Document {ordinal}: <TEXT> has no closing </TEXT>")
    return SubmissionDocument(
        ordinal=ordinal,
        document_type=fields["TYPE"],
        sequence=fields.get("SEQUENCE"),
        filename=fields.get("FILENAME"),
        description=fields.get("DESCRIPTION"),
        document_start=start,
        document_end=end_tag + len(DOCUMENT_END_TAG),
        content_start=content_start,
        content_end=content_end,
    )


def split_submission_documents(raw: bytes) -> list[SubmissionDocument]:
    """Split raw full-submission bytes into its `<DOCUMENT>` blocks, in order.

    Block tags count only at the start of a line (LF or CRLF files), and a
    block ends at the last `</DOCUMENT>` line before the next `<DOCUMENT>`
    line, so tag text inside content cannot split a block. Raises ValueError
    naming the block ordinal when a block is unterminated or has no <TYPE>.
    """

    documents: list[SubmissionDocument] = []
    start = _find_line_tag(raw, DOCUMENT_START_TAG, 0, len(raw))
    while start != -1:
        ordinal = len(documents) + 1
        next_start = _find_line_tag(
            raw, DOCUMENT_START_TAG, start + len(DOCUMENT_START_TAG), len(raw)
        )
        block_limit = len(raw) if next_start == -1 else next_start
        end_tag = _rfind_line_tag(raw, DOCUMENT_END_TAG, start, block_limit)
        if end_tag == -1:
            raise ValueError(f"Document {ordinal}: missing </DOCUMENT>")
        documents.append(_parse_block(raw, ordinal, start, end_tag))
        start = next_start
    if not documents:
        raise ValueError("Submission contains no <DOCUMENT> blocks")
    return documents


def classify_content_kind(filename: str | None) -> ContentKind:
    """Classify a document's content by its filename extension."""

    if filename is None:
        return "unknown"
    extension = PurePosixPath(filename).suffix.lower()
    return _CONTENT_KIND_BY_EXTENSION.get(extension, "unknown")


def _xml_root_name(content_head: bytes) -> str | None:
    """Return the local name of the first XML element after EDGAR wrappers."""

    for match in _XML_START_TAG.finditer(content_head):
        name = match.group(1)
        if name not in _EDGAR_CONTENT_WRAPPERS:
            return name.decode("ascii").rsplit(":", maxsplit=1)[-1]
    return None


def classify_role(
    *,
    document_type: str,
    filename: str | None,
    content_kind: ContentKind,
    content_head: bytes,
    is_primary: bool,
) -> DocumentRole:
    """Classify a document's semantic role; the first matching rule wins.

    `content_head` is the start of the `<TEXT>` content and is only sniffed
    for XML documents, to recognise XBRL instances.
    """

    if is_primary:
        return "primary_document"
    xbrl_role = _XBRL_ROLE_BY_TYPE.get(document_type)
    if xbrl_role is not None:
        return xbrl_role
    if content_kind == "xml" and (
        (filename or "").lower().endswith("_htm.xml")
        or _xml_root_name(content_head) == "xbrl"
    ):
        return "xbrl_instance"
    if document_type.startswith("EX-"):
        return "exhibit"
    if document_type == "GRAPHIC" or content_kind == "graphic":
        return "graphic"
    return "generated" if document_type in _SEC_GENERATED_TYPES else "other"


def build_document_records(
    parent: FilingRecord,
    raw: bytes,
    processed_at: datetime,
) -> list[DocumentRecord]:
    """Describe every `<DOCUMENT>` block of a filing's full submission.

    `raw` must be the parent's full-submission bytes, already verified against
    `parent.full_submission_sha256`. The block whose TYPE equals the parent
    form is the primary document; there must be exactly one such block apart
    from PDFs. EDGAR accepts a PDF only as an unofficial copy and lets it carry
    the form's TYPE (e.g. Caterpillar's 10-Q 0000018230-21-000260), so a PDF is
    never the primary document.
    """

    documents = split_submission_documents(raw)
    candidates = [
        doc
        for doc in documents
        if doc.document_type == parent.form
        and PurePosixPath(doc.filename or "").suffix.lower() != _UNOFFICIAL_COPY_SUFFIX
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one primary document of TYPE {parent.form!r}; "
            f"found {len(candidates)}"
        )

    records: list[DocumentRecord] = []
    for document in documents:
        content = (
            None
            if document.content_start is None
            else raw[document.content_start : document.content_end]
        )
        content_kind = classify_content_kind(document.filename)
        records.append(
            DocumentRecord(
                document_id=document_id_for(parent.filing_id, document.ordinal),
                filing_id=parent.filing_id,
                filing_date=parent.filing_date,
                document_ordinal=document.ordinal,
                sequence=document.sequence,
                document_type=document.document_type,
                filename=document.filename,
                description=document.description,
                role=classify_role(
                    document_type=document.document_type,
                    filename=document.filename,
                    content_kind=content_kind,
                    content_head=(content or b"")[:_XML_SNIFF_BYTES],
                    is_primary=document is candidates[0],
                ),
                content_kind=content_kind,
                source_path=parent.full_submission_path,
                source_submission_sha256=parent.full_submission_sha256,
                document_start_byte=document.document_start,
                document_end_byte=document.document_end,
                content_start_byte=document.content_start,
                content_end_byte=document.content_end,
                content_size_bytes=0 if content is None else len(content),
                content_sha256=None if content is None else sha256_bytes(content),
                processed_at=processed_at,
            )
        )
    return records


def check_document_offsets(records: Sequence[DocumentRecord], raw: bytes) -> None:
    """Re-read each record's spans from `raw` and check tags and content SHA-256.

    Records must be in ordinal order; spans must be in bounds and must not
    overlap. Raises ValueError naming the first offending ordinal.
    """

    previous_end = 0
    for record in records:
        where = f"Document {record.document_ordinal}"
        start, end = record.document_start_byte, record.document_end_byte
        if not previous_end <= start < end <= len(raw):
            raise ValueError(f"{where}: span {start}..{end} out of order or bounds")
        if not (
            raw.startswith(DOCUMENT_START_TAG, start)
            and raw.endswith(DOCUMENT_END_TAG, start, end)
        ):
            raise ValueError(f"{where}: span does not cover <DOCUMENT>...</DOCUMENT>")
        content_start, content_end = record.content_start_byte, record.content_end_byte
        if content_start is not None and content_end is not None:
            if not (
                raw.endswith(TEXT_START_TAG, start, content_start)
                and raw.startswith(TEXT_END_TAG, content_end)
            ):
                raise ValueError(f"{where}: content is not delimited by <TEXT> tags")
            if sha256_bytes(raw[content_start:content_end]) != record.content_sha256:
                raise ValueError(f"{where}: content SHA-256 does not match its slice")
        previous_end = end

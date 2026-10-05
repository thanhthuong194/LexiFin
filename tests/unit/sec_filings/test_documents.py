from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime

import pyarrow as pa
import pytest
from pydantic import ValidationError

from src.data_pipeline.contracts.sec_filings.documents import (
    DOCUMENTS_ARROW_SCHEMA,
    DocumentRecord,
)
from src.data_pipeline.contracts.sec_filings.filings import FilingRecord
from src.data_pipeline.transforms.sec_filings.documents import (
    build_document_records,
    check_document_offsets,
    classify_content_kind,
    classify_role,
    split_submission_documents,
)

pytestmark = pytest.mark.unit

FILING_ID = "sec:0000320193-25-000079"
SOURCE_PATH = (
    "bronze/sec_filings/sec-edgar-filings/0000320193/10-K/"
    "0000320193-25-000079/full-submission.txt"
)
PROCESSED_AT = datetime(2026, 10, 5, 4, 0, tzinfo=UTC)
XBRL_INSTANCE_TEXT = (
    "\n<XBRL>\n<?xml version='1.0' encoding='ASCII'?>\n"
    "<!--Created by hand-->\n"
    '<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"/>\n</XBRL>\n'
)


def _block(*header_lines: str, text: str | None) -> str:
    lines = ["<DOCUMENT>", *header_lines]
    if text is not None:
        lines.append(f"<TEXT>{text}</TEXT>")
    lines.append("</DOCUMENT>")
    return "\n".join(lines)


# (block text, expected <TEXT> content) in submission order.
BLOCKS = [
    (
        _block(
            "<TYPE>10-K",
            "<SEQUENCE>1",
            "<FILENAME>aapl-20250927.htm",
            "<DESCRIPTION>10-K",
            text="\n<html>annual report</html>\n",
        ),
        "\n<html>annual report</html>\n",
    ),
    (
        _block(
            "<TYPE>EX-21.1",
            "<SEQUENCE>2",
            "<FILENAME>ex21.htm",
            text="\n<html>subsidiaries</html>\n",
        ),
        "\n<html>subsidiaries</html>\n",
    ),
    (
        _block("<TYPE>GRAPHIC", text="\nbegin 644 x\nM``\nend\n"),
        "\nbegin 644 x\nM``\nend\n",
    ),
    (
        _block(
            "<TYPE>EX-101.SCH",
            "<SEQUENCE>4",
            "<FILENAME>aapl-20250927.xsd",
            "<DESCRIPTION>XBRL TAXONOMY EXTENSION SCHEMA",
            text="\n<XBRL>\n<xs:schema/>\n</XBRL>\n",
        ),
        "\n<XBRL>\n<xs:schema/>\n</XBRL>\n",
    ),
    (
        _block(
            "<TYPE>XML",
            "<SEQUENCE>5",
            "<FILENAME>instance.xml",
            text=XBRL_INSTANCE_TEXT,
        ),
        XBRL_INSTANCE_TEXT,
    ),
    (
        _block(
            "<TYPE>ZIP",
            "<SEQUENCE>7",
            "<FILENAME>0000320193-25-000079-xbrl.zip",
            text=None,
        ),
        None,
    ),
]


def _submission(newline: str = "\n") -> bytes:
    header = (
        "<SEC-DOCUMENT>x.txt\n<SEC-HEADER>\nFILED AS OF DATE:\t20251031\n</SEC-HEADER>"
    )
    text = "\n".join([header, *(block for block, _ in BLOCKS), "</SEC-DOCUMENT>\n"])
    return text.replace("\n", newline).encode()


def _parent(raw: bytes, form: str = "10-K") -> FilingRecord:
    return FilingRecord(
        filing_id=FILING_ID,
        accession_number="0000320193-25-000079",
        cik="0000320193",
        form=form,
        filing_date=date(2025, 10, 31),
        report_date=None,
        accepted_at=None,
        public_document_count=None,
        full_submission_path=SOURCE_PATH,
        primary_document_path=None,
        full_submission_sha256=hashlib.sha256(raw).hexdigest(),
        primary_document_sha256=None,
        ingestion_run_id=None,
        processed_at=PROCESSED_AT,
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["LF", "CRLF"])
def test_split_finds_every_block_with_exact_offsets(newline: str) -> None:
    raw = _submission(newline)

    documents = split_submission_documents(raw)

    assert [doc.ordinal for doc in documents] == [1, 2, 3, 4, 5, 6]
    assert [doc.document_type for doc in documents] == [
        "10-K",
        "EX-21.1",
        "GRAPHIC",
        "EX-101.SCH",
        "XML",
        "ZIP",
    ]
    for document, (block, content) in zip(documents, BLOCKS, strict=True):
        exact_block = block.replace("\n", newline).encode()
        assert raw[document.document_start : document.document_end] == exact_block
        if content is None:
            assert (document.content_start, document.content_end) == (None, None)
        else:
            exact_content = content.replace("\n", newline).encode()
            assert raw[document.content_start : document.content_end] == exact_content
    graphic, exhibit = documents[2], documents[1]
    assert (graphic.sequence, graphic.filename, graphic.description) == (
        None,
        None,
        None,
    )
    assert (exhibit.sequence, exhibit.filename) == ("2", "ex21.htm")
    assert exhibit.description is None


def test_build_document_records_derives_ids_checksums_roles_and_kinds() -> None:
    raw = _submission()
    parent = _parent(raw)

    records = build_document_records(parent, raw, PROCESSED_AT)

    assert [record.document_id for record in records] == [
        f"{FILING_ID}:document:000{ordinal}" for ordinal in range(1, 7)
    ]
    assert [record.role for record in records] == [
        "primary_document",
        "exhibit",
        "graphic",
        "xbrl_schema",
        "xbrl_instance",
        "generated",
    ]
    assert [record.content_kind for record in records] == [
        "html",
        "html",
        "unknown",
        "xml",
        "xml",
        "binary",
    ]
    primary = records[0]
    content = b"\n<html>annual report</html>\n"
    assert primary.content_size_bytes == len(content)
    assert primary.content_sha256 == hashlib.sha256(content).hexdigest()
    without_text = records[5]
    assert (without_text.content_start_byte, without_text.content_end_byte) == (
        None,
        None,
    )
    assert (without_text.content_size_bytes, without_text.content_sha256) == (0, None)
    assert {record.source_path for record in records} == {SOURCE_PATH}
    assert {record.source_submission_sha256 for record in records} == {
        parent.full_submission_sha256
    }
    assert {record.filing_date for record in records} == {date(2025, 10, 31)}
    assert build_document_records(parent, raw, PROCESSED_AT) == records
    check_document_offsets(records, raw)


@pytest.mark.parametrize(
    ("document_type", "filename", "content_head", "is_primary", "role"),
    [
        ("EX-99.1", "ex99.htm", b"", True, "primary_document"),
        ("EX-101.CAL", "x_cal.xml", b"", False, "xbrl_calculation"),
        ("EX-101.DEF", "x_def.xml", b"", False, "xbrl_definition"),
        ("EX-101.LAB", "x_lab.xml", b"", False, "xbrl_label"),
        ("EX-101.PRE", "x_pre.xml", b"", False, "xbrl_presentation"),
        ("EX-101.INS", "x.xml", b"", False, "xbrl_instance"),
        ("XML", "aapl-20250927_htm.xml", b"", False, "xbrl_instance"),
        ("XML", "instance.xml", XBRL_INSTANCE_TEXT.encode(), False, "xbrl_instance"),
        (
            "XML",
            "FilingSummary.xml",
            b"\n<XML>\n<?xml version='1.0'?>\n<FilingSummary>",
            False,
            "generated",
        ),
        ("EX-99.1", "ex99.htm", b"", False, "exhibit"),
        ("EX-99.1", "chart.jpg", b"", False, "exhibit"),
        ("GRAPHIC", None, b"", False, "graphic"),
        ("COVER", "cover.png", b"", False, "graphic"),
        ("XML", "R2.htm", b"", False, "generated"),
        ("JSON", "MetaLinks.json", b"", False, "generated"),
        ("EXCEL", "Financial_Report.xlsx", b"", False, "generated"),
        ("CORRESP", "letter.txt", b"", False, "other"),
    ],
)
def test_classify_role_applies_rules_in_precedence_order(
    document_type: str,
    filename: str | None,
    content_head: bytes,
    is_primary: bool,
    role: str,
) -> None:
    assert (
        classify_role(
            document_type=document_type,
            filename=filename,
            content_kind=classify_content_kind(filename),
            content_head=content_head,
            is_primary=is_primary,
        )
        == role
    )


@pytest.mark.parametrize(
    ("filename", "kind"),
    [
        ("a.htm", "html"),
        ("A.HTML", "html"),
        ("a.xml", "xml"),
        ("a.xsd", "xml"),
        ("MetaLinks.json", "json"),
        ("a.txt", "text"),
        ("report.css", "text"),
        ("Show.js", "text"),
        ("a.jpg", "graphic"),
        ("a.jpeg", "graphic"),
        ("a.png", "graphic"),
        ("a.gif", "graphic"),
        ("a.svg", "graphic"),
        ("a.zip", "binary"),
        ("a.pdf", "binary"),
        ("Financial_Report.xlsx", "binary"),
        ("a.bin", "unknown"),
        ("README", "unknown"),
        (None, "unknown"),
    ],
)
def test_classify_content_kind_uses_filename_extension(
    filename: str | None, kind: str
) -> None:
    assert classify_content_kind(filename) == kind


@pytest.mark.parametrize(
    ("submission", "message"),
    [
        (
            _block("<TYPE>10-K", text="a") + "\n" + _block("<SEQUENCE>2", text="b"),
            "Document 2: missing <TYPE>",
        ),
        ("<DOCUMENT>\n<TYPE>10-K\n<TEXT>a</TEXT>\n", "Document 1: missing </DOCUMENT>"),
        ("<DOCUMENT>\n<TYPE>10-K\n<TEXT>a\n</DOCUMENT>\n", "Document 1: <TEXT> has no"),
        ("<SEC-DOCUMENT>\n</SEC-DOCUMENT>\n", "no <DOCUMENT> blocks"),
    ],
)
def test_split_rejects_malformed_blocks_naming_the_ordinal(
    submission: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        split_submission_documents(submission.encode())


@pytest.mark.parametrize("primary_types", [["EX-99.1"], ["10-K", "10-K"]])
def test_build_requires_exactly_one_primary_document(primary_types: list[str]) -> None:
    raw = "\n".join(
        _block(f"<TYPE>{document_type}", text="x") for document_type in primary_types
    ).encode()

    with pytest.raises(ValueError, match="exactly one primary document"):
        build_document_records(_parent(raw), raw, PROCESSED_AT)


def test_check_document_offsets_rejects_a_shifted_content_span() -> None:
    raw = _submission()
    records = build_document_records(_parent(raw), raw, PROCESSED_AT)
    shifted = records[1].model_copy(
        update={"content_start_byte": records[1].content_start_byte + 1}
    )

    with pytest.raises(ValueError, match="Document 2: content is not delimited"):
        check_document_offsets([records[0], shifted], raw)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"document_id": f"{FILING_ID}:document:0002"}, "document_id must be derived"),
        ({"content_size_bytes": 3}, "content_size_bytes must equal"),
        ({"content_sha256": None}, "must all be set or all null"),
        ({"content_start_byte": 0}, "content offsets must lie inside"),
    ],
)
def test_document_record_rejects_inconsistent_fields(
    update: dict[str, object], message: str
) -> None:
    raw = _submission()
    record = build_document_records(_parent(raw), raw, PROCESSED_AT)[0]

    with pytest.raises(ValidationError, match=message):
        DocumentRecord.model_validate({**record.model_dump(), **update})


def test_records_match_exact_documents_arrow_schema() -> None:
    raw = _submission()
    records = build_document_records(_parent(raw), raw, PROCESSED_AT)

    rows = pa.Table.from_pylist(
        [record.model_dump() for record in records], schema=DOCUMENTS_ARROW_SCHEMA
    )

    assert list(DocumentRecord.model_fields) == DOCUMENTS_ARROW_SCHEMA.names
    assert rows.schema.equals(DOCUMENTS_ARROW_SCHEMA)
    assert [(f.name, str(f.type), f.nullable) for f in DOCUMENTS_ARROW_SCHEMA] == [
        ("document_id", "string", False),
        ("filing_id", "string", False),
        ("filing_date", "date32[day]", False),
        ("document_ordinal", "int32", False),
        ("sequence", "string", True),
        ("document_type", "string", False),
        ("filename", "string", True),
        ("description", "string", True),
        ("role", "string", False),
        ("content_kind", "string", False),
        ("source_path", "string", False),
        ("source_submission_sha256", "string", False),
        ("document_start_byte", "int64", False),
        ("document_end_byte", "int64", False),
        ("content_start_byte", "int64", True),
        ("content_end_byte", "int64", True),
        ("content_size_bytes", "int64", False),
        ("content_sha256", "string", True),
        ("processed_at", "timestamp[us, tz=UTC]", False),
    ]

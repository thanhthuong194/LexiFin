from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pytest

from src.data_pipeline.contracts.sec_filings.filing_entities import (
    FILING_ENTITIES_ARROW_SCHEMA,
    FilingEntityRecord,
)
from src.data_pipeline.contracts.sec_filings.filings import FilingRecord
from src.data_pipeline.transforms.sec_filings.filing_entities import (
    build_filing_entity_records,
    check_entity_offsets,
    parse_entity_blocks,
)
from src.data_pipeline.transforms.sec_filings.filings import (
    RawSecHeader,
    read_raw_sec_header,
)

pytestmark = pytest.mark.unit

FILING_ID = "sec:0000320193-25-000079"
SOURCE_PATH = (
    "bronze/sec_filings/sec-edgar-filings/0000320193/8-K/"
    "0000320193-25-000079/full-submission.txt"
)
PROCESSED_AT = datetime(2026, 10, 5, 6, 0, tzinfo=UTC)

SUBMISSION_FIELDS = """<ACCEPTANCE-DATETIME>20251031060126
ACCESSION NUMBER:\t\t0000320193-25-000079
CONFORMED SUBMISSION TYPE:\t8-K
ITEM INFORMATION:\t\tResults of Operations and Financial Condition
FILED AS OF DATE:\t\t20251031

"""
FILER_BLOCK = """FILER:

\tCOMPANY DATA:\t
\t\tCOMPANY CONFORMED NAME:\t\t\tApple Inc.
\t\tCENTRAL INDEX KEY:\t\t\t0000320193
\t\tSTANDARD INDUSTRIAL CLASSIFICATION:\tELECTRONIC COMPUTERS [3571]
\t\tORGANIZATION NAME:           \t06 Technology
\t\tEIN:\t\t\t\t942404110
\t\tSTATE OF INCORPORATION:\t\t\tCA
\t\tFISCAL YEAR END:\t\t\t0927

\tFILING VALUES:
\t\tFORM TYPE:\t\t8-K
\t\tSEC ACT:\t\t1934 Act
\t\tSEC FILE NUMBER:\t001-36743
\t\tFILM NUMBER:\t\t251451234

\tBUSINESS ADDRESS:\t
\t\tSTREET 1:\t\tONE APPLE PARK WAY
\t\tSTREET 2:\t\tBUILDING 1
\t\tCITY:\t\t\tCUPERTINO
\t\tSTATE:\t\t\tCA
\t\tZIP:\t\t\t95014
\t\tBUSINESS PHONE:\t\t(408) 996-1010

\tMAIL ADDRESS:\t
\t\tSTREET 1:\t\tONE APPLE PARK WAY
\t\tCITY:\t\t\tCUPERTINO
\t\tSTATE OR COUNTRY:\t\tCA
\t\tZIP:\t\t\t95014

\tFORMER COMPANY:\t
\t\tFORMER CONFORMED NAME:\tAPPLE INC
\t\tDATE OF NAME CHANGE:\t20070109

\tFORMER COMPANY:\t
\t\tFORMER CONFORMED NAME:\tAPPLE COMPUTER INC
\t\tDATE OF NAME CHANGE:\t19970808

"""
SUBJECT_BLOCK = """SUBJECT COMPANY:

\tCOMPANY DATA:\t
\t\tCOMPANY CONFORMED NAME:\t\t\tTarget Co
\t\tCENTRAL INDEX KEY:\t\t\t789019
\t\tSTANDARD INDUSTRIAL CLASSIFICATION:\t[7372]

\tBUSINESS ADDRESS:\t
\t\tPROVINCE COUNTRY:\t\tA1

"""
FILED_BY_BLOCK = """FILED-BY:

\tOWNER DATA:\t
\t\tCOMPANY CONFORMED NAME:\t\t\tInvestor LP
\t\tCENTRAL INDEX KEY:\t\t\t0001045810
"""
BLOCKS = [FILER_BLOCK, SUBJECT_BLOCK, FILED_BY_BLOCK]


def _submission(blocks: list[str] = BLOCKS, newline: str = "\n") -> bytes:
    text = (
        "<SEC-DOCUMENT>0000320193-25-000079.txt : 20251031\n"
        "<SEC-HEADER>0000320193-25-000079.hdr.sgml : 20251031\n"
        + SUBMISSION_FIELDS
        + "".join(blocks)
        + "</SEC-HEADER>\n"
        # An entity label inside a document body must not become an entity.
        + "<DOCUMENT>\n<TYPE>8-K\n<TEXT>\nFILER:\n</TEXT>\n</DOCUMENT>\n"
        + "</SEC-DOCUMENT>\n"
    )
    return text.replace("\n", newline).encode()


def _header(tmp_path: Path, raw: bytes) -> RawSecHeader:
    path = tmp_path / "full-submission.txt"
    path.write_bytes(raw)
    return read_raw_sec_header(path)


def _parent(form: str = "8-K", cik: str = "0000320193") -> FilingRecord:
    return FilingRecord(
        filing_id=FILING_ID,
        accession_number="0000320193-25-000079",
        cik=cik,
        form=form,
        filing_date=date(2025, 10, 31),
        report_date=None,
        accepted_at=None,
        public_document_count=None,
        full_submission_path=SOURCE_PATH,
        primary_document_path=None,
        full_submission_sha256="a" * 64,
        primary_document_sha256=None,
        ingestion_run_id=None,
        processed_at=PROCESSED_AT,
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["LF", "CRLF"])
def test_parse_finds_every_role_block_with_exact_spans(
    tmp_path: Path, newline: str
) -> None:
    raw = _submission(newline=newline)
    header = _header(tmp_path, raw)

    blocks = parse_entity_blocks(header)

    assert [block.ordinal for block in blocks] == [1, 2, 3]
    assert [block.source_role for block in blocks] == [
        "FILER",
        "SUBJECT COMPANY",
        "FILED-BY",
    ]
    assert [block.entity_role for block in blocks] == [
        "filer",
        "subject_company",
        "filed_by",
    ]
    for block, text in zip(blocks, BLOCKS, strict=True):
        exact_block = text.replace("\n", newline).encode()
        assert raw[block.start_byte : block.end_byte] == exact_block
    assert raw[blocks[-1].end_byte :].startswith(b"</SEC-HEADER>")
    assert "FORMER CONFORMED NAME" not in str(blocks[0].values)


def test_build_records_parses_sections_and_normalises_values(tmp_path: Path) -> None:
    raw = _submission()
    header = _header(tmp_path, raw)

    records = build_filing_entity_records(_parent(), header, PROCESSED_AT)

    assert [record.filing_entity_id for record in records] == [
        f"{FILING_ID}:entity:0001",
        f"{FILING_ID}:entity:0002",
        f"{FILING_ID}:entity:0003",
    ]
    filer, subject, filed_by = records
    assert filer.model_dump(
        exclude={
            "filing_entity_id",
            "entity_start_byte",
            "entity_end_byte",
            "entity_block_sha256",
            "processed_at",
        }
    ) == {
        "filing_id": FILING_ID,
        "filing_date": date(2025, 10, 31),
        "entity_ordinal": 1,
        "entity_role": "filer",
        "source_role": "FILER",
        "cik": "0000320193",
        "company_name": "Apple Inc.",
        "sic": "3571",
        "sic_description": "ELECTRONIC COMPUTERS",
        "organization_name": "06 Technology",
        "ein": "942404110",
        "state_of_incorporation": "CA",
        "fiscal_year_end": "0927",
        "form_type": "8-K",
        "sec_act": "1934 Act",
        "file_number": "001-36743",
        "film_number": "251451234",
        "business_street_1": "ONE APPLE PARK WAY",
        "business_street_2": "BUILDING 1",
        "business_city": "CUPERTINO",
        "business_state_or_country": "CA",
        "business_zip": "95014",
        "business_phone": "(408) 996-1010",
        "mailing_street_1": "ONE APPLE PARK WAY",
        "mailing_street_2": None,
        "mailing_city": "CUPERTINO",
        "mailing_state_or_country": "CA",
        "mailing_zip": "95014",
        "source_path": SOURCE_PATH,
        "source_submission_sha256": "a" * 64,
    }
    assert (subject.cik, subject.sic, subject.sic_description) == (
        "0000789019",
        "7372",
        None,
    )
    assert subject.business_state_or_country == "A1"
    assert (filed_by.entity_role, filed_by.company_name) == ("filed_by", "Investor LP")
    assert filed_by.form_type is None
    for record in records:
        span = raw[record.entity_start_byte : record.entity_end_byte]
        assert record.entity_block_sha256 == hashlib.sha256(span).hexdigest()
    check_entity_offsets(records, header)
    assert build_filing_entity_records(_parent(), header, PROCESSED_AT) == records


@pytest.mark.parametrize(
    ("label", "role"),
    [
        ("FILER", "filer"),
        ("FILED BY", "filed_by"),
        ("FILED-BY", "filed_by"),
        ("SUBJECT COMPANY", "subject_company"),
        ("SUBJECT-COMPANY", "subject_company"),
        ("REPORTING OWNER", "reporting_owner"),
        ("REPORTING-OWNER", "reporting_owner"),
        ("ISSUER", "issuer"),
        ("SERIAL COMPANY", "serial_company"),
        ("SERIAL-COMPANY", "serial_company"),
        ("FILED FOR", "filed_for"),
        ("FILED-FOR", "filed_for"),
    ],
)
def test_role_labels_map_to_normalised_roles(
    tmp_path: Path, label: str, role: str
) -> None:
    raw = _submission([FILER_BLOCK, FILED_BY_BLOCK.replace("FILED-BY", label)])

    blocks = parse_entity_blocks(_header(tmp_path, raw))

    assert (blocks[1].source_role, blocks[1].entity_role) == (label, role)


@pytest.mark.parametrize(
    ("blocks", "parent", "message"),
    [
        (
            [FILER_BLOCK.replace("\t\tCENTRAL INDEX KEY:\t\t\t0000320193\n", "")],
            _parent(),
            r"Entity 1 \(FILER\): missing CENTRAL INDEX KEY",
        ),
        (
            [FILER_BLOCK, FILED_BY_BLOCK.replace("Investor LP", "")],
            _parent(),
            r"Entity 2 \(FILED-BY\): missing COMPANY CONFORMED NAME",
        ),
        (BLOCKS, _parent(form="10-K"), r"Entity 1 \(FILER\): FORM TYPE '8-K' differs"),
        (BLOCKS, _parent(cik="0000789019"), "No FILER block carries the filing CIK"),
        ([SUBJECT_BLOCK, FILED_BY_BLOCK], _parent(), "No FILER block carries"),
        (
            [
                FILER_BLOCK.replace(
                    "ELECTRONIC COMPUTERS [3571]", "ELECTRONIC COMPUTERS"
                )
            ],
            _parent(),
            "Entity 1 .*unparseable STANDARD INDUSTRIAL CLASSIFICATION",
        ),
        (
            [FILER_BLOCK, "MYSTERY ROLE:\n\n"],
            _parent(),
            r"Entity 2 \(MYSTERY ROLE\): unknown entity-role label",
        ),
        (
            [FILER_BLOCK.replace("SEC ACT:", "SEC ACTS:")],
            _parent(),
            "unknown 'SEC ACTS' in FILING VALUES",
        ),
        (
            [FILER_BLOCK + "\tMAIL ADDRESS:\t\n\t\tCITY:\t\t\tPALO ALTO\n"],
            _parent(),
            "repeated MAIL ADDRESS section",
        ),
        (
            [FILER_BLOCK + "ITEM INFORMATION:\t\tOther Events\n"],
            _parent(),
            "unexpected header line",
        ),
        ([], _parent(), "no entity-role blocks"),
    ],
)
def test_build_rejects_invalid_entity_blocks_with_context(
    tmp_path: Path, blocks: list[str], parent: FilingRecord, message: str
) -> None:
    header = _header(tmp_path, _submission(blocks))

    with pytest.raises(ValueError, match=message):
        build_filing_entity_records(parent, header, PROCESSED_AT)


def test_check_entity_offsets_rejects_a_shifted_span(tmp_path: Path) -> None:
    header = _header(tmp_path, _submission())
    records = build_filing_entity_records(_parent(), header, PROCESSED_AT)
    shifted = records[1].model_copy(
        update={"entity_start_byte": records[1].entity_start_byte + 1}
    )

    with pytest.raises(ValueError, match="Entity 2 .*does not start where"):
        check_entity_offsets([records[0], shifted, records[2]], header)


def test_records_match_exact_filing_entities_arrow_schema(tmp_path: Path) -> None:
    header = _header(tmp_path, _submission())
    records = build_filing_entity_records(_parent(), header, PROCESSED_AT)

    rows = pa.Table.from_pylist(
        [record.model_dump() for record in records],
        schema=FILING_ENTITIES_ARROW_SCHEMA,
    )

    assert list(FilingEntityRecord.model_fields) == FILING_ENTITIES_ARROW_SCHEMA.names
    assert rows.schema.equals(FILING_ENTITIES_ARROW_SCHEMA)
    required = {
        "filing_entity_id",
        "filing_id",
        "filing_date",
        "entity_ordinal",
        "entity_role",
        "source_role",
        "cik",
        "company_name",
        "source_path",
        "source_submission_sha256",
        "entity_start_byte",
        "entity_end_byte",
        "entity_block_sha256",
        "processed_at",
    }
    assert {f.name for f in FILING_ENTITIES_ARROW_SCHEMA if not f.nullable} == required
    assert {
        f.name: str(f.type)
        for f in FILING_ENTITIES_ARROW_SCHEMA
        if str(f.type) != "string"
    } == {
        "filing_date": "date32[day]",
        "entity_ordinal": "int32",
        "entity_start_byte": "int64",
        "entity_end_byte": "int64",
        "processed_at": "timestamp[us, tz=UTC]",
    }


def test_newer_header_fields_and_submission_flags_are_tolerated(
    tmp_path: Path,
) -> None:
    # Newer EDGAR headers: a CONFIRMING COPY flag before the first block, an
    # LEI in COMPANY DATA and a non-US flag in the addresses.
    filer = FILER_BLOCK.replace(
        "\t\tFISCAL YEAR END:",
        "\t\tLEGAL ENTITY IDENTIFIER:\t\tHWUPKR0MPOU8FGXBT394\n\t\tFISCAL YEAR END:",
    ).replace("\t\tZIP:", "\t\tADDRESS IS A NON US LOCATION:\tNO\n\t\tZIP:")
    raw = _submission([filer]).replace(
        b"FILED AS OF DATE:", b"CONFIRMING COPY:\t\nFILED AS OF DATE:"
    )

    (record,) = build_filing_entity_records(
        _parent(), _header(tmp_path, raw), PROCESSED_AT
    )

    assert (record.entity_ordinal, record.source_role) == (1, "FILER")
    assert (record.business_zip, record.mailing_zip) == ("95014", "95014")
    assert raw[record.entity_start_byte :].startswith(b"FILER:")

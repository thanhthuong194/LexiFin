from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pytest

from src.data_pipeline.connectors.sec_filings.bronze import discover_filings
from src.data_pipeline.contracts.ingestion import FilingManifestEntry, RawArtifact
from src.data_pipeline.contracts.sec_filings.filings import (
    FILINGS_ARROW_SCHEMA,
    FilingRecord,
)
from src.data_pipeline.transforms.sec_filings.filings import (
    build_filing_record,
    index_manifest,
    normalize_cik,
    parse_sec_header,
    read_sec_header,
)

pytestmark = pytest.mark.unit

CIK = "0000320193"
ACCESSION = "0000320193-25-000079"
PROCESSED_AT = datetime(2026, 10, 4, 8, 0, tzinfo=UTC)


def _only_filing(data_dir: Path):
    (filing,) = discover_filings(data_dir)
    return filing


def test_discovery_finds_supported_forms_in_deterministic_order(
    tmp_path: Path, write_sec_filing
) -> None:
    for cik, form, accession in [
        ("0000789019", "8-K", "0000789019-25-000002"),
        (CIK, "8-K", "0000320193-25-000010"),
        (CIK, "10-Q", "0000320193-25-000005"),
        (CIK, "10-K", "0000320193-25-000079"),
        (CIK, "10-K", "0000320193-24-000123"),
        (CIK, "DEF 14A", "0000320193-25-000001"),
    ]:
        write_sec_filing(
            tmp_path, cik=cik, form=form, accession=accession, filed="20250131"
        )
    filings_root = tmp_path / "bronze" / "sec_filings" / "sec-edgar-filings"
    (filings_root / ".DS_Store").write_bytes(b"finder")
    (filings_root / CIK / "10-K" / ".partial").mkdir()

    found = [(f.cik, f.form, f.accession_number) for f in discover_filings(tmp_path)]

    assert found == [
        (CIK, "10-K", "0000320193-24-000123"),
        (CIK, "10-K", "0000320193-25-000079"),
        (CIK, "10-Q", "0000320193-25-000005"),
        (CIK, "8-K", "0000320193-25-000010"),
        ("0000789019", "8-K", "0000789019-25-000002"),
    ]


def test_discovery_rejects_file_where_accession_folder_expected(
    tmp_path: Path, write_sec_filing
) -> None:
    directory = write_sec_filing(
        tmp_path, cik=CIK, form="10-K", accession=ACCESSION, filed="20251031"
    )
    (directory.parent / "stray.txt").write_text("not a filing")

    with pytest.raises(ValueError, match="Expected only directories"):
        discover_filings(tmp_path)


def test_parse_sec_header_reads_submission_fields(
    tmp_path: Path, write_sec_filing
) -> None:
    directory = write_sec_filing(
        tmp_path,
        cik=CIK,
        form="10-K",
        accession=ACCESSION,
        filed="20251031",
        header_cik="320193",
    )

    header = parse_sec_header(read_sec_header(directory / "full-submission.txt"))

    assert header.accession_number == ACCESSION
    assert header.form == "10-K"
    assert header.ciks == (CIK,)
    assert header.filing_date == date(2025, 10, 31)
    assert header.report_date == date(2024, 12, 31)
    assert header.accepted_at == datetime.fromisoformat("2025-10-31T16:01:02")
    assert header.accepted_at.tzinfo is None
    assert header.public_document_count == 6


def test_normalize_cik_zero_pads_and_rejects_non_digits() -> None:
    assert normalize_cik("320193") == CIK
    assert normalize_cik(CIK) == CIK
    with pytest.raises(ValueError, match="Invalid CIK"):
        normalize_cik("AAPL")
    with pytest.raises(ValueError, match="Invalid CIK"):
        normalize_cik("12345678901")


@pytest.mark.parametrize(
    "field",
    [
        "ACCESSION NUMBER",
        "CONFORMED SUBMISSION TYPE",
        "CENTRAL INDEX KEY",
        "FILED AS OF DATE",
    ],
)
def test_parse_sec_header_requires_mandatory_fields(
    tmp_path: Path, write_sec_filing, field: str
) -> None:
    directory = write_sec_filing(
        tmp_path, cik=CIK, form="10-K", accession=ACCESSION, filed="20251031"
    )
    header = read_sec_header(directory / "full-submission.txt")
    without_field = "\n".join(
        line for line in header.splitlines() if not line.strip().startswith(field)
    )

    with pytest.raises(ValueError, match=f"missing '{field}'"):
        parse_sec_header(without_field)


def test_read_sec_header_fails_without_header_block(tmp_path: Path) -> None:
    path = tmp_path / "full-submission.txt"
    path.write_text("<DOCUMENT>\n<TYPE>10-K\n</DOCUMENT>\n")

    with pytest.raises(ValueError, match="SEC header block"):
        read_sec_header(path)


def test_build_filing_record_derives_ids_paths_checksums_and_run_id(
    tmp_path: Path, write_sec_filing
) -> None:
    write_sec_filing(
        tmp_path, cik=CIK, form="10-K", accession=ACCESSION, filed="20251031"
    )
    filing = _only_filing(tmp_path)
    full_path = f"bronze/sec_filings/sec-edgar-filings/{CIK}/10-K/{ACCESSION}/full-submission.txt"
    primary_path = full_path.replace("full-submission.txt", "primary-document.html")
    full_sha = hashlib.sha256((tmp_path / full_path).read_bytes()).hexdigest()
    primary_sha = hashlib.sha256(b"<html>primary</html>").hexdigest()

    def manifest_entry(run_id: str, full_sha256: str) -> FilingManifestEntry:
        return FilingManifestEntry(
            run_id=run_id,
            filing_id=f"{CIK}:10-K:{ACCESSION}",
            cik=CIK,
            accession_number=ACCESSION,
            form="10-K",
            status="success",
            artifacts=(
                RawArtifact(
                    role="full_submission",
                    relative_path=full_path,
                    sha256=full_sha256,
                    size_bytes=1,
                ),
                RawArtifact(
                    role="primary_document",
                    relative_path=primary_path,
                    sha256=primary_sha,
                    size_bytes=1,
                ),
            ),
            recorded_at=PROCESSED_AT,
        )

    manifest = index_manifest(
        [
            manifest_entry("run-with-older-bytes", "0" * 64),
            manifest_entry("run-that-wrote-current-bytes", full_sha),
            manifest_entry("later-rerecord", full_sha),
        ]
    )

    record = build_filing_record(
        filing, data_dir=tmp_path, manifest=manifest, processed_at=PROCESSED_AT
    )

    assert record.filing_id == f"sec:{ACCESSION}"
    assert record.accession_number == ACCESSION
    assert (record.cik, record.form) == (CIK, "10-K")
    assert record.full_submission_path == full_path
    assert record.primary_document_path == primary_path
    assert record.full_submission_sha256 == full_sha
    assert record.primary_document_sha256 == primary_sha
    assert record.ingestion_run_id == "run-that-wrote-current-bytes"
    assert record.processed_at == PROCESSED_AT


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"header_accession": "0000320193-25-000080"}, "Header accession"),
        ({"header_form": "10-Q"}, "Header form"),
        ({"header_cik": "0000789019"}, "Folder CIK"),
    ],
)
def test_build_filing_record_rejects_header_that_disagrees_with_folder(
    tmp_path: Path, write_sec_filing, override: dict[str, str], message: str
) -> None:
    write_sec_filing(
        tmp_path,
        cik=CIK,
        form="10-K",
        accession=ACCESSION,
        filed="20251031",
        **override,
    )

    with pytest.raises(ValueError, match=message):
        build_filing_record(
            _only_filing(tmp_path),
            data_dir=tmp_path,
            manifest={},
            processed_at=PROCESSED_AT,
        )


def test_build_filing_record_allows_missing_primary_document_and_manifest(
    tmp_path: Path, write_sec_filing
) -> None:
    write_sec_filing(
        tmp_path,
        cik=CIK,
        form="8-K",
        accession=ACCESSION,
        filed="20251031",
        primary_html=None,
    )

    record = build_filing_record(
        _only_filing(tmp_path),
        data_dir=tmp_path,
        manifest={},
        processed_at=PROCESSED_AT,
    )

    assert record.primary_document_path is None
    assert record.primary_document_sha256 is None
    assert record.ingestion_run_id is None


def test_build_filing_record_rejects_empty_primary_document(
    tmp_path: Path, write_sec_filing
) -> None:
    write_sec_filing(
        tmp_path,
        cik=CIK,
        form="8-K",
        accession=ACCESSION,
        filed="20251031",
        primary_html=b"",
    )

    with pytest.raises(ValueError, match="Missing or empty file"):
        build_filing_record(
            _only_filing(tmp_path),
            data_dir=tmp_path,
            manifest={},
            processed_at=PROCESSED_AT,
        )


def test_record_fields_match_exact_arrow_schema(
    tmp_path: Path, write_sec_filing
) -> None:
    write_sec_filing(
        tmp_path, cik=CIK, form="10-Q", accession=ACCESSION, filed="20250801"
    )
    record = build_filing_record(
        _only_filing(tmp_path),
        data_dir=tmp_path,
        manifest={},
        processed_at=PROCESSED_AT,
    )

    rows = pa.Table.from_pylist([record.model_dump()], schema=FILINGS_ARROW_SCHEMA)

    assert list(FilingRecord.model_fields) == FILINGS_ARROW_SCHEMA.names
    assert rows.schema.equals(FILINGS_ARROW_SCHEMA)
    assert [(f.name, str(f.type), f.nullable) for f in FILINGS_ARROW_SCHEMA] == [
        ("filing_id", "string", False),
        ("accession_number", "string", False),
        ("cik", "string", False),
        ("form", "string", False),
        ("filing_date", "date32[day]", False),
        ("report_date", "date32[day]", True),
        ("accepted_at", "timestamp[us]", True),
        ("public_document_count", "int32", True),
        ("full_submission_path", "string", False),
        ("primary_document_path", "string", True),
        ("full_submission_sha256", "string", False),
        ("primary_document_sha256", "string", True),
        ("ingestion_run_id", "string", True),
        ("processed_at", "timestamp[us, tz=UTC]", False),
    ]

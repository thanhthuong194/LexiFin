from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from src.data_pipeline.connectors.sec_filings.bronze import BronzeFiling
from src.data_pipeline.contracts.ingestion import FilingManifestEntry
from src.data_pipeline.contracts.sec_filings.filings import FilingRecord, filing_id_for
from src.data_pipeline.storage.bronze import SecBronze, sha256_file

# Header lines look like "ACCESSION NUMBER:\t\t0000320193-24-000123", indented
# under FILER/SUBJECT COMPANY blocks for per-entity keys such as the CIK.
SEC_HEADER_FIELD = re.compile(r"^\s*([A-Z][A-Z0-9 -]*?):\s*(.*?)\s*$")
_ACCEPTANCE_DATETIME = re.compile(r"^<ACCEPTANCE-DATETIME>\s*(\d{14})\s*$")

ManifestIndex = Mapping[str, tuple[FilingManifestEntry, ...]]


@dataclass(frozen=True)
class SecHeader:
    """Submission-level fields from the SGML `<SEC-HEADER>` of a full submission."""

    accession_number: str
    form: str
    ciks: tuple[str, ...]
    filing_date: date
    report_date: date | None
    accepted_at: datetime | None
    public_document_count: int | None


@dataclass(frozen=True)
class RawSecHeader:
    """The raw bytes between the `<SEC-HEADER>` and `</SEC-HEADER>` lines.

    `start_byte` is the offset of `raw[0]` in the full submission, so positions
    inside the header map back to the Bronze file.
    """

    start_byte: int
    raw: bytes

    @property
    def end_byte(self) -> int:
        """Offset of the `</SEC-HEADER>` line: the exclusive end of the header."""

        return self.start_byte + len(self.raw)


def read_raw_sec_header(full_submission: Path) -> RawSecHeader:
    """Read the SEC header of a full submission as raw bytes with its offset.

    Reads line by line and stops at the end tag, so the (large) documents that
    follow the header are never loaded.
    """

    lines: list[bytes] = []
    started = False
    offset = start_byte = 0
    with full_submission.open("rb") as file_handle:
        for line in file_handle:
            if line.startswith(b"<SEC-HEADER>"):
                if started:
                    raise ValueError("SEC header has more than one <SEC-HEADER> line")
                started = True
                start_byte = offset + len(line)
            elif line.startswith(b"</SEC-HEADER>"):
                if not started:
                    break
                return RawSecHeader(start_byte=start_byte, raw=b"".join(lines))
            elif line.startswith(b"<DOCUMENT>"):
                break
            elif started:
                lines.append(line)
            offset += len(line)
    raise ValueError("SEC header block <SEC-HEADER>...</SEC-HEADER> not found")


def read_sec_header(full_submission: Path) -> str:
    """Return the SEC header text of a full submission.

    EDGAR headers are ASCII; any other byte is replaced rather than failing
    the whole filing.
    """

    return read_raw_sec_header(full_submission).raw.decode("utf-8", errors="replace")


def _single(fields: Mapping[str, list[str]], key: str) -> str | None:
    values = set(fields.get(key, ()))
    if len(values) > 1:
        raise ValueError(f"SEC header has conflicting {key!r} values: {sorted(values)}")
    return values.pop() if values else None


def _required(fields: Mapping[str, list[str]], key: str) -> str:
    value = _single(fields, key)
    if value is None:
        raise ValueError(f"SEC header is missing {key!r}")
    return value


def _parse_date(value: str, key: str) -> date:
    try:
        return date.fromisoformat(value)  # EDGAR writes basic ISO dates: YYYYMMDD
    except ValueError as error:
        raise ValueError(f"Invalid {key!r} date: {value!r}") from error


def normalize_cik(value: str) -> str:
    """Return a CIK as the canonical 10-digit, zero-padded string."""

    if not value.isdigit() or len(value) > 10:
        raise ValueError(f"Invalid CIK: {value!r}")
    return value.zfill(10)


def parse_sec_header(header: str) -> SecHeader:
    """Parse the submission fields LexiFin uses from SEC header text."""

    fields: dict[str, list[str]] = {}
    acceptance: set[str] = set()
    for line in header.splitlines():
        if match := _ACCEPTANCE_DATETIME.match(line):
            acceptance.add(match.group(1))
        elif (match := SEC_HEADER_FIELD.match(line)) and match.group(2):
            fields.setdefault(match.group(1), []).append(match.group(2))

    if len(acceptance) > 1:
        raise ValueError(f"SEC header has conflicting acceptance times: {acceptance}")
    ciks = tuple(
        dict.fromkeys(normalize_cik(v) for v in fields.get("CENTRAL INDEX KEY", ()))
    )
    if not ciks:
        raise ValueError("SEC header is missing 'CENTRAL INDEX KEY'")
    document_count = _single(fields, "PUBLIC DOCUMENT COUNT")
    report_date = _single(fields, "CONFORMED PERIOD OF REPORT")
    return SecHeader(
        accession_number=_required(fields, "ACCESSION NUMBER"),
        form=_required(fields, "CONFORMED SUBMISSION TYPE"),
        ciks=ciks,
        filing_date=_parse_date(
            _required(fields, "FILED AS OF DATE"), "FILED AS OF DATE"
        ),
        report_date=(
            _parse_date(report_date, "CONFORMED PERIOD OF REPORT")
            if report_date
            else None
        ),
        # EDGAR gives no offset for the acceptance time, so it stays naive.
        accepted_at=(
            datetime.strptime(acceptance.pop(), "%Y%m%d%H%M%S")  # noqa: DTZ007
            if acceptance
            else None
        ),
        public_document_count=int(document_count) if document_count else None,
    )


def check_header_matches_folder(header: SecHeader, filing: BronzeFiling) -> None:
    """Reject a header whose accession, form or CIK disagrees with its folder."""

    if header.accession_number != filing.accession_number:
        raise ValueError(
            f"Header accession {header.accession_number!r} does not match "
            f"folder {filing.accession_number!r}"
        )
    if header.form != filing.form:
        raise ValueError(
            f"Header form {header.form!r} does not match folder {filing.form!r}"
        )
    # A submission can list several filers; the folder CIK must be one of them.
    if filing.cik not in header.ciks:
        raise ValueError(
            f"Folder CIK {filing.cik!r} is not in header CIKs {header.ciks}"
        )


def index_manifest(entries: Iterable[FilingManifestEntry]) -> ManifestIndex:
    """Group successful manifest entries by accession, keeping append order."""

    index: dict[str, list[FilingManifestEntry]] = {}
    for entry in entries:
        if entry.status == "success":
            index.setdefault(entry.accession_number, []).append(entry)
    return {accession: tuple(group) for accession, group in index.items()}


def _ingestion_run_id(
    manifest: ManifestIndex,
    accession_number: str,
    artifacts: set[tuple[str, str]],
) -> str | None:
    """Return the run that wrote exactly the current artifacts, if recorded.

    The first matching entry wins: later entries for the same bytes only
    re-recorded a filing an earlier run had already downloaded.
    """

    return next(
        (
            entry.run_id
            for entry in manifest.get(accession_number, ())
            if {(a.relative_path, a.sha256) for a in entry.artifacts} == artifacts
        ),
        None,
    )


def _recorded_sha256(
    manifest: ManifestIndex, accession_number: str, stored_path: str, path: Path
) -> str:
    """The file's SHA-256 as the manifest recorded it at download, when every
    successful entry of the filing that lists the path records one hash and
    the file still has the recorded size; otherwise the file is hashed.

    Bronze files are never rewritten, so the recorded hash is the file's. The
    documents stage reads every full submission whole and rejects one whose
    bytes do not match it; `audit_silver.py --verify-source-hashes` re-hashes
    every file.
    """

    recorded = {
        (artifact.sha256, artifact.size_bytes)
        for entry in manifest.get(accession_number, ())
        for artifact in entry.artifacts
        if artifact.relative_path == stored_path
    }
    if len(recorded) == 1:
        ((sha256, size),) = recorded
        if size == path.stat().st_size:
            return sha256
    return sha256_file(path)


def _nonempty_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty file: {path}")
    return path


def build_filing_record(
    filing: BronzeFiling,
    *,
    bronze: SecBronze,
    manifest: ManifestIndex,
    processed_at: datetime,
) -> FilingRecord:
    """Validate one Bronze filing directory and describe it as a FilingRecord.

    Raises OSError or ValueError (including pydantic.ValidationError) when the
    directory or its SEC header is invalid.
    """

    full_submission = _nonempty_file(filing.full_submission)
    header = parse_sec_header(read_sec_header(full_submission))
    check_header_matches_folder(header, filing)

    full_path = bronze.stored(full_submission)
    full_sha256 = _recorded_sha256(
        manifest, header.accession_number, full_path, full_submission
    )
    artifacts = {(full_path, full_sha256)}
    primary_path = primary_sha256 = None
    if filing.primary_document.exists():
        primary_document = _nonempty_file(filing.primary_document)
        primary_path = bronze.stored(primary_document)
        primary_sha256 = _recorded_sha256(
            manifest, header.accession_number, primary_path, primary_document
        )
        artifacts.add((primary_path, primary_sha256))

    # model_validate (not the constructor) because `form` is only narrowed to
    # FormType by validation; the folder form was already checked above.
    return FilingRecord.model_validate(
        {
            "filing_id": filing_id_for(header.accession_number),
            "accession_number": header.accession_number,
            "cik": filing.cik,
            "form": filing.form,
            "filing_date": header.filing_date,
            "report_date": header.report_date,
            "accepted_at": header.accepted_at,
            "public_document_count": header.public_document_count,
            "full_submission_path": full_path,
            "primary_document_path": primary_path,
            "full_submission_sha256": full_sha256,
            "primary_document_sha256": primary_sha256,
            "ingestion_run_id": _ingestion_run_id(
                manifest, header.accession_number, artifacts
            ),
            "processed_at": processed_at,
        }
    )

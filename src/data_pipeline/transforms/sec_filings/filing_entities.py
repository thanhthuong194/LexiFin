from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from src.data_pipeline.contracts.sec_filings.filing_entities import (
    EntityRole,
    FilingEntityRecord,
    filing_entity_id_for,
)
from src.data_pipeline.contracts.sec_filings.filings import FilingRecord
from src.data_pipeline.storage.bronze import sha256_bytes
from src.data_pipeline.transforms.sec_filings.filings import (
    SEC_HEADER_FIELD,
    RawSecHeader,
    normalize_cik,
)

_ENTITY_ROLE_BY_LABEL: dict[str, EntityRole] = {
    "FILER": "filer",
    "FILED BY": "filed_by",
    "FILED-BY": "filed_by",
    "SUBJECT COMPANY": "subject_company",
    "SUBJECT-COMPANY": "subject_company",
    "REPORTING OWNER": "reporting_owner",
    "REPORTING-OWNER": "reporting_owner",
    "ISSUER": "issuer",
    "SERIAL COMPANY": "serial_company",
    "SERIAL-COMPANY": "serial_company",
    "FILED FOR": "filed_for",
    "FILED-FOR": "filed_for",
}

_ENTITY_DATA_FIELDS = {
    "COMPANY CONFORMED NAME": "company_name",
    "CENTRAL INDEX KEY": "cik",
    "STANDARD INDUSTRIAL CLASSIFICATION": "sic",
    "ORGANIZATION NAME": "organization_name",
    "IRS NUMBER": "ein",
    "EIN": "ein",
    "STATE OF INCORPORATION": "state_of_incorporation",
    "FISCAL YEAR END": "fiscal_year_end",
}


def _address_fields(prefix: str) -> dict[str, str]:
    return {
        "STREET 1": f"{prefix}_street_1",
        "STREET 2": f"{prefix}_street_2",
        "CITY": f"{prefix}_city",
        "STATE": f"{prefix}_state_or_country",
        "STATE OR COUNTRY": f"{prefix}_state_or_country",
        "PROVINCE COUNTRY": f"{prefix}_state_or_country",
        "ZIP": f"{prefix}_zip",
    }


# Nested sections of an entity block and the record field each source key fills.
_SECTION_FIELDS: dict[str, Mapping[str, str]] = {
    "COMPANY DATA": _ENTITY_DATA_FIELDS,
    "OWNER DATA": _ENTITY_DATA_FIELDS,
    "FILING VALUES": {
        "FORM TYPE": "form_type",
        "SEC ACT": "sec_act",
        "SEC FILE NUMBER": "file_number",
        "FILM NUMBER": "film_number",
    },
    "BUSINESS ADDRESS": {
        **_address_fields("business"),
        "BUSINESS PHONE": "business_phone",
    },
    "MAIL ADDRESS": _address_fields("mailing"),
}
# Name history inside an entity block (may repeat). It stays in Bronze, reachable
# through the block's offsets; normalising former names is a later stage.
_FORMER_COMPANY = "FORMER COMPANY"
_SIC = re.compile(r"(?P<description>.*?)\s*\[(?P<code>\d+)\]")


@dataclass(frozen=True)
class EntityBlock:
    """One entity-role block of a SEC header with its byte span in the file.

    `values` maps FilingEntityRecord field names to trimmed header values.
    """

    ordinal: int
    source_role: str
    entity_role: EntityRole
    start_byte: int
    end_byte: int
    values: Mapping[str, str]


def _header_lines(header: RawSecHeader) -> Iterator[tuple[int, str]]:
    """Yield each header line (without its LF/CRLF) and its offset in the file."""

    offset = header.start_byte
    for raw_line in header.raw.splitlines(keepends=True):
        yield offset, raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        offset += len(raw_line)


def _role_label(line: str) -> str | None:
    """Return the label of an unindented `LABEL:` line without a value."""

    match = SEC_HEADER_FIELD.match(line)
    if match is None or line[:1].isspace() or match.group(2):
        return None
    return match.group(1)


def _block_values(lines: Sequence[str]) -> dict[str, str]:
    """Map the indented section and field lines of one entity block to fields."""

    values: dict[str, str] = {}
    section: str | None = None
    seen_sections: set[str] = set()
    for line in lines:
        if not line.strip():
            continue
        match = SEC_HEADER_FIELD.match(line)
        if match is None or not line[:1].isspace():
            raise ValueError(f"unexpected header line {line.strip()!r}")
        key, value = match.groups()
        if not value and (key in _SECTION_FIELDS or key == _FORMER_COMPANY):
            if key in seen_sections and key != _FORMER_COMPANY:
                raise ValueError(f"repeated {key} section")
            section = key
            seen_sections.add(key)
            continue
        if section is None:
            raise ValueError(f"{key!r} appears outside a section")
        if section == _FORMER_COMPANY:
            continue
        field = _SECTION_FIELDS[section].get(key)
        if field is None:
            raise ValueError(f"unknown {key!r} in {section}")
        if value and values.setdefault(field, value) != value:
            raise ValueError(f"conflicting values for {key!r}")
    return values


def parse_entity_blocks(header: RawSecHeader) -> list[EntityBlock]:
    """Split a SEC header into its entity-role blocks, in order of appearance.

    A block starts at an unindented `ROLE:` line and runs to the next such line
    or to the end of the header; lines before the first block are submission
    fields and are not part of any block. Raises ValueError naming the block
    ordinal and role for an unknown role, section or key, a repeated section
    or conflicting values.
    """

    starts: list[tuple[int, str]] = []
    bodies: list[list[str]] = []
    for offset, line in _header_lines(header):
        label = _role_label(line)
        if label is not None:
            starts.append((offset, label))
            bodies.append([])
        elif bodies:
            bodies[-1].append(line)

    if not starts:
        return []
    ends = [offset for offset, _ in starts[1:]] + [header.end_byte]
    blocks: list[EntityBlock] = []
    for ordinal, ((start, label), end, body) in enumerate(
        zip(starts, ends, bodies, strict=True), start=1
    ):
        try:
            role = _ENTITY_ROLE_BY_LABEL.get(label)
            if role is None:
                raise ValueError("unknown entity-role label")
            values = _block_values(body)
        except ValueError as error:
            raise ValueError(f"Entity {ordinal} ({label}): {error}") from error
        blocks.append(EntityBlock(ordinal, label, role, start, end, values))
    return blocks


def _split_sic(value: str) -> tuple[str, str | None]:
    """Split `ELECTRONIC COMPUTERS [3571]` into code `3571` and its description."""

    match = _SIC.fullmatch(value)
    if match is None:
        raise ValueError(f"unparseable STANDARD INDUSTRIAL CLASSIFICATION {value!r}")
    return match["code"], match["description"] or None


def _entity_record(
    parent: FilingRecord,
    header: RawSecHeader,
    block: EntityBlock,
    processed_at: datetime,
) -> FilingEntityRecord:
    values: dict[str, str | None] = dict(block.values)
    if "cik" not in values:
        raise ValueError("missing CENTRAL INDEX KEY")
    if "company_name" not in values:
        raise ValueError("missing COMPANY CONFORMED NAME")
    form_type = values.get("form_type")
    if form_type is not None and form_type != parent.form:
        raise ValueError(
            f"FORM TYPE {form_type!r} differs from filing form {parent.form!r}"
        )
    values["cik"] = normalize_cik(block.values["cik"])
    if "sic" in block.values:
        values["sic"], values["sic_description"] = _split_sic(block.values["sic"])

    block_bytes = header.raw[
        block.start_byte - header.start_byte : block.end_byte - header.start_byte
    ]
    return FilingEntityRecord.model_validate(
        {
            "filing_entity_id": filing_entity_id_for(parent.filing_id, block.ordinal),
            "filing_id": parent.filing_id,
            "filing_date": parent.filing_date,
            "entity_ordinal": block.ordinal,
            "entity_role": block.entity_role,
            "source_role": block.source_role,
            **values,
            "source_path": parent.full_submission_path,
            "source_submission_sha256": parent.full_submission_sha256,
            "entity_start_byte": block.start_byte,
            "entity_end_byte": block.end_byte,
            "entity_block_sha256": sha256_bytes(block_bytes),
            "processed_at": processed_at,
        }
    )


def build_filing_entity_records(
    parent: FilingRecord,
    header: RawSecHeader,
    processed_at: datetime,
) -> list[FilingEntityRecord]:
    """Describe every entity-role block in a filing's SEC header.

    `header` must be read from the parent's full submission. The header needs
    at least one block, every FORM TYPE given must equal the parent form, and
    a FILER block must carry the parent CIK. Raises ValueError naming the
    entity ordinal and role otherwise.
    """

    blocks = parse_entity_blocks(header)
    if not blocks:
        raise ValueError("SEC header has no entity-role blocks")
    records: list[FilingEntityRecord] = []
    for block in blocks:
        try:
            records.append(_entity_record(parent, header, block, processed_at))
        except ValueError as error:
            raise ValueError(
                f"Entity {block.ordinal} ({block.source_role}): {error}"
            ) from error
    if not any(r.entity_role == "filer" and r.cik == parent.cik for r in records):
        raise ValueError(f"No FILER block carries the filing CIK {parent.cik}")
    return records


def check_entity_offsets(
    records: Sequence[FilingEntityRecord], header: RawSecHeader
) -> None:
    """Re-read each record's span from the raw header; check label and SHA-256.

    Spans must tile the header from the first role label to `</SEC-HEADER>`:
    each starts where the previous one ends. Raises ValueError naming the
    first offending ordinal.
    """

    expected_start: int | None = None
    for record in records:
        where = f"Entity {record.entity_ordinal} ({record.source_role})"
        start, end = record.entity_start_byte, record.entity_end_byte
        if not header.start_byte <= start < end <= header.end_byte:
            raise ValueError(f"{where}: span {start}..{end} is outside the SEC header")
        if expected_start is not None and start != expected_start:
            raise ValueError(f"{where}: span does not start where the previous ends")
        block = header.raw[start - header.start_byte : end - header.start_byte]
        if not block.startswith(f"{record.source_role}:".encode()):
            raise ValueError(f"{where}: span does not start at its role label")
        if sha256_bytes(block) != record.entity_block_sha256:
            raise ValueError(f"{where}: entity_block_sha256 does not match its span")
        expected_start = end
    if records and records[-1].entity_end_byte != header.end_byte:
        raise ValueError("Last entity span does not end at </SEC-HEADER>")

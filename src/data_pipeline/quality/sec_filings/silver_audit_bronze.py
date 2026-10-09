from __future__ import annotations

from collections.abc import Sequence

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.filing_entities import FilingEntityRecord
from src.data_pipeline.contracts.sec_filings.filings import FilingRecord
from src.data_pipeline.quality.sec_filings.silver_audit_report import SilverAuditReport
from src.data_pipeline.quality.sec_filings.silver_audit_sql import ScratchDatabase
from src.data_pipeline.storage.bronze import (
    SecBronze,
    sha256_bytes,
    sha256_file,
)
from src.data_pipeline.storage.sec_filings.documents import DOCUMENTS_TABLE
from src.data_pipeline.storage.sec_filings.filing_entities import FILING_ENTITIES_TABLE
from src.data_pipeline.storage.sec_filings.filings import FILINGS_TABLE
from src.data_pipeline.transforms.sec_filings.documents import check_document_offsets
from src.data_pipeline.transforms.sec_filings.filing_entities import (
    check_entity_offsets,
)
from src.data_pipeline.transforms.sec_filings.filings import read_raw_sec_header


def _paths_problem(filing: FilingRecord, bronze: SecBronze) -> str | None:
    for path in (filing.full_submission_path, filing.primary_document_path):
        if path is None:
            continue
        try:
            if not bronze.path(path).is_file():
                return f"{path} does not exist"
        except ValueError as error:
            return str(error)
    return None


def _hashes_problem(
    filing: FilingRecord,
    documents: Sequence[DocumentRecord],
    entities: Sequence[FilingEntityRecord],
    bronze: SecBronze,
) -> str | None:
    """Re-hash one filing's Bronze files and re-read every stored byte span.

    One full submission is held in memory at a time.
    """

    try:
        path = bronze.path(filing.full_submission_path)
        raw = path.read_bytes()
        if sha256_bytes(raw) != filing.full_submission_sha256:
            return "full submission SHA-256 differs"
        if filing.primary_document_path is not None and (
            sha256_file(bronze.path(filing.primary_document_path))
            != filing.primary_document_sha256
        ):
            return "primary document SHA-256 differs"
        check_document_offsets(documents, raw)
        check_entity_offsets(entities, read_raw_sec_header(path))
    except (OSError, ValueError) as error:
        return str(error)
    return None


def _stored_filing(
    scratch: ScratchDatabase, filing_id: str
) -> tuple[FilingRecord, list[DocumentRecord], list[FilingEntityRecord]]:
    """A filing with its documents and entities, in ordinal order, as records."""

    row = scratch.stored_rows(FILINGS_TABLE, "filing_id", filing_id)[0]
    documents = scratch.stored_rows(DOCUMENTS_TABLE, "filing_id", filing_id)
    entities = scratch.stored_rows(FILING_ENTITIES_TABLE, "filing_id", filing_id)
    return (
        FilingRecord.model_validate(row),
        sorted(
            (DocumentRecord.model_validate(d) for d in documents),
            key=lambda document: document.document_ordinal,
        ),
        sorted(
            (FilingEntityRecord.model_validate(e) for e in entities),
            key=lambda entity: entity.entity_ordinal,
        ),
    )


def check_bronze_files(
    scratch: ScratchDatabase,
    bronze: SecBronze,
    report: SilverAuditReport,
    *,
    verify_hashes: bool,
) -> None:
    """Every filing's Bronze files exist inside the Bronze directory; with `verify_hashes`,
    they also match their SHA-256 and every document and entity span.

    Filings are read one at a time from the scratch database, where the
    filing, document and entity tables are stored whole.
    """

    with (
        report.check(
            "bronze_paths",
            FILINGS_TABLE,
            "every filing's full submission and primary document exist inside the Bronze directory",
        ) as paths,
        report.check(
            "source_hashes",
            FILINGS_TABLE,
            "every full submission and primary document matches its SHA-256; every "
            "document span, <TEXT> slice and entity block matches its size and SHA-256",
        ) as hashes,
    ):
        if not verify_hashes:
            hashes.skip("Bronze is re-hashed only with --verify-source-hashes")
        for filing_id in scratch.filing_ids():
            try:
                filing, documents, entities = _stored_filing(scratch, filing_id)
            except ValueError:
                continue  # rows that break their contract fail row_contract
            paths.rows_checked += 1
            problem = _paths_problem(filing, bronze)
            if problem is not None:
                paths.found(filing_id, problem)
            elif verify_hashes:
                hashes.rows_checked += 1
                problem = _hashes_problem(filing, documents, entities, bronze)
                if problem is not None:
                    hashes.found(filing_id, problem)

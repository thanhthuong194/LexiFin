from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from src.data_pipeline.contracts.ingestion import FilingManifestEntry, FormType
from src.data_pipeline.pipelines.sec_state import load_manifest
from src.data_pipeline.storage.bronze import SecBronze

SUPPORTED_FORMS: frozenset[str] = frozenset(get_args(FormType))
FULL_SUBMISSION_NAME = "full-submission.txt"
PRIMARY_DOCUMENT_NAME = "primary-document.html"


@dataclass(frozen=True)
class BronzeFiling:
    """One accession directory found under the SEC Bronze download tree.

    `cik`, `form` and `accession_number` are the folder names as found on disk;
    they are not yet validated against the SEC header.
    """

    cik: str
    form: str
    accession_number: str
    directory: Path

    @property
    def full_submission(self) -> Path:
        return self.directory / FULL_SUBMISSION_NAME

    @property
    def primary_document(self) -> Path:
        return self.directory / PRIMARY_DOCUMENT_NAME


def _visible_directories(parent: Path) -> list[Path]:
    """Return non-hidden children of a filing-tree level, which must be folders."""

    children = sorted(
        path for path in parent.iterdir() if not path.name.startswith(".")
    )
    unexpected = [path for path in children if not path.is_dir()]
    if unexpected:
        raise ValueError(f"Expected only directories in {parent}: {unexpected[0]}")
    return children


def discover_filings(bronze: SecBronze) -> tuple[BronzeFiling, ...]:
    """Find every supported `<CIK>/<FORM>/<ACCESSION>` directory in SEC Bronze.

    Results are sorted by CIK, form and accession number. Hidden entries are
    ignored and unsupported form folders are skipped; any other file where the
    layout expects a directory raises ValueError.
    """

    filings_root = bronze.root / "sec-edgar-filings"
    if not filings_root.is_dir():
        raise FileNotFoundError(f"SEC Bronze filings folder not found: {filings_root}")

    # Each level is listed in sorted order, so the result is already ordered.
    return tuple(
        BronzeFiling(
            cik=cik_dir.name,
            form=form_dir.name,
            accession_number=accession_dir.name,
            directory=accession_dir,
        )
        for cik_dir in _visible_directories(filings_root)
        for form_dir in _visible_directories(cik_dir)
        if form_dir.name in SUPPORTED_FORMS
        for accession_dir in _visible_directories(form_dir)
    )


def load_filing_manifest(bronze: SecBronze) -> tuple[FilingManifestEntry, ...]:
    """Load the append-only SEC Bronze filing manifest; empty when absent."""

    return load_manifest(bronze.root / "_metadata" / "filing_manifest.jsonl")

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.data_pipeline.pipelines.sec_filings.document_tables import (
    DocumentTablesStageSummary,
    run_document_tables_stage,
)
from src.data_pipeline.pipelines.sec_filings.documents import (
    DocumentsStageSummary,
    run_documents_stage,
)
from src.data_pipeline.pipelines.sec_filings.filing_entities import (
    FilingEntitiesStageSummary,
    run_filing_entities_stage,
)
from src.data_pipeline.pipelines.sec_filings.filings import (
    FilingsStageSummary,
    run_filings_stage,
)
from src.data_pipeline.pipelines.sec_filings.narrative import (
    NarrativeStageSummary,
    run_narrative_stage,
)
from src.utils.logger import get_logger
from src.utils.timer import timed_operation

logger = get_logger(__name__)


@dataclass(frozen=True)
class SilverRunResult:
    """Stage summaries of one SEC Bronze -> Silver run.

    Stages run in order and stop at the first rejected one, so a later stage
    is None when an earlier stage was rejected: child rows are only built once
    their parents are committed.
    """

    filings: FilingsStageSummary
    documents: DocumentsStageSummary | None
    filing_entities: FilingEntitiesStageSummary | None
    narrative: NarrativeStageSummary | None
    document_tables: DocumentTablesStageSummary | None

    @property
    def completed(self) -> bool:
        """Whether every stage ran and committed (or had nothing to commit)."""

        return all(
            stage is not None and stage.status == "completed"
            for stage in (
                self.filings,
                self.documents,
                self.filing_entities,
                self.narrative,
                self.document_tables,
            )
        )


def run_sec_filings_silver(data_dir: Path) -> SilverRunResult:
    """Build or incrementally update the SEC Silver tables from all SEC Bronze.

    Stages run parent first: `sec_filings.filings` from Bronze, then
    `sec_filings.documents` and `sec_filings.filing_entities` from the
    committed filings, then `sec_filings.sections` and
    `sec_filings.content_blocks` from the committed primary HTML documents,
    then `sec_filings.document_tables` and `sec_filings.document_table_cells`
    from the committed `table` blocks. Each stage validates its whole batch before one append,
    so a rejected stage writes nothing, later stages do not run, and a rerun
    completes whatever is missing. Unexpected errors are logged and re-raised.
    """

    data_dir = data_dir.expanduser().resolve()
    processed_at = datetime.now(UTC)
    documents = filing_entities = narrative = document_tables = None
    logger.info("sec_silver_started", data_dir=str(data_dir))
    try:
        with timed_operation("pipeline.total"):
            filings = run_filings_stage(data_dir, processed_at)
            if filings.status == "completed":
                documents = run_documents_stage(data_dir, processed_at)
            if documents is not None and documents.status == "completed":
                filing_entities = run_filing_entities_stage(data_dir, processed_at)
            if filing_entities is not None and filing_entities.status == "completed":
                narrative = run_narrative_stage(data_dir, processed_at)
            if narrative is not None and narrative.status == "completed":
                document_tables = run_document_tables_stage(data_dir, processed_at)
    except Exception as error:
        logger.error(
            "sec_silver_failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise

    result = SilverRunResult(
        filings=filings,
        documents=documents,
        filing_entities=filing_entities,
        narrative=narrative,
        document_tables=document_tables,
    )
    statuses = {
        "filings_status": filings.status,
        "documents_status": None if documents is None else documents.status,
        "filing_entities_status": (
            None if filing_entities is None else filing_entities.status
        ),
        "narrative_status": None if narrative is None else narrative.status,
        "document_tables_status": (
            None if document_tables is None else document_tables.status
        ),
    }
    if result.completed:
        logger.info("sec_silver_completed", **statuses)
    else:
        logger.error("sec_silver_failed", **statuses)
    return result

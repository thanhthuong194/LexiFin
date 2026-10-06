from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.data_pipeline.pipelines.sec_filings.document_links import (
    DocumentLinksStageSummary,
    run_document_links_stage,
)
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
from src.data_pipeline.pipelines.sec_filings.xbrl_core import (
    XbrlCoreStageSummary,
    run_xbrl_core_stage,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_enrichment import (
    XbrlEnrichmentStageSummary,
    run_xbrl_enrichment_stage,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_facts import (
    XbrlFactsStageSummary,
    run_xbrl_facts_stage,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_linkbases import (
    XbrlLinkbasesStageSummary,
    run_xbrl_linkbases_stage,
)
from src.data_pipeline.transforms.sec_filings.xbrl_processor import ProcessorSettings
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
    document_links: DocumentLinksStageSummary | None
    xbrl_core: XbrlCoreStageSummary | None
    xbrl_facts: XbrlFactsStageSummary | None
    xbrl_enrichment: XbrlEnrichmentStageSummary | None
    xbrl_linkbases: XbrlLinkbasesStageSummary | None

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
                self.document_links,
                self.xbrl_core,
                self.xbrl_facts,
                self.xbrl_enrichment,
                self.xbrl_linkbases,
            )
        )


def run_sec_filings_silver(
    data_dir: Path,
    *,
    taxonomy_cache_dir: Path | None = None,
    allow_taxonomy_network: bool = False,
    http_user_agent: str | None = None,
) -> SilverRunResult:
    """Build or incrementally update the SEC Silver tables from all SEC Bronze.

    Stages run parent first: `sec_filings.filings` from Bronze, then
    `sec_filings.documents` and `sec_filings.filing_entities` from the
    committed filings, then `sec_filings.sections` and
    `sec_filings.content_blocks` from the committed primary HTML documents,
    then `sec_filings.document_tables` and `sec_filings.document_table_cells`
    from the committed `table` blocks, then `sec_filings.document_links` from
    every committed HTML document, then the XBRL core tables
    (`xbrl_instances`, `xbrl_contexts`, `xbrl_units`) from the filings' Inline
    XBRL and XML instance documents, then `xbrl_concepts` and `xbrl_facts`
    from those instances, then `xbrl_dimensions` and `xbrl_fact_footnotes`
    from their contexts and facts, and last `xbrl_labels`, `xbrl_networks`
    and `xbrl_relationships` from their DTSs. The XBRL processor
    resolves taxonomies from `taxonomy_cache_dir` (default
    `<data_dir>/_cache/arelle`) and downloads missing ones, identifying
    itself with `http_user_agent`, only when `allow_taxonomy_network` is set. Each stage validates its whole batch
    before its appends, so a rejected stage writes nothing, later stages do
    not run, and a rerun completes whatever is missing. Unexpected errors are
    logged and re-raised.
    """

    data_dir = data_dir.expanduser().resolve()
    taxonomy = ProcessorSettings(
        cache_dir=(taxonomy_cache_dir or data_dir / "_cache" / "arelle")
        .expanduser()
        .resolve(),
        allow_network=allow_taxonomy_network,
        http_user_agent=http_user_agent,
    )
    processed_at = datetime.now(UTC)
    stages: list[tuple[str, Callable[[], Any]]] = [
        ("filings", lambda: run_filings_stage(data_dir, processed_at)),
        ("documents", lambda: run_documents_stage(data_dir, processed_at)),
        ("filing_entities", lambda: run_filing_entities_stage(data_dir, processed_at)),
        ("narrative", lambda: run_narrative_stage(data_dir, processed_at)),
        ("document_tables", lambda: run_document_tables_stage(data_dir, processed_at)),
        ("document_links", lambda: run_document_links_stage(data_dir, processed_at)),
        ("xbrl_core", lambda: run_xbrl_core_stage(data_dir, processed_at)),
        ("xbrl_facts", lambda: run_xbrl_facts_stage(data_dir, processed_at, taxonomy)),
        (
            "xbrl_enrichment",
            lambda: run_xbrl_enrichment_stage(data_dir, processed_at, taxonomy),
        ),
        (
            "xbrl_linkbases",
            lambda: run_xbrl_linkbases_stage(data_dir, processed_at, taxonomy),
        ),
    ]
    summaries: dict[str, Any] = dict.fromkeys(name for name, _ in stages)
    logger.info("sec_silver_started", data_dir=str(data_dir))
    try:
        with timed_operation("pipeline.total"):
            for name, run_stage in stages:
                summaries[name] = run_stage()
                # Child rows are only built once their parents are committed.
                if summaries[name].status != "completed":
                    break
    except Exception as error:
        logger.error(
            "sec_silver_failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise

    result = SilverRunResult(**summaries)
    statuses = {
        f"{name}_status": None if summary is None else summary.status
        for name, summary in summaries.items()
    }
    if result.completed:
        logger.info("sec_silver_completed", **statuses)
    else:
        logger.error("sec_silver_failed", **statuses)
    return result

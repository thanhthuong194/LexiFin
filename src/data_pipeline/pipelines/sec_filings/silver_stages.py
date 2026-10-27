from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol

from src.data_pipeline.pipelines.sec_filings.document_links import (
    run_document_links_stage,
)
from src.data_pipeline.pipelines.sec_filings.document_tables import (
    run_document_tables_stage,
)
from src.data_pipeline.pipelines.sec_filings.documents import run_documents_stage
from src.data_pipeline.pipelines.sec_filings.exhibit_index import (
    run_exhibit_index_stage,
)
from src.data_pipeline.pipelines.sec_filings.filing_entities import (
    run_filing_entities_stage,
)
from src.data_pipeline.pipelines.sec_filings.filings import run_filings_stage
from src.data_pipeline.pipelines.sec_filings.narrative import run_narrative_stage
from src.data_pipeline.pipelines.sec_filings.xbrl_core import run_xbrl_core_stage
from src.data_pipeline.pipelines.sec_filings.xbrl_enrichment import (
    run_xbrl_enrichment_stage,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_facts import run_xbrl_facts_stage
from src.data_pipeline.pipelines.sec_filings.xbrl_linkbases import (
    run_xbrl_linkbases_stage,
)
from src.data_pipeline.quality.sec_filings.silver_audit import (
    AuditOptions,
    audit_silver,
)
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.storage.iceberg import SilverTableSpec
from src.data_pipeline.storage.sec_filings.content_blocks import CONTENT_BLOCKS_SPEC
from src.data_pipeline.storage.sec_filings.document_links import DOCUMENT_LINKS_SPEC
from src.data_pipeline.storage.sec_filings.document_table_cells import (
    DOCUMENT_TABLE_CELLS_SPEC,
)
from src.data_pipeline.storage.sec_filings.document_tables import DOCUMENT_TABLES_SPEC
from src.data_pipeline.storage.sec_filings.documents import DOCUMENTS_SPEC
from src.data_pipeline.storage.sec_filings.exhibits import EXHIBITS_SPEC
from src.data_pipeline.storage.sec_filings.filing_entities import FILING_ENTITIES_SPEC
from src.data_pipeline.storage.sec_filings.filings import FILINGS_SPEC
from src.data_pipeline.storage.sec_filings.sections import SECTIONS_SPEC
from src.data_pipeline.storage.sec_filings.xbrl_core import (
    XBRL_CONTEXTS_SPEC,
    XBRL_INSTANCES_SPEC,
    XBRL_UNITS_SPEC,
)
from src.data_pipeline.storage.sec_filings.xbrl_enrichment import (
    XBRL_DIMENSIONS_SPEC,
    XBRL_FACT_FOOTNOTES_SPEC,
)
from src.data_pipeline.storage.sec_filings.xbrl_facts import (
    XBRL_CONCEPTS_SPEC,
    XBRL_FACTS_SPEC,
)
from src.data_pipeline.storage.sec_filings.xbrl_linkbases import (
    XBRL_LABELS_SPEC,
    XBRL_NETWORKS_SPEC,
    XBRL_RELATIONSHIPS_SPEC,
)
from src.data_pipeline.transforms.sec_filings.xbrl_processor import ProcessorSettings

# Statuses of a stage that let its dependents run: a table stage completed
# (written, or nothing to write) or the integrity audit passed.
SUCCEEDED = frozenset({"completed", "PASS", "WARN"})


class StageOutcome(Protocol):
    """The part of a stage's summary the pipeline reads."""

    @property
    def status(self) -> str: ...


@dataclass(frozen=True)
class StageContext:
    """What every stage of one run receives."""

    data_dir: Path
    bronze: SecBronze
    processed_at: datetime
    taxonomy: ProcessorSettings
    # How the integrity audit runs (its Bronze directory is `bronze`).
    audit: AuditOptions = field(default_factory=AuditOptions)


@dataclass(frozen=True)
class SilverStage:
    """One step of the Bronze -> Silver pipeline.

    `dependencies` name the stages whose tables it reads, `owned_tables` the
    tables it writes. A stage reads only committed parent tables, so it can
    run in a later process than its dependencies.
    """

    name: str
    dependencies: tuple[str, ...]
    owned_tables: tuple[SilverTableSpec, ...]
    run: Callable[[StageContext], StageOutcome]


_TABLE_STAGES = (
    SilverStage(
        "filings",
        (),
        (FILINGS_SPEC,),
        lambda c: run_filings_stage(c.data_dir, c.processed_at, c.bronze),
    ),
    SilverStage(
        "documents",
        ("filings",),
        (DOCUMENTS_SPEC,),
        lambda c: run_documents_stage(c.data_dir, c.processed_at, c.bronze),
    ),
    SilverStage(
        "filing_entities",
        ("filings",),
        (FILING_ENTITIES_SPEC,),
        lambda c: run_filing_entities_stage(c.data_dir, c.processed_at, c.bronze),
    ),
    SilverStage(
        "narrative",
        ("documents",),
        (SECTIONS_SPEC, CONTENT_BLOCKS_SPEC),
        lambda c: run_narrative_stage(c.data_dir, c.processed_at, c.bronze),
    ),
    SilverStage(
        "document_tables",
        ("documents", "narrative"),
        (DOCUMENT_TABLES_SPEC, DOCUMENT_TABLE_CELLS_SPEC),
        lambda c: run_document_tables_stage(c.data_dir, c.processed_at, c.bronze),
    ),
    SilverStage(
        "document_links",
        ("documents",),
        (DOCUMENT_LINKS_SPEC,),
        lambda c: run_document_links_stage(c.data_dir, c.processed_at, c.bronze),
    ),
    SilverStage(
        "exhibit_index",
        ("filings", "narrative", "document_tables", "document_links"),
        (EXHIBITS_SPEC,),
        lambda c: run_exhibit_index_stage(c.data_dir, c.processed_at),
    ),
    SilverStage(
        "xbrl_core",
        ("filings", "documents"),
        (XBRL_INSTANCES_SPEC, XBRL_CONTEXTS_SPEC, XBRL_UNITS_SPEC),
        lambda c: run_xbrl_core_stage(c.data_dir, c.processed_at, c.bronze),
    ),
    SilverStage(
        "xbrl_facts",
        # Facts are aligned with the stored narrative blocks and table cells.
        ("documents", "narrative", "document_tables", "xbrl_core"),
        (XBRL_CONCEPTS_SPEC, XBRL_FACTS_SPEC),
        lambda c: run_xbrl_facts_stage(
            c.data_dir, c.processed_at, c.taxonomy, c.bronze
        ),
    ),
    SilverStage(
        "xbrl_enrichment",
        ("documents", "xbrl_core", "xbrl_facts"),
        (XBRL_DIMENSIONS_SPEC, XBRL_FACT_FOOTNOTES_SPEC),
        lambda c: run_xbrl_enrichment_stage(
            c.data_dir, c.processed_at, c.taxonomy, c.bronze
        ),
    ),
    SilverStage(
        "xbrl_linkbases",
        ("documents", "xbrl_core", "xbrl_facts"),
        (XBRL_LABELS_SPEC, XBRL_NETWORKS_SPEC, XBRL_RELATIONSHIPS_SPEC),
        lambda c: run_xbrl_linkbases_stage(
            c.data_dir, c.processed_at, c.taxonomy, c.bronze
        ),
    ),
)
# The stages in run order; the read-only integrity audit of every table last.
SILVER_STAGES = (
    *_TABLE_STAGES,
    SilverStage(
        "integrity",
        tuple(stage.name for stage in _TABLE_STAGES),
        (),
        lambda c: audit_silver(
            c.data_dir,
            replace(c.audit, bronze_dir=c.bronze.root),
        ),
    ),
)


def check_stage_order(stages: Sequence[SilverStage]) -> None:
    """Raise ValueError unless `stages` is a valid run order.

    Names must be unique, every dependency must be an earlier stage, and
    each table must have one owner.
    """

    seen: set[str] = set()
    owners: dict[str, str] = {}
    for stage in stages:
        if stage.name in seen:
            raise ValueError(f"stage {stage.name!r} is registered twice")
        late = [name for name in stage.dependencies if name not in seen]
        if late:
            raise ValueError(
                f"stage {stage.name!r} depends on {late}, not run before it"
            )
        for spec in stage.owned_tables:
            owner = owners.setdefault(spec.identifier, stage.name)
            if owner != stage.name:
                raise ValueError(
                    f"{spec.identifier} is owned by {owner!r} and {stage.name!r}"
                )
        seen.add(stage.name)


check_stage_order(SILVER_STAGES)


def select_stages(
    stages: Sequence[SilverStage],
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> tuple[SilverStage, ...]:
    """The stages from `from_stage` through `to_stage` (default: all), in run order.

    Raises ValueError for an unknown name or a `from_stage` after `to_stage`.
    """

    names = [stage.name for stage in stages]
    for name in (from_stage, to_stage):
        if name is not None and name not in names:
            raise ValueError(f"unknown stage {name!r}; stages are {names}")
    start = 0 if from_stage is None else names.index(from_stage)
    end = len(names) - 1 if to_stage is None else names.index(to_stage)
    if start > end:
        raise ValueError(f"stage {from_stage!r} runs after stage {to_stage!r}")
    return tuple(stages[start : end + 1])


def prerequisite_tables(
    stages: Sequence[SilverStage], selected: Sequence[SilverStage]
) -> tuple[SilverTableSpec, ...]:
    """Tables of the stages that `selected` depends on, directly or not, but does not run.

    They must already exist with their contract definition.
    """

    by_name = {stage.name: stage for stage in stages}
    chosen = {stage.name for stage in selected}
    needed: set[str] = set()
    pending = [name for stage in selected for name in stage.dependencies]
    while pending:
        name = pending.pop()
        if name not in chosen and name not in needed:
            needed.add(name)
            pending.extend(by_name[name].dependencies)
    return tuple(
        spec for stage in stages if stage.name in needed for spec in stage.owned_tables
    )

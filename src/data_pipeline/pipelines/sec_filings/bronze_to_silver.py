from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from pyiceberg.exceptions import NoSuchTableError

from src.data_pipeline.pipelines.sec_filings.document_links import (
    DocumentLinksStageSummary,
)
from src.data_pipeline.pipelines.sec_filings.document_tables import (
    DocumentTablesStageSummary,
)
from src.data_pipeline.pipelines.sec_filings.documents import DocumentsStageSummary
from src.data_pipeline.pipelines.sec_filings.exhibit_index import (
    ExhibitIndexStageSummary,
)
from src.data_pipeline.pipelines.sec_filings.filing_entities import (
    FilingEntitiesStageSummary,
)
from src.data_pipeline.pipelines.sec_filings.filings import FilingsStageSummary
from src.data_pipeline.pipelines.sec_filings.narrative import NarrativeStageSummary
from src.data_pipeline.pipelines.sec_filings.silver_stages import (
    SILVER_STAGES,
    SUCCEEDED,
    SilverStage,
    StageContext,
    StageOutcome,
    prerequisite_tables,
    select_stages,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_core import XbrlCoreStageSummary
from src.data_pipeline.pipelines.sec_filings.xbrl_enrichment import (
    XbrlEnrichmentStageSummary,
)
from src.data_pipeline.pipelines.sec_filings.xbrl_facts import XbrlFactsStageSummary
from src.data_pipeline.pipelines.sec_filings.xbrl_linkbases import (
    XbrlLinkbasesStageSummary,
)
from src.data_pipeline.quality.sec_filings.silver_report import SilverAuditReport
from src.data_pipeline.storage.iceberg import (
    SilverTableSpec,
    open_silver_catalog,
    silver_catalog_paths,
)
from src.data_pipeline.transforms.sec_filings.xbrl_processor import ProcessorSettings
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)


@dataclass(frozen=True)
class TableState:
    """A table's current snapshot and row count, read from its metadata."""

    snapshot_id: int | None
    rows: int


@dataclass(frozen=True)
class SilverRunResult:
    """One Bronze -> Silver run: each stage's summary and the run's totals.

    A stage field holds that stage's summary, or None when it did not run:
    it was not requested, a prerequisite table was missing, or an earlier
    stage failed (stages stop at the first rejected stage or failed audit, so
    child rows are only built from committed parents). Table states are read
    from metadata before and after the run.
    """

    run_id: str
    data_dir: Path
    started_at: datetime
    finished_at: datetime
    stages_requested: tuple[str, ...]
    failed_stage: str | None
    prerequisite_problems: tuple[str, ...]
    tables_before: dict[str, TableState | None]
    tables_after: dict[str, TableState | None]
    filings: FilingsStageSummary | None = None
    documents: DocumentsStageSummary | None = None
    filing_entities: FilingEntitiesStageSummary | None = None
    narrative: NarrativeStageSummary | None = None
    document_tables: DocumentTablesStageSummary | None = None
    document_links: DocumentLinksStageSummary | None = None
    exhibit_index: ExhibitIndexStageSummary | None = None
    xbrl_core: XbrlCoreStageSummary | None = None
    xbrl_facts: XbrlFactsStageSummary | None = None
    xbrl_enrichment: XbrlEnrichmentStageSummary | None = None
    xbrl_linkbases: XbrlLinkbasesStageSummary | None = None
    integrity: SilverAuditReport | None = None

    def stage(self, name: str) -> StageOutcome | None:
        """The summary of a stage, by registry name; None when it did not run."""

        outcome: StageOutcome | None = getattr(self, name)
        return outcome

    @property
    def stages_completed(self) -> tuple[str, ...]:
        """Requested stages that ran and succeeded, in run order."""

        return tuple(
            name
            for name in self.stages_requested
            if (outcome := self.stage(name)) is not None and outcome.status in SUCCEEDED
        )

    @property
    def stages_skipped(self) -> tuple[str, ...]:
        """Requested stages that did not run."""

        return tuple(name for name in self.stages_requested if self.stage(name) is None)

    @property
    def completed(self) -> bool:
        """Whether every requested stage ran and succeeded."""

        return self.stages_completed == self.stages_requested

    @property
    def status(self) -> Literal["completed", "failed"]:
        """`completed` exactly when every requested stage succeeded."""

        return "completed" if self.completed else "failed"

    @property
    def rows_inserted(self) -> dict[str, int]:
        """Rows each written table gained in this run."""

        return {
            table: after.rows - (0 if before is None else before.rows)
            for table, before in self.tables_before.items()
            if (after := self.tables_after[table]) is not None
            and after.rows != (0 if before is None else before.rows)
        }

    @property
    def rows_skipped(self) -> int:
        """Rows already stored in the tables of the stages that succeeded.

        Those stages left them unchanged: a stored group is skipped, never
        rewritten.
        """

        succeeded = set(self.stages_completed)
        owned = {
            spec.identifier
            for stage in SILVER_STAGES
            if stage.name in succeeded
            for spec in stage.owned_tables
        }
        return sum(
            state.rows
            for table, state in self.tables_before.items()
            if table in owned and state is not None
        )

    def summary(self) -> dict[str, Any]:
        """The run's totals, as logged and printed; no row payloads."""

        audit = self.integrity
        failed = None if self.failed_stage is None else self.stage(self.failed_stage)
        return {
            "run_id": self.run_id,
            "data_dir": str(self.data_dir),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "duration_seconds": round(
                (self.finished_at - self.started_at).total_seconds(), 3
            ),
            "status": self.status,
            "stages_requested": list(self.stages_requested),
            "stages_completed": list(self.stages_completed),
            "stages_skipped": list(self.stages_skipped),
            "failed_stage": self.failed_stage,
            # Stages reject whole groups (filings, documents, instances...)
            # before writing; the failed stage's own counters say which.
            "failed_stage_summary": (
                vars(failed)
                if failed is not None and not isinstance(failed, SilverAuditReport)
                else None
            ),
            "prerequisite_problems": list(self.prerequisite_problems),
            "tables_created": [
                table
                for table, before in self.tables_before.items()
                if before is None and self.tables_after[table] is not None
            ],
            "rows_inserted": self.rows_inserted,
            "rows_inserted_total": sum(self.rows_inserted.values()),
            "rows_skipped": self.rows_skipped,
            "snapshot_changes": {
                table: [
                    None if before is None else before.snapshot_id,
                    None if after is None else after.snapshot_id,
                ]
                for table, before in self.tables_before.items()
                if (before and before.snapshot_id)
                != ((after := self.tables_after[table]) and after.snapshot_id)
            },
            "integrity_checks_run": None if audit is None else len(audit.checks_run),
            "integrity_checks_passed": (
                None if audit is None else len(audit.checks_run) - audit.checks_failed
            ),
            "integrity_checks_failed": None if audit is None else audit.checks_failed,
            "integrity_failures": (
                [] if audit is None else [asdict(f) for f in audit.failures.values()]
            ),
            "warnings": None if audit is None else audit.warnings,
        }


def _catalog_exists(data_dir: Path) -> bool:
    return all(path.exists() for path in silver_catalog_paths(data_dir))


def _table_states(
    data_dir: Path, specs: Sequence[SilverTableSpec]
) -> dict[str, TableState | None]:
    """Each table's state, None for a table not created yet; creates nothing."""

    states: dict[str, TableState | None] = dict.fromkeys(s.identifier for s in specs)
    if not _catalog_exists(data_dir):
        return states
    with open_silver_catalog(data_dir) as catalog:
        for spec in specs:
            try:
                snapshot = catalog.load_table(spec.identifier).current_snapshot()
            except NoSuchTableError:
                continue
            if snapshot is None or snapshot.summary is None:
                states[spec.identifier] = TableState(None, 0)
                continue
            rows = snapshot.summary["total-records"]
            if rows is None:
                raise ValueError(f"{spec.identifier} snapshot has no total-records")
            states[spec.identifier] = TableState(snapshot.snapshot_id, int(rows))
    return states


def _prerequisite_problems(
    data_dir: Path, specs: Sequence[SilverTableSpec]
) -> list[str]:
    """Why tables a run reads but does not write are missing or off-contract."""

    if not specs:
        return []
    if not _catalog_exists(data_dir):
        return [f"no Silver catalog under {data_dir}"]
    problems = []
    with open_silver_catalog(data_dir) as catalog:
        for spec in specs:
            try:
                table = catalog.load_table(spec.identifier)
            except NoSuchTableError:
                problems.append(f"{spec.identifier} does not exist")
                continue
            problem = spec.definition_problem(table)
            if problem is not None:
                problems.append(f"{spec.identifier} {problem}")
    return problems


def _run_stage(
    stage: SilverStage, context: StageContext, outcomes: dict[str, StageOutcome]
) -> bool:
    """Run one stage and keep its summary; return whether it succeeded."""

    logger.info("silver_stage_started", dependencies=list(stage.dependencies))
    started = perf_counter()
    try:
        with timed_operation(f"silver_pipeline.{stage.name}"):
            outcome = stage.run(context)
    except Exception as error:
        logger.error(
            "silver_stage_failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    outcomes[stage.name] = outcome
    fields = {
        "status": outcome.status,
        "duration_seconds": round(perf_counter() - started, 3),
    }
    if outcome.status in SUCCEEDED:
        logger.info("silver_stage_completed", **fields)
        return True
    logger.error("silver_stage_failed", **fields)
    return False


def _run_stages(
    stages: Sequence[SilverStage],
    context: StageContext,
    outcomes: dict[str, StageOutcome],
    blocked: str | None,
) -> str | None:
    """Run stages in order until one fails; return the failed stage's name.

    Once one fails, or when `blocked` gives a reason, the rest are skipped.
    """

    failed = None
    for stage in stages:
        with log_context(stage=stage.name):
            if blocked is not None:
                logger.warning("silver_stage_skipped", reason=blocked)
            elif not _run_stage(stage, context, outcomes):
                failed = stage.name
                blocked = f"stage {stage.name!r} failed"
    return failed


def run_sec_filings_silver(
    data_dir: Path,
    *,
    from_stage: str | None = None,
    to_stage: str | None = None,
    taxonomy: ProcessorSettings | None = None,
) -> SilverRunResult:
    """Build or incrementally update the SEC Silver tables from all SEC Bronze.

    Runs the stages of SILVER_STAGES from `from_stage` through `to_stage`
    (default: all, ending with the read-only integrity audit) in their
    dependency order. A stage before `from_stage` is not run: the tables of
    every stage the run depends on must already exist with their contract
    definition, otherwise nothing runs. Each stage validates its whole batch
    before its appends, so a rejected stage writes nothing, later stages do
    not run, and a rerun completes whatever is missing; there is no global
    rollback. The XBRL stages resolve taxonomies as `taxonomy` says (default:
    offline, from `<data_dir>/_cache/arelle`). An unknown stage or an empty
    range raises ValueError; unexpected errors are logged and re-raised.
    """

    data_dir = data_dir.expanduser().resolve()
    selected = select_stages(SILVER_STAGES, from_stage, to_stage)
    started_at = datetime.now(UTC)
    context = StageContext(
        data_dir=data_dir,
        processed_at=started_at,
        taxonomy=taxonomy
        or ProcessorSettings(
            cache_dir=data_dir / "_cache" / "arelle",
            allow_network=False,
            http_user_agent=None,
        ),
    )
    run_id = uuid4().hex
    written = [spec for stage in selected for spec in stage.owned_tables]
    outcomes: dict[str, StageOutcome] = {}
    with log_context(
        source="sec_filings",
        pipeline="bronze_to_silver",
        run_id=run_id,
        data_dir=str(data_dir),
    ):
        logger.info(
            "silver_pipeline_started", stages_requested=[s.name for s in selected]
        )
        try:
            with timed_operation("silver_pipeline.total"):
                before = _table_states(data_dir, written)
                problems = _prerequisite_problems(
                    data_dir, prerequisite_tables(SILVER_STAGES, selected)
                )
                failed_stage = _run_stages(
                    selected,
                    context,
                    outcomes,
                    "prerequisite tables are missing or invalid" if problems else None,
                )
                after = _table_states(data_dir, written)
        except Exception as error:
            logger.error(
                "silver_pipeline_failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        result = SilverRunResult(
            run_id=run_id,
            data_dir=data_dir,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            stages_requested=tuple(stage.name for stage in selected),
            failed_stage=failed_stage,
            prerequisite_problems=tuple(problems),
            tables_before=before,
            tables_after=after,
            **outcomes,  # type: ignore[arg-type]  # keyed by stage name, one field per stage
        )
        if result.completed:
            logger.info("silver_pipeline_completed", **result.summary())
        else:
            logger.error("silver_pipeline_failed", **result.summary())
    return result

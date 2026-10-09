from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any, Literal

from src.utils.logger import get_logger

logger = get_logger(__name__)

Status = Literal["PASS", "WARN", "FAIL", "SKIP"]
# Identifiers kept per check: a systematic defect is counted in full but
# cannot flood the report.
SAMPLE_IDS = 5
_RANK: dict[Status, int] = {"PASS": 0, "SKIP": 0, "WARN": 1, "FAIL": 2}


@dataclass(frozen=True)
class AuditCheck:
    """The outcome of one check of one table, or of the layer when `table` is None.

    `message` states the rule, followed by the first problem found;
    `failed_count` counts failing rows or groups, of which `sample_ids`
    keeps the first SAMPLE_IDS.
    """

    check_name: str
    status: Status
    table: str | None
    rows_checked: int
    failed_count: int
    message: str
    sample_ids: tuple[str, ...]
    duration_seconds: float


@dataclass
class CheckRun:
    """What a running check has found.

    The check gets `severity` (FAIL or WARN) when `failed_count` is positive,
    PASS otherwise, and SKIP when `skip` was called.
    """

    message: str
    severity: Status = "FAIL"
    rows_checked: int = 0
    failed_count: int = 0
    sample_ids: list[str] = field(default_factory=list)
    first_problem: str | None = None
    skipped: str | None = None

    def found(
        self, identifier: object, problem: str | None = None, count: int = 1
    ) -> None:
        """Count a failing row or group (`count` rows for a repeated key)."""

        self.failed_count += count
        if len(self.sample_ids) < SAMPLE_IDS:
            self.sample_ids.append(str(identifier))
        if self.first_problem is None and problem is not None:
            self.first_problem = problem

    def skip(self, reason: str) -> None:
        """Mark the check as not applicable, with the reason."""

        self.skipped = reason


@dataclass(frozen=True)
class TableMetadata:
    """A table as the catalog describes it when the audit starts.

    `fields` lists `name type required|optional` in column order; row and
    file counts come from the current snapshot's summary.
    """

    identifier: str
    location: str
    metadata_location: str
    snapshot_id: int | None
    snapshot_count: int
    schema_id: int
    fields: tuple[str, ...]
    partition_spec: tuple[str, ...]
    sort_order: str
    row_count: int
    data_file_count: int
    data_file_bytes: int


@dataclass
class FilingSample:
    """One sampled filing and the evidence printed for a reviewer."""

    form: str
    filing_id: str
    lines: list[str] = field(default_factory=list)


@dataclass
class SilverAuditReport:
    """Outcome of one read-only audit of the SEC Silver tables."""

    data_dir: str
    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float = 0.0
    tables: dict[str, TableMetadata] = field(default_factory=dict)
    checks: list[AuditCheck] = field(default_factory=list)
    samples: list[FilingSample] = field(default_factory=list)
    rows_scanned: int = 0
    snapshot_changes: dict[str, str] = field(default_factory=dict)

    @contextmanager
    def check(
        self,
        check_name: str,
        table: str | None,
        message: str,
        *,
        severity: Status = "FAIL",
    ) -> Iterator[CheckRun]:
        """Run one check: time it, then record and log its outcome.

        An exception inside the block propagates and records nothing: the
        audit could not run that check.
        """

        run = CheckRun(message, severity)
        started = perf_counter()
        yield run
        status: Status = (
            "SKIP"
            if run.skipped is not None
            else run.severity
            if run.failed_count
            else "PASS"
        )
        text = run.skipped or run.message
        if run.first_problem is not None:
            text = f"{text}; first problem: {run.first_problem}"
        result = AuditCheck(
            check_name,
            status,
            table,
            run.rows_checked,
            run.failed_count,
            text,
            tuple(run.sample_ids),
            round(perf_counter() - started, 3),
        )
        self.checks.append(result)
        log = {"FAIL": logger.error, "WARN": logger.warning}.get(status, logger.info)
        log("silver_integrity_check", **vars(result))

    @property
    def status(self) -> Status:
        """FAIL when any check failed, else WARN when any warned, else PASS."""

        return self._worst(check.status for check in self.checks)

    def table_status(self, identifier: str) -> Status:
        """The worst status among the checks of one table."""

        return self._worst(c.status for c in self.checks if c.table == identifier)

    @staticmethod
    def _worst(statuses: Iterable[Status]) -> Status:
        worst: Status = "PASS"
        for status in statuses:
            if _RANK[status] > _RANK[worst]:
                worst = status
        return worst

    def summary(self) -> dict[str, Any]:
        """The fields of the final structured log event."""

        checks = Counter(check.status for check in self.checks)
        tables = Counter(self.table_status(name) for name in self.tables)
        return {
            "status": self.status,
            "tables_discovered": len(self.tables),
            "tables_passed": tables["PASS"],
            "tables_warned": tables["WARN"],
            "tables_failed": tables["FAIL"],
            "checks_passed": checks["PASS"],
            "checks_warned": checks["WARN"],
            "checks_failed": checks["FAIL"],
            "checks_skipped": checks["SKIP"],
            "total_rows_scanned": self.rows_scanned,
            "samples_checked": len(self.samples),
            "snapshot_changes": self.snapshot_changes,
            "duration_seconds": self.duration_seconds,
            "failures": [vars(c) for c in self.checks if c.status == "FAIL"],
            "warnings": [vars(c) for c in self.checks if c.status == "WARN"],
        }

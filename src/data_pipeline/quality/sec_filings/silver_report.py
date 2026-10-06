from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from src.utils.logger import get_logger

logger = get_logger(__name__)

Severity = Literal["error", "warning", "info"]
# Every occurrence is counted, but only the first issues of a check are kept
# and logged, so a systematic defect cannot flood the report.
ISSUES_PER_CHECK = 20


@dataclass(frozen=True)
class AuditIssue:
    """One finding: what failed, where, and the record or group it concerns."""

    severity: Severity
    check_name: str
    table: str
    message: str
    identifier: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SilverAuditReport:
    """Outcome of one read-only audit of the SEC Silver tables.

    `issues` keeps the first ISSUES_PER_CHECK issues of each check, while
    `issue_counts` counts every occurrence by (severity, check name). The
    audit passes when no error was found; warnings and info do not fail it.
    """

    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float = 0.0
    tables_expected: list[str] = field(default_factory=list)
    tables_found: list[str] = field(default_factory=list)
    tables_missing: list[str] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    issues: list[AuditIssue] = field(default_factory=list)
    issue_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    row_counts: dict[str, int] = field(default_factory=dict)
    snapshot_ids_before: dict[str, int | None] = field(default_factory=dict)
    snapshot_ids_after: dict[str, int | None] = field(default_factory=dict)
    duplicate_counts: dict[str, int] = field(default_factory=dict)
    orphan_counts: dict[str, int] = field(default_factory=dict)
    group_violation_counts: Counter[str] = field(default_factory=Counter)
    source_hash_failures: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> int:
        """Number of error occurrences, including those not kept in `issues`."""

        return sum(
            n for (severity, _), n in self.issue_counts.items() if severity == "error"
        )

    @property
    def warnings(self) -> int:
        """Number of warning occurrences, including those not kept in `issues`."""

        return sum(
            n for (severity, _), n in self.issue_counts.items() if severity == "warning"
        )

    @property
    def status(self) -> Literal["passed", "failed"]:
        """`passed` exactly when the audit found no error."""

        return "failed" if self.errors else "passed"

    def ran(self, check_name: str) -> None:
        """Record that a check was run."""

        if check_name not in self.checks_run:
            self.checks_run.append(check_name)

    def add(
        self,
        severity: Severity,
        check_name: str,
        table: str,
        message: str,
        identifier: str | None = None,
        **details: Any,
    ) -> None:
        """Count an issue; keep and log it while its check has few issues."""

        self.ran(check_name)
        self.issue_counts[(severity, check_name)] += 1
        if self.issue_counts[(severity, check_name)] > ISSUES_PER_CHECK:
            return
        issue = AuditIssue(severity, check_name, table, message, identifier, details)
        self.issues.append(issue)
        log = {"error": logger.error, "warning": logger.warning, "info": logger.info}
        log[severity](
            f"silver_audit_{severity}",
            check_name=check_name,
            table=table,
            message=message,
            identifier=identifier,
            **details,
        )

    def group_violation(
        self, table: str, group_identifier: str, message: str, **details: Any
    ) -> None:
        """Record an incomplete or inconsistent group of rows as an error."""

        self.group_violation_counts[table] += 1
        self.add("error", "group", table, message, group_identifier, **details)

    def summary(self) -> dict[str, Any]:
        """The fields of the final structured log event."""

        return {
            "status": self.status,
            "tables_expected": len(self.tables_expected),
            "tables_checked": len(self.tables_found),
            "tables_missing": self.tables_missing,
            "checks_run": len(self.checks_run),
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_seconds": self.duration_seconds,
            "row_counts": self.row_counts,
            "duplicate_counts": self.duplicate_counts,
            "orphan_counts": self.orphan_counts,
            "group_violation_counts": dict(self.group_violation_counts),
            "source_hash_failures": self.source_hash_failures,
            "issue_counts": {
                f"{severity}:{check}": n
                for (severity, check), n in sorted(self.issue_counts.items())
            },
            "metrics": self.metrics,
            "snapshots_unchanged": self.snapshot_ids_after == self.snapshot_ids_before,
        }

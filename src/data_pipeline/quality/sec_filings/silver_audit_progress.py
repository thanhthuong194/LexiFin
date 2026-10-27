from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import duckdb

from src.data_pipeline.quality.sec_filings.silver_audit_report import (
    SilverAuditReport,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# The scratch database of a resumable audit, inside its work directory.
SCRATCH_FILE = "audit.duckdb"
# Code whose change makes stored steps meaningless: the checks and the
# record contracts they validate rows against.
_CODE_DIRECTORIES = (
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parents[2] / "contracts" / "sec_filings",
)


def audit_fingerprint(report: SilverAuditReport, settings: Mapping[str, Any]) -> str:
    """What a stored step depends on: every table's snapshot and row count,
    the audit settings and the audit code. A resumed audit must match it."""

    digest = hashlib.sha256()
    tables = {
        identifier: [table.snapshot_id, table.row_count]
        for identifier, table in sorted(report.tables.items())
    }
    digest.update(json.dumps([tables, settings], sort_keys=True).encode())
    for directory in _CODE_DIRECTORIES:
        for path in sorted(directory.glob("*.py")):
            digest.update(path.name.encode() + b"\x00" + path.read_bytes())
    return digest.hexdigest()


def _stored_fingerprint(path: Path) -> str | None:
    try:
        with duckdb.connect(str(path), read_only=True) as connection:
            row = connection.execute("SELECT value FROM audit_fingerprint").fetchone()
    except duckdb.Error:
        return None
    return row[0] if row else None


def discard_scratch(path: Path) -> None:
    """Remove a scratch database, its write-ahead log and its spill files."""

    for file in (path, path.with_name(f"{path.name}.wal")):
        file.unlink(missing_ok=True)
    shutil.rmtree(path.with_name(f"{path.name}.tmp"), ignore_errors=True)


def prepare_scratch(work_dir: Path, fingerprint: str) -> Path:
    """The scratch database path of `work_dir`; a database left by an audit
    with another fingerprint (or unreadable) is removed first."""

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / SCRATCH_FILE
    if path.exists():
        stored = _stored_fingerprint(path)
        if stored != fingerprint:
            logger.info(
                "silver_integrity_scratch_discarded",
                path=str(path),
                reason="unreadable" if stored is None else "fingerprint changed",
            )
            discard_scratch(path)
    return path


class AuditProgress:
    """The completed steps of an audit, stored in its scratch database.

    A step is a table scan or one set-based check. When it completes, the
    checks it added to the report (and `state`, for a scan) are committed;
    a later audit with the same fingerprint replays them into its report
    instead of running the step again.
    """

    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        report: SilverAuditReport,
        fingerprint: str,
    ) -> None:
        self.connection = connection
        self.report = report
        self.started_at = len(report.checks)
        connection.execute("CREATE TABLE IF NOT EXISTS audit_fingerprint (value TEXT)")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS audit_progress "
            "(step TEXT PRIMARY KEY, checks BLOB, state BLOB)"
        )
        if connection.execute("SELECT 1 FROM audit_fingerprint").fetchone() is None:
            connection.execute(
                "INSERT INTO audit_fingerprint VALUES (?)", (fingerprint,)
            )
        self.replayed = 0

    def replay(self, step: str) -> tuple[bool, Any]:
        """Whether `step` completed earlier, with its state; its checks are
        appended to the report. Otherwise later checks belong to `step`."""

        row = self.connection.execute(
            "SELECT checks, state FROM audit_progress WHERE step = ?", (step,)
        ).fetchone()
        if row is None:
            self.started_at = len(self.report.checks)
            return False, None
        checks = pickle.loads(row[0])
        self.report.checks.extend(checks)
        for check in checks:
            logger.info("silver_integrity_check_resumed", step=step, **vars(check))
        self.replayed += 1
        return True, pickle.loads(row[1])

    def skip(self, step: str) -> bool:
        """`replay` for a step without state."""

        return self.replay(step)[0]

    def complete(self, step: str, state: Any = None) -> None:
        """Commit `step` with the checks added since it was not replayed."""

        checks = self.report.checks[self.started_at :]
        self.connection.execute(
            "INSERT INTO audit_progress VALUES (?, ?, ?)",
            (step, pickle.dumps(checks), pickle.dumps(state)),
        )
        self.started_at = len(self.report.checks)


class NoProgress:
    """The progress of an audit that starts over every time."""

    replayed = 0

    def replay(self, _step: str) -> tuple[bool, Any]:
        return False, None

    def skip(self, _step: str) -> bool:
        return False

    def complete(self, _step: str, _state: Any = None) -> None:
        return None

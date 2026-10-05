#!/usr/bin/env python3
"""Stop hook for Claude Code and Codex: run the repository quality gate.

When the agent tries to end its turn, this runs `scripts/check.sh changed`.
On failure it exits 2 with the check output on stderr; both tools then keep
the agent working with that text as feedback. After MAX_ATTEMPTS consecutive
failures in one session it allows the stop and warns the user, so a failure
the agent cannot fix (for example, a pre-existing one) does not loop forever.
A per-session counter file is used instead of `stop_hook_active`, which some
Claude Code versions do not set reliably.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

GATE_SCRIPT = "scripts/check.sh"
GATE_MODE = "changed"
MAX_ATTEMPTS = 3
MAX_FEEDBACK_CHARS = 6000


def read_payload() -> dict[str, Any]:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def shorten(text: str, limit: int = MAX_FEEDBACK_CHARS) -> str:
    """Keep the start (first failures) and the end (failure summary) of long output."""
    if len(text) <= limit:
        return text
    head = limit * 3 // 4
    return f"{text[:head]}\n... [output truncated] ...\n{text[-(limit - head) :]}"


def find_repo_root(cwd: str) -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return Path(result.stdout.strip()) if result.returncode == 0 else None


def main() -> int:
    payload = read_payload()
    root = find_repo_root(payload.get("cwd") or os.getcwd())
    if root is None or not (root / GATE_SCRIPT).is_file():
        return 0

    session_id = payload.get("session_id", "unknown")
    attempts_file = Path(tempfile.gettempdir()) / f"agent-stop-gate-{session_id}"

    result = subprocess.run(
        [str(root / GATE_SCRIPT), GATE_MODE],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        attempts_file.unlink(missing_ok=True)
        return 0

    attempts = int(attempts_file.read_text()) + 1 if attempts_file.exists() else 1
    if attempts > MAX_ATTEMPTS:
        attempts_file.unlink(missing_ok=True)
        warning = (
            f"Quality gate still failing after {MAX_ATTEMPTS} attempts. "
            f"Run `{GATE_SCRIPT} {GATE_MODE}` and review before merging."
        )
        print(json.dumps({"systemMessage": warning}))
        return 0

    attempts_file.write_text(str(attempts))
    output = shorten(result.stdout + result.stderr)
    print(
        f"Quality gate failed (attempt {attempts}/{MAX_ATTEMPTS}).\n"
        "Fix the reported problems in code you changed for this task, then finish. "
        "If a failure existed before your change or is outside the task's scope, "
        "do not fix it; state it in your report.\n\n"
        f"{output}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())

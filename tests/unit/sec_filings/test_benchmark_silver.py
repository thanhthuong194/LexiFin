from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from scripts.sec_filings import benchmark_silver as benchmark

pytestmark = pytest.mark.unit

MIB = 2**20


def _bronze(root: Path, filings: int) -> list[str]:
    """A Bronze directory with one success filing (two files) per manifest line."""

    lines = []
    for number in range(filings):
        accession = f"0000000000-26-{number:06d}"
        directory = f"bronze/sec_filings/sec-edgar-filings/0000000001/10-K/{accession}"
        artifacts = []
        for name in ("full-submission.txt", "primary-document.html"):
            path = root / directory.removeprefix("bronze/sec_filings/") / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * (number + 1))
            artifacts.append(
                {"relative_path": f"{directory}/{name}", "size_bytes": number + 1}
            )
        entry = {
            "filing_id": f"sec:{accession}",
            "form": "8-K" if number % 2 else "10-K",
            "status": "success",
            "artifacts": artifacts,
        }
        lines.append(json.dumps(entry) + "\n")
    (root / benchmark.MANIFEST).parent.mkdir(parents=True)
    (root / benchmark.MANIFEST).write_text("".join(lines), encoding="utf-8")
    return lines


def test_samples_are_nested_and_close_to_their_fraction() -> None:
    ids = [f"sec:{n}" for n in range(4000)]
    quarter = {i for i in ids if benchmark.in_sample(i, 0.25)}
    half = {i for i in ids if benchmark.in_sample(i, 0.5)}

    assert quarter < half
    assert 900 < len(quarter) < 1100
    assert all(benchmark.in_sample(i, 1) for i in ids)


def test_sample_bronze_links_the_sampled_filings_without_writing_the_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bronze"
    lines = _bronze(source, 40)
    target = tmp_path / "sample"

    sample = benchmark.sample_bronze(source, 0.5, target)

    kept = (target / benchmark.MANIFEST).read_text(encoding="utf-8").splitlines(True)
    assert 0 < len(kept) < 40
    assert set(kept) <= set(lines)
    entries = [json.loads(line) for line in kept]
    assert sample.filings == len(entries)
    assert sample.forms == Counter(e["form"] for e in entries)
    assert sample.bronze_bytes == sum(
        a["size_bytes"] for e in entries for a in e["artifacts"]
    )
    linked = target / entries[0]["artifacts"][0]["relative_path"].removeprefix(
        "bronze/sec_filings/"
    )
    original = source / linked.relative_to(target)
    assert linked.stat().st_ino == original.stat().st_ino
    assert (source / benchmark.MANIFEST).read_text(encoding="utf-8") == "".join(lines)


def test_whole_bronze_is_used_in_place(tmp_path: Path) -> None:
    source = tmp_path / "bronze"
    _bronze(source, 3)

    sample = benchmark.sample_bronze(source, 1.0, tmp_path / "sample")

    assert sample.root == source
    assert sample.filings == 3
    assert not (tmp_path / "sample").exists()


def test_run_measured_reports_exit_code_output_and_peak_memory(tmp_path: Path) -> None:
    code = "import time; b = bytearray(96 * 2**20); time.sleep(0.5); print('done'); raise SystemExit(3)"
    log = tmp_path / "log.txt"

    run = benchmark.run_measured([sys.executable, "-c", code], {}, log, 0.05)

    assert run.exit_code == 3
    assert log.read_text().strip() == "done"
    assert run.peak_rss_bytes >= 96 * MIB
    assert run.peak_tree_rss_bytes >= 96 * MIB
    assert run.wall_seconds >= 0.5


def test_tree_memory_includes_child_processes(tmp_path: Path) -> None:
    child = "import time; b = bytearray(96 * 2**20); time.sleep(0.6)"
    parent = (
        f"import subprocess, sys; subprocess.run([sys.executable, '-c', {child!r}])"
    )

    run = benchmark.run_measured(
        [sys.executable, "-c", parent], {}, tmp_path / "log", 0.05
    )

    assert run.exit_code == 0
    assert run.peak_tree_rss_bytes >= 96 * MIB


def test_read_stage_log_keeps_summary_and_time_inside_the_stage(tmp_path: Path) -> None:
    events = [
        {
            "event": "operation_completed",
            "operation": "silver_pipeline.total",
            "duration_ms": 9000,
        },
        {
            "event": "operation_completed",
            "operation": "silver_pipeline.xbrl_facts",
            "duration_ms": 9000,
        },
        {
            "event": "operation_completed",
            "operation": "xbrl_facts.read",
            "duration_ms": 1500,
        },
        {"event": "xbrl_facts_extracted", "load_ms": 2000, "extract_ms": 500},
        {"event": "xbrl_facts_extracted", "load_ms": 1000, "extract_ms": 500},
        {"event": "silver_pipeline_completed", "rows_inserted": {"a": 2, "b": 3}},
    ]
    log = tmp_path / "stage.jsonl"
    log.write_text(
        "\n".join(map(json.dumps, events))
        + "\nstatus: completed\nrows_inserted: {'a': 2}\n"
    )

    parsed = benchmark.read_stage_log(log)

    assert parsed.operations == {"xbrl_facts.read": 1.5}
    assert parsed.phases == {
        "xbrl_facts_extracted.load": 3.0,
        "xbrl_facts_extracted.extract": 1.0,
    }
    assert parsed.summary is not None
    assert parsed.summary["rows_inserted"] == {"a": 2, "b": 3}


def _scenario(
    filings: int, wall: float, samples: tuple[tuple[float, int], ...]
) -> benchmark.Scenario:
    sample = benchmark.BronzeSample(
        f"{filings}", Path("/b"), filings, filings * MIB, Counter()
    )
    peak = max(rss for _, rss in samples)
    process = benchmark.ProcessRun(0, wall, wall, peak, peak, samples)
    log = benchmark.StageLog(summary={"rows_inserted": {"t": filings}})
    stage = benchmark.StageResult(f"build-{filings}", "filings", process, log)
    return benchmark.Scenario(f"build-{filings}", sample, filings, (stage,))


def _filings_row(report: str) -> str:
    return next(
        line
        for line in report.splitlines()
        if line.startswith("| filings |") and "/" in line
    )


def test_report_projects_time_linearly_and_shows_peak_memory_per_half_run(
    tmp_path: Path,
) -> None:
    settings = benchmark.Settings(tmp_path, tmp_path, False, 1, 1.0, {})
    samples = ((0.0, 100 * MIB), (1.0, 300 * MIB), (2.0, 200 * MIB), (3.0, 280 * MIB))
    builds = [_scenario(50, 5.0, samples), _scenario(100, 10.0, samples)]

    report = benchmark.render_report(
        [*builds, _scenario(200, 20.0, samples)], [], settings, tmp_path, 1000
    )

    # 0.1 s per filing from 50 to 200 filings: 20 + 80 s at 1000 filings;
    # the halves split at 1.5 s.
    assert _filings_row(report).endswith("| 300 / 280 | 100.0 s |")
    assert "## Lỗi" not in report

"""Benchmark the SEC Silver build: time, CPU, memory, throughput, scaling, reruns.

Every stage of every scenario runs as its own `build_silver.py --from-stage S
--to-stage S` process, so its peak memory is its own and not the high-water
mark of the stages before it. Scenarios, in order:

- one build from scratch per Bronze sample (`--fractions`, smallest first);
  a sample keeps the filings whose filing_id hash falls under the fraction,
  so samples are nested and spread over companies and forms;
- a rerun of the largest sample with nothing new (incremental no-op);
- a build of the largest sample over the Silver of the second largest
  (incremental delta: only the filings it lacks are new).

Writes `report.md`, one log per stage process under `logs/` and the memory
samples (`rss_samples.csv`) under `--work-dir`. Linux only (/proc, wait4).
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_pipeline.pipelines.sec_filings.silver_stages import SILVER_STAGES
from src.data_pipeline.storage.bronze import SecBronze

BUILD_SCRIPT = REPOSITORY_ROOT / "scripts" / "sec_filings" / "build_silver.py"
MANIFEST = Path("_metadata") / "filing_manifest.jsonl"
PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")
MIB = 2**20
# Operations every stage process logs around the whole stage.
_STAGE_OPERATIONS = "silver_pipeline."


@dataclass(frozen=True)
class BronzeSample:
    """The Bronze a scenario reads: a nested sample of `--bronze-dir`."""

    label: str
    root: Path
    filings: int
    bronze_bytes: int
    forms: Counter[str]


@dataclass(frozen=True)
class ProcessRun:
    """What one measured process used.

    `peak_rss_bytes` is the kernel's high-water mark of the largest single
    process (itself or a waited-for descendant); `peak_tree_rss_bytes` is the
    largest sampled sum over the process and its live descendants, which is
    what the machine had to hold when work runs in several processes.
    """

    exit_code: int
    wall_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int
    peak_tree_rss_bytes: int
    samples: tuple[tuple[float, int], ...]


@dataclass
class StageLog:
    """What a stage process logged: its run summary and where time went."""

    summary: dict[str, Any] | None = None
    operations: Counter[str] = field(default_factory=Counter)
    phases: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class StageResult:
    scenario: str
    stage: str
    process: ProcessRun
    log: StageLog

    @property
    def rows_inserted(self) -> int:
        summary = self.log.summary or {}
        return sum((summary.get("rows_inserted") or {}).values())


@dataclass(frozen=True)
class Scenario:
    name: str
    sample: BronzeSample
    new_filings: int
    stages: tuple[StageResult, ...]

    @property
    def failed(self) -> StageResult | None:
        return next((s for s in self.stages if s.process.exit_code != 0), None)

    @property
    def wall_seconds(self) -> float:
        return sum(s.process.wall_seconds for s in self.stages)

    @property
    def peak_rss_bytes(self) -> int:
        return max(s.process.peak_rss_bytes for s in self.stages)

    @property
    def peak_tree_rss_bytes(self) -> int:
        return max(s.process.peak_tree_rss_bytes for s in self.stages)

    def stage(self, name: str) -> StageResult | None:
        return next((s for s in self.stages if s.stage == name), None)


@dataclass(frozen=True)
class Settings:
    work_dir: Path
    taxonomy_cache_dir: Path
    allow_taxonomy_network: bool
    xbrl_workers: int
    sample_seconds: float
    env: dict[str, str]


def in_sample(filing_id: str, fraction: float) -> bool:
    """Whether a filing is in the `fraction` sample; smaller samples are subsets."""

    digest = hashlib.sha256(filing_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") < fraction * 2**64


def sample_label(fraction: float) -> str:
    return f"{fraction * 100:g}pct"


def sample_bronze(bronze_dir: Path, fraction: float, target: Path) -> BronzeSample:
    """The `fraction` sample of a Bronze directory, laid out under `target`.

    The sample's manifest keeps the original lines of the sampled filings and
    their files are hard links (a copy across file systems), so nothing under
    `bronze_dir` is written. The whole Bronze (fraction 1) is used in place.
    """

    source = SecBronze(bronze_dir)
    lines = (bronze_dir / MANIFEST).read_text(encoding="utf-8").splitlines(True)
    parsed = ((line, json.loads(line)) for line in lines if line.strip())
    entries = [(l, e) for l, e in parsed if in_sample(e["filing_id"], fraction)]
    root = bronze_dir if fraction == 1 else target
    if fraction != 1:
        sample = SecBronze(target)
        (target / MANIFEST).parent.mkdir(parents=True)
        (target / MANIFEST).write_text("".join(l for l, _ in entries), encoding="utf-8")
        for _, entry in entries:
            for artifact in entry["artifacts"]:
                _link(
                    source.path(artifact["relative_path"]),
                    sample.path(artifact["relative_path"]),
                )
    filings = [entry for _, entry in entries if entry["status"] == "success"]
    return BronzeSample(
        label=sample_label(fraction),
        root=root,
        filings=len(filings),
        bronze_bytes=sum(a["size_bytes"] for e in filings for a in e["artifacts"]),
        forms=Counter(entry["form"] for entry in filings),
    )


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():  # a filing retried after a failure keeps its paths
        return
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:  # the sample is on another file system
            raise
        shutil.copy2(source, target)


def tree_rss_bytes(root_pid: int) -> int:
    """Resident memory of a process and all its live descendants."""

    children: defaultdict[int, list[int]] = defaultdict(list)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:  # the process exited while /proc was listed
            continue
        # The command name may hold spaces and parentheses; ppid follows the last ')'.
        children[int(stat.rsplit(")", 1)[1].split()[1])].append(int(entry.name))
    total, pending = 0, [root_pid]
    while pending:
        pid = pending.pop()
        pending.extend(children[pid])
        try:
            resident_pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
        except OSError:  # exited since the listing
            continue
        total += resident_pages * PAGE_BYTES
    return total


def run_measured(
    command: Sequence[str], env: dict[str, str], log_path: Path, sample_seconds: float
) -> ProcessRun:
    """Run a command with its output in `log_path`, measuring time and memory."""

    samples: list[tuple[float, int]] = []
    stop = threading.Event()
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env
        )

        def sample() -> None:
            while True:
                samples.append(
                    (time.monotonic() - started, tree_rss_bytes(process.pid))
                )
                if stop.wait(sample_seconds):
                    return

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        # wait4 reports the child's own usage (with its waited-for descendants);
        # Popen.wait would not.
        _, status, usage = os.wait4(process.pid, 0)
        wall_seconds = time.monotonic() - started
        stop.set()
        sampler.join()
    # Popen did not reap the child itself; tell it the outcome.
    process.returncode = os.waitstatus_to_exitcode(status)
    return ProcessRun(
        exit_code=process.returncode,
        wall_seconds=wall_seconds,
        cpu_seconds=usage.ru_utime + usage.ru_stime,
        peak_rss_bytes=usage.ru_maxrss * 1024,  # kilobytes on Linux
        peak_tree_rss_bytes=max(rss for _, rss in samples),
        samples=tuple(samples),
    )


def read_stage_log(path: Path) -> StageLog:
    """The run summary and time per operation and per phase of a stage's log.

    Phases are the `<phase>_ms` totals stages log once per batch (Arelle load,
    extraction...), summed over batches.
    """

    log = StageLog()
    with path.open(encoding="utf-8", errors="replace") as lines:
        for line in lines:
            if not line.startswith("{"):  # the printed summary, tracebacks
                continue
            event = json.loads(line)
            name = event.get("event")
            if name in ("silver_pipeline_completed", "silver_pipeline_failed") and (
                "rows_inserted" in event
            ):
                log.summary = event
            if name == "operation_completed" and not event["operation"].startswith(
                _STAGE_OPERATIONS
            ):
                log.operations[event["operation"]] += event["duration_ms"] / 1000
            for key, value in event.items():
                if key.endswith("_ms") and key != "duration_ms":
                    log.phases[f"{name}.{key.removesuffix('_ms')}"] += value / 1000
    return log


def run_scenario(
    name: str,
    silver_dir: Path,
    sample: BronzeSample,
    new_filings: int,
    settings: Settings,
) -> Scenario:
    """Run every stage in its own process until one fails."""

    results: list[StageResult] = []
    for stage in SILVER_STAGES:
        command = [
            sys.executable,
            str(BUILD_SCRIPT),
            "--data-dir",
            str(silver_dir),
            "--bronze-dir",
            str(sample.root),
            "--taxonomy-cache-dir",
            str(settings.taxonomy_cache_dir),
            "--from-stage",
            stage.name,
            "--to-stage",
            stage.name,
            "--xbrl-workers",
            str(settings.xbrl_workers),
            *(["--allow-taxonomy-network"] if settings.allow_taxonomy_network else []),
        ]
        log_path = settings.work_dir / "logs" / f"{name}.{stage.name}.jsonl"
        process = run_measured(command, settings.env, log_path, settings.sample_seconds)
        results.append(StageResult(name, stage.name, process, read_stage_log(log_path)))
        print(
            f"{name:>16} {stage.name:<16} exit {process.exit_code} "
            f"{process.wall_seconds:8.1f} s {process.peak_rss_bytes / MIB:8.0f} MiB",
            flush=True,
        )
        if process.exit_code != 0:
            break
    _write_samples(settings.work_dir / "rss_samples.csv", results)
    return Scenario(name, sample, new_filings, tuple(results))


def _write_samples(path: Path, results: Sequence[StageResult]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new:
            writer.writerow(("scenario", "stage", "elapsed_seconds", "tree_rss_mib"))
        for result in results:
            for elapsed, rss in result.process.samples:
                writer.writerow(
                    (
                        result.scenario,
                        result.stage,
                        round(elapsed, 2),
                        round(rss / MIB, 1),
                    )
                )


def _seconds(value: float) -> str:
    if value >= 3600:
        return f"{value / 3600:.1f} h"
    return f"{value / 60:.1f} min" if value >= 120 else f"{value:.1f} s"


def _mib(value: float) -> str:
    return f"{value / MIB:,.0f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows),
        "",
    ]


def _git_revision() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    dirty = " (có thay đổi chưa commit)" if git("status", "--porcelain") else ""
    return git("rev-parse", "--short", "HEAD") + dirty


def _machine() -> str:
    meminfo = Path("/proc/meminfo").read_text().split("\n", 1)[0]
    total_kib = int(meminfo.split()[1])
    return f"{os.cpu_count()} CPU, RAM {total_kib / 2**20:.1f} GiB, Python {sys.version.split()[0]}"


def _header(
    builds: Sequence[Scenario], settings: Settings, bronze_dir: Path
) -> list[str]:
    network = "online (tải taxonomy còn thiếu, thời gian không so được với offline)"
    return [
        f"# Silver benchmark {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
        "",
        f"- Code: `{_git_revision()}`; máy: {_machine()}",
        f"- Bronze: `{bronze_dir}`; taxonomy: `{settings.taxonomy_cache_dir}`, "
        + (network if settings.allow_taxonomy_network else "offline")
        + f"; XBRL workers: {settings.xbrl_workers}",
        (
            f"- Mức dữ liệu: {', '.join(b.sample.label for b in builds)} filing của "
            "Bronze (mẫu lồng nhau theo hash filing_id)"
        ),
        (
            "- Mỗi stage chạy trong một process riêng: thời gian gồm cả khởi động process "
            "(import, mở catalog: ~1–3 s mỗi stage). RAM tối đa = high-water mark của "
            "kernel cho process lớn nhất; RAM cây = tổng RSS của process và mọi process "
            f"con, lấy mẫu mỗi {settings.sample_seconds:g} s."
        ),
        "",
    ]


def _totals(builds: Sequence[Scenario]) -> list[str]:
    rows = []
    for build in builds:
        sample, wall = build.sample, build.wall_seconds
        forms = "/".join(str(sample.forms[f]) for f in ("10-K", "10-Q", "8-K"))
        rows.append(
            (
                sample.label,
                sample.filings,
                _mib(sample.bronze_bytes),
                forms,
                _seconds(wall),
                _seconds(sum(s.process.cpu_seconds for s in build.stages)),
                _mib(build.peak_rss_bytes),
                _mib(build.peak_tree_rss_bytes),
                f"{sample.filings / wall * 60:.1f}",
                f"{sample.bronze_bytes / MIB / wall * 60:.0f}",
                f"{sum(s.rows_inserted for s in build.stages):,}",
            )
        )
    return [
        "## 1. Tổng thời gian, RAM tối đa, throughput (build từ đầu)",
        "",
        *_table(
            (
                "Mức",
                "Filing",
                "Bronze (MiB)",
                "10-K/10-Q/8-K",
                "Thời gian",
                "CPU",
                "RAM tối đa (MiB)",
                "RAM cây (MiB)",
                "Filing/phút",
                "MiB Bronze/phút",
                "Row ghi",
            ),
            rows,
        ),
    ]


def _stages(build: Scenario) -> list[str]:
    total = build.wall_seconds
    rows = [
        (
            s.stage,
            f"{s.process.wall_seconds:.1f}",
            f"{s.process.wall_seconds / total:.0%}",
            f"{s.process.cpu_seconds / s.process.wall_seconds:.2f}",
            _mib(s.process.peak_rss_bytes),
            f"{s.rows_inserted:,}",
            f"{s.rows_inserted / s.process.wall_seconds:,.0f}",
            f"{s.process.wall_seconds / build.sample.filings * 1000:.0f}",
        )
        for s in build.stages
    ]
    return [
        f"## 2. Theo stage, mức {build.sample.label} ({build.sample.filings} filing)",
        "",
        "CPU/wall ≈ 1: stage bị giới hạn bởi CPU một lõi; thấp hơn nhiều: chờ I/O.",
        "",
        *_table(
            (
                "Stage",
                "Thời gian (s)",
                "% tổng",
                "CPU/wall",
                "RAM tối đa (MiB)",
                "Row ghi",
                "Row/s",
                "ms/filing",
            ),
            rows,
        ),
    ]


def _slope(low: float, high: float, low_filings: int, high_filings: int) -> float:
    """Change per filing between two samples."""

    return (high - low) / (high_filings - low_filings)


def _halves(samples: Sequence[tuple[float, int]]) -> tuple[int, int]:
    """Peak tree RSS in the first and in the second half of a run."""

    middle = samples[-1][0] / 2
    first = max((rss for elapsed, rss in samples if elapsed < middle), default=0)
    second = max((rss for elapsed, rss in samples if elapsed >= middle), default=0)
    return first, second


def _scaling(builds: Sequence[Scenario], project_filings: int | None) -> list[str]:
    if len(builds) < 2:
        return ["## 3. Scaling", "", "Cần ít nhất hai mức `--fractions`.", ""]
    low, high = builds[0], builds[-1]
    if low.sample.filings == high.sample.filings:
        return [
            "## 3. Scaling",
            "",
            "Mức nhỏ nhất và lớn nhất có cùng số filing: chọn `--fractions` cách xa hơn.",
            "",
        ]
    header = [
        *(f"t {b.sample.label} (s)" for b in builds),
        *(f"RAM {b.sample.label} (MiB)" for b in builds),
        "RAM nửa đầu / nửa sau (MiB)",
    ]
    if project_filings:
        header.append(f"t dự báo {project_filings:,} filing")
    rows = []
    for name in (stage.name for stage in SILVER_STAGES):
        results = [r for b in builds if (r := b.stage(name))]
        if len(results) != len(builds):
            continue
        walls = [r.process.wall_seconds for r in results]
        first, second = _halves(results[-1].process.samples)
        row = [
            name,
            *(f"{w:.1f}" for w in walls),
            *(_mib(r.process.peak_rss_bytes) for r in results),
            f"{_mib(first)} / {_mib(second)}",
        ]
        if project_filings:
            slope = _slope(walls[0], walls[-1], low.sample.filings, high.sample.filings)
            extra = project_filings - high.sample.filings
            row.append(_seconds(walls[-1] + max(slope, 0) * extra))
        rows.append(row)
    return [
        "## 3. Scaling theo stage",
        "",
        (
            f"Thời gian dự báo: tuyến tính giữa mức nhỏ nhất ({low.sample.label}) và lớn "
            f"nhất ({high.sample.label}); chỉ đúng khi tỉ lệ form và kích thước filing giống "
            "mẫu. RAM không được ngoại suy: stage dựng và commit theo lô, nên peak là của lô "
            "nặng nhất, và giữa các mức nhỏ RAM còn tăng vì lô chưa đầy. RAM nửa đầu / nửa "
            "sau là peak RSS (cả cây process) trong hai nửa thời gian của lần chạy lớn nhất: "
            "chỉ khi stage chạy nhiều lô, nửa sau cao hơn hẳn nửa đầu mới là dấu hiệu RAM leo "
            "theo dữ liệu (với một lô, hay stage ngắn mà nửa đầu là lúc khởi động, nửa sau "
            "luôn cao hơn)."
        ),
        "",
        *_table(["Stage", *header], rows),
    ]


def _bottlenecks(build: Scenario, top: int = 3) -> list[str]:
    lines = [
        f"## 4. Bottleneck, mức {build.sample.label}",
        "",
        (
            "Các stage chậm nhất và thao tác/pha tốn thời gian nhất trong log của chúng "
            "(thao tác có thể lồng nhau; pha `*_ms` là tổng qua các lô)."
        ),
        "",
    ]
    slowest = sorted(build.stages, key=lambda s: -s.process.wall_seconds)[:top]
    for result in slowest:
        parts = (result.log.phases + result.log.operations).most_common(5)
        lines.append(
            f"- **{result.stage}** ({result.process.wall_seconds:.0f} s, "
            f"CPU/wall {result.process.cpu_seconds / result.process.wall_seconds:.2f}): "
            + ", ".join(f"`{name}` {seconds:.0f} s" for name, seconds in parts)
        )
    return [*lines, ""]


def _reruns(reruns: Sequence[Scenario]) -> list[str]:
    if not reruns:
        return []
    rows = [
        (
            r.name,
            r.sample.label,
            r.new_filings,
            _seconds(r.wall_seconds),
            _mib(r.peak_rss_bytes),
            f"{sum(s.rows_inserted for s in r.stages):,}",
            _seconds(
                sum(s.process.wall_seconds for s in r.stages if s.stage != "integrity")
            ),
        )
        for r in reruns
    ]
    stage_rows = [
        (
            name,
            *(
                f"{s.process.wall_seconds:.1f}" if (s := r.stage(name)) else "-"
                for r in reruns
            ),
        )
        for name in (stage.name for stage in SILVER_STAGES)
    ]
    return [
        "## 5. Chạy lại (incremental)",
        "",
        (
            "`rerun`: chạy lại khi không có filing mới. `delta`: Silver của mức nhỏ hơn "
            "được cập nhật lên mức lớn nhất (chỉ phần filing chưa có là mới)."
        ),
        "",
        *_table(
            (
                "Kịch bản",
                "Bronze",
                "Filing mới",
                "Thời gian",
                "RAM tối đa (MiB)",
                "Row ghi",
                "Không tính integrity",
            ),
            rows,
        ),
        *_table(("Stage", *(f"{r.name} (s)" for r in reruns)), stage_rows),
    ]


def _failures(scenarios: Sequence[Scenario], work_dir: Path) -> list[str]:
    failed = [(s, f) for s in scenarios if (f := s.failed)]
    if not failed:
        return []
    return [
        "## Lỗi",
        "",
        *(
            f"- `{s.name}` dừng ở stage `{f.stage}` (exit {f.process.exit_code}); "
            f"log: `{work_dir / 'logs' / f'{s.name}.{f.stage}.jsonl'}`"
            for s, f in failed
        ),
        "",
    ]


def render_report(
    builds: Sequence[Scenario],
    reruns: Sequence[Scenario],
    settings: Settings,
    bronze_dir: Path,
    project_filings: int | None,
) -> str:
    complete = [b for b in builds if b.failed is None]
    lines = [
        *_header(builds, settings, bronze_dir),
        *_failures([*builds, *reruns], settings.work_dir),
    ]
    lines += _totals(builds)
    if complete:
        lines += [
            *_stages(complete[-1]),
            *_scaling(complete, project_filings),
            *_bottlenecks(complete[-1]),
        ]
    lines += _reruns(reruns)
    lines.append(
        f"Log từng process: `{settings.work_dir / 'logs'}`; RSS theo thời gian: "
        f"`{settings.work_dir / 'rss_samples.csv'}`."
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--bronze-dir",
        type=Path,
        required=True,
        help="SEC Bronze directory (sec-edgar-filings/, _metadata/); only read",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="New or empty directory for samples, Silver builds, logs, report",
    )
    parser.add_argument(
        "--taxonomy-cache-dir",
        type=Path,
        required=True,
        help="XBRL taxonomy cache shared by every build",
    )
    parser.add_argument(
        "--allow-taxonomy-network",
        action="store_true",
        help="Download missing taxonomy files (slower: times are not offline times)",
    )
    parser.add_argument(
        "--xbrl-workers",
        type=int,
        default=1,
        help="Passed to build_silver.py --xbrl-workers (default: 1)",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
        help="Bronze samples to build from scratch (default: 0.25 0.5 1)",
    )
    parser.add_argument(
        "--skip-incremental",
        action="store_true",
        help="Only the builds from scratch: no rerun or delta scenario",
    )
    parser.add_argument(
        "--project-filings",
        type=int,
        default=None,
        help="Also project each stage's time and memory to this many filings",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=1.0,
        help="Memory sampling interval (default: 1)",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="TMPDIR of the stages, where the audit spills (default: <work-dir>/tmp)",
    )
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("needs Linux (/proc and wait4)")
    if any(not 0 < f <= 1 for f in args.fractions):
        parser.error("--fractions must be in (0, 1]")
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        parser.error(f"--work-dir {args.work_dir} is not empty")
    return args


def main() -> int:
    """Run the scenarios and write the report; exit 1 when a stage failed."""

    args = parse_args()
    work_dir = args.work_dir.expanduser().resolve()
    bronze_dir = args.bronze_dir.expanduser().resolve()
    tmp_dir = (args.tmp_dir or work_dir / "tmp").expanduser().resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        work_dir=work_dir,
        taxonomy_cache_dir=args.taxonomy_cache_dir.expanduser().resolve(),
        allow_taxonomy_network=args.allow_taxonomy_network,
        xbrl_workers=args.xbrl_workers,
        sample_seconds=args.sample_seconds,
        env={**os.environ, "TMPDIR": str(tmp_dir)},
    )
    builds: list[Scenario] = []
    for fraction in sorted(set(args.fractions)):
        sample = sample_bronze(
            bronze_dir, fraction, work_dir / "bronze" / sample_label(fraction)
        )
        if sample.filings == 0:
            print(
                f"--fractions {fraction:g}: the sample holds no filing", file=sys.stderr
            )
            return 2
        silver = work_dir / "silver" / sample.label
        builds.append(
            run_scenario(
                f"build-{sample.label}", silver, sample, sample.filings, settings
            )
        )
        if builds[-1].failed:
            break
    reruns: list[Scenario] = []
    if not args.skip_incremental and builds[-1].failed is None:
        largest = builds[-1]
        silver = work_dir / "silver" / largest.sample.label
        reruns.append(run_scenario("rerun", silver, largest.sample, 0, settings))
        if len(builds) > 1:
            smaller = builds[-2]
            reruns.append(
                run_scenario(
                    "delta",
                    work_dir / "silver" / smaller.sample.label,
                    largest.sample,
                    largest.sample.filings - smaller.sample.filings,
                    settings,
                )
            )
    report = work_dir / "report.md"
    report.write_text(
        render_report(builds, reruns, settings, bronze_dir, args.project_filings),
        encoding="utf-8",
    )
    print(f"report: {report}")
    return 1 if any(s.failed for s in [*builds, *reruns]) else 0


if __name__ == "__main__":
    raise SystemExit(main())

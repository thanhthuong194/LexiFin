#!/usr/bin/env bash
# Build or benchmark SEC Silver on a large Bronze, everything under one OUT directory.
#
#   scripts/sec_filings/run_silver.sh all    BRONZE OUT   warm, bench, build, then status; rerun to resume
#   scripts/sec_filings/run_silver.sh warm   BRONZE OUT   fill the taxonomy cache from a 2% Bronze sample (network)
#   scripts/sec_filings/run_silver.sh bench  BRONZE OUT   benchmark nested samples -> OUT/bench/real/report.md
#   scripts/sec_filings/run_silver.sh build  BRONZE OUT [build_silver.py options]
#                                                         build OUT; rerun the same command to resume
#   scripts/sec_filings/run_silver.sh status BRONZE OUT   last stage events, errors and memory
#
# BRONZE holds sec-edgar-filings/ and _metadata/; the tables land in
# OUT/silver/sec_filings/, next to OUT/_catalog, the taxonomy cache
# (_cache/arelle), logs/, tmp/ (TMPDIR: the audit spills there; tmp/audit keeps
# its scratch database, so an interrupted audit resumes) and bench/.
# warm and bench are skipped once their report exists. build runs offline: when
# a stage stops on taxonomy files missing from the cache, they are downloaded
# and the build resumes (at most TAXONOMY_RETRIES times, default 20). Every Python step
# runs under a hard memory cap (MEMORY_MAX, default 12G): over it, the step is
# killed instead of the machine swapping. WORKERS (default 3) is --xbrl-workers,
# AUDIT_WORKERS (default 4) --audit-workers.
set -euo pipefail

if (($# < 3)); then
  sed -n '2,16p' "$0"
  exit 2
fi
command=$1
bronze=$(realpath "$2")
out=$(realpath -m "$3")
shift 3
workers=${WORKERS:-3}
manifest="$bronze/_metadata/filing_manifest.jsonl"
if [[ ! -d "$bronze/sec-edgar-filings" || ! -f "$manifest" ]]; then
  echo "not an SEC Bronze directory (needs sec-edgar-filings/ and $manifest): $bronze" >&2
  exit 2
fi
# An unmounted disk must not turn into a new directory on the system disk.
if [[ ! -d "$(dirname "$out")" ]]; then
  echo "parent of OUT does not exist (disk not mounted?): $(dirname "$out")" >&2
  exit 2
fi
echo "bronze: $bronze ($(wc -l <"$manifest") manifest lines)"
echo "out:    $out (file system $(df --output=source "$(dirname "$out")" | tail -n 1))"
echo "workers: $workers, memory cap: ${MEMORY_MAX:-12G}"
cd "$(dirname "$0")/../.."
mkdir -p "$out/logs" "$out/tmp"
export TMPDIR="$out/tmp"

capped() {
  systemd-run --user --scope -q -p "MemoryMax=${MEMORY_MAX:-12G}" -p MemorySwapMax=0 "$@"
}

# Benchmark into a work directory, unless an earlier run already wrote its report.
benchmark() {
  local dir=$1
  shift
  if [[ -f "$dir/report.md" ]]; then
    echo "skipped: $dir/report.md exists"
    return
  fi
  if [[ -e "$dir" ]]; then
    echo "$dir has no report.md (an earlier run stopped): remove it to rerun" >&2
    return 2
  fi
  capped uv run python scripts/sec_filings/benchmark_silver.py --work-dir "$dir" \
    --bronze-dir "$bronze" --taxonomy-cache-dir "$out/_cache/arelle" "$@"
}

# Fetch the taxonomies the bench samples use (and most of the build's), online.
warm() {
  local report="$out/bench/taxonomy.json"
  if [[ -f "$report" ]] && grep -q '"unavailable": \[\]' "$report"; then
    echo "skipped: $report exists"
    return
  fi
  capped uv run python scripts/sec_filings/warm_taxonomy.py \
    --taxonomy-cache-dir "$out/_cache/arelle" --bronze-dir "$bronze" \
    --fraction "${TAXONOMY_FRACTION:-0.02}" --report "$report"
}

bench() {
  benchmark "$out/bench/real" --xbrl-workers "$workers" \
    --fractions 0.005 0.01 0.02 --project-filings 62000
}

build() {
  local status attempt start attempt_log="$out/logs/build.attempt.jsonl"
  for ((attempt = 0; ; attempt++)); do
    status=0
    start=1
    [[ -f "$out/logs/build.jsonl" ]] && start=$(($(wc -l <"$out/logs/build.jsonl") + 1))
    # Appended, so a resumed build keeps the log of the runs before it.
    capped /usr/bin/time -v -a -o "$out/logs/build.time" \
      uv run python scripts/sec_filings/build_silver.py \
      --bronze-dir "$bronze" --data-dir "$out" --xbrl-workers "$workers" \
      --audit-work-dir "$out/tmp/audit" --audit-workers "${AUDIT_WORKERS:-4}" "$@" \
      >>"$out/logs/build.jsonl" 2>&1 || status=$?
    tail -n "+$start" "$out/logs/build.jsonl" >"$attempt_log"
    if ((status != 1 || attempt >= ${TAXONOMY_RETRIES:-20})) ||
      ! grep -q '"xbrl_taxonomy_cache_miss"' "$attempt_log"; then
      break
    fi
    echo "taxonomy files missing from the cache: downloading, then resuming the build"
    capped uv run python scripts/sec_filings/warm_taxonomy.py \
      --taxonomy-cache-dir "$out/_cache/arelle" --from-log "$attempt_log" \
      >>"$out/logs/taxonomy.log" 2>&1 || {
      echo "taxonomy download failed; see $out/logs/taxonomy.log" >&2
      break
    }
  done
  rm -f "$attempt_log"
  if ((status == 0)); then
    echo "build completed; log: $out/logs/build.jsonl"
  else
    echo "build failed (exit $status); last errors in $out/logs/build.jsonl:" >&2
    grep '"level": "error"' "$out/logs/build.jsonl" | tail -n 3 | cut -c 1-300 >&2 || true
  fi
  return "$status"
}

status() {
  local log="$out/logs/build.jsonl"
  grep -E '"event": "silver_(stage|pipeline)_(started|completed|failed)"' "$log" |
    tail -n 5 | cut -c 1-300 || true
  echo "errors: $(grep -c '"level": "error"' "$log" || true)"
  grep '"level": "error"' "$log" | tail -n 3 | cut -c 1-300 || true
  free -h | head -n 2
}

case "$command" in
  all)
    warm
    bench
    build "$@"
    status
    ;;
  warm | bench | status) "$command" ;;
  build) build "$@" ;;
  *)
    echo "unknown command: $command (all, warm, bench, build, status)" >&2
    exit 2
    ;;
esac

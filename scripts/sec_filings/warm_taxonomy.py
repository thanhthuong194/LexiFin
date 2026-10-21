"""Fill the Arelle taxonomy cache before an offline Silver build.

The taxonomy schemas come from the submissions of a Bronze sample (every
schema URL they mention, so a later offline build of these filings finds its
DTS cached), or from the `xbrl_taxonomy_cache_miss` events of a build log.
Only URIs not yet cached are loaded, each with its DTS, online. Exit 0 when
everything is cached afterwards, 1 when files stayed unavailable.

    uv run python scripts/sec_filings/warm_taxonomy.py --taxonomy-cache-dir C \
        (--bronze-dir B --fraction 0.02 | --from-log build.jsonl) [--report R]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_silver import MANIFEST, in_sample
from dotenv import load_dotenv

from src.data_pipeline.connectors.sec_filings.sec_edgar import sec_user_agent
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.transforms.sec_filings.xbrl_processor import (
    ProcessorSettings,
    fetch_taxonomies,
)

# Absolute schema URLs, as in xs:import, linkbaseRef and loc hrefs; the W3C
# schemas ship with the processor.
SCHEMA_URL = re.compile(rb"https?://[A-Za-z0-9.\-]+/[^\s\"'<>#]*?\.xsd")
BUNDLED_HOSTS = {"www.w3.org"}


def scan_bronze(bronze_dir: Path, fraction: float) -> set[str]:
    """Schema URLs in the full submissions of the `fraction` sample."""

    source = SecBronze(bronze_dir)
    uris: set[str] = set()
    with (bronze_dir / MANIFEST).open(encoding="utf-8") as manifest:
        entries = [json.loads(line) for line in manifest if line.strip()]
    sampled = [
        e
        for e in entries
        if e["status"] == "success" and in_sample(e["filing_id"], fraction)
    ]
    for count, entry in enumerate(sampled, 1):
        for artifact in entry["artifacts"]:
            if artifact["role"] == "full_submission":
                content = source.path(artifact["relative_path"]).read_bytes()
                uris.update(u.decode("ascii") for u in SCHEMA_URL.findall(content))
        if count % 100 == 0 or count == len(sampled):
            print(f"scanned {count}/{len(sampled)} filings, {len(uris)} URLs")
    return uris


def read_misses(log: Path) -> set[str]:
    """URIs of the `xbrl_taxonomy_cache_miss` events of a JSON-lines log."""

    uris: set[str] = set()
    with log.open(encoding="utf-8", errors="replace") as lines:
        for line in lines:
            if '"xbrl_taxonomy_cache_miss"' not in line:
                continue
            try:
                uris.update(json.loads(line).get("uris", []))
            except json.JSONDecodeError:
                continue
    return uris


def is_cached(uri: str, cache_dir: Path) -> bool:
    parts = urlsplit(uri)
    return (cache_dir / parts.scheme / parts.netloc / parts.path.lstrip("/")).is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--taxonomy-cache-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bronze-dir", type=Path)
    source.add_argument("--from-log", type=Path)
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.02,
        help="Bronze sample to scan (nested like benchmark_silver.py samples)",
    )
    parser.add_argument("--report", type=Path, help="JSON summary to write")
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")

    load_dotenv(REPOSITORY_ROOT / ".env")
    try:
        user_agent = sec_user_agent(
            os.getenv("SEC_COMPANY_NAME", ""), os.getenv("SEC_EMAIL", "")
        )
    except ValueError as error:
        raise SystemExit(f"{error}; add them to .env to download taxonomies") from error
    cache_dir = args.taxonomy_cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.from_log:
        found = read_misses(args.from_log)
    else:
        found = scan_bronze(args.bronze_dir.expanduser().resolve(), args.fraction)
    found = {u for u in found if urlsplit(u).netloc not in BUNDLED_HOSTS}
    missing = sorted(u for u in found if not is_cached(u, cache_dir))
    print(f"{len(found)} schema URLs, {len(missing)} not cached")
    settings = ProcessorSettings(
        cache_dir=cache_dir, allow_network=True, http_user_agent=user_agent
    )
    unavailable = []
    for number, uri in enumerate(missing, 1):
        print(f"[{number}/{len(missing)}] {uri}", flush=True)
        unavailable += fetch_taxonomies([uri], settings)
    unavailable = sorted(set(unavailable))
    for uri in unavailable:
        print(f"unavailable: {uri}", file=sys.stderr)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {"urls": sorted(found), "fetched": missing, "unavailable": unavailable},
                indent=2,
            ),
            encoding="utf-8",
        )
    return 1 if unavailable else 0


if __name__ == "__main__":
    raise SystemExit(main())

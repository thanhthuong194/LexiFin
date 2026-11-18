from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from qdrant_client import QdrantClient, models

from src.data_pipeline.storage.iceberg import (
    open_gold_catalog_read_only,
    table_rules_version,
)
from src.data_pipeline.storage.native_rag.chunks import CHUNKS_TABLE
from src.llm.config import load_profiles
from src.rag_engine.indexing.audit import (
    MIN_INDEXED_SHARE,
    VECTORS_PER_POINT,
    AuditRequest,
    audit_qdrant,
)
from src.rag_engine.indexing.pipeline import LoadRequest, run_qdrant_load
from src.rag_engine.indexing.qdrant_store import (
    SEC_FILINGS,
    collection_name,
    switch_alias,
)
from src.utils.logger import configure_logging

# Seconds per Qdrant request: an upsert of 512 points with text payloads can
# take long while the optimizers run.
QDRANT_TIMEOUT = 120
INDEX_TIMEOUT = 3600
LARGE_INDEX_TIMEOUT = 14400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load native_rag chunks + embeddings (Gold) into Qdrant, then audit"
    )
    parser.add_argument("--gold-dir", type=Path, default=REPOSITORY_ROOT / "data")
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant server (default http://localhost:6333)",
    )
    target.add_argument(
        "--qdrant-path", type=Path, help="Embedded Qdrant stored in this directory"
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Embedding profile whose model's vectors are loaded (default: default)",
    )
    parser.add_argument(
        "--model-id", help="Model ID of the vectors (overrides --profile)"
    )
    parser.add_argument("--collection", help="Collection name (default: derived)")
    parser.add_argument(
        "--alias",
        default=SEC_FILINGS.name,
        help="Corpus alias, switched after a load whose audit has no FAIL and whose "
        "HNSW is built",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Original vectors on disk, int8 copies in RAM (500-company corpus)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only audit (the alias is not touched)",
    )
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument(
        "--sample", type=int, default=20, help="Points the audit samples"
    )
    parser.add_argument(
        "--index-timeout",
        type=int,
        help="Seconds to wait for indexing before the audit (default 3600; "
        "14400 with --large: HNSW over millions of points takes hours)",
    )
    return parser.parse_args()


def wait_until_indexed(client: QdrantClient, name: str, timeout: int) -> str:
    """Wait until the optimizers are idle (green) and HNSW holds nearly every
    vector, or the timeout passes; describe the state."""

    deadline = time.monotonic() + timeout
    while True:
        info = client.get_collection(name)
        points = info.points_count or 0
        indexed = info.indexed_vectors_count or 0
        expected = VECTORS_PER_POINT * points
        ready = info.status == models.CollectionStatus.GREEN and (
            not points or indexed >= MIN_INDEXED_SHARE * expected
        )
        if ready or time.monotonic() > deadline:
            status = getattr(info.status, "value", info.status)
            return f"{status}, {indexed}/{expected} vectors indexed"
        time.sleep(5)


def main() -> int:
    """Exit 0 when the load completed and the audit has no FAIL, else 1."""

    args = parse_args()
    configure_logging()
    # The HTTP client logs one INFO line per request.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    model_id = args.model_id or load_profiles().embedding[args.profile].model
    embedded = args.qdrant_path is not None
    client = (
        QdrantClient(path=str(args.qdrant_path))
        if embedded
        else QdrantClient(url=args.qdrant_url, timeout=QDRANT_TIMEOUT)
    )
    request = LoadRequest(
        model_id=model_id,
        collection=args.collection,
        large=args.large,
    )
    if args.check_only:
        with open_gold_catalog_read_only(args.gold_dir) as gold:
            version = table_rules_version(gold.load_table(CHUNKS_TABLE))
        name = args.collection or collection_name(SEC_FILINGS, version, model_id)
    else:
        summary = run_qdrant_load(args.gold_dir, client, request)
        print(json.dumps(asdict(summary), indent=2))
        if summary.status != "completed":
            return 1
        if args.skip_audit:
            return 0
        name = summary.collection
    if not embedded:
        timeout = args.index_timeout or (
            LARGE_INDEX_TIMEOUT if args.large else INDEX_TIMEOUT
        )
        print(f"collection: {wait_until_indexed(client, name, timeout)}")
    report = audit_qdrant(
        args.gold_dir,
        client,
        AuditRequest(
            model_id=model_id,
            collection=name,
            corpus=SEC_FILINGS,
            embedded=embedded,
            sample=args.sample,
        ),
    )
    for check in report.checks:
        print(f"{check.status:4} {check.name}: {check.detail}")
        for sample in check.samples:
            print(f"       {sample}")
    print(f"audit: {report.status}")
    if args.check_only:
        return 1 if report.status == "FAIL" else 0
    # Only a fully audited collection goes live: an ANN check skipped
    # because HNSW is not built yet blocks the switch too.
    unbuilt = any(c.name == "ann_recall" and c.status == "SKIP" for c in report.checks)
    if report.status == "FAIL" or (unbuilt and not embedded):
        print(
            f"alias {args.alias} left unchanged; once indexed, run again to switch it"
        )
        return 1
    switch_alias(client, name, args.alias)
    print(f"alias {args.alias} -> {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

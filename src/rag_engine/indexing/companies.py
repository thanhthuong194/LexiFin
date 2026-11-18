"""Company directory of a chunk collection: one vectorless point per company.

Retrieval maps a company named in a question to its `cik` (the tenant
filter) from this directory, so it reads Qdrant only. It sits next to the
chunk collection it describes (`<collection>__companies`), so switching the
corpus alias switches the directory with it. Rewritten on every load from
the points actually loaded (a load can leave chunks without a vector out):
companies and chunk counts from a `cik` facet, name, tickers and latest
filing date from each company's newest point; Gold adds the older names a
company filed under.
"""

from __future__ import annotations

import uuid
from typing import Any

from pyiceberg.table import Table
from qdrant_client import QdrantClient, models

from src.data_pipeline.storage.iceberg import scan_batches
from src.rag_engine.indexing.retry import retried

_COMPANY_NAMESPACE = uuid.UUID("0b8a4f5e-2c61-5d3e-9f7a-4e1c2b3d5a60")
# Facet size: far above the 500-company universe.
MAX_COMPANIES = 100_000


def companies_collection(chunk_collection: str) -> str:
    return f"{chunk_collection}__companies"


def company_point_id(cik: str) -> str:
    return str(uuid.uuid5(_COMPANY_NAMESPACE, cik))


def gold_names(chunks: Table) -> dict[str, set[str]]:
    """Every company name each `cik` filed under in Gold."""

    found: dict[str, set[str]] = {}
    for batch in scan_batches(chunks, ("cik", "company_name")):
        for cik, name in zip(
            batch["cik"].to_pylist(), batch["company_name"].to_pylist(), strict=True
        ):
            found.setdefault(cik, set()).add(name)
    return found


def collection_companies(
    client: QdrantClient, chunk_collection: str, names: dict[str, set[str]]
) -> dict[str, dict[str, Any]]:
    """Directory payload per `cik` of the points in `chunk_collection`."""

    facet = retried(
        lambda: client.facet(
            chunk_collection, key="cik", limit=MAX_COMPANIES, exact=True
        ),
        "facet cik",
    )
    found: dict[str, dict[str, Any]] = {}
    for hit in sorted(facet.hits, key=lambda h: str(h.value)):
        cik = str(hit.value)
        latest = _latest_payload(client, chunk_collection, cik)
        if not latest:  # points deleted since the facet: next load lists it
            continue
        found[cik] = {
            "cik": cik,
            "company_name": latest["company_name"],
            "tickers": latest["tickers"],
            "names": sorted(names.get(cik, set()) | {latest["company_name"]}),
            "latest_filing_date": str(latest["filing_date"])[:10],
            "chunk_count": hit.count,
        }
    return found


def _latest_payload(
    client: QdrantClient, chunk_collection: str, cik: str
) -> dict[str, Any]:
    """Name, tickers and filing date of the company's newest point."""

    scope = models.Filter(
        must=[models.FieldCondition(key="cik", match=models.MatchValue(value=cik))]
    )
    fields = ["chunk_id", "company_name", "tickers", "filing_date"]
    newest, _ = retried(
        lambda: client.scroll(
            chunk_collection,
            scroll_filter=scope,
            order_by=models.OrderBy(key="filing_date", direction=models.Direction.DESC),
            limit=1,
            with_payload=fields,
        ),
        "latest point",
    )
    if not newest or not newest[0].payload:
        return {}
    # Several filings on that day: the smallest chunk ID decides, so reruns
    # (and the audit) always agree.
    day = models.Filter(
        must=[
            *(scope.must or []),
            models.FieldCondition(
                key="filing_date",
                range=models.DatetimeRange(
                    gte=newest[0].payload["filing_date"],
                    lte=newest[0].payload["filing_date"],
                ),
            ),
        ]
    )
    same_day, _ = retried(
        lambda: client.scroll(
            chunk_collection, scroll_filter=day, limit=10_000, with_payload=fields
        ),
        "latest points",
    )
    payloads = [p.payload for p in same_day if p.payload] or [newest[0].payload]
    return min(payloads, key=lambda p: str(p["chunk_id"]))


def load_companies(
    client: QdrantClient, chunk_collection: str, names: dict[str, set[str]]
) -> int:
    """Rewrite the directory of `chunk_collection`; drop companies it no
    longer holds. Each server call is retried on its own (creating the index
    is idempotent, so a create that timed out after succeeding still gets
    its index). Returns the number of companies."""

    name = companies_collection(chunk_collection)
    if not retried(lambda: client.collection_exists(name), "directory exists"):
        retried(
            lambda: client.create_collection(name, vectors_config={}),
            "create directory",
        )
    retried(
        lambda: client.create_payload_index(
            name, "cik", models.PayloadSchemaType.KEYWORD
        ),
        "directory index",
    )
    companies = collection_companies(client, chunk_collection, names)
    points = [
        models.PointStruct(id=company_point_id(cik), vector={}, payload=row)
        for cik, row in companies.items()
    ]
    if points:
        retried(lambda: client.upsert(name, points, wait=True), "directory upsert")
    gone = (
        models.Filter(
            must_not=[
                models.FieldCondition(
                    key="cik", match=models.MatchAny(any=list(companies))
                )
            ]
        )
        if companies
        else models.Filter()  # no company left: clear the directory
    )
    retried(
        lambda: client.delete(
            name, points_selector=models.FilterSelector(filter=gone), wait=True
        ),
        "directory delete",
    )
    return len(companies)


def stored_companies(
    client: QdrantClient, chunk_collection: str
) -> list[dict[str, Any]]:
    """Every directory payload of `chunk_collection` (empty when missing)."""

    name = companies_collection(chunk_collection)
    if not client.collection_exists(name):
        return []
    found: list[dict[str, Any]] = []
    offset: Any = None
    while True:
        points, offset = client.scroll(
            name, limit=1000, offset=offset, with_payload=True
        )
        found += [point.payload for point in points if point.payload]
        if offset is None:
            return found

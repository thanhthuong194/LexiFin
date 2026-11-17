"""Qdrant collection layout for one corpus: vectors, payload, indexes, alias.

One collection per (corpus, chunk rules version, index version, embedding
model), e.g. `sec_filings__chunks_v4__index_v1__qwen_qwen3_embedding_0_6b`; the application reads
through the corpus alias (`sec_filings`), switched once a collection is
loaded, indexed and audited (`switch_alias`). Gold stays the source of truth: a collection is rebuilt
from `native_rag.chunks` + `native_rag.embeddings` at any time.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any

from qdrant_client import QdrantClient, models

DENSE = "dense"
SPARSE = "bm25"
# Point IDs: UUID5 of the chunk ID in this namespace (stable across loads).
_POINT_NAMESPACE = uuid.UUID("6f1d3a52-4b7e-5c0a-9a43-2d1e8b6f0c11")


@dataclass(frozen=True)
class CollectionSettings:
    """Index parameters; `large` keeps original vectors on disk with int8
    copies in RAM (rescored), for the 500-company corpus."""

    dimensions: int
    large: bool = False
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    search_hnsw_ef: int = 128
    quantile: float = 0.99
    oversampling: float = 2.0
    indexing_threshold: int = 20_000


@dataclass(frozen=True)
class Corpus:
    """What every point of a corpus says about where it comes from."""

    name: str
    jurisdiction: str
    language: str


SEC_FILINGS = Corpus("sec_filings", "US", "en")

# Payload fields with an index, and their type. `cik` is the tenant: most
# questions name a company.
KEYWORD_FIELDS = (
    "tickers",
    "form",
    "document_type",
    "section_topic",
    "chunk_kind",
    "fiscal_period",
    "filing_id",
    "document_id",
    "section_id",
    "corpus",
)
INTEGER_FIELDS = ("fiscal_year", "chunk_ordinal")
DATETIME_FIELDS = ("filing_date", "period_end_date")
TENANT_FIELD = "cik"


# Version of what a point is built from besides the chunk text: payload
# fields and the BM25 vector (tokenizer, stopwords, AVG_DOC_WORDS). Bump it
# when either changes: chunk hashes do not change, so an incremental load into
# the old collection would skip every point.
INDEX_VERSION = 1


def collection_name(corpus: Corpus, rules_version: int, model_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")
    return f"{corpus.name}__chunks_v{rules_version}__index_v{INDEX_VERSION}__{slug}"


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def rfc3339(day: date | None) -> str | None:
    return f"{day.isoformat()}T00:00:00Z" if day else None


def payload(chunk: dict[str, Any], corpus: Corpus) -> dict[str, Any]:
    """Point payload from a `native_rag.chunks` row: filter, citation and
    context-expansion fields, the text, and the hash it was built from."""

    keys = (
        "chunk_id",
        "filing_id",
        "document_id",
        "section_id",
        "cik",
        "tickers",
        "company_name",
        "form",
        "document_type",
        "fiscal_year",
        "fiscal_period",
        "section_topic",
        "section_title",
        "subheading",
        "chunk_kind",
        "table_title",
        "chunk_ordinal",
        "part",
        "part_count",
        "accession_number",
        "sec_url",
        "context",
        "text",
        "token_count",
        "chunk_sha256",
    )
    found = {key: chunk[key] for key in keys}
    found["filing_date"] = rfc3339(chunk["filing_date"])
    found["period_end_date"] = rfc3339(chunk["period_end_date"])
    found["corpus"] = corpus.name
    found["jurisdiction"] = corpus.jurisdiction
    found["language"] = corpus.language
    return found


def ensure_collection(
    client: QdrantClient, name: str, settings: CollectionSettings
) -> bool:
    """Create the collection unless it exists; return whether it was created.

    A new collection starts with indexing off (bulk load); `enable_indexing`
    turns it on. An existing one must have exactly the requested layout
    (`collection_problems`), else ValueError: delete it and load again.
    Payload indexes are (re)created on every run, which is idempotent, so a
    crash between creating the collection and its indexes heals.
    """

    created = False
    if client.collection_exists(name):
        problems = collection_problems(client, name, settings)
        if problems:
            raise ValueError(
                f"collection {name} differs from the requested layout "
                f"({'; '.join(problems)}): delete it and load again"
            )
    else:
        _create(client, name, settings)
        created = True
    _create_indexes(client, name)
    return created


def _create(client: QdrantClient, name: str, settings: CollectionSettings) -> None:
    quantization = (
        models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=settings.quantile,
                always_ram=True,
            )
        )
        if settings.large
        else None
    )
    client.create_collection(
        name,
        vectors_config={
            DENSE: models.VectorParams(
                size=settings.dimensions,
                distance=models.Distance.COSINE,
                on_disk=settings.large,
                hnsw_config=models.HnswConfigDiff(
                    m=settings.hnsw_m, ef_construct=settings.hnsw_ef_construct
                ),
            )
        },
        sparse_vectors_config={
            SPARSE: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=settings.large),
                modifier=models.Modifier.IDF,
            )
        },
        hnsw_config=models.HnswConfigDiff(
            m=settings.hnsw_m, ef_construct=settings.hnsw_ef_construct
        ),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        quantization_config=quantization,
        on_disk_payload=settings.large,
    )


def _create_indexes(client: QdrantClient, name: str) -> None:
    client.create_payload_index(
        name,
        TENANT_FIELD,
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD, is_tenant=True
        ),
    )
    for field in KEYWORD_FIELDS:
        client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)
    for field in INTEGER_FIELDS:
        client.create_payload_index(name, field, models.PayloadSchemaType.INTEGER)
    for field in DATETIME_FIELDS:
        client.create_payload_index(name, field, models.PayloadSchemaType.DATETIME)


def collection_problems(
    client: QdrantClient, name: str, settings: CollectionSettings
) -> list[str]:
    """How an existing collection's layout differs from `settings`."""

    config = client.get_collection(name).config
    params = config.params
    vectors = params.vectors if isinstance(params.vectors, dict) else {}
    dense = vectors.get(DENSE)
    sparse = (params.sparse_vectors or {}).get(SPARSE)
    found = {
        "dense size": dense.size if dense else None,
        "distance": dense.distance if dense else None,
        "dense on disk": bool(dense.on_disk) if dense else None,
        "sparse IDF": sparse.modifier == models.Modifier.IDF if sparse else None,
        "quantized": config.quantization_config is not None,
        "hnsw m": _hnsw(dense, config).m,
        "hnsw ef_construct": _hnsw(dense, config).ef_construct,
    }
    wanted = {
        "dense size": settings.dimensions,
        "distance": models.Distance.COSINE,
        "dense on disk": settings.large,
        "sparse IDF": True,
        "quantized": settings.large,
        "hnsw m": settings.hnsw_m,
        "hnsw ef_construct": settings.hnsw_ef_construct,
    }
    return [
        f"{key} {found[key]} != {wanted[key]}"
        for key in wanted
        if found[key] != wanted[key]
    ]


def _hnsw(
    dense: models.VectorParams | None, config: models.CollectionConfig
) -> models.HnswConfigDiff | models.HnswConfig:
    """The dense vector's HNSW settings (they override the collection's)."""

    return dense.hnsw_config if dense and dense.hnsw_config else config.hnsw_config


def is_large(client: QdrantClient, name: str) -> bool:
    """Whether the collection keeps int8 copies (search must rescore)."""

    return client.get_collection(name).config.quantization_config is not None


def enable_indexing(
    client: QdrantClient, name: str, settings: CollectionSettings
) -> None:
    """Turn indexing on (the optimizers build HNSW in the background)."""

    client.update_collection(
        name,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=settings.indexing_threshold
        ),
    )


def switch_alias(client: QdrantClient, name: str, alias: str) -> None:
    """Point the corpus alias at `name` in one atomic operation."""

    current = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
    if current.get(alias) == name:
        return
    operations: list[models.AliasOperations] = []
    if alias in current:
        operations.append(
            models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(alias_name=alias)
            )
        )
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=name, alias_name=alias)
        )
    )
    client.update_collection_aliases(change_aliases_operations=operations)


def search_params(
    settings: CollectionSettings, *, exact: bool = False
) -> models.SearchParams:
    return models.SearchParams(
        hnsw_ef=settings.search_hnsw_ef,
        exact=exact,
        quantization=models.QuantizationSearchParams(
            rescore=True, oversampling=settings.oversampling
        )
        if settings.large
        else None,
    )

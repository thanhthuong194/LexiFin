from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from pyiceberg.catalog import Catalog

from src.data_pipeline.storage.iceberg import scan_batches
from src.data_pipeline.storage.sec_filings.documents import (
    DOCUMENTS_TABLE,
    is_narrative_document,
)
from src.data_pipeline.storage.sec_filings.exhibits import EXHIBITS_TABLE
from src.data_pipeline.storage.sec_filings.filing_entities import FILING_ENTITIES_TABLE
from src.data_pipeline.storage.sec_filings.filings import FILINGS_TABLE
from src.data_pipeline.storage.sec_filings.xbrl_core import (
    XBRL_CONTEXTS_TABLE,
    XBRL_INSTANCES_TABLE,
)
from src.data_pipeline.storage.sec_filings.xbrl_facts import (
    XBRL_CONCEPTS_TABLE,
    XBRL_FACTS_TABLE,
)

DEI_CONCEPTS = (
    "EntityRegistrantName",
    "TradingSymbol",
    "Security12bTitle",
    "DocumentPeriodEndDate",
    "DocumentFiscalYearFocus",
    "DocumentFiscalPeriodFocus",
)
_DEI_NAMESPACE = "xbrl.sec.gov/dei/"
_TICKER = re.compile(r"^[A-Z][A-Z.\-]{0,9}$")
# Securities whose symbols are not the company's shares (notes, debentures...).
_DEBT_TITLE = re.compile(r"note|debenture|bond|%|due \d{4}", re.IGNORECASE)
_FISCAL_PERIODS = frozenset({"FY", "Q1", "Q2", "Q3", "Q4"})


@dataclass(frozen=True)
class FilingMeta:
    """Filing-level fields every chunk of a filing carries."""

    filing_id: str
    accession_number: str
    cik: str
    form: str
    filing_date: date
    company_name: str
    tickers: tuple[str, ...]
    period_end_date: date | None
    fiscal_year: int | None
    fiscal_period: str | None


@dataclass(frozen=True)
class DocumentMeta:
    """A narrative document's fields used by its chunks."""

    document_id: str
    filing_id: str
    filing_date: date
    document_type: str
    filename: str | None
    label: str  # Exhibit path title: `EX-99.1: Press release`, or the type.


@dataclass
class _Dei:
    # Values per concept from the filing's own (non-dimensional) contexts.
    values: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    # Every context's values per concept, to pair symbols with security titles.
    by_context: dict[str, dict[str, list[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )


def _most_common(values: Counter[str]) -> str | None:
    if not values:
        return None
    return min(values.items(), key=lambda item: (-item[1], item[0]))[0]


def _parse_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def _tickers(dei: _Dei) -> tuple[str, ...]:
    """Trading symbols of the company's shares: each context's symbols count
    unless a security title in that context names debt (notes...)."""

    symbols: set[str] = set()
    for facts in dei.by_context.values():
        if any(_DEBT_TITLE.search(t) for t in facts.get("Security12bTitle", [])):
            continue
        for raw in facts.get("TradingSymbol", []):
            symbol = raw.strip().upper()
            if _TICKER.match(symbol):
                symbols.add(symbol)
    return tuple(sorted(symbols))


def _dimensional_contexts(catalog: Catalog, keys: Collection[str]) -> set[str]:
    """Which of `keys` have a segment or scenario (dimensional contexts, such
    as one registrant of a multi-registrant filing)."""

    wanted = pa.array(sorted(keys), type=pa.string())
    found: set[str] = set()
    contexts = catalog.load_table(XBRL_CONTEXTS_TABLE)
    for batch in scan_batches(contexts, ("context_key", "segment_xml", "scenario_xml")):
        matching = batch.filter(pc.is_in(batch["context_key"], value_set=wanted))
        for row in matching.to_pylist():
            if row["segment_xml"] or row["scenario_xml"]:
                found.add(row["context_key"])
    return found


def _dei_concepts(catalog: Catalog) -> dict[str, str]:
    """`concept_id` -> local name of the dei concepts read, every dei year."""

    names: dict[str, str] = {}
    concepts = catalog.load_table(XBRL_CONCEPTS_TABLE)
    for batch in scan_batches(concepts, ("concept_id", "namespace_uri", "local_name")):
        for row in batch.to_pylist():
            if (
                row["local_name"] in DEI_CONCEPTS
                and _DEI_NAMESPACE in row["namespace_uri"]
            ):
                names[row["concept_id"]] = row["local_name"]
    return names


def _instance_filings(catalog: Catalog) -> dict[str, str]:
    filing_of: dict[str, str] = {}
    instances = catalog.load_table(XBRL_INSTANCES_TABLE)
    for batch in scan_batches(instances, ("instance_id", "filing_id")):
        for row in batch.to_pylist():
            filing_of[row["instance_id"]] = row["filing_id"]
    return filing_of


def read_dei(catalog: Catalog) -> dict[str, _Dei]:
    """The cover-page dei facts of every filing, in one pass over the facts.

    Registrant name and period fields count only facts of contexts without a
    segment or scenario: a multi-registrant filing (Duke Energy) reports each
    subsidiary's name under a `LegalEntityAxis` context.
    """

    names = _dei_concepts(catalog)
    filing_of = _instance_filings(catalog)
    facts_found: list[tuple[str, str, str, str]] = []
    wanted = pa.array(sorted(names), type=pa.string())
    facts = catalog.load_table(XBRL_FACTS_TABLE)
    columns = (
        "instance_id",
        "concept_id",
        "context_key",
        "normalized_value",
        "raw_value",
    )
    for batch in scan_batches(facts, columns):
        matching = batch.filter(pc.is_in(batch["concept_id"], value_set=wanted))
        for row in matching.to_pylist():
            filing_id = filing_of.get(row["instance_id"])
            value = (row["normalized_value"] or row["raw_value"] or "").strip()
            if filing_id is not None and value:
                facts_found.append(
                    (filing_id, names[row["concept_id"]], row["context_key"], value)
                )
    dimensional = _dimensional_contexts(catalog, {f[2] for f in facts_found})
    found: dict[str, _Dei] = defaultdict(_Dei)
    for filing_id, name, context_key, value in sorted(facts_found):
        dei = found[filing_id]
        if context_key not in dimensional:
            dei.values[name][value] += 1
        dei.by_context[context_key][name].append(value)
    return dict(found)


def read_filing_meta(catalog: Catalog) -> dict[str, FilingMeta]:
    """Filing metadata for chunks: Silver filings + filer entity + dei facts.

    `company_name` is dei `EntityRegistrantName`, else the conformed name of
    the filer entity whose CIK is the filing's; `period_end_date` is dei
    `DocumentPeriodEndDate`, else `report_date`; fiscal year/period come from
    dei and are null for 8-Ks.
    """

    dei_by_filing = read_dei(catalog)
    conformed: dict[str, tuple[int, str]] = {}
    entities = catalog.load_table(FILING_ENTITIES_TABLE)
    columns = ("filing_id", "entity_ordinal", "entity_role", "cik", "company_name")
    filings = catalog.load_table(FILINGS_TABLE)
    rows = [
        row
        for batch in scan_batches(
            filings,
            (
                "filing_id",
                "accession_number",
                "cik",
                "form",
                "filing_date",
                "report_date",
            ),
        )
        for row in batch.to_pylist()
    ]
    filing_cik = {row["filing_id"]: row["cik"] for row in rows}
    for batch in scan_batches(entities, columns):
        for row in batch.to_pylist():
            if row["entity_role"] != "filer" or row["cik"] != filing_cik.get(
                row["filing_id"]
            ):
                continue
            current = conformed.get(row["filing_id"])
            if current is None or row["entity_ordinal"] < current[0]:
                conformed[row["filing_id"]] = (
                    row["entity_ordinal"],
                    row["company_name"],
                )

    names = {
        row["filing_id"]: _most_common(
            dei_by_filing.get(row["filing_id"], _Dei()).values["EntityRegistrantName"]
        )
        or conformed.get(row["filing_id"], (0, row["cik"]))[1]
        for row in rows
    }
    canonical = _canonical_names(rows, names)
    meta: dict[str, FilingMeta] = {}
    for row in rows:
        filing_id = row["filing_id"]
        dei = dei_by_filing.get(filing_id, _Dei())
        name = canonical[filing_id]
        is_8k = row["form"] == "8-K"
        year = _most_common(dei.values["DocumentFiscalYearFocus"])
        period = _most_common(dei.values["DocumentFiscalPeriodFocus"])
        meta[filing_id] = FilingMeta(
            filing_id=filing_id,
            accession_number=row["accession_number"],
            cik=row["cik"],
            form=row["form"],
            filing_date=row["filing_date"],
            company_name=name,
            tickers=_tickers(dei),
            period_end_date=_parse_date(
                _most_common(dei.values["DocumentPeriodEndDate"])
            )
            or row["report_date"],
            fiscal_year=int(year) if year and year.isdigit() and not is_8k else None,
            fiscal_period=period if period in _FISCAL_PERIODS and not is_8k else None,
        )
    return meta


_NAME_NOISE = re.compile(r"[^a-z0-9]+")
_NAME_SUFFIXES = re.compile(r"(corporation|corp|incorporated|inc|company|co|ltd|plc)$")


def _name_key(name: str) -> str:
    key = _NAME_NOISE.sub("", name.lower())
    return _NAME_SUFFIXES.sub("", key)


def _canonical_names(
    rows: Sequence[dict[str, Any]], names: Mapping[str, str]
) -> dict[str, str]:
    """One spelling per company name: filings of a CIK whose names differ only
    in case, punctuation or suffix (`MICROSOFT CORP`, `Microsoft Corporation`)
    get the spelling of the earliest filing; a real rename (another key)
    keeps its own name. Only the earliest filing decides (v5 preferred mixed
    case, so a later mixed-case filing renamed chunks already stored), so new
    filings never change the choice and chunks stored earlier keep matching."""

    first: dict[tuple[str, str], tuple[date, str, str]] = {}
    for row in rows:
        name = " ".join(names[row["filing_id"]].split())
        group = (row["cik"], _name_key(name))
        rank = (row["filing_date"], row["filing_id"], name)
        if group not in first or rank < first[group]:
            first[group] = rank
    chosen = {group: rank[2] for group, rank in first.items()}
    result: dict[str, str] = {}
    for row in rows:
        name = " ".join(names[row["filing_id"]].split())
        result[row["filing_id"]] = chosen[(row["cik"], _name_key(name))] or name
    return result


def read_document_meta(catalog: Catalog) -> dict[str, DocumentMeta]:
    """Every narrative document, with the label of EX-99 exhibits.

    The label is the Exhibit Index description of the entry that resolves to
    the document, else `documents.description` when it differs from the
    type, else the type.
    """

    index_titles: dict[str, str] = {}
    exhibits = catalog.load_table(EXHIBITS_TABLE)
    for batch in scan_batches(
        exhibits,
        ("resolved_document_id", "exhibit_ordinal", "description", "filing_id"),
    ):
        for row in sorted(
            batch.to_pylist(), key=lambda r: (r["filing_id"], r["exhibit_ordinal"])
        ):
            target, text = row["resolved_document_id"], row["description"]
            if target and text and target not in index_titles:
                index_titles[target] = " ".join(text.split())

    documents = catalog.load_table(DOCUMENTS_TABLE)
    columns = (
        "document_id",
        "filing_id",
        "filing_date",
        "document_type",
        "filename",
        "description",
        "role",
        "content_kind",
    )
    meta: dict[str, DocumentMeta] = {}
    for batch in scan_batches(documents, columns):
        for row in batch.to_pylist():
            if not is_narrative_document(
                row["role"], row["content_kind"], row["document_type"]
            ):
                continue
            meta[row["document_id"]] = DocumentMeta(
                document_id=row["document_id"],
                filing_id=row["filing_id"],
                filing_date=row["filing_date"],
                document_type=row["document_type"],
                filename=row["filename"],
                label=_label(row, index_titles.get(row["document_id"])),
            )
    return meta


def _label(row: dict[str, str], index_title: str | None) -> str:
    kind = row["document_type"]
    description = " ".join((row["description"] or "").split())
    title = index_title or (description if description.upper() != kind.upper() else "")
    return f"{kind}: {title}" if title else kind


def sec_url(cik: str, accession_number: str, filename: str | None) -> str:
    """The document's page on EDGAR (the filing index when no filename)."""

    folder = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession_number.replace('-', '')}"
    )
    return f"{folder}/{filename}" if filename else f"{folder}/"

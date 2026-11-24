"""Gold and Silver reads for the eval sampler: chunk metadata, texts and
corpus stats from Gold; numeric XBRL facts and where they show from Silver."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal, NamedTuple

import pyarrow as pa
import pyarrow.compute as pc
from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.storage.iceberg import (
    filing_dates_filter,
    iter_rows,
    read_rows_in,
    scan_batches,
    snapshot_total,
    values_filter,
)
from src.data_pipeline.storage.sec_filings.xbrl_core import (
    XBRL_CONTEXTS_TABLE,
    XBRL_INSTANCES_TABLE,
    XBRL_UNITS_TABLE,
)
from src.data_pipeline.storage.sec_filings.xbrl_facts import (
    XBRL_CONCEPTS_TABLE,
    XBRL_FACTS_TABLE,
)
from src.evals.testset import CompanyRef, XbrlFact
from src.rag_engine.retrieval.companies import ALIASES, normalize_name


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


def _names(text: str) -> tuple[str, ...]:
    return tuple(text.split())


# us-gaap local names numeric tasks ask for.
KEY_CONCEPTS = _words(
    "Revenues RevenueFromContractWithCustomerExcludingAssessedTax SalesRevenueNet "
    "CostOfRevenue CostOfGoodsAndServicesSold GrossProfit ResearchAndDevelopmentExpense "
    "OperatingExpenses OperatingIncomeLoss IncomeTaxExpenseBenefit NetIncomeLoss "
    "EarningsPerShareBasic EarningsPerShareDiluted CashAndCashEquivalentsAtCarryingValue "
    "AccountsReceivableNetCurrent InventoryNet Goodwill Assets Liabilities "
    "LongTermDebtNoncurrent StockholdersEquity NetCashProvidedByUsedInOperatingActivities "
    "PaymentsToAcquirePropertyPlantAndEquipment PaymentsForRepurchaseOfCommonStock "
    "PaymentsOfDividends ShareBasedCompensation AllocatedShareBasedCompensationExpense"
)
# (name, tickers, keys): US companies, large public ones first. A name is
# dropped when a corpus company has its ticker or name, or a chunk mentions
# one of its keys (case-sensitive, at a word start; default the name).
OUTSIDE_COMPANIES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("Exxon Mobil", ("XOM",), ("Exxon",)),
    ("Coca-Cola", ("KO",), ()),
    ("JPMorgan Chase", ("JPM",), ("JPMorgan", "J.P. Morgan")),
    ("Walmart", ("WMT",), ("Walmart", "Wal-Mart")),
    ("Johnson & Johnson", ("JNJ",), ()),
    ("Procter & Gamble", ("PG",), ()),
    ("Chevron", ("CVX",), ()),
    ("Pfizer", ("PFE",), ()),
    ("Boeing", ("BA",), ()),
    ("Caterpillar", ("CAT",), ()),
    ("Home Depot", ("HD",), ()),
    ("McDonald's", ("MCD",), ("McDonald",)),
    ("Starbucks", ("SBUX",), ()),
    ("UnitedHealth Group", ("UNH",), ("UnitedHealth",)),
    ("Eli Lilly", ("LLY",), ("Lilly",)),
    ("Berkshire Hathaway", ("BRK.B", "BRK.A"), ("Berkshire",)),
    ("PepsiCo", ("PEP",), ()),
    ("Goldman Sachs", ("GS",), ()),
    ("Lockheed Martin", ("LMT",), ("Lockheed",)),
    ("Deere & Company", ("DE",), ("Deere",)),
    ("General Motors", ("GM",), ()),
    ("Ford Motor", ("F",), ()),
    ("Delta Air Lines", ("DAL",), ("Delta Air",)),
    ("Tesla", ("TSLA",), ()),
    ("Netflix", ("NFLX",), ()),
    ("Snowflake", ("SNOW",), ()),
    ("Rivian", ("RIVN",), ()),
    ("Roblox", ("RBLX",), ()),
    ("Pinterest", ("PINS",), ()),
    ("Peloton", ("PTON",), ()),
    ("Chewy", ("CHWY",), ()),
    ("Publix Super Markets", (), ("Publix",)),
    ("Cargill", (), ()),
    ("Koch Industries", (), ("Koch",)),
    ("Bechtel", (), ()),
    ("Chick-fil-A", (), ()),
    ("Fidelity Investments", (), ("Fidelity",)),
    ("SpaceX", (), ()),
    ("Stripe", (), ()),
    ("Databricks", (), ()),
)
# Topics a company plausibly never discusses; a company whose chunks contain
# one of a probe's keys (lowercase) is not asked about it.
TOPIC_PROBES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("proved oil and gas reserves", ("proved reserves", "oil and gas reserves")),
    ("FDA approval of its drug candidates", ("drug candidate", "clinical trial")),
    ("coal mine reclamation liabilities", ("coal mining", "mine reclamation")),
    ("crop insurance and farm subsidy payments", ("crop insurance", "farm subsid")),
    ("FDIC deposit insurance assessments", ("fdic", "deposit insurance")),
    ("tobacco Master Settlement Agreement payments", ("tobacco", "master settlement")),
    ("casino gaming licenses in Macau", ("casino", "macau")),
    ("Medicare Advantage star ratings", ("medicare advantage", "star rating")),
    ("nuclear plant decommissioning trusts", ("decommissioning trust", "nuclear")),
    ("timber harvest volumes from its timberlands", ("timberland", "timber harvest")),
)
# Coverage dimensions, in `ChunkMeta.dims` / `FactCandidate.dims` order.
CHUNK_DIMS = (
    "cik",
    "section_topic",
    "doc_class",
    "chunk_kind",
    "split",
    "year",
    "period",
)
FACT_DIMS = ("cik", "form", "concept", "period", "year")
# Text shorter than this that only points elsewhere or is boilerplate is skipped.
STUB_CHARS = 1500
_STOPWORDS = _words(
    "a an and as at by for from in of on or the to condensed consolidated unaudited "
    "audited inc corp corporation company millions thousands except per share shares "
    "amounts data ended months year years following table january february march "
    "april may june july august september october november december"
)
_LIST_LINE = re.compile(r"^\s*(?:[•●◦▪■–—]|\([a-z]{1,4}\))\s+\S", re.MULTILINE)
_BULLET_START = re.compile(r"^\s*[•●◦▪■–—*]")
# A heading that is a table row or an address line: ends in a long number or `*`.
_ROW_HEADING = re.compile(r"(?:\d[\d,]{4,}|\*)\s*$")
_POINTER = re.compile(
    r"incorporated (?:herein )?by reference|\brefer to\b|is set forth in|"
    r"appears in|may be found|will be (?:included|contained) in|"
    r"(?:is|are) described in",
    re.IGNORECASE,
)
_BOILERPLATE = re.compile(
    r"no changes? in (?:our|the company['’]s) internal control|"
    r"disclosure controls and procedures\b[^.]{0,200}\b(?:were|are) effective|"
    r"(?:facilities|properties)[^.]{0,80}(?:suitable|adequate|good operating condition)",
    re.IGNORECASE,
)
_FORWARD_LOOKING = re.compile(r"forward-looking statements?|safe harbor", re.IGNORECASE)
_STATEMENT = re.compile(
    r"balance sheets?|statements? of (?:consolidated )?(?:operations|income|earnings|"
    r"cash flows?|comprehensive income)|income statements?|cash flows? statements?",
    re.IGNORECASE,
)
_NOTE_HEADING = re.compile(r"›\s*note\s+(\d{1,2})\b", re.IGNORECASE)
_NOTE_REF = re.compile(r"\bNotes? (\d{1,2})\b")
_EXHIBIT_REF = re.compile(r"\bExhibit (99\.\d{1,2})\b")
_YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_FISCAL_YEAR = re.compile(r"\b(?:fiscal|FY) ?(?:year )?(\d{4})\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")
_CELL_ID_INFIX = ":table:cell:"
_UNIT_PREFIXES = {
    "http://www.xbrl.org/2003/iso4217": "iso4217",
    "http://www.xbrl.org/2003/instance": "xbrli",
}


def rank(seed: int, key: str) -> int:
    """Stable pseudo-random order of `key` for `seed`."""

    digest = hashlib.sha256(f"{seed}\x00{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def exhibit_number(number: str) -> str:
    """`99.01` and `99.1` -> `99.1` (`EX-99.01` is how some filers type 99.1)."""

    major, _, minor = number.partition(".")
    return f"{major}.{int(minor or 0)}"


def _quarter(day: date) -> int:
    return (day.month - 1) // 3 + 1


def _title_key(raw: str | None, company_name: str, max_chars: int) -> str:
    """Lowercase content words of a heading, without the company's name or
    parentheses, so one table matches across periods and companies; empty
    for headings that are bullets, table rows or longer than `max_chars`."""

    if (
        not raw
        or len(raw) > max_chars
        or _BULLET_START.match(raw)
        or _ROW_HEADING.search(raw)
    ):
        return ""
    company = set(_WORD.findall(company_name.lower()))
    words = [
        word
        for word in re.findall(r"[a-z]+", re.sub(r"\([^)]*\)", " ", raw.lower()))
        if word not in _STOPWORDS and word not in company
    ]
    key = " ".join(words[:8])
    return key if len(key) >= 5 else ""


# ALIASES that name a segment or product, not the company: "AWS net income"
# asks for a segment's number (v1 writers had to write "AWS parent Amazon").
_PRODUCT_ALIASES = frozenset({"aws", "youtube", "instagram", "whatsapp"})


def company_aliases(name: str) -> tuple[str, ...]:
    """Retrieval ALIASES (`google` -> `alphabet`) that name this company."""

    target = normalize_name(name)
    return tuple(
        sorted(
            alias.title()
            for alias, company in ALIASES.items()
            if company == target and alias not in _PRODUCT_ALIASES
        )
    )


def company_ref(cik: str, name: str, tickers: Sequence[str]) -> CompanyRef:
    return CompanyRef(
        cik=cik, name=name, tickers=tuple(tickers), aliases=company_aliases(name)
    )


@dataclass(frozen=True, slots=True)
class ChunkMeta:
    """A Gold chunk's metadata (no text); the last fields describe its text."""

    chunk_id: str
    filing_id: str
    document_id: str
    block_ids: tuple[str, ...]
    cik: str
    company_name: str
    tickers: tuple[str, ...]
    form: str
    document_type: str
    section_topic: str
    section_title: str
    subheading: str | None
    table_title: str | None
    chunk_kind: str
    fiscal_year: int | None
    fiscal_period: str | None
    period_end_date: date | None
    filing_date: date
    part: int | None
    part_count: int | None
    chars: int
    list_lines: int = 0
    text_hash: int = 0  # of its first and last 300 chars: near-duplicates match
    note: str | None = None  # number of the financial statement note it is in
    note_refs: tuple[str, ...] = ()  # notes its text cites ("see Note 12")
    exhibit_refs: tuple[str, ...] = ()  # exhibits its text cites ("99.1")
    title: str = field(init=False)  # `_title_key` of its table title / subheading

    def __post_init__(self) -> None:
        table = self.chunk_kind == "table"
        raw = self.table_title if table else self.subheading
        title = _title_key(raw, self.company_name, 400 if table else 100)
        object.__setattr__(self, "title", title)

    @property
    def doc_class(self) -> str:
        return "EX-99" if self.document_type.startswith("EX-99") else self.form

    @property
    def year(self) -> int:
        return self.fiscal_year or self.filing_date.year

    @property
    def period(self) -> str:
        """Fiscal period; for 8-Ks the calendar quarter filed (`filed_Q3`)."""

        return self.fiscal_period or f"filed_Q{_quarter(self.filing_date)}"

    @property
    def calendar_period(self) -> str:
        """Calendar year (10-K) or quarter (10-Q) mostly covered, so fiscal
        calendars align (NVIDIA FY2025 ends January 2025: `CY2024`); for
        8-Ks the quarter filed."""

        if self.period_end_date is None or self.fiscal_period is None:
            return f"filed_{self.filing_date.year}Q{_quarter(self.filing_date)}"
        covered = self.period_end_date - timedelta(days=45)
        if self.fiscal_period == "FY":
            return f"CY{covered.year}"
        return f"CY{covered.year}Q{_quarter(covered)}"

    @property
    def parts(self) -> tuple[int, int]:
        return self.part or 0, self.part_count or 0

    @property
    def is_statement(self) -> bool:
        """A primary statement (balance sheet, income, cash flows), not a note."""

        heading = f"{self.table_title or ''} {self.subheading or ''}"
        return self.note is None and bool(_STATEMENT.search(heading))

    def dims(self) -> tuple[str, ...]:
        split = f"part{min(self.part, 3)}" if self.part else "whole"
        return (
            self.cik,
            self.section_topic,
            self.doc_class,
            self.chunk_kind,
            split,
            str(self.year),
            self.period,
        )


@dataclass(frozen=True, slots=True)
class ChunkText:
    context: str
    text: str


class FilingInfo(NamedTuple):
    cik: str
    form: str
    fiscal_year: int | None
    fiscal_period: str | None
    period_end_date: date | None
    filing_date: date


class Occurrence(NamedTuple):
    block_id: str
    cell_id: str | None
    raw_value: str


@dataclass(frozen=True, slots=True)
class FactCandidate:
    """A non-dimensional numeric XBRL fact of a 10-K/10-Q and every block
    showing it; `fact.block_id` is set to the passage's occurrence once picked."""

    fact: XbrlFact
    filing_id: str
    filing: FilingInfo
    occurrences: tuple[Occurrence, ...]
    is_current: bool

    def dims(self) -> tuple[str, ...]:
        info = self.filing
        return (
            info.cik,
            info.form,
            self.fact.concept,
            info.fiscal_period or "",
            str(info.fiscal_year),
        )


@dataclass
class CorpusStats:
    """Facts about the whole Gold table, gathered in the first chunk pass."""

    companies: dict[str, CompanyRef] = field(default_factory=dict)
    first_year: dict[str, int] = field(default_factory=dict)
    filings: dict[str, FilingInfo] = field(default_factory=dict)
    # Per company: years its chunks mention, indexes of TOPIC_PROBES they match.
    years: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    probe_hits: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    # Indexes of OUTSIDE_COMPANIES some chunk mentions.
    mentioned: set[int] = field(default_factory=set)
    rows: int = 0


# --- Gold reads ------------------------------------------------------------------

# ChunkMeta fields copied as they are from Gold.
_META_FIELDS = _names(
    "chunk_id filing_id document_id cik company_name form document_type "
    "section_topic section_title subheading table_title chunk_kind fiscal_year "
    "fiscal_period period_end_date filing_date part part_count"
)
_META_COLUMNS = (*_META_FIELDS, "block_ids", "tickers", "context", "text")
# Columns `record` reads from every row.
_STATS_COLUMNS = _names(
    "cik company_name tickers fiscal_year filing_id form fiscal_period "
    "period_end_date filing_date chunk_kind text"
)


def keep_text(chunk_kind: str, topic: str, heading: str, text: str) -> bool:
    """Text-based exclusions applied while reading: disclaimers, contact
    lines and table rows taken as headings, stubs pointing elsewhere and
    short controls/properties boilerplate."""

    if (
        "@" in heading
        or _FORWARD_LOOKING.search(heading)
        or _ROW_HEADING.search(heading)
    ):
        return False
    if chunk_kind != "text":
        return True
    if len(_FORWARD_LOOKING.findall(text)) >= 2:
        return False
    short = len(text) < STUB_CHARS
    if _POINTER.search(text) and (short or topic == "governance"):
        return False
    # Short controls passages are all the same certification language.
    return not (short and (topic == "controls" or _BOILERPLATE.search(text)))


def _chunk_meta(row: Mapping[str, Any]) -> ChunkMeta:
    text: str = row["text"]
    note = _NOTE_HEADING.search(row["context"])
    return ChunkMeta(
        **{name: row[name] for name in _META_FIELDS},
        block_ids=tuple(row["block_ids"]),
        tickers=tuple(row["tickers"]),
        chars=len(text),
        list_lines=len(_LIST_LINE.findall(text)) if row["chunk_kind"] == "text" else 0,
        text_hash=rank(0, text[:300] + text[-300:]),
        note=note.group(1) if note else None,
        note_refs=tuple(dict.fromkeys(_NOTE_REF.findall(text))),
        exhibit_refs=tuple(
            dict.fromkeys(exhibit_number(n) for n in _EXHIBIT_REF.findall(text))
        ),
    )


def _mentions(text: str, key: str) -> bool:
    """`key` in `text` at a word start (`ExxonMobil` mentions `Exxon`)."""

    return key in text and re.search(rf"(?<![\w]){re.escape(key)}", text) is not None


def record(stats: CorpusStats, row: Mapping[str, Any]) -> None:
    """Add one chunk row to the corpus stats (substring checks first: this
    runs on every row)."""

    text: str = row["text"]
    cik: str = row["cik"]
    if cik not in stats.companies:
        stats.companies[cik] = company_ref(cik, row["company_name"], row["tickers"])
    if row["fiscal_year"] is not None:
        stats.first_year[cik] = min(stats.first_year.get(cik, 9999), row["fiscal_year"])
    if row["filing_id"] not in stats.filings:
        stats.filings[row["filing_id"]] = FilingInfo(
            cik,
            row["form"],
            row["fiscal_year"],
            row["fiscal_period"],
            row["period_end_date"],
            row["filing_date"],
        )
    # Years whose values a chunk may state: any year in a table, "fiscal 2015" in text.
    found = (
        _YEAR.findall(text)
        if row["chunk_kind"] == "table"
        else _FISCAL_YEAR.findall(text)
    )
    stats.years[cik].update(int(y) for y in found)
    hits = stats.probe_hits[cik]
    if len(hits) < len(TOPIC_PROBES):
        lower = text.lower()
        for index, (_, keys) in enumerate(TOPIC_PROBES):
            if index not in hits and any(key in lower for key in keys):
                hits.add(index)
    for index, (name, _, keys) in enumerate(OUTSIDE_COMPANIES):
        if index not in stats.mentioned and any(
            _mentions(text, key) for key in keys or (name,)
        ):
            stats.mentioned.add(index)


def read_candidates(
    chunks: Table, seed: int, max_candidates: int
) -> tuple[list[ChunkMeta], CorpusStats]:
    """First pass: metadata of a hash sample of chunks, plus corpus stats from
    every row."""

    total = snapshot_total(chunks, "total-records")
    threshold = (1 << 64) * min(1.0, max_candidates / max(total, 1))
    stats = CorpusStats(rows=total)
    candidates: list[ChunkMeta] = []
    for batch in scan_batches(chunks, _META_COLUMNS):
        for row in batch.select(_STATS_COLUMNS).to_pylist():
            record(stats, row)
        sampled = [
            rank(seed, chunk_id) < threshold
            for chunk_id in batch["chunk_id"].to_pylist()
        ]
        if not any(sampled):
            continue
        for row in batch.filter(pa.array(sampled)).to_pylist():
            heading = row["subheading"] or ""
            if keep_text(row["chunk_kind"], row["section_topic"], heading, row["text"]):
                candidates.append(_chunk_meta(row))
    return candidates, stats


def read_chunk_texts(
    chunks: Table,
    filing_dates: Mapping[str, date],
    column: Literal["chunk_id", "block_ids"],
    values: Collection[str],
) -> Iterator[tuple[ChunkMeta, ChunkText]]:
    """Chunks of the filings of `filing_dates` (`filing_id` -> `filing_date`)
    whose `column` holds one of `values` (any of them for `block_ids`), with
    texts; only matching rows are decoded to Python."""

    if not filing_dates or not values:
        return
    filing_ids = pa.array(sorted(filing_dates), type=pa.string())
    wanted = pa.array(sorted(values), type=pa.string())
    files = filing_dates_filter(set(filing_dates.values()))
    for batch in scan_batches(chunks, _META_COLUMNS, files=files):
        mask = pc.is_in(batch["filing_id"], value_set=filing_ids)
        if column == "block_ids":
            lists = batch["block_ids"]
            hit = pc.is_in(pc.list_flatten(lists), value_set=wanted)
            rows = pc.filter(pc.list_parent_indices(lists), hit)
            positions = pa.array(range(batch.num_rows), type=rows.type)
            mask = pc.and_(mask, pc.is_in(positions, value_set=rows))
        else:
            mask = pc.and_(mask, pc.is_in(batch[column], value_set=wanted))
        for row in batch.filter(mask).to_pylist():
            yield _chunk_meta(row), ChunkText(row["context"], row["text"])


# --- Silver reads ----------------------------------------------------------------

_FACT_COLUMNS = _names(
    "fact_id instance_id concept_id context_key unit_key block_id cell_id raw_value "
    "numeric_value decimals is_nil is_hidden"
)
_FactKey = tuple[str, str, str]  # instance, concept, context
_Period = tuple[date | None, date | None, date | None]  # start, end, instant


def cell_block(cell_id: str | None) -> str | None:
    """The `table` block of a cell: cell IDs are `{block_id}:table:cell:NNNNNN`
    (`table_id_for`, `cell_id_for`); facts in cells carry no `block_id`."""

    if cell_id is None or _CELL_ID_INFIX not in cell_id:
        return None
    return cell_id.rsplit(_CELL_ID_INFIX, 1)[0]


def readable_unit(expression: str) -> str:
    """`{http://www.xbrl.org/2003/iso4217}USD` -> `iso4217:USD`; ratios as
    `iso4217:USD/xbrli:shares` (outer parentheses dropped)."""

    def prefixed(match: re.Match[str]) -> str:
        uri, local = match.group(1), match.group(2)
        return f"{_UNIT_PREFIXES.get(uri, uri)}:{local}"

    text = re.sub(r"\{([^}]*)\}([\w.-]+)", prefixed, expression)
    return re.sub(r"\(([^()]*)\)", r"\1", text)


def _fact_instances(
    silver: Catalog, filings: Mapping[str, FilingInfo], seed: int, limit: int
) -> dict[str, str]:
    """`instance_id` -> `filing_id` of 10-K/10-Q instances, at most `limit`
    spread evenly over companies (lowest hashes)."""

    by_company: dict[str, list[tuple[str, str]]] = defaultdict(list)
    table = silver.load_table(XBRL_INSTANCES_TABLE)
    for row in iter_rows(table, ("instance_id", "filing_id")):
        info = filings.get(row["filing_id"])
        if info is not None and info.form in ("10-K", "10-Q"):
            by_company[info.cik].append((row["instance_id"], row["filing_id"]))
    per_company = max(1, limit // max(len(by_company), 1))
    chosen: dict[str, str] = {}
    for ids in by_company.values():
        chosen.update(sorted(ids, key=lambda p: rank(seed, p[0]))[:per_company])
    return chosen


def _key_concepts(silver: Catalog) -> dict[str, str]:
    """`concept_id` -> local name of KEY_CONCEPTS, every us-gaap year."""

    table = silver.load_table(XBRL_CONCEPTS_TABLE)
    rows = read_rows_in(
        table, "local_name", KEY_CONCEPTS, columns=("concept_id", "namespace_uri")
    )
    return {
        row["concept_id"]: row["local_name"]
        for row in rows
        if "fasb.org/us-gaap/" in row["namespace_uri"]
    }


def _shown_facts(
    silver: Catalog, concepts: Collection[str], instances: Collection[str]
) -> tuple[dict[_FactKey, dict[str, Any]], dict[_FactKey, list[Occurrence]]]:
    """Visible numeric facts shown in a block: the first row and every
    occurrence (one per block) of each (instance, concept, context)."""

    first: dict[_FactKey, dict[str, Any]] = {}
    occurrences: dict[_FactKey, list[Occurrence]] = defaultdict(list)
    table = silver.load_table(XBRL_FACTS_TABLE)
    rows = read_rows_in(
        table,
        "concept_id",
        concepts,
        columns=_FACT_COLUMNS,
        files=values_filter("instance_id", instances),
    )
    for row in rows:
        if row["instance_id"] not in instances:
            continue
        block_id = row["block_id"] or cell_block(row["cell_id"])
        if row["is_nil"] or row["is_hidden"] or not block_id:
            continue
        if row["numeric_value"] is None or not row["raw_value"]:
            continue
        key = (row["instance_id"], row["concept_id"], row["context_key"])
        first.setdefault(key, row)
        if all(o.block_id != block_id for o in occurrences[key]):
            occurrences[key].append(
                Occurrence(block_id, row["cell_id"], row["raw_value"])
            )
    return first, occurrences


def _plain_contexts(silver: Catalog, keys: Collection[str]) -> dict[str, _Period]:
    """Periods of the contexts of `keys` without segment/scenario."""

    columns = _names(
        "context_key start_date end_date instant_date segment_xml scenario_xml"
    )
    table = silver.load_table(XBRL_CONTEXTS_TABLE)
    return {
        row["context_key"]: (row["start_date"], row["end_date"], row["instant_date"])
        for row in read_rows_in(
            table,
            "context_key",
            keys,
            columns=columns,
            files=values_filter("context_key", keys),
        )
        if not row["segment_xml"] and not row["scenario_xml"]
    }


def _unit_expressions(silver: Catalog, keys: Collection[str]) -> dict[str, str]:
    table = silver.load_table(XBRL_UNITS_TABLE)
    return {
        row["unit_key"]: readable_unit(row["normalized_expression"])
        for row in read_rows_in(
            table,
            "unit_key",
            keys,
            columns=("normalized_expression",),
            files=values_filter("unit_key", keys),
        )
    }


def fact_candidate(
    row: Mapping[str, Any],
    occurrences: Sequence[Occurrence],
    filing_id: str,
    filing: FilingInfo,
    context: _Period,
) -> FactCandidate:
    """A fact row (with `concept` local name and readable `unit`) and its
    filing; current when its context ends on the filing's period end."""

    start, end, instant = context
    fact = XbrlFact(
        fact_id=row["fact_id"],
        concept=row["concept"],
        value=row["numeric_value"],
        unit=row["unit"],
        decimals=row["decimals"],
        period_start=start,
        period_end=end,
        instant=instant,
        block_id=occurrences[0].block_id,
        cell_id=occurrences[0].cell_id,
        block_ids=tuple(o.block_id for o in occurrences),
    )
    period_end = end or instant
    current = period_end is not None and period_end == filing.period_end_date
    return FactCandidate(fact, filing_id, filing, tuple(occurrences), current)


def read_fact_candidates(
    silver: Catalog, filings: Mapping[str, FilingInfo], seed: int, max_instances: int
) -> list[FactCandidate]:
    """Non-dimensional numeric facts of KEY_CONCEPTS shown in a block, from
    10-K/10-Q instances."""

    instances = _fact_instances(silver, filings, seed, max_instances)
    concepts = _key_concepts(silver)
    first, occurrences = _shown_facts(silver, concepts, instances)
    contexts = _plain_contexts(silver, {key[2] for key in first})
    units = _unit_expressions(
        silver, {r["unit_key"] for r in first.values() if r["unit_key"]}
    )
    return [
        fact_candidate(
            row
            | {"concept": concepts[key[1]], "unit": units.get(row["unit_key"] or "")},
            occurrences[key],
            instances[key[0]],
            filings[instances[key[0]]],
            contexts[key[2]],
        )
        for key, row in sorted(first.items())
        if key[2] in contexts
    ]

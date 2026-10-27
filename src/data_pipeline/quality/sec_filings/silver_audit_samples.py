from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lxml import etree
from pydantic import BaseModel

from src.data_pipeline.contracts.sec_filings.content_blocks import ContentBlockRecord
from src.data_pipeline.contracts.sec_filings.document_table_cells import (
    DocumentTableCellRecord,
)
from src.data_pipeline.contracts.sec_filings.document_tables import DocumentTableRecord
from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.filing_entities import FilingEntityRecord
from src.data_pipeline.contracts.sec_filings.filings import FilingRecord
from src.data_pipeline.contracts.sec_filings.sections import SectionRecord
from src.data_pipeline.contracts.sec_filings.xbrl_core import (
    XbrlContextRecord,
    XbrlInstanceRecord,
    XbrlUnitRecord,
)
from src.data_pipeline.contracts.sec_filings.xbrl_dimensions import (
    XbrlDimensionRecord,
)
from src.data_pipeline.contracts.sec_filings.xbrl_facts import XbrlFactRecord
from src.data_pipeline.contracts.sec_filings.xbrl_networks import (
    XbrlNetworkRecord,
    XbrlRelationshipRecord,
)
from src.data_pipeline.quality.sec_filings.document_tables import (
    cells_problem,
    table_blocks_sync_problem,
    tables_problem,
)
from src.data_pipeline.quality.sec_filings.documents import documents_problem
from src.data_pipeline.quality.sec_filings.narrative import (
    narrative_problem,
    sections_problem,
)
from src.data_pipeline.quality.sec_filings.silver_audit_report import (
    CheckRun,
    FilingSample,
    SilverAuditReport,
)
from src.data_pipeline.quality.sec_filings.silver_audit_scan import (
    Keep,
    contract_row,
)
from src.data_pipeline.quality.sec_filings.xbrl_core import instance_problem
from src.data_pipeline.quality.sec_filings.xbrl_linkbases import network_problem
from src.data_pipeline.storage.bronze import (
    SecBronze,
    sha256_bytes,
    sha256_file,
)
from src.data_pipeline.storage.sec_filings.content_blocks import CONTENT_BLOCKS_TABLE
from src.data_pipeline.storage.sec_filings.document_table_cells import (
    DOCUMENT_TABLE_CELLS_TABLE,
)
from src.data_pipeline.storage.sec_filings.document_tables import DOCUMENT_TABLES_TABLE
from src.data_pipeline.storage.sec_filings.documents import (
    DOCUMENTS_TABLE,
    read_document_content,
)
from src.data_pipeline.storage.sec_filings.filing_entities import FILING_ENTITIES_TABLE
from src.data_pipeline.storage.sec_filings.filings import FILINGS_TABLE
from src.data_pipeline.storage.sec_filings.sections import SECTIONS_TABLE
from src.data_pipeline.storage.sec_filings.xbrl_core import (
    XBRL_CONTEXTS_TABLE,
    XBRL_INSTANCES_TABLE,
    XBRL_UNITS_TABLE,
)
from src.data_pipeline.storage.sec_filings.xbrl_enrichment import (
    XBRL_DIMENSIONS_TABLE,
)
from src.data_pipeline.storage.sec_filings.xbrl_facts import XBRL_FACTS_TABLE
from src.data_pipeline.storage.sec_filings.xbrl_linkbases import (
    XBRL_NETWORKS_TABLE,
    XBRL_RELATIONSHIPS_TABLE,
)
from src.data_pipeline.transforms.sec_filings.document_tables import (
    build_document_tables,
)
from src.data_pipeline.transforms.sec_filings.documents import (
    build_document_records,
    check_document_offsets,
)
from src.data_pipeline.transforms.sec_filings.filing_entities import (
    build_filing_entity_records,
    check_entity_offsets,
)
from src.data_pipeline.transforms.sec_filings.filings import read_raw_sec_header
from src.data_pipeline.transforms.sec_filings.html_blocks import (
    ExtractedBlocks,
    extract_blocks,
)
from src.data_pipeline.transforms.sec_filings.html_dom import parse_html
from src.data_pipeline.transforms.sec_filings.html_tables import (
    build_grid,
    extract_table,
)
from src.data_pipeline.transforms.sec_filings.narrative import (
    build_document_narrative,
    detect_section_headings,
)

# Facts printed per sample: the first numeric facts without and with
# dimensions, then the first non-numeric ones, in fact order.
_FACTS_PER_KIND = 2
_TEXT_WIDTH = 60
# Local names of concepts, by concept_id.
ConceptNames = Callable[[Collection[str]], Mapping[str, str]]


def _clip(text: str | None) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= _TEXT_WIDTH else f"{text[: _TEXT_WIDTH - 3]}..."


# The column on which each table's rows are matched to the sampled filings,
# in scan order: document tables by filing, XBRL tables through their
# instance, context or network.
_SAMPLE_MATCH = {
    DOCUMENTS_TABLE: "filing_id",
    FILING_ENTITIES_TABLE: "filing_id",
    SECTIONS_TABLE: "filing_id",
    CONTENT_BLOCKS_TABLE: "filing_id",
    DOCUMENT_TABLES_TABLE: "filing_id",
    DOCUMENT_TABLE_CELLS_TABLE: "filing_id",
    XBRL_INSTANCES_TABLE: "filing_id",
    XBRL_CONTEXTS_TABLE: "instance_id",
    XBRL_UNITS_TABLE: "instance_id",
    XBRL_FACTS_TABLE: "instance_id",
    XBRL_NETWORKS_TABLE: "instance_id",
    XBRL_DIMENSIONS_TABLE: "context_key",
    XBRL_RELATIONSHIPS_TABLE: "network_id",
}
# The key a kept row adds to the set that later tables are matched on.
_SAMPLE_KEYS = {
    XBRL_INSTANCES_TABLE: "instance_id",
    XBRL_CONTEXTS_TABLE: "context_key",
    XBRL_NETWORKS_TABLE: "network_id",
}


class SampleRows:
    """The stored rows of the sampled filings, kept while every table is scanned.

    When filings are added, the `per_form` latest filings of each form
    (ties broken by filing_id) are chosen. Later tables keep only the rows
    that `_SAMPLE_MATCH` links to a chosen filing, so tables must be added
    parents first, in SILVER_TABLES order. Nothing is read twice, and only
    the chosen filings' rows are held.
    """

    def __init__(self, forms: Sequence[str], per_form: int) -> None:
        self.per_form = per_form
        self.chosen: dict[str, list[dict[str, Any]]] = {form: [] for form in forms}
        self.keys: dict[str, set[str]] = defaultdict(set)
        self.rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add(self, identifier: str, rows: Sequence[dict[str, Any]]) -> None:
        """Keep the rows of one scanned batch that belong to the sample."""

        if identifier == FILINGS_TABLE:
            self._choose(rows)
            return
        column = _SAMPLE_MATCH.get(identifier)
        if column is None:
            return
        kept = [row for row in rows if row[column] in self.keys[column]]
        self.rows[identifier].extend(kept)
        if identifier in _SAMPLE_KEYS:
            key = _SAMPLE_KEYS[identifier]
            self.keys[key].update(row[key] for row in kept)

    def keep(self, identifier: str) -> Keep | None:
        """The rows of a table's scan that `add` needs: every filing, the rows
        matching the chosen filings' keys, or none."""

        if identifier == FILINGS_TABLE:
            return Keep(None)
        column = _SAMPLE_MATCH.get(identifier)
        if column is None:
            return None
        return Keep(column, frozenset(self.keys[column]))

    def _choose(self, rows: Sequence[dict[str, Any]]) -> None:
        for row in rows:
            if row["form"] in self.chosen:
                self.chosen[row["form"]].append(row)
        for candidates in self.chosen.values():
            candidates.sort(
                key=lambda r: (-r["filing_date"].toordinal(), r["filing_id"])
            )
            del candidates[self.per_form :]
        filings = [row for candidates in self.chosen.values() for row in candidates]
        self.rows[FILINGS_TABLE] = filings
        self.keys["filing_id"] = {row["filing_id"] for row in filings}

    def select(
        self, identifier: str, column: str, values: Collection[str]
    ) -> list[dict[str, Any]]:
        """The kept rows of a table whose `column` is one of `values`."""

        return [row for row in self.rows[identifier] if row[column] in values]


def _records[Record: BaseModel](
    model: type[Record], identifier: str, rows: Sequence[dict[str, Any]]
) -> list[Record]:
    return [model.model_validate(contract_row(identifier, row)) for row in rows]


def _differences(
    stored: Sequence[BaseModel], rebuilt: Sequence[BaseModel], key: str
) -> list[str]:
    """Keys of the records that differ between Silver and a rebuild from Bronze."""

    def by_key(records: Sequence[BaseModel]) -> dict[str, dict[str, Any]]:
        return {
            getattr(record, key): record.model_dump(exclude={"processed_at"})
            for record in records
        }

    old, new = by_key(stored), by_key(rebuilt)
    return sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))


def _report_differences(run: CheckRun, label: str, differences: list[str]) -> None:
    for identifier in differences:
        run.found(
            identifier, f"{label} {identifier} differs from its rebuild from Bronze"
        )


def _unreadable(
    report: SilverAuditReport,
    check_name: str,
    table: str,
    identifier: str,
    error: Exception,
) -> None:
    """Fail a sample check whose stored rows or Bronze source cannot be read."""

    with report.check(
        check_name,
        table,
        "the sampled rows validate against their contracts and their Bronze "
        "source is readable",
    ) as run:
        run.found(identifier, " ".join(str(error).split()))


@dataclass(frozen=True)
class _StoredFiling:
    """A sampled filing's stored filing, document and entity rows."""

    filing: FilingRecord
    documents: list[DocumentRecord]
    entities: list[FilingEntityRecord]


def _check_submission(
    stored: _StoredFiling,
    bronze: SecBronze,
    report: SilverAuditReport,
    sample: FilingSample,
) -> bytes | None:
    """Re-hash the Bronze files; return the submission bytes when they match."""

    filing = stored.filing
    with report.check(
        "sample_bronze_files",
        FILINGS_TABLE,
        "the full submission lies inside the Bronze directory and matches full_submission_sha256; "
        "the primary document file matches primary_document_sha256",
    ) as run:
        run.rows_checked = 1
        try:
            raw = bronze.path(filing.full_submission_path).read_bytes()
            if sha256_bytes(raw) != filing.full_submission_sha256:
                raise ValueError("full submission SHA-256 differs")
            if filing.primary_document_path is not None and (
                sha256_file(bronze.path(filing.primary_document_path))
                != filing.primary_document_sha256
            ):
                raise ValueError("primary document SHA-256 differs")
        except (OSError, ValueError) as error:
            run.found(filing.filing_id, f"{filing.full_submission_path}: {error}")
            return None
    sample.lines.append(
        f"Bronze: {filing.full_submission_path} ({len(raw):,} bytes, SHA-256 verified)"
    )
    return raw


def _check_documents(
    stored: _StoredFiling, raw: bytes, bronze: SecBronze, report: SilverAuditReport
) -> None:
    """Documents and entities rebuilt from the submission bytes equal Silver."""

    filing, processed_at = stored.filing, report.started_at
    with report.check(
        "sample_documents",
        DOCUMENTS_TABLE,
        "documents rebuilt from the <DOCUMENT> blocks equal the stored rows; "
        "every span and <TEXT> slice matches its size and SHA-256; one primary document",
    ) as run:
        run.rows_checked = len(stored.documents)
        try:
            rebuilt = build_document_records(filing, raw, processed_at)
            _report_differences(
                run, "document", _differences(stored.documents, rebuilt, "document_id")
            )
            check_document_offsets(stored.documents, raw)
            problem = documents_problem(filing, stored.documents)
            if problem is not None:
                run.found(filing.filing_id, problem)
        except ValueError as error:
            run.found(filing.filing_id, str(error))
    with report.check(
        "sample_entities",
        FILING_ENTITIES_TABLE,
        "entities rebuilt from the SEC header equal the stored rows; spans tile "
        "the header and match entity_block_sha256",
    ) as run:
        run.rows_checked = len(stored.entities)
        try:
            path = bronze.path(filing.full_submission_path)
            header = read_raw_sec_header(path)
            rebuilt_entities = build_filing_entity_records(filing, header, processed_at)
            _report_differences(
                run,
                "entity",
                _differences(stored.entities, rebuilt_entities, "filing_entity_id"),
            )
            check_entity_offsets(stored.entities, header)
        except (OSError, ValueError) as error:
            run.found(filing.filing_id, str(error))


@dataclass(frozen=True)
class _Narrative:
    """A primary document's stored sections, blocks, tables and cells."""

    document: DocumentRecord
    sections: list[SectionRecord]
    blocks: list[ContentBlockRecord]
    tables: list[DocumentTableRecord]
    cells: list[DocumentTableCellRecord]


def _read_narrative(rows: SampleRows, document: DocumentRecord) -> _Narrative:
    def read[Record: BaseModel](identifier: str, model: type[Record]) -> list[Record]:
        found = rows.select(identifier, "document_id", {document.document_id})
        return _records(model, identifier, found)

    return _Narrative(
        document,
        sorted(read(SECTIONS_TABLE, SectionRecord), key=lambda s: s.section_ordinal),
        sorted(read(CONTENT_BLOCKS_TABLE, ContentBlockRecord), key=lambda b: b.block_ordinal),
        sorted(read(DOCUMENT_TABLES_TABLE, DocumentTableRecord), key=lambda t: t.table_ordinal),
        sorted(read(DOCUMENT_TABLE_CELLS_TABLE, DocumentTableCellRecord), key=lambda c: c.cell_id),
    )  # fmt: skip


def _outline(narrative: _Narrative) -> list[str]:
    """One line per section: ordinal, kind, code, level, parent, block range, title."""

    ordinals = {s.section_id: s.section_ordinal for s in narrative.sections}
    lines = [
        "  #  kind               code              level parent blocks       title"
    ]
    for s in narrative.sections:
        parent = ordinals.get(s.parent_section_id or "", "-")
        blocks = f"{s.start_block_ordinal}-{s.end_block_ordinal - 1}"
        lines.append(
            f"  {s.section_ordinal:<2} {s.section_kind:<18} {s.section_code or '-':<17} "
            f"{s.level:<5} {parent!s:<6} {blocks:<12} {_clip(s.title)}"
        )
    return lines


def _check_narrative(
    narrative: _Narrative,
    html: ExtractedBlocks,
    report: SilverAuditReport,
    sample: FilingSample,
) -> None:
    """Sections and blocks rebuilt from the primary document equal Silver."""

    document = narrative.document
    with report.check(
        "sample_narrative",
        SECTIONS_TABLE,
        "sections and blocks rebuilt from the primary HTML equal the stored rows "
        "(block text hashes included); section_text_sha256 recomputed from the "
        "stored blocks; every block in its deepest section",
    ) as run:
        run.rows_checked = len(narrative.sections) + len(narrative.blocks)
        headings = detect_section_headings(document.document_type, html.blocks)
        rebuilt = build_document_narrative(
            document, html.blocks, headings, report.started_at
        )
        _report_differences(
            run,
            "section",
            _differences(narrative.sections, rebuilt.sections, "section_id"),
        )
        _report_differences(
            run, "block", _differences(narrative.blocks, rebuilt.blocks, "block_id")
        )
        problem = sections_problem(document, narrative.sections) or narrative_problem(
            document, narrative.sections, narrative.blocks
        )
        if problem is not None:
            run.found(document.document_id, problem)
    roots_only = len(narrative.sections) == 1
    sample.lines.append(
        f"narrative: {len(narrative.sections)} sections "
        f"({rebuilt.duplicate_section_codes} repeated codes"
        f"{', root only' if roots_only else ''}), {len(narrative.blocks)} blocks"
    )
    sample.lines.extend(_outline(narrative))


def _cells_outside_their_table(
    narrative: _Narrative, table_xpaths: Mapping[str, str]
) -> list[str]:
    """Cells whose innermost enclosing table block is not their own table.

    A nested table with visible cells is its own block, so its cells must
    not also be cells of the table around it.
    """

    by_depth = sorted(table_xpaths.items(), key=lambda item: -len(item[1]))
    wrong = []
    for cell in narrative.cells:
        owner = next(
            (t for t, xpath in by_depth if cell.source_xpath.startswith(f"{xpath}/")),
            None,
        )
        if owner != cell.table_id:
            wrong.append(cell.cell_id)
    return wrong


def _check_tables(
    narrative: _Narrative,
    root: etree._Element,
    html: ExtractedBlocks,
    report: SilverAuditReport,
    sample: FilingSample,
) -> None:
    """Tables and cells rebuilt from the DOM equal Silver; nested cells stay in their table."""

    table_blocks = [b for b in narrative.blocks if b.block_type == "table"]
    elements = list(root.iter("table"))
    with report.check(
        "sample_tables",
        DOCUMENT_TABLES_TABLE,
        "the DOM's tables with visible cells are exactly the table blocks; tables "
        "and cells rebuilt from them equal the stored rows; table_content_sha256 "
        "and block text recomputed from the cells; nested tables keep their own cells",
    ) as run:
        run.rows_checked = len(narrative.tables) + len(narrative.cells)
        if not table_blocks and not narrative.tables:
            run.skip("the primary document has no table blocks")
            return
        problem = table_blocks_sync_problem(html.blocks, table_blocks)
        if problem is not None:
            run.found(narrative.document.document_id, problem)
            return
        tree = root.getroottree()
        by_xpath = {tree.getpath(element): element for element in elements}
        layouts = [
            build_grid(extract_table(by_xpath[b.source_xpath])) for b in table_blocks
        ]
        tables, cells = build_document_tables(table_blocks, layouts, report.started_at)
        _report_differences(
            run, "table", _differences(narrative.tables, tables, "table_id")
        )
        _report_differences(
            run, "cell", _differences(narrative.cells, cells, "cell_id")
        )
        problem = tables_problem(
            narrative.document, table_blocks, narrative.tables
        ) or cells_problem(table_blocks, narrative.tables, narrative.cells)
        if problem is not None:
            run.found(narrative.document.document_id, problem)
        xpaths = {t.table_id: t.source_xpath for t in narrative.tables}
        for cell_id in _cells_outside_their_table(narrative, xpaths):
            run.found(cell_id, f"{cell_id} lies in another table block")
    nested = sum(b.inside_table for b in table_blocks)
    sample.lines.append(
        f"tables: {len(elements)} <table> elements ({html.empty_tables} without a visible "
        f"cell), {len(table_blocks)} table blocks ({nested} nested), "
        f"{len(narrative.tables)} stored tables, {len(narrative.cells)} cells"
    )


def _check_primary(
    rows: SampleRows,
    document: DocumentRecord,
    bronze: SecBronze,
    report: SilverAuditReport,
    sample: FilingSample,
) -> None:
    """Parse the verified primary HTML once; check narrative and tables against it."""

    try:
        narrative = _read_narrative(rows, document)
        root = parse_html(read_document_content(document, bronze))
    except (OSError, ValueError) as error:
        _unreadable(
            report, "sample_narrative", SECTIONS_TABLE, document.document_id, error
        )
        return
    html = extract_blocks(root)
    _check_narrative(narrative, html, report, sample)
    _check_tables(narrative, root, html, report, sample)


@dataclass(frozen=True)
class _Xbrl:
    """The stored XBRL rows of a sampled filing."""

    instances: list[XbrlInstanceRecord]
    networks: list[XbrlNetworkRecord]
    relationships: list[XbrlRelationshipRecord]
    facts: list[XbrlFactRecord]
    contexts: dict[str, XbrlContextRecord]
    units: dict[str, XbrlUnitRecord]
    dimensions: dict[str, list[XbrlDimensionRecord]]


def _read_xbrl(rows: SampleRows, filing_id: str) -> _Xbrl:
    def read[Record: BaseModel](
        identifier: str, model: type[Record], column: str, values: Collection[str]
    ) -> list[Record]:
        return _records(model, identifier, rows.select(identifier, column, values))

    instances = read(XBRL_INSTANCES_TABLE, XbrlInstanceRecord, "filing_id", {filing_id})
    ids = {instance.instance_id for instance in instances}
    networks = read(XBRL_NETWORKS_TABLE, XbrlNetworkRecord, "instance_id", ids)
    contexts = read(XBRL_CONTEXTS_TABLE, XbrlContextRecord, "instance_id", ids)
    dimensions: dict[str, list[XbrlDimensionRecord]] = defaultdict(list)
    for dimension in read(
        XBRL_DIMENSIONS_TABLE,
        XbrlDimensionRecord,
        "context_key",
        {c.context_key for c in contexts},
    ):
        dimensions[dimension.context_key].append(dimension)
    return _Xbrl(
        instances,
        networks,
        read(XBRL_RELATIONSHIPS_TABLE, XbrlRelationshipRecord, "network_id", {n.network_id for n in networks}),
        sorted(read(XBRL_FACTS_TABLE, XbrlFactRecord, "instance_id", ids), key=lambda f: (f.instance_id, f.fact_order)),
        {c.context_key: c for c in contexts},
        {u.unit_key: u for u in read(XBRL_UNITS_TABLE, XbrlUnitRecord, "instance_id", ids)},
        dict(dimensions),
    )  # fmt: skip


def _representative_facts(xbrl: _Xbrl) -> list[XbrlFactRecord]:
    """The first numeric facts without and with dimensions, then non-numeric ones."""

    kinds: dict[tuple[str, bool], list[XbrlFactRecord]] = defaultdict(list)
    for fact in xbrl.facts:
        kind = (fact.fact_kind, fact.context_key in xbrl.dimensions)
        if len(kinds[kind]) < _FACTS_PER_KIND:
            kinds[kind].append(fact)
    order = [("numeric", False), ("numeric", True), ("non_numeric", False)]
    return [fact for kind in order for fact in kinds.get(kind, [])]


def _fact_line(fact: XbrlFactRecord, xbrl: _Xbrl, names: Mapping[str, str]) -> str:
    def name(concept_id: str | None) -> str:
        return names.get(concept_id or "", str(concept_id))

    context = xbrl.contexts[fact.context_key]
    period = (
        str(context.instant_date)
        if context.instant_date is not None
        else f"{context.start_date}..{context.end_date}"
        if context.start_date is not None
        else context.period_type
    )
    unit = xbrl.units[fact.unit_key].normalized_expression if fact.unit_key else "-"
    members = ", ".join(
        f"{name(d.dimension_concept_id)}={name(d.member_concept_id) if d.member_concept_id else d.typed_member_value}"
        for d in xbrl.dimensions.get(fact.context_key, [])
    )
    place = "cell" if fact.cell_id else "block" if fact.block_id else "unaligned"
    value = "(nil)" if fact.is_nil else _clip(fact.normalized_value or fact.raw_value)
    return (
        f"  fact {fact.fact_order}: {name(fact.concept_id)} = {value} [{unit}] "
        f"period {period}; {members or 'no dimensions'}; {place}"
    )


def _check_xbrl(
    rows: SampleRows,
    stored: _StoredFiling,
    report: SilverAuditReport,
    sample: FilingSample,
    concept_names: ConceptNames,
) -> None:
    """Instances against the verified documents, networks against their relationships."""

    try:
        xbrl = _read_xbrl(rows, stored.filing.filing_id)
    except ValueError as error:
        _unreadable(
            report, "sample_xbrl", XBRL_INSTANCES_TABLE, stored.filing.filing_id, error
        )
        return
    with report.check(
        "sample_xbrl",
        XBRL_INSTANCES_TABLE,
        "source_set_sha256 matches the Bronze-verified documents; each network's "
        "relationship count, ordinals and network_hash recomputed; fact, context "
        "and dimension hashes recomputed by their contracts",
    ) as run:
        run.rows_checked = len(xbrl.instances) + len(xbrl.networks) + len(xbrl.facts)
        if not xbrl.instances:
            run.skip("the filing has no XBRL instance")
            return
        documents = {d.document_id: d for d in stored.documents}
        for instance in xbrl.instances:
            problem = instance_problem(instance, documents)
            if problem is not None:
                run.found(instance.instance_id, problem)
        by_network: dict[str, list[XbrlRelationshipRecord]] = defaultdict(list)
        for relationship in xbrl.relationships:
            by_network[relationship.network_id].append(relationship)
        for network in xbrl.networks:
            problem = network_problem(network, by_network.get(network.network_id, []))
            if problem is not None:
                run.found(network.network_id, problem)
    shown = _representative_facts(xbrl)
    concept_ids = {f.concept_id for f in shown} | {
        concept_id
        for f in shown
        for d in xbrl.dimensions.get(f.context_key, [])
        for concept_id in (d.dimension_concept_id, d.member_concept_id)
        if concept_id is not None
    }
    names = concept_names(concept_ids)
    sample.lines.append(
        f"xbrl: {len(xbrl.instances)} instance, {len(xbrl.facts)} facts, "
        f"{len(xbrl.contexts)} contexts, {len(xbrl.units)} units, "
        f"{len(xbrl.networks)} networks, {len(xbrl.relationships)} relationships"
    )
    sample.lines.extend(_fact_line(fact, xbrl, names) for fact in shown)


def _read_filing(rows: SampleRows, filing_id: str) -> _StoredFiling:
    def read[Record: BaseModel](identifier: str, model: type[Record]) -> list[Record]:
        return _records(
            model, identifier, rows.select(identifier, "filing_id", {filing_id})
        )

    (filing,) = read(FILINGS_TABLE, FilingRecord)
    return _StoredFiling(
        filing,
        sorted(read(DOCUMENTS_TABLE, DocumentRecord), key=lambda d: d.document_ordinal),
        sorted(read(FILING_ENTITIES_TABLE, FilingEntityRecord), key=lambda e: e.entity_ordinal),
    )  # fmt: skip


def check_sample(
    rows: SampleRows,
    sample: FilingSample,
    bronze: SecBronze,
    report: SilverAuditReport,
    concept_names: ConceptNames,
) -> None:
    """Trace one sampled filing from Silver back to its Bronze bytes.

    Re-hashes the submission, rebuilds documents, entities, sections, blocks,
    tables and cells from Bronze with the stage code and compares them with
    the stored rows, re-runs the group validators on the stored rows, checks
    the XBRL instances and networks, and adds the outline and representative
    facts (named through `concept_names`) to `sample.lines` for a reviewer.
    """

    try:
        stored = _read_filing(rows, sample.filing_id)
    except ValueError as error:
        _unreadable(
            report, "sample_filing_rows", FILINGS_TABLE, sample.filing_id, error
        )
        return
    filing = stored.filing
    entity = stored.entities[0].company_name if stored.entities else "-"
    sample.lines.append(
        f"{filing.form} {filing.accession_number} filed {filing.filing_date} by {entity}"
    )
    raw = _check_submission(stored, bronze, report, sample)
    if raw is None:
        return
    _check_documents(stored, raw, bronze, report)
    primary = next((d for d in stored.documents if d.role == "primary_document"), None)
    if primary is not None and primary.content_kind == "html":
        sample.lines.append(
            f"primary document: {primary.filename} ({primary.document_id})"
        )
        _check_primary(rows, primary, bronze, report, sample)
    _check_xbrl(rows, stored, report, sample, concept_names)

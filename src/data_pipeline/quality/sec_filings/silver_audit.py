from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pyarrow as pa
import pyarrow.compute as pc
from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import NoSuchNamespaceError, NoSuchTableError
from pyiceberg.table import Table

from src.data_pipeline.quality.sec_filings.silver_constraints import (
    DOCUMENT_RELATIONS,
    XBRL_RELATIONS,
    check_keys,
    check_references,
    check_source_hashes,
)
from src.data_pipeline.quality.sec_filings.silver_groups import check_groups
from src.data_pipeline.quality.sec_filings.silver_report import SilverAuditReport
from src.data_pipeline.quality.sec_filings.silver_xbrl import check_xbrl_groups
from src.data_pipeline.storage.iceberg import (
    SilverTableSpec,
    current_snapshot_id,
    open_silver_catalog,
    silver_catalog_paths,
    silver_table_location,
)
from src.data_pipeline.storage.sec_filings.content_blocks import (
    CONTENT_BLOCKS_SPEC,
    CONTENT_BLOCKS_TABLE,
)
from src.data_pipeline.storage.sec_filings.document_links import (
    DOCUMENT_LINKS_SPEC,
    DOCUMENT_LINKS_TABLE,
)
from src.data_pipeline.storage.sec_filings.document_table_cells import (
    DOCUMENT_TABLE_CELLS_SPEC,
    DOCUMENT_TABLE_CELLS_TABLE,
)
from src.data_pipeline.storage.sec_filings.document_tables import (
    DOCUMENT_TABLES_SPEC,
    DOCUMENT_TABLES_TABLE,
)
from src.data_pipeline.storage.sec_filings.documents import (
    DOCUMENTS_SPEC,
    DOCUMENTS_TABLE,
)
from src.data_pipeline.storage.sec_filings.exhibits import EXHIBITS_SPEC
from src.data_pipeline.storage.sec_filings.filing_entities import FILING_ENTITIES_SPEC
from src.data_pipeline.storage.sec_filings.filings import FILINGS_SPEC, FILINGS_TABLE
from src.data_pipeline.storage.sec_filings.sections import (
    SECTIONS_SPEC,
    SECTIONS_TABLE,
)
from src.data_pipeline.storage.sec_filings.xbrl_core import (
    XBRL_CONTEXTS_SPEC,
    XBRL_INSTANCES_SPEC,
    XBRL_UNITS_SPEC,
)
from src.data_pipeline.storage.sec_filings.xbrl_enrichment import (
    XBRL_DIMENSIONS_SPEC,
    XBRL_FACT_FOOTNOTES_SPEC,
)
from src.data_pipeline.storage.sec_filings.xbrl_facts import (
    XBRL_CONCEPTS_SPEC,
    XBRL_FACTS_SPEC,
)
from src.data_pipeline.storage.sec_filings.xbrl_linkbases import (
    XBRL_LABELS_SPEC,
    XBRL_NETWORKS_SPEC,
    XBRL_RELATIONSHIPS_SPEC,
)
from src.utils.logger import get_logger, log_context
from src.utils.timer import timed_operation

logger = get_logger(__name__)

NAMESPACE = "sec_filings"


@dataclass(frozen=True)
class AuditedTable:
    """An audited table: its definition and the columns that identify a row.

    `unique` lists the primary key first, then every other column set of the
    table's grain that must hold each value once.
    """

    spec: SilverTableSpec
    unique: tuple[tuple[str, ...], ...]

    @property
    def identifier(self) -> str:
        """The table's catalog identifier."""

        return self.spec.identifier


# The filing, document and HTML tables, then the XBRL tables: the audit
# checks the first group, then keeps only their keys to check the second.
DOCUMENT_TABLES = (
    AuditedTable(FILINGS_SPEC, (("filing_id",), ("accession_number",))),
    AuditedTable(DOCUMENTS_SPEC, (("document_id",), ("filing_id", "document_ordinal"))),
    AuditedTable(FILING_ENTITIES_SPEC, (("filing_entity_id",), ("filing_id", "entity_ordinal"))),
    AuditedTable(SECTIONS_SPEC, (("section_id",), ("document_id", "section_ordinal"))),
    AuditedTable(CONTENT_BLOCKS_SPEC, (("block_id",), ("document_id", "block_ordinal"))),
    AuditedTable(DOCUMENT_TABLES_SPEC, (("table_id",), ("block_id",), ("document_id", "table_ordinal"))),
    AuditedTable(DOCUMENT_TABLE_CELLS_SPEC, (("cell_id",), ("table_id", "cell_ordinal"))),
    AuditedTable(DOCUMENT_LINKS_SPEC, (("link_id",), ("source_document_id", "link_ordinal"))),
    AuditedTable(EXHIBITS_SPEC, (("exhibit_id",), ("filing_id", "exhibit_ordinal"))),
)  # fmt: skip
XBRL_TABLES = (
    AuditedTable(XBRL_INSTANCES_SPEC, (("instance_id",),)),
    AuditedTable(XBRL_CONTEXTS_SPEC, (("context_key",), ("instance_id", "context_ref"))),
    AuditedTable(XBRL_UNITS_SPEC, (("unit_key",), ("instance_id", "unit_ref"))),
    AuditedTable(XBRL_CONCEPTS_SPEC, (("concept_id",), ("namespace_uri", "local_name"))),
    AuditedTable(XBRL_FACTS_SPEC, (("fact_id",), ("instance_id", "fact_order"))),
    AuditedTable(XBRL_DIMENSIONS_SPEC, (("dimension_id",), ("context_key", "dimension_ordinal"), ("context_key", "dimension_concept_id"))),
    AuditedTable(XBRL_FACT_FOOTNOTES_SPEC, (("footnote_relation_id",), ("fact_id", "footnote_ordinal"))),
    AuditedTable(XBRL_LABELS_SPEC, (("label_id",),)),
    AuditedTable(XBRL_NETWORKS_SPEC, (("network_id",), ("instance_id", "network_ordinal"), ("instance_id", "linkbase_type", "role_uri"))),
    AuditedTable(XBRL_RELATIONSHIPS_SPEC, (("relationship_id",), ("network_id", "relationship_ordinal"))),
)  # fmt: skip
SILVER_TABLES = DOCUMENT_TABLES + XBRL_TABLES
# Columns of the first group that the XBRL checks reference.
_XBRL_PARENT_KEYS = {
    FILINGS_TABLE: ("filing_id",),
    DOCUMENTS_TABLE: ("document_id", "filing_id"),
    CONTENT_BLOCKS_TABLE: ("block_id", "document_id"),
    DOCUMENT_TABLE_CELLS_TABLE: ("cell_id", "document_id"),
}


# Narrative tables must be rebuilt in this order when the persisted table
# semantics are outdated (one block per <table>, cells in their own table).
TABLE_SEMANTICS_REBUILD = (
    SECTIONS_TABLE,
    CONTENT_BLOCKS_TABLE,
    DOCUMENT_TABLES_TABLE,
    DOCUMENT_TABLE_CELLS_TABLE,
)


def _check_catalog(catalog: Catalog, report: SilverAuditReport) -> dict[str, Table]:
    """Load every expected table; a missing one is an error, never created."""

    report.ran("catalog")
    try:
        present = {".".join(name) for name in catalog.list_tables(NAMESPACE)}
    except NoSuchNamespaceError:
        present = set()
    found = {}
    for expected in SILVER_TABLES:
        if expected.identifier not in present:
            report.tables_missing.append(expected.identifier)
            report.add("error", "missing_table", expected.identifier, "not in catalog")
            continue
        table = catalog.load_table(expected.identifier)
        found[expected.identifier] = table
        report.tables_found.append(expected.identifier)
        report.snapshot_ids_before[expected.identifier] = current_snapshot_id(table)
    expected_ids = {expected.identifier for expected in SILVER_TABLES}
    for identifier in sorted(present - expected_ids):
        report.add("warning", "unexpected_table", identifier, "not an audited table")
    for namespace in catalog.list_namespaces():
        if namespace != (NAMESPACE,):
            report.add("info", "other_namespace", ".".join(namespace), "not audited")
    logger.info(
        "silver_audit_catalog_checked",
        tables_found=len(found),
        tables_missing=report.tables_missing,
    )
    return found


def _check_definitions(
    found: Mapping[str, Table], data_dir: Path, report: SilverAuditReport
) -> set[str]:
    """Schema, partition spec and location of each table; return those matching."""

    valid = set()
    for expected in SILVER_TABLES:
        table = found.get(expected.identifier)
        if table is None:
            continue
        report.ran(f"definition:{expected.identifier}")
        problem = expected.spec.definition_problem(table)
        if problem is None:
            valid.add(expected.identifier)
        else:
            report.add("error", "table_definition", expected.identifier, problem)
        location = silver_table_location(data_dir, expected.identifier)
        if Path(table.location()) != location:
            report.add(
                "error",
                "table_location",
                expected.identifier,
                f"location {table.location()} is not {location}",
            )
        logger.info(
            "silver_audit_table_checked",
            table=expected.identifier,
            snapshot_id=report.snapshot_ids_before[expected.identifier],
            definition_problem=problem,
        )
    return valid


def _value_counts(table: pa.Table, *columns: str) -> dict[str, int]:
    counts = table.group_by(list(columns)).aggregate([([], "count_all")])
    return {
        ":".join(str(row[column]) for column in columns): row["count_all"]
        for row in counts.to_pylist()
    }


def _check_semantics(data: Mapping[str, pa.Table], report: SilverAuditReport) -> None:
    """Flag the retired `table_cell` block layout; collect content and link metrics.

    Unresolved and ambiguous links are valid data, reported as warnings.
    """

    report.ran("table_semantics")
    blocks = data[CONTENT_BLOCKS_TABLE]
    block_types = _value_counts(blocks, "block_type")
    legacy = block_types.get("table_cell", 0)
    if legacy:
        report.add(
            "error",
            "table_semantics",
            CONTENT_BLOCKS_TABLE,
            f"{legacy} blocks use the retired table_cell layout",
            rebuild_order=list(TABLE_SEMANTICS_REBUILD),
        )
    cells = data[DOCUMENT_TABLE_CELLS_TABLE]
    links = data[DOCUMENT_LINKS_TABLE]
    statuses = _value_counts(links, "resolution_status")
    for status in ("unresolved", "ambiguous"):
        if statuses.get(status):
            report.add(
                "warning",
                f"{status}_links",
                DOCUMENT_LINKS_TABLE,
                f"{statuses[status]} links are {status}",
            )
    report.metrics.update(
        block_types=block_types,
        table_blocks=block_types.get("table", 0),
        document_tables=data[DOCUMENT_TABLES_TABLE].num_rows,
        document_table_cells=cells.num_rows,
        empty_cells=pc.sum(pc.equal(cells["text"], "")).as_py() or 0,
        links_by_kind_and_status=_value_counts(links, "link_kind", "resolution_status"),
        link_sources=pc.count_distinct(links["source_document_id"]).as_py(),
    )


def _verify_snapshots(catalog: Catalog, report: SilverAuditReport) -> None:
    """Reload every table: a snapshot that moved means the audit saw no stable state."""

    report.ran("snapshot_verify")
    for identifier in report.tables_found:
        try:
            after = current_snapshot_id(catalog.load_table(identifier))
        except NoSuchTableError:
            after = None
        report.snapshot_ids_after[identifier] = after
        if after != report.snapshot_ids_before[identifier]:
            report.add(
                "error",
                "snapshot_changed",
                identifier,
                f"snapshot {report.snapshot_ids_before[identifier]} became {after}",
            )


def _read(
    found: Mapping[str, Table],
    tables: Sequence[AuditedTable],
    report: SilverAuditReport,
) -> dict[str, pa.Table]:
    """Each table at the snapshot the catalog check loaded."""

    data = {}
    for table in tables:
        rows = found[table.identifier].scan().to_arrow()
        # PyIceberg gives an empty scan large Arrow types; joins need the
        # contract's types on both sides.
        data[table.identifier] = (
            rows if rows.num_rows else table.spec.arrow_schema.empty_table()
        )
    report.row_counts.update((name, rows.num_rows) for name, rows in data.items())
    return data


def _check_rows(
    found: Mapping[str, Table],
    data_dir: Path,
    report: SilverAuditReport,
    *,
    verify_source_hashes: bool,
) -> None:
    """Keys, references, groups and semantics of every table, one group at a time."""

    with timed_operation("silver_audit.documents"):
        data = _read(found, DOCUMENT_TABLES, report)
        check_keys(data, {t.identifier: t.unique for t in DOCUMENT_TABLES}, report)
        check_references(data, DOCUMENT_RELATIONS, report)
        parents = check_groups(data, data_dir, report)
        _check_semantics(data, report)
    if verify_source_hashes:
        with timed_operation("silver_audit.source_hashes"):
            check_source_hashes(parents, data_dir, report)
    keys = {name: data[name].select(list(c)) for name, c in _XBRL_PARENT_KEYS.items()}
    del data  # the XBRL checks only need the keys above
    with timed_operation("silver_audit.xbrl"):
        xbrl = _read(found, XBRL_TABLES, report) | keys
        check_keys(xbrl, {t.identifier: t.unique for t in XBRL_TABLES}, report)
        check_references(xbrl, XBRL_RELATIONS, report)
        check_xbrl_groups(xbrl, parents.documents, report)


def _audit(
    catalog: Catalog,
    data_dir: Path,
    report: SilverAuditReport,
    *,
    verify_source_hashes: bool,
) -> None:
    with timed_operation("silver_audit.catalog"):
        found = _check_catalog(catalog, report)
    with timed_operation("silver_audit.schemas"):
        valid = _check_definitions(found, data_dir, report)
    if len(valid) == len(SILVER_TABLES):
        _check_rows(found, data_dir, report, verify_source_hashes=verify_source_hashes)
    else:
        report.add(
            "info",
            "row_checks_skipped",
            NAMESPACE,
            "row checks need every audited table with its contract definition",
        )
    with timed_operation("silver_audit.snapshot_verify"):
        _verify_snapshots(catalog, report)


def validate_silver(
    data_dir: Path, *, verify_source_hashes: bool = False
) -> SilverAuditReport:
    """Audit the SEC Silver tables of `data_dir` without writing anything.

    Checks that every audited table exists with its contract schema, partition
    spec and location; then, when all do, primary keys and grain keys,
    foreign keys and copied lineage, row contracts, group completeness and
    table semantics, and with `verify_source_hashes` the Bronze bytes behind
    filings, documents and entities. Snapshot IDs must not move during the
    audit. Findings are returned in the report; a missing catalog or an
    unexpected error is logged as `silver_integrity_failed` and raised.
    """

    data_dir = data_dir.expanduser().resolve()
    started = perf_counter()
    report = SilverAuditReport(
        started_at=datetime.now(UTC),
        tables_expected=[expected.identifier for expected in SILVER_TABLES],
    )
    with log_context(source="sec_filings"):
        logger.info(
            "silver_integrity_started",
            data_dir=str(data_dir),
            verify_source_hashes=verify_source_hashes,
        )
        try:
            for path in silver_catalog_paths(data_dir):
                if not path.exists():  # opening would create it
                    raise FileNotFoundError(f"Silver catalog path is missing: {path}")
            with (
                timed_operation("silver_audit.total"),
                open_silver_catalog(data_dir) as catalog,
            ):
                _audit(
                    catalog,
                    data_dir,
                    report,
                    verify_source_hashes=verify_source_hashes,
                )
        except Exception as error:
            logger.error(
                "silver_integrity_failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        report.finished_at = datetime.now(UTC)
        report.duration_seconds = round(perf_counter() - started, 3)
        if report.status == "passed":
            logger.info("silver_integrity_completed", **report.summary())
        else:
            logger.error("silver_integrity_failed", **report.summary())
    return report

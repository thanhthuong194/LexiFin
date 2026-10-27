from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
from lxml import etree

from src.data_pipeline.contracts.sec_filings.xbrl_core import source_set_sha256
from src.data_pipeline.quality.sec_filings.silver_audit_progress import (
    AuditProgress,
    NoProgress,
)
from src.data_pipeline.quality.sec_filings.silver_audit_report import (
    SilverAuditReport,
    Status,
)
from src.data_pipeline.quality.sec_filings.silver_audit_scan import local_path
from src.data_pipeline.quality.sec_filings.silver_model import (
    RELATIONS,
    SILVER_TABLES,
    TABLE_SEMANTICS_REBUILD,
    Relation,
)
from src.data_pipeline.quality.sec_filings.xbrl_linkbases import (
    RELATIONSHIP_ATTRIBUTE_LINKBASES,
)
from src.data_pipeline.storage.sec_filings.content_blocks import CONTENT_BLOCKS_TABLE
from src.data_pipeline.storage.sec_filings.document_links import DOCUMENT_LINKS_TABLE
from src.data_pipeline.storage.sec_filings.document_table_cells import (
    DOCUMENT_TABLE_CELLS_TABLE,
)
from src.data_pipeline.storage.sec_filings.document_tables import DOCUMENT_TABLES_TABLE
from src.data_pipeline.storage.sec_filings.documents import (
    DOCUMENTS_SPEC,
    DOCUMENTS_TABLE,
)
from src.data_pipeline.storage.sec_filings.exhibits import EXHIBITS_TABLE
from src.data_pipeline.storage.sec_filings.filing_entities import (
    FILING_ENTITIES_SPEC,
    FILING_ENTITIES_TABLE,
)
from src.data_pipeline.storage.sec_filings.filings import FILINGS_SPEC, FILINGS_TABLE
from src.data_pipeline.storage.sec_filings.sections import SECTIONS_TABLE
from src.data_pipeline.storage.sec_filings.xbrl_core import (
    XBRL_CONTEXTS_TABLE,
    XBRL_INSTANCES_TABLE,
    XBRL_UNITS_TABLE,
)
from src.data_pipeline.storage.sec_filings.xbrl_enrichment import (
    XBRL_DIMENSIONS_TABLE,
    XBRL_FACT_FOOTNOTES_TABLE,
)
from src.data_pipeline.storage.sec_filings.xbrl_facts import (
    XBRL_CONCEPTS_TABLE,
    XBRL_FACTS_TABLE,
)
from src.data_pipeline.storage.sec_filings.xbrl_linkbases import (
    XBRL_LABELS_TABLE,
    XBRL_NETWORKS_TABLE,
    XBRL_RELATIONSHIPS_TABLE,
)
from src.data_pipeline.transforms.sec_filings.xbrl_dimensions import (
    expected_dimension_count,
)

type Progress = AuditProgress | NoProgress
# Derived from a context's segment and scenario XML while it is scanned; null
# when that XML does not parse.
EXPECTED_DIMENSIONS = "expected_dimension_count"
# One row per entry of xbrl_instances.source_document_ids, in list order.
_INSTANCE_SOURCES = "instance_sources"


def _name(identifier: str) -> str:
    """The scratch table of a Silver table: its name without the namespace."""

    return identifier.split(".", 1)[1]


@dataclass(frozen=True)
class GroupRule:
    """Rows of `table` grouped by `group` have ordinals exactly 1..N.

    When the contract has a `declared` count column, every row of the group
    holds N in it.
    """

    table: str
    group: str
    ordinal: str
    declared: str | None = None


# The ordinal groups of the record contracts and stage validators.
GROUP_RULES = (
    GroupRule(DOCUMENTS_TABLE, "filing_id", "document_ordinal"),
    GroupRule(FILING_ENTITIES_TABLE, "filing_id", "entity_ordinal"),
    GroupRule(SECTIONS_TABLE, "document_id", "section_ordinal", "document_section_count"),
    GroupRule(CONTENT_BLOCKS_TABLE, "document_id", "block_ordinal", "document_block_count"),
    GroupRule(DOCUMENT_TABLES_TABLE, "document_id", "table_ordinal", "document_table_count"),
    GroupRule(DOCUMENT_TABLE_CELLS_TABLE, "table_id", "cell_ordinal", "table_cell_count"),
    GroupRule(DOCUMENT_LINKS_TABLE, "source_document_id", "link_ordinal", "document_link_count"),
    GroupRule(EXHIBITS_TABLE, "filing_id", "exhibit_ordinal", "filing_exhibit_count"),
    GroupRule(XBRL_FACTS_TABLE, "instance_id", "fact_order"),
    GroupRule(XBRL_DIMENSIONS_TABLE, "context_key", "dimension_ordinal", "context_dimension_count"),
    GroupRule(XBRL_FACT_FOOTNOTES_TABLE, "fact_id", "footnote_ordinal", "fact_footnote_count"),
    GroupRule(XBRL_NETWORKS_TABLE, "instance_id", "network_ordinal", "instance_network_count"),
    GroupRule(XBRL_RELATIONSHIPS_TABLE, "network_id", "relationship_ordinal", "network_relationship_count"),
)  # fmt: skip


@dataclass(frozen=True)
class DeclaredCount:
    """`parent.declared` counts the `child` rows whose `child_key` is the parent's key."""

    parent: str
    parent_key: str
    declared: str
    child: str
    child_key: str


# Child counts that a parent row declares. A count of zero allows no child;
# contracts require at least one cell per table and relationship per network.
DECLARED_COUNTS = (
    DeclaredCount(XBRL_INSTANCES_TABLE, "instance_id", "fact_count", XBRL_FACTS_TABLE, "instance_id"),
    DeclaredCount(XBRL_INSTANCES_TABLE, "instance_id", "context_count", XBRL_CONTEXTS_TABLE, "instance_id"),
    DeclaredCount(XBRL_INSTANCES_TABLE, "instance_id", "unit_count", XBRL_UNITS_TABLE, "instance_id"),
    DeclaredCount(DOCUMENT_TABLES_TABLE, "table_id", "cell_count", DOCUMENT_TABLE_CELLS_TABLE, "table_id"),
    DeclaredCount(XBRL_NETWORKS_TABLE, "network_id", "relationship_count", XBRL_RELATIONSHIPS_TABLE, "network_id"),
    DeclaredCount(XBRL_CONTEXTS_TABLE, "context_key", EXPECTED_DIMENSIONS, XBRL_DIMENSIONS_TABLE, "context_key"),
)  # fmt: skip


@dataclass(frozen=True)
class SqlRule:
    """A check whose `sql` selects one identifier per failing row or group."""

    check_name: str
    table: str
    message: str
    sql: str
    severity: Status = "FAIL"


def _primary_html_only(identifier: str) -> SqlRule:
    return SqlRule(
        "primary_html_documents_only",
        identifier,
        "rows belong to primary HTML documents only",
        f"SELECT DISTINCT x.document_id FROM {_name(identifier)} x "
        "JOIN documents d ON d.document_id = x.document_id "
        "WHERE d.role != 'primary_document' OR d.content_kind != 'html'",
    )


def _relationship_attributes() -> SqlRule:
    misplaced = " OR ".join(
        f"(r.{column} IS NOT NULL AND n.linkbase_type != '{linkbase_type}')"
        for column, linkbase_type in RELATIONSHIP_ATTRIBUTE_LINKBASES.items()
    )
    return SqlRule(
        "relationship_attributes",
        XBRL_RELATIONSHIPS_TABLE,
        "a relationship holds only the attributes of its network's linkbase type",
        "SELECT r.relationship_id FROM xbrl_relationships r JOIN xbrl_networks n "
        f"ON n.network_id = r.network_id WHERE {misplaced}",
    )


def _dts_source(identifier: str, key: str) -> SqlRule:
    return SqlRule(
        "dts_source_document",
        identifier,
        "source_document_id is a URL or a stored document",
        f"SELECT x.{key} FROM {_name(identifier)} x "
        "WHERE instr(x.source_document_id, '://') = 0 AND NOT EXISTS "
        "(SELECT 1 FROM documents d WHERE d.document_id = x.source_document_id)",
    )


# Cardinality, hierarchy and semantics rules of the model (see
# docs/data/sec_filings_silver_model.md), in table order.
SQL_RULES = (
    SqlRule("has_documents", FILINGS_TABLE, "every filing has documents", "SELECT f.filing_id FROM filings f WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.filing_id = f.filing_id)"),
    SqlRule("has_entities", FILINGS_TABLE, "every filing has entity rows", "SELECT f.filing_id FROM filings f WHERE NOT EXISTS (SELECT 1 FROM filing_entities e WHERE e.filing_id = f.filing_id)"),
    SqlRule("one_primary_document", FILINGS_TABLE, "every filing has exactly one primary_document", "SELECT f.filing_id FROM filings f WHERE (SELECT COUNT(*) FROM documents d WHERE d.filing_id = f.filing_id AND d.role = 'primary_document') != 1"),
    SqlRule("primary_document_form", DOCUMENTS_TABLE, "the primary document's type is the filing form", "SELECT d.document_id FROM documents d JOIN filings f ON f.filing_id = d.filing_id WHERE d.role = 'primary_document' AND d.document_type != f.form"),
    SqlRule("public_document_count", DOCUMENTS_TABLE, "the header PUBLIC DOCUMENT COUNT equals the stored <DOCUMENT> blocks (the header is known to overcount: warning only)", "SELECT f.filing_id FROM filings f WHERE f.public_document_count != (SELECT COUNT(*) FROM documents d WHERE d.filing_id = f.filing_id)", "WARN"),
    SqlRule("has_narrative", SECTIONS_TABLE, "every primary HTML document has sections", "SELECT d.document_id FROM documents d WHERE d.role = 'primary_document' AND d.content_kind = 'html' AND NOT EXISTS (SELECT 1 FROM sections s WHERE s.document_id = d.document_id)"),
    _primary_html_only(SECTIONS_TABLE),
    SqlRule("one_root", SECTIONS_TABLE, "every document has exactly one root section (null parent)", "SELECT document_id FROM sections GROUP BY document_id HAVING SUM(parent_section_id IS NULL) != 1"),
    SqlRule("no_self_parent", SECTIONS_TABLE, "no section is its own parent", "SELECT section_id FROM sections WHERE parent_section_id = section_id"),
    SqlRule("parent_shallower", SECTIONS_TABLE, "a parent's level is below its child's, so levels strictly decrease up every parent chain and no chain can cycle", "SELECT c.section_id FROM sections c JOIN sections p ON p.section_id = c.parent_section_id WHERE p.level >= c.level"),
    SqlRule("range_in_parent", SECTIONS_TABLE, "a section's block range lies inside its parent's", "SELECT c.section_id FROM sections c JOIN sections p ON p.section_id = c.parent_section_id WHERE c.start_block_ordinal < p.start_block_ordinal OR c.end_block_ordinal > p.end_block_ordinal"),
    SqlRule("increasing_starts", SECTIONS_TABLE, "detected sections start in strictly increasing block order", "SELECT section_id FROM (SELECT section_id, section_ordinal, start_block_ordinal, LAG(start_block_ordinal) OVER (PARTITION BY document_id ORDER BY section_ordinal) AS previous FROM sections) WHERE section_ordinal > 2 AND start_block_ordinal <= previous"),
    SqlRule("root_spans_blocks", SECTIONS_TABLE, "the root section spans exactly the document's blocks", "SELECT s.section_id FROM sections s WHERE s.parent_section_id IS NULL AND s.end_block_ordinal != 1 + (SELECT COUNT(*) FROM content_blocks b WHERE b.document_id = s.document_id)"),
    SqlRule("root_only_document", SECTIONS_TABLE, "documents have a detected section besides the root (warning only)", "SELECT document_id FROM sections GROUP BY document_id HAVING COUNT(*) = 1", "WARN"),
    SqlRule("repeated_section_code", SECTIONS_TABLE, "no section code repeats under the same parent code (warning only)", "SELECT c.document_id || ' ' || c.section_code FROM sections c JOIN sections p ON p.section_id = c.parent_section_id GROUP BY c.document_id, p.section_code, c.section_code HAVING COUNT(*) > 1", "WARN"),
    _primary_html_only(CONTENT_BLOCKS_TABLE),
    SqlRule("block_in_section_range", CONTENT_BLOCKS_TABLE, "a block lies inside its section's block range", "SELECT b.block_id FROM content_blocks b JOIN sections s ON s.section_id = b.section_id WHERE b.block_ordinal < s.start_block_ordinal OR b.block_ordinal >= s.end_block_ordinal"),
    SqlRule("block_in_deepest_section", CONTENT_BLOCKS_TABLE, "no deeper section of the document contains the block", "SELECT b.block_id FROM content_blocks b JOIN sections s ON s.section_id = b.section_id WHERE EXISTS (SELECT 1 FROM sections d WHERE d.document_id = b.document_id AND d.level > s.level AND d.start_block_ordinal <= b.block_ordinal AND b.block_ordinal < d.end_block_ordinal)"),
    SqlRule("no_table_cell_blocks", CONTENT_BLOCKS_TABLE, f"no block uses the retired table_cell layout (rebuild under control, in order: {', '.join(TABLE_SEMANTICS_REBUILD)})", "SELECT block_id FROM content_blocks WHERE block_type = 'table_cell'"),
    SqlRule("table_block_source", CONTENT_BLOCKS_TABLE, "a table block has source_tag 'table' and an XPath ending at a <table>", "SELECT block_id FROM content_blocks WHERE block_type = 'table' AND (source_tag != 'table' OR rtrim(source_xpath, '0123456789[]') NOT LIKE '%/table')"),
    SqlRule("table_block_has_table", CONTENT_BLOCKS_TABLE, "every table block has its document_tables row", "SELECT b.block_id FROM content_blocks b WHERE b.block_type = 'table' AND NOT EXISTS (SELECT 1 FROM document_tables t WHERE t.block_id = b.block_id)"),
    _primary_html_only(DOCUMENT_TABLES_TABLE),
    SqlRule("header_cell_count", DOCUMENT_TABLES_TABLE, "header_cell_count equals the table's <th> cells", "SELECT t.table_id FROM document_tables t WHERE t.header_cell_count != (SELECT COUNT(*) FROM document_table_cells c WHERE c.table_id = t.table_id AND c.is_header)"),
    SqlRule("cells_fill_grid", DOCUMENT_TABLES_TABLE, "cells lie within row_count rows and span exactly column_count columns", "SELECT t.table_id FROM document_tables t JOIN (SELECT table_id, MAX(row_ordinal) AS last_row, MAX(column_ordinal + colspan - 1) AS last_column FROM document_table_cells GROUP BY table_id) g ON g.table_id = t.table_id WHERE g.last_row > t.row_count OR g.last_column != t.column_count"),
    SqlRule("table_on_table_block", DOCUMENT_TABLES_TABLE, "every document table belongs to a table block", "SELECT t.table_id FROM document_tables t JOIN content_blocks b ON b.block_id = t.block_id WHERE b.block_type != 'table'"),
    SqlRule("links_from_html", DOCUMENT_LINKS_TABLE, "links come from HTML documents", "SELECT DISTINCT l.source_document_id FROM document_links l JOIN documents d ON d.document_id = l.source_document_id WHERE d.content_kind != 'html'"),
    SqlRule("resolved_links", DOCUMENT_LINKS_TABLE, "links resolve to a stored document (unresolved and ambiguous links are valid data: warning only)", "SELECT link_id FROM document_links WHERE resolution_status IN ('unresolved', 'ambiguous')", "WARN"),
    SqlRule("exhibit_from_primary", EXHIBITS_TABLE, "Exhibit Index entries are read from the primary document", "SELECT x.exhibit_id FROM exhibits x JOIN documents d ON d.document_id = x.source_document_id WHERE d.role != 'primary_document'"),
    SqlRule("source_documents_in_filing", XBRL_INSTANCES_TABLE, "every source document of an instance is a stored document of its filing", f"SELECT s.instance_id FROM {_INSTANCE_SOURCES} s JOIN xbrl_instances i ON i.instance_id = s.instance_id LEFT JOIN documents d ON d.document_id = s.document_id WHERE d.document_id IS NULL OR d.filing_id != i.filing_id"),
    SqlRule("source_documents_ordered", XBRL_INSTANCES_TABLE, "an instance lists its source documents in document order", f"SELECT s.instance_id FROM {_INSTANCE_SOURCES} s JOIN {_INSTANCE_SOURCES} n ON n.instance_id = s.instance_id AND n.position = s.position + 1 JOIN documents a ON a.document_id = s.document_id JOIN documents b ON b.document_id = n.document_id WHERE a.document_ordinal >= b.document_ordinal"),
    SqlRule("context_xml_parses", XBRL_CONTEXTS_TABLE, "segment and scenario XML parse", f"SELECT context_key FROM xbrl_contexts WHERE {EXPECTED_DIMENSIONS} IS NULL"),
    SqlRule("fact_document_in_instance", XBRL_FACTS_TABLE, "a fact comes from a source document of its instance", f"SELECT f.fact_id FROM xbrl_facts f WHERE NOT EXISTS (SELECT 1 FROM {_INSTANCE_SOURCES} s WHERE s.instance_id = f.instance_id AND s.document_id = f.document_id)"),
    _dts_source(XBRL_LABELS_TABLE, "label_id"),
    _dts_source(XBRL_RELATIONSHIPS_TABLE, "relationship_id"),
    SqlRule("footnote_resource_consistent", XBRL_FACT_FOOTNOTES_TABLE, "rows of one footnote resource share its location and content", "SELECT footnote_resource_id FROM xbrl_fact_footnotes GROUP BY footnote_resource_id HAVING COUNT(DISTINCT source_xpath || ' ' || footnote_content_sha256) > 1"),
    _relationship_attributes(),
    SqlRule("dts_document_in_filing", XBRL_RELATIONSHIPS_TABLE, "a relationship read from a filing document comes from its instance's filing", "SELECT r.relationship_id FROM xbrl_relationships r JOIN xbrl_networks n ON n.network_id = r.network_id JOIN xbrl_instances i ON i.instance_id = n.instance_id JOIN documents d ON d.document_id = r.source_document_id WHERE d.filing_id != i.filing_id"),
)  # fmt: skip

# Columns the rules above and the sample report read, besides keys,
# relations, groups and counts.
_READ_COLUMNS = {
    # Every column: Bronze checks rebuild these rows as records.
    FILINGS_TABLE: tuple(FILINGS_SPEC.arrow_schema.names),
    DOCUMENTS_TABLE: tuple(DOCUMENTS_SPEC.arrow_schema.names),
    FILING_ENTITIES_TABLE: tuple(FILING_ENTITIES_SPEC.arrow_schema.names),
    SECTIONS_TABLE: (
        "section_code",
        "level",
        "start_block_ordinal",
        "end_block_ordinal",
    ),
    CONTENT_BLOCKS_TABLE: ("block_type", "source_tag", "source_xpath"),
    DOCUMENT_TABLES_TABLE: ("header_cell_count", "row_count", "column_count"),
    DOCUMENT_TABLE_CELLS_TABLE: (
        "is_header",
        "row_ordinal",
        "column_ordinal",
        "colspan",
    ),
    DOCUMENT_LINKS_TABLE: ("resolution_status",),
    XBRL_INSTANCES_TABLE: ("source_set_sha256",),
    XBRL_CONCEPTS_TABLE: ("local_name",),
    XBRL_LABELS_TABLE: ("source_document_id",),
    XBRL_FACT_FOOTNOTES_TABLE: (
        "footnote_resource_id",
        "source_xpath",
        "footnote_content_sha256",
    ),
    XBRL_RELATIONSHIPS_TABLE: (
        "source_document_id",
        *RELATIONSHIP_ATTRIBUTE_LINKBASES,
    ),
}


def _expected_dimensions(row: Mapping[str, Any]) -> int | None:
    try:
        return expected_dimension_count(row["segment_xml"], row["scenario_xml"])
    except etree.XMLSyntaxError:
        return None


# Columns computed from a scanned row, by table: (column, function of the row).
_DERIVED: dict[str, tuple[str, Callable[[Mapping[str, Any]], object]]] = {
    XBRL_CONTEXTS_TABLE: (EXPECTED_DIMENSIONS, _expected_dimensions),
}


def scratch_columns() -> dict[str, tuple[str, ...]]:
    """The Silver columns each check of this module reads, by table.

    Keys, relation keys and lineage, group and count columns, and the
    columns the rules name; text and other wide columns stay out.
    """

    needed: dict[str, list[str]] = {table.identifier: [] for table in SILVER_TABLES}
    for table in SILVER_TABLES:
        for key in table.unique:
            needed[table.identifier].extend(key)
    for relation in RELATIONS:
        needed[relation.child] += [relation.child_id, relation.child_key]
        needed[relation.child] += [column for column, _ in relation.lineage]
        needed[relation.parent] += [relation.parent_key]
        needed[relation.parent] += [column for _, column in relation.lineage]
    for rule in GROUP_RULES:
        needed[rule.table] += [rule.group, rule.ordinal, *filter(None, [rule.declared])]
    for count in DECLARED_COUNTS:
        needed[count.parent].append(count.parent_key)
        if count.declared != EXPECTED_DIMENSIONS:
            needed[count.parent].append(count.declared)
        needed[count.child].append(count.child_key)
    for identifier, columns in _READ_COLUMNS.items():
        needed[identifier].extend(columns)
    return {name: tuple(dict.fromkeys(columns)) for name, columns in needed.items()}


# DuckDB's memory budget; joins and sorts beyond it spill to its temp directory.
_MEMORY_LIMIT = "4GB"
# Rows fetched at a time from a query result.
_FETCH_ROWS = 10_000


def _arrow_schema(identifier: str, columns: Sequence[str]) -> pa.Schema:
    schema = next(
        t.spec for t in SILVER_TABLES if t.identifier == identifier
    ).arrow_schema
    fields = [schema.field(column) for column in columns]
    if identifier in _DERIVED:
        fields.append(pa.field(_DERIVED[identifier][0], pa.int64()))
    return pa.schema(fields)


class ScratchDatabase:
    """A DuckDB copy of the columns the set-based checks read.

    It lives in a file the caller owns, never in the Iceberg catalog. Rows
    are stored by column and compressed, joins and aggregations run as hash
    operations on every core, and DuckDB spills what does not fit in
    _MEMORY_LIMIT to temporary files next to the database, so duplicate,
    orphan and group checks run in bounded memory whatever the table sizes.
    Every statement commits on its own and DuckDB replays its write-ahead
    log when the file is opened again, so after a crash or power loss the
    database holds exactly the statements that completed.
    """

    def __init__(self, path: Path) -> None:
        self.connection = duckdb.connect(str(path))
        spill = path.with_name(f"{path.name}.tmp")
        self.connection.execute(
            f"SET memory_limit = '{_MEMORY_LIMIT}'; "
            f"SET temp_directory = '{spill}'; "
            "SET preserve_insertion_order = false"
        )
        self.columns = scratch_columns()
        self.schemas = {
            identifier: _arrow_schema(identifier, columns)
            for identifier, columns in self.columns.items()
        }
        # Rows stored per Silver table.
        self.row_counts = dict.fromkeys(self.columns, 0)
        for identifier in self.columns:
            self._create(identifier)

    def _create(self, identifier: str) -> None:
        self._from_arrow(
            f"CREATE TABLE IF NOT EXISTS {_name(identifier)} AS SELECT * FROM {{}}",
            self.schemas[identifier].empty_table(),
        )
        if identifier == XBRL_INSTANCES_TABLE:
            self.connection.execute(
                f"CREATE TABLE IF NOT EXISTS {_INSTANCE_SOURCES} "
                "(instance_id VARCHAR, position INTEGER, document_id VARCHAR)"
            )

    def _from_arrow(self, sql: str, table: pa.Table) -> None:
        """Run `sql` with `{}` naming the Arrow table."""

        self.connection.register("arrow_rows", table)
        try:
            self.connection.execute(sql.format("arrow_rows"))
        finally:
            self.connection.unregister("arrow_rows")

    def close(self) -> None:
        """Close the connection; the caller deletes the file."""

        self.connection.close()

    def reset(self, identifier: str) -> None:
        """Drop the rows of a table whose scan did not complete."""

        self.connection.execute(f"DROP TABLE {_name(identifier)}")
        if identifier == XBRL_INSTANCES_TABLE:
            self.connection.execute(f"DROP TABLE {_INSTANCE_SOURCES}")
        self._create(identifier)
        self.row_counts[identifier] = 0

    def insert(self, identifier: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """Store the checked columns of some rows given as dicts."""

        columns = self.columns[identifier]
        stored = pa.Table.from_pylist(
            [{column: row[column] for column in columns} for row in rows],
            schema=pa.schema([self.schemas[identifier].field(c) for c in columns]),
        )
        derive = _DERIVED.get(identifier)
        if derive is not None:
            stored = stored.append_column(
                self.schemas[identifier].field(derive[0]),
                pa.array(map(derive[1], rows), pa.int64()),
            )
        self._from_arrow(
            f"INSERT INTO {_name(identifier)} BY NAME SELECT * FROM {{}}", stored
        )
        if identifier == XBRL_INSTANCES_TABLE:
            sources = pa.Table.from_pylist(
                [
                    {
                        "instance_id": row["instance_id"],
                        "position": position,
                        "document_id": document_id,
                    }
                    for row in rows
                    for position, document_id in enumerate(row["source_document_ids"])
                ],
                schema=pa.schema(
                    [
                        ("instance_id", pa.string()),
                        ("position", pa.int32()),
                        ("document_id", pa.string()),
                    ]
                ),
            )
            self._from_arrow(
                f"INSERT INTO {_INSTANCE_SOURCES} BY NAME SELECT * FROM {{}}", sources
            )
        self.row_counts[identifier] += len(rows)

    def load(
        self,
        identifier: str,
        paths: Sequence[str],
        rows: int,
        derived: Mapping[str, Sequence[object]] | None = None,
    ) -> None:
        """Copy the checked columns of a table's data files straight from
        Parquet; `rows` is how many they hold. `derived` gives the derived
        column's values of each file's rows in file order."""

        if not paths:
            return
        files = ", ".join(
            "'" + local_path(path).replace("'", "''") + "'" for path in paths
        )
        source = f"read_parquet([{files}])"
        selected = ", ".join(f"p.{column}" for column in self.columns[identifier])
        derive = _DERIVED.get(identifier)
        if derive is None:
            self.connection.execute(
                f"INSERT INTO {_name(identifier)} BY NAME "
                f"SELECT {selected} FROM {source} p"
            )
        else:
            by_file = [(local_path(path), v) for path, v in (derived or {}).items()]
            values = pa.table(
                {
                    "filename": [name for name, column in by_file for _ in column],
                    "file_row_number": [
                        number for _, column in by_file for number in range(len(column))
                    ],
                    "value": pa.array(
                        [value for _, column in by_file for value in column], pa.int64()
                    ),
                }
            )
            self._from_arrow(
                f"INSERT INTO {_name(identifier)} BY NAME SELECT {selected}, "
                f"d.value AS {derive[0]} FROM read_parquet([{files}], "
                "filename = 'scratch_file', file_row_number = true) p "
                "LEFT JOIN {} d ON d.filename = p.scratch_file "
                "AND d.file_row_number = p.file_row_number",
                values,
            )
        if identifier == XBRL_INSTANCES_TABLE:
            self.connection.execute(
                f"INSERT INTO {_INSTANCE_SOURCES} BY NAME SELECT instance_id, "
                "unnest(range(len(source_document_ids))) AS position, "
                f"unnest(source_document_ids) AS document_id FROM {source}"
            )
        self.row_counts[identifier] = rows

    def rows(self, sql: str, *parameters: object) -> Iterator[tuple[Any, ...]]:
        """The rows of a query, fetched lazily on a cursor of their own."""

        cursor = self.connection.cursor()
        cursor.execute(sql, parameters)
        while chunk := cursor.fetchmany(_FETCH_ROWS):
            yield from chunk
        cursor.close()

    def stored_rows(
        self, identifier: str, column: str, value: object
    ) -> list[dict[str, Any]]:
        """The stored columns of the rows whose `column` equals `value`, as dicts."""

        cursor = self.connection.execute(
            f"SELECT * FROM {_name(identifier)} WHERE {column} = ?", (value,)
        )
        names = [description[0] for description in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

    def filing_ids(self) -> Iterator[str]:
        """Every stored filing_id, in order."""

        return (row[0] for row in self.rows("SELECT filing_id FROM filings ORDER BY 1"))

    def local_names(self, concept_ids: Collection[str]) -> dict[str, str]:
        """The local names of concepts, by concept_id."""

        rows = self.connection.execute(
            "SELECT concept_id, local_name FROM xbrl_concepts "
            "WHERE list_contains(?, concept_id)",
            (sorted(concept_ids),),
        ).fetchall()
        return dict(rows)


def _check_unique_keys(
    scratch: ScratchDatabase, report: SilverAuditReport, progress: Progress
) -> None:
    for table in SILVER_TABLES:
        rows = scratch.row_counts[table.identifier]
        for key in table.unique:
            columns = ", ".join(key)
            value = " || '|' || ".join(f"CAST({column} AS VARCHAR)" for column in key)
            step = f"unique:{table.identifier}:{'+'.join(key)}"
            if progress.skip(step):
                continue
            with report.check(
                f"unique:{'+'.join(key)}",
                table.identifier,
                f"{'+'.join(key)} identifies one row",
            ) as run:
                run.rows_checked = rows
                for repeated, count in scratch.rows(
                    f"SELECT {value}, COUNT(*) "
                    f"FROM {_name(table.identifier)} GROUP BY {columns} "
                    "HAVING COUNT(*) > 1"
                ):
                    run.found(repeated, f"{repeated} appears {count} times", count - 1)
            progress.complete(step)


def _check_relations(
    scratch: ScratchDatabase, report: SilverAuditReport, progress: Progress
) -> None:
    for relation in RELATIONS:
        child, parent = _name(relation.child), _name(relation.parent)
        rows = scratch.row_counts[relation.child]
        step = f"orphan:{relation.name}"
        if not progress.skip(step):
            _check_orphans(scratch, report, relation, rows)
            progress.complete(step)
        step = f"lineage:{relation.name}"
        if not relation.lineage or progress.skip(step):
            continue
        copied = ", ".join(f"{c} = {parent}.{p}" for c, p in relation.lineage)
        differs = " OR ".join(
            f"c.{c} IS DISTINCT FROM p.{p}" for c, p in relation.lineage
        )
        with report.check(
            f"lineage:{relation.name}",
            relation.child,
            f"rows with a parent repeat its values: {copied}",
        ) as run:
            run.rows_checked = rows
            for (child_id,) in scratch.rows(
                f"SELECT c.{relation.child_id} FROM {child} c JOIN {parent} p "
                f"ON p.{relation.parent_key} = c.{relation.child_key} WHERE {differs}"
            ):
                run.found(child_id)
        progress.complete(step)


def _check_orphans(
    scratch: ScratchDatabase, report: SilverAuditReport, relation: Relation, rows: int
) -> None:
    child, parent = _name(relation.child), _name(relation.parent)
    with report.check(
        f"orphan:{relation.name}",
        relation.child,
        f"every non-null {relation.child_key} has a {parent}.{relation.parent_key}",
    ) as run:
        run.rows_checked = rows
        for child_id, key in scratch.rows(
            f"SELECT c.{relation.child_id}, c.{relation.child_key} FROM {child} c "
            f"WHERE c.{relation.child_key} IS NOT NULL AND NOT EXISTS (SELECT 1 "
            f"FROM {parent} p WHERE p.{relation.parent_key} = c.{relation.child_key})"
        ):
            run.found(child_id, f"{child_id}: no {parent} row {key}")


def _check_groups(
    scratch: ScratchDatabase, report: SilverAuditReport, progress: Progress
) -> None:
    for rule in GROUP_RULES:
        complete = f"MIN({rule.ordinal}) = 1 AND MAX({rule.ordinal}) = COUNT(*) AND COUNT(DISTINCT {rule.ordinal}) = COUNT(*)"
        declared = "NULL, NULL"
        if rule.declared is not None:
            complete += f" AND MIN({rule.declared}) = COUNT(*) AND MAX({rule.declared}) = COUNT(*)"
            declared = f"MIN({rule.declared}), MAX({rule.declared})"
        step = f"complete_group:{rule.table}:{rule.group}"
        if progress.skip(step):
            continue
        with report.check(
            "complete_group",
            rule.table,
            f"per {rule.group}, {rule.ordinal} runs 1..N"
            + (f" and every row declares {rule.declared} = N" if rule.declared else ""),
        ) as run:
            run.rows_checked = scratch.row_counts[rule.table]
            for group, rows, low, high, distinct, *counts in scratch.rows(
                f"SELECT {rule.group}, COUNT(*), MIN({rule.ordinal}), "
                f"MAX({rule.ordinal}), COUNT(DISTINCT {rule.ordinal}), {declared} "
                f"FROM {_name(rule.table)} GROUP BY {rule.group} HAVING NOT ({complete})"
            ):
                run.found(
                    group,
                    f"{group}: {rows} rows, {distinct} distinct ordinals "
                    f"{low}..{high}, declared {counts}",
                )
        progress.complete(step)


def _check_declared_counts(
    scratch: ScratchDatabase, report: SilverAuditReport, progress: Progress
) -> None:
    for count in DECLARED_COUNTS:
        parent, child = _name(count.parent), _name(count.child)
        step = f"declared_count:{count.parent}:{count.declared}"
        if progress.skip(step):
            continue
        with report.check(
            f"declared_count:{count.declared}",
            count.parent,
            f"{count.declared} equals the {child} rows of the {parent} row",
        ) as run:
            run.rows_checked = scratch.row_counts[count.parent]
            for key, declared, rows in scratch.rows(
                f"SELECT * FROM (SELECT p.{count.parent_key}, p.{count.declared} "
                f"AS declared, (SELECT COUNT(*) FROM {child} c WHERE "
                f"c.{count.child_key} = p.{count.parent_key}) AS found FROM {parent} p) "
                "WHERE declared != found"
            ):
                run.found(key, f"{key}: declares {declared}, has {rows}")
        progress.complete(step)


def _check_rules(
    scratch: ScratchDatabase, report: SilverAuditReport, progress: Progress
) -> None:
    for rule in SQL_RULES:
        step = f"rule:{rule.table}:{rule.check_name}"
        if progress.skip(step):
            continue
        with report.check(
            rule.check_name, rule.table, rule.message, severity=rule.severity
        ) as run:
            run.rows_checked = scratch.row_counts[rule.table]
            for (identifier,) in scratch.rows(rule.sql):
                run.found(identifier)
        progress.complete(step)


def _check_source_sets(
    scratch: ScratchDatabase, report: SilverAuditReport, progress: Progress
) -> None:
    """Recompute each instance's source_set_sha256 from its documents' hashes."""

    if progress.skip("source_set_sha256"):
        return
    with report.check(
        "source_set_sha256",
        XBRL_INSTANCES_TABLE,
        "source_set_sha256 hashes the instance's source documents and their content_sha256",
    ) as run:
        rows = scratch.rows(
            f"SELECT s.instance_id, i.source_set_sha256, s.document_id, d.content_sha256 "
            f"FROM {_INSTANCE_SOURCES} s JOIN xbrl_instances i ON i.instance_id = s.instance_id "
            "LEFT JOIN documents d ON d.document_id = s.document_id "
            "ORDER BY s.instance_id, s.position"
        )
        for instance_id, group in groupby(rows, key=itemgetter(0)):
            sources = list(group)
            run.rows_checked += 1
            pairs = [(row[2], row[3] or "") for row in sources]
            if source_set_sha256(pairs) != sources[0][1]:
                run.found(instance_id)
    progress.complete("source_set_sha256")


def run_sql_checks(
    scratch: ScratchDatabase,
    report: SilverAuditReport,
    progress: Progress | None = None,
) -> None:
    """Keys, foreign keys and lineage, groups, declared counts and model rules.

    Every check reads the scratch database, so memory does not grow with
    the tables; each keeps only a few failing identifiers. A check that
    `progress` holds is replayed, not run again.
    """

    progress = progress or NoProgress()
    _check_unique_keys(scratch, report, progress)
    _check_relations(scratch, report, progress)
    _check_groups(scratch, report, progress)
    _check_declared_counts(scratch, report, progress)
    _check_rules(scratch, report, progress)
    _check_source_sets(scratch, report, progress)


def deriver(identifier: str) -> Callable[[Mapping[str, Any]], object] | None:
    """The function computing a table's derived scratch column from a row."""

    derive = _DERIVED.get(identifier)
    return None if derive is None else derive[1]

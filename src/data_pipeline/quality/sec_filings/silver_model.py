from __future__ import annotations

from dataclasses import dataclass

from src.data_pipeline.storage.iceberg import SilverTableSpec
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
from src.data_pipeline.storage.sec_filings.exhibits import EXHIBITS_SPEC, EXHIBITS_TABLE
from src.data_pipeline.storage.sec_filings.filing_entities import (
    FILING_ENTITIES_SPEC,
    FILING_ENTITIES_TABLE,
)
from src.data_pipeline.storage.sec_filings.filings import FILINGS_SPEC, FILINGS_TABLE
from src.data_pipeline.storage.sec_filings.sections import (
    SECTIONS_SPEC,
    SECTIONS_TABLE,
)
from src.data_pipeline.storage.sec_filings.xbrl_core import (
    XBRL_CONTEXTS_SPEC,
    XBRL_CONTEXTS_TABLE,
    XBRL_INSTANCES_SPEC,
    XBRL_INSTANCES_TABLE,
    XBRL_UNITS_SPEC,
    XBRL_UNITS_TABLE,
)
from src.data_pipeline.storage.sec_filings.xbrl_enrichment import (
    XBRL_DIMENSIONS_SPEC,
    XBRL_DIMENSIONS_TABLE,
    XBRL_FACT_FOOTNOTES_SPEC,
    XBRL_FACT_FOOTNOTES_TABLE,
)
from src.data_pipeline.storage.sec_filings.xbrl_facts import (
    XBRL_CONCEPTS_SPEC,
    XBRL_CONCEPTS_TABLE,
    XBRL_FACTS_SPEC,
    XBRL_FACTS_TABLE,
)
from src.data_pipeline.storage.sec_filings.xbrl_linkbases import (
    XBRL_LABELS_SPEC,
    XBRL_LABELS_TABLE,
    XBRL_NETWORKS_SPEC,
    XBRL_NETWORKS_TABLE,
    XBRL_RELATIONSHIPS_SPEC,
    XBRL_RELATIONSHIPS_TABLE,
)


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


# Every Silver table with its unique keys, parents before children: the order
# the audit scans them in (docs/data/sec_filings_silver_model.md).
SILVER_TABLES = (
    AuditedTable(FILINGS_SPEC, (("filing_id",), ("accession_number",))),
    AuditedTable(DOCUMENTS_SPEC, (("document_id",), ("filing_id", "document_ordinal"))),
    AuditedTable(FILING_ENTITIES_SPEC, (("filing_entity_id",), ("filing_id", "entity_ordinal"))),
    AuditedTable(SECTIONS_SPEC, (("section_id",), ("document_id", "section_ordinal"))),
    AuditedTable(CONTENT_BLOCKS_SPEC, (("block_id",), ("document_id", "block_ordinal"))),
    AuditedTable(DOCUMENT_TABLES_SPEC, (("table_id",), ("block_id",), ("document_id", "table_ordinal"))),
    AuditedTable(DOCUMENT_TABLE_CELLS_SPEC, (("cell_id",), ("table_id", "cell_ordinal"))),
    AuditedTable(DOCUMENT_LINKS_SPEC, (("link_id",), ("source_document_id", "link_ordinal"), ("source_document_id", "source_xpath"))),
    AuditedTable(EXHIBITS_SPEC, (("exhibit_id",), ("filing_id", "exhibit_ordinal"))),
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
# Narrative tables must be rebuilt in this order when the persisted table
# semantics are outdated (one block per <table>, cells in their own table).
TABLE_SEMANTICS_REBUILD = (
    SECTIONS_TABLE,
    CONTENT_BLOCKS_TABLE,
    DOCUMENT_TABLES_TABLE,
    DOCUMENT_TABLE_CELLS_TABLE,
)


@dataclass(frozen=True)
class Relation:
    """A foreign key from `child.child_key` to `parent.parent_key`.

    `lineage` pairs (child column, parent column) that a matched child must
    repeat exactly; `child_id` names the child's key, used in issue reports.
    """

    child: str
    child_key: str
    parent: str
    parent_key: str
    lineage: tuple[tuple[str, str], ...]
    child_id: str

    @property
    def name(self) -> str:
        """`<child>.<column> -> <parent>` without the namespace."""

        child, parent = (t.split(".", 1)[1] for t in (self.child, self.parent))
        return f"{child}.{self.child_key} -> {parent}"


_FROM_DOCUMENT = (
    ("filing_id", "filing_id"),
    ("filing_date", "filing_date"),
    ("source_document_sha256", "content_sha256"),
)
_SAME_DOCUMENT = (
    ("document_id", "document_id"),
    ("filing_id", "filing_id"),
    ("filing_date", "filing_date"),
    ("source_document_sha256", "source_document_sha256"),
)
# The foreign keys of the filing, document and HTML tables, parent first. A
# null key (root section, unresolved link or exhibit entry) is no orphan; a
# resolved target stays in its filing.
_DOCUMENT_RELATIONS = (
    Relation(
        DOCUMENTS_TABLE,
        "filing_id",
        FILINGS_TABLE,
        "filing_id",
        (
            ("filing_date", "filing_date"),
            ("source_path", "full_submission_path"),
            ("source_submission_sha256", "full_submission_sha256"),
        ),
        "document_id",
    ),
    Relation(
        FILING_ENTITIES_TABLE,
        "filing_id",
        FILINGS_TABLE,
        "filing_id",
        (
            ("filing_date", "filing_date"),
            ("source_path", "full_submission_path"),
            ("source_submission_sha256", "full_submission_sha256"),
        ),
        "filing_entity_id",
    ),
    Relation(
        SECTIONS_TABLE,
        "document_id",
        DOCUMENTS_TABLE,
        "document_id",
        _FROM_DOCUMENT,
        "section_id",
    ),
    Relation(
        SECTIONS_TABLE,
        "parent_section_id",
        SECTIONS_TABLE,
        "section_id",
        (("document_id", "document_id"),),
        "section_id",
    ),
    Relation(
        CONTENT_BLOCKS_TABLE,
        "document_id",
        DOCUMENTS_TABLE,
        "document_id",
        _FROM_DOCUMENT,
        "block_id",
    ),
    Relation(
        CONTENT_BLOCKS_TABLE,
        "section_id",
        SECTIONS_TABLE,
        "section_id",
        (("document_id", "document_id"),),
        "block_id",
    ),
    Relation(
        DOCUMENT_TABLES_TABLE,
        "block_id",
        CONTENT_BLOCKS_TABLE,
        "block_id",
        (*_SAME_DOCUMENT, ("source_xpath", "source_xpath")),
        "table_id",
    ),
    Relation(
        DOCUMENT_TABLE_CELLS_TABLE,
        "table_id",
        DOCUMENT_TABLES_TABLE,
        "table_id",
        _SAME_DOCUMENT,
        "cell_id",
    ),
    Relation(
        DOCUMENT_LINKS_TABLE,
        "source_document_id",
        DOCUMENTS_TABLE,
        "document_id",
        _FROM_DOCUMENT,
        "link_id",
    ),
    Relation(
        DOCUMENT_LINKS_TABLE,
        "target_document_id",
        DOCUMENTS_TABLE,
        "document_id",
        (("filing_id", "filing_id"),),
        "link_id",
    ),
    Relation(
        EXHIBITS_TABLE,
        "filing_id",
        FILINGS_TABLE,
        "filing_id",
        (("filing_date", "filing_date"),),
        "exhibit_id",
    ),
    Relation(
        EXHIBITS_TABLE,
        "source_document_id",
        DOCUMENTS_TABLE,
        "document_id",
        (("filing_id", "filing_id"),),
        "exhibit_id",
    ),
    Relation(
        EXHIBITS_TABLE,
        "resolved_document_id",
        DOCUMENTS_TABLE,
        "document_id",
        (("filing_id", "filing_id"),),
        "exhibit_id",
    ),
)


def _xbrl_relation(
    child: str,
    child_key: str,
    parent: str,
    child_id: str,
    lineage: tuple[tuple[str, str], ...] = (),
) -> Relation:
    """A foreign key to a parent's key column of the same name (or `<parent>_id`)."""

    parent_key = {
        FILINGS_TABLE: "filing_id",
        DOCUMENTS_TABLE: "document_id",
        CONTENT_BLOCKS_TABLE: "block_id",
        DOCUMENT_TABLE_CELLS_TABLE: "cell_id",
        XBRL_INSTANCES_TABLE: "instance_id",
        XBRL_CONTEXTS_TABLE: "context_key",
        XBRL_UNITS_TABLE: "unit_key",
        XBRL_CONCEPTS_TABLE: "concept_id",
        XBRL_FACTS_TABLE: "fact_id",
        XBRL_NETWORKS_TABLE: "network_id",
    }[parent]
    return Relation(child, child_key, parent, parent_key, lineage, child_id)


# The foreign keys of the XBRL tables. XBRL children do not repeat filing
# columns; a fact's context and unit belong to its own instance, and its block
# or cell (when aligned) to its own document. Labels and relationships name
# their DTS file by document_id or URL, checked by the audit's own rules.
_XBRL_RELATIONS = (
    _xbrl_relation(XBRL_INSTANCES_TABLE, "filing_id", FILINGS_TABLE, "instance_id"),
    _xbrl_relation(XBRL_CONTEXTS_TABLE, "instance_id", XBRL_INSTANCES_TABLE, "context_key"),
    _xbrl_relation(XBRL_UNITS_TABLE, "instance_id", XBRL_INSTANCES_TABLE, "unit_key"),
    _xbrl_relation(XBRL_FACTS_TABLE, "instance_id", XBRL_INSTANCES_TABLE, "fact_id"),
    _xbrl_relation(XBRL_FACTS_TABLE, "document_id", DOCUMENTS_TABLE, "fact_id"),
    _xbrl_relation(XBRL_FACTS_TABLE, "concept_id", XBRL_CONCEPTS_TABLE, "fact_id"),
    _xbrl_relation(XBRL_FACTS_TABLE, "context_key", XBRL_CONTEXTS_TABLE, "fact_id", (("instance_id", "instance_id"),)),
    _xbrl_relation(XBRL_FACTS_TABLE, "unit_key", XBRL_UNITS_TABLE, "fact_id", (("instance_id", "instance_id"),)),
    _xbrl_relation(XBRL_FACTS_TABLE, "block_id", CONTENT_BLOCKS_TABLE, "fact_id", (("document_id", "document_id"),)),
    _xbrl_relation(XBRL_FACTS_TABLE, "cell_id", DOCUMENT_TABLE_CELLS_TABLE, "fact_id", (("document_id", "document_id"),)),
    _xbrl_relation(XBRL_DIMENSIONS_TABLE, "context_key", XBRL_CONTEXTS_TABLE, "dimension_id"),
    _xbrl_relation(XBRL_DIMENSIONS_TABLE, "dimension_concept_id", XBRL_CONCEPTS_TABLE, "dimension_id"),
    _xbrl_relation(XBRL_DIMENSIONS_TABLE, "member_concept_id", XBRL_CONCEPTS_TABLE, "dimension_id"),
    _xbrl_relation(XBRL_FACT_FOOTNOTES_TABLE, "fact_id", XBRL_FACTS_TABLE, "footnote_relation_id"),
    _xbrl_relation(XBRL_LABELS_TABLE, "concept_id", XBRL_CONCEPTS_TABLE, "label_id"),
    _xbrl_relation(XBRL_NETWORKS_TABLE, "instance_id", XBRL_INSTANCES_TABLE, "network_id"),
    _xbrl_relation(XBRL_RELATIONSHIPS_TABLE, "network_id", XBRL_NETWORKS_TABLE, "relationship_id"),
    _xbrl_relation(XBRL_RELATIONSHIPS_TABLE, "source_concept_id", XBRL_CONCEPTS_TABLE, "relationship_id"),
    _xbrl_relation(XBRL_RELATIONSHIPS_TABLE, "target_concept_id", XBRL_CONCEPTS_TABLE, "relationship_id"),
)  # fmt: skip
RELATIONS = _DOCUMENT_RELATIONS + _XBRL_RELATIONS

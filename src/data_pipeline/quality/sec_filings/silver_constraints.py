from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from src.data_pipeline.quality.sec_filings.silver_groups import ParentRecords
from src.data_pipeline.quality.sec_filings.silver_report import SilverAuditReport
from src.data_pipeline.storage.bronze import (
    resolve_relative_path,
    sha256_bytes,
    sha256_file,
)
from src.data_pipeline.storage.sec_filings.content_blocks import CONTENT_BLOCKS_TABLE
from src.data_pipeline.storage.sec_filings.document_links import DOCUMENT_LINKS_TABLE
from src.data_pipeline.storage.sec_filings.document_table_cells import (
    DOCUMENT_TABLE_CELLS_TABLE,
)
from src.data_pipeline.storage.sec_filings.document_tables import DOCUMENT_TABLES_TABLE
from src.data_pipeline.storage.sec_filings.documents import DOCUMENTS_TABLE
from src.data_pipeline.storage.sec_filings.exhibits import EXHIBITS_TABLE
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
from src.data_pipeline.transforms.sec_filings.documents import check_document_offsets
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PARENT_PREFIX = "parent."


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
DOCUMENT_RELATIONS = (
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
# their DTS file by document_id or URL, checked with the XBRL groups.
XBRL_RELATIONS = (
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


def _orphans(child: pa.Table, parent: pa.Table, relation: Relation) -> pa.Table:
    """Child rows with a non-null key that is missing from the parent table."""

    keys = child.column(relation.child_key)
    parent_keys = parent.column(relation.parent_key).combine_chunks()
    missing = pc.invert(pc.is_in(keys, value_set=parent_keys))
    return child.filter(pc.and_(pc.is_valid(keys), missing))


def _lineage_mismatches(
    child: pa.Table, parent: pa.Table, relation: Relation
) -> pa.Table:
    """Matched child rows whose lineage columns differ from their parent's."""

    child_columns = [relation.child_id, relation.child_key]
    child_columns += [column for column, _ in relation.lineage]
    parent_columns = [relation.parent_key, *(column for _, column in relation.lineage)]
    left = child.select(list(dict.fromkeys(child_columns)))
    right = parent.select(list(dict.fromkeys(parent_columns)))
    right = right.rename_columns([_PARENT_PREFIX + name for name in right.column_names])
    joined = left.join(
        right,
        keys=relation.child_key,
        right_keys=_PARENT_PREFIX + relation.parent_key,
        join_type="inner",
    )
    differs = [
        pc.invert(
            pc.fill_null(
                pc.equal(joined[column], joined[_PARENT_PREFIX + parent_column]), False
            )
        )
        for column, parent_column in relation.lineage
    ]
    return joined.filter(reduce(pc.or_, differs))


def check_references(
    data: Mapping[str, pa.Table],
    relations: Sequence[Relation],
    report: SilverAuditReport,
) -> None:
    """Check each foreign key of `relations` and the lineage it carries.

    A child key missing from its parent is an orphan; a matched child that
    does not repeat its parent's lineage is a lineage mismatch. Both are
    errors. Works on whole columns, without per-row loops over valid data;
    `data` needs each relation's key, lineage and child ID columns.
    """

    for relation in relations:
        report.ran(f"reference:{relation.name}")
        child, parent = data[relation.child], data[relation.parent]
        orphans = _orphans(child, parent, relation)
        report.orphan_counts[relation.name] = orphans.num_rows
        for row in orphans.select([relation.child_id, relation.child_key]).to_pylist():
            report.add(
                "error",
                "orphan",
                relation.child,
                f"{relation.name}: no parent row",
                row[relation.child_id],
                parent_table=relation.parent,
                **{relation.child_key: row[relation.child_key]},
            )
        mismatches = (
            _lineage_mismatches(child, parent, relation)
            if relation.lineage
            else child.slice(0, 0)
        )
        for row in mismatches.to_pylist():
            report.add(
                "error",
                "lineage",
                relation.child,
                f"{relation.name}: lineage differs from the parent row",
                row[relation.child_id],
                parent_table=relation.parent,
                **{
                    column: [str(row[column]), str(row[_PARENT_PREFIX + parent_column])]
                    for column, parent_column in relation.lineage
                    if row[column] != row[_PARENT_PREFIX + parent_column]
                },
            )
        logger.info(
            "silver_audit_relationship_checked",
            relation=relation.name,
            rows=child.num_rows,
            orphans=orphans.num_rows,
            lineage_mismatches=mismatches.num_rows,
        )


def _repeated(table: pa.Table, key: Sequence[str]) -> list[tuple[str, int]]:
    """Each value (joined with "|" for several columns) held by several rows."""

    counts = table.group_by(list(key)).aggregate([([], "count_all")])
    repeated = counts.filter(pc.greater(counts["count_all"], 1)).to_pylist()
    return [
        ("|".join(str(row[column]) for column in key), row["count_all"])
        for row in repeated
    ]


def check_keys(
    data: Mapping[str, pa.Table],
    unique_keys: Mapping[str, Sequence[Sequence[str]]],
    report: SilverAuditReport,
) -> None:
    """Each unique key of each table identifies one row.

    `unique_keys` maps table identifiers to their keys: the primary key and
    the column sets of the table's grain. Every repeated value is an error.
    """

    for identifier, keys in unique_keys.items():
        for key in keys:
            name = f"{identifier.split('.', 1)[1]}.{'+'.join(key)}"
            report.ran(f"unique:{name}")
            repeated = _repeated(data[identifier], key)
            report.duplicate_counts[name] = sum(count - 1 for _, count in repeated)
            for value, count in repeated:
                report.add(
                    "error",
                    "duplicate_key",
                    identifier,
                    f"{'+'.join(key)} appears {count} times",
                    value,
                )


def check_source_hashes(
    parents: ParentRecords, data_dir: Path, report: SilverAuditReport
) -> None:
    """Re-hash each filing's Bronze files and re-check its stored byte spans.

    One full submission is held in memory at a time. A filing fails at its
    first mismatch: submission or primary-document SHA-256, a document span or
    `<TEXT>` slice (size and SHA-256), or an entity block SHA-256.
    """

    report.ran("source_hashes")
    documents = defaultdict(list)
    for document in sorted(
        parents.documents.values(), key=lambda d: d.document_ordinal
    ):
        documents[document.filing_id].append(document)
    for filing in parents.filings:
        try:
            raw = resolve_relative_path(
                filing.full_submission_path, data_dir
            ).read_bytes()
            if not raw:
                raise ValueError("full submission is empty")
            if sha256_bytes(raw) != filing.full_submission_sha256:
                raise ValueError("full submission SHA-256 differs")
            if filing.primary_document_path is not None and (
                sha256_file(
                    resolve_relative_path(filing.primary_document_path, data_dir)
                )
                != filing.primary_document_sha256
            ):
                raise ValueError("primary document SHA-256 differs")
            check_document_offsets(documents[filing.filing_id], raw)
            for entity in parents.entities.get(filing.filing_id, []):
                block = raw[entity.entity_start_byte : entity.entity_end_byte]
                if sha256_bytes(block) != entity.entity_block_sha256:
                    raise ValueError(
                        f"entity {entity.entity_ordinal}: block SHA-256 differs"
                    )
        except (OSError, ValueError) as error:
            report.source_hash_failures += 1
            report.add(
                "error",
                "source_hash",
                FILINGS_TABLE,
                str(error),
                filing.filing_id,
                path=filing.full_submission_path,
            )
    logger.info(
        "silver_audit_source_hash_checked",
        filings=len(parents.filings),
        documents=len(parents.documents),
        failures=report.source_hash_failures,
    )

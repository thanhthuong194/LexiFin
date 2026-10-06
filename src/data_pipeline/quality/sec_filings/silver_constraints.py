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
from src.data_pipeline.storage.sec_filings.filing_entities import FILING_ENTITIES_TABLE
from src.data_pipeline.storage.sec_filings.filings import FILINGS_TABLE
from src.data_pipeline.storage.sec_filings.sections import SECTIONS_TABLE
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
# The ERD of the SEC Silver layer, parent first. A null target_document_id is
# an unresolved link, not an orphan; a resolved target stays in its filing.
RELATIONS = (
    Relation(
        DOCUMENTS_TABLE,
        "filing_id",
        FILINGS_TABLE,
        "filing_id",
        (
            ("filing_date", "filing_date"),
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
)


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


def check_references(data: Mapping[str, pa.Table], report: SilverAuditReport) -> None:
    """Check every foreign key of RELATIONS and the lineage it carries.

    A child key missing from its parent is an orphan; a matched child that
    does not repeat its parent's lineage is a lineage mismatch. Both are
    errors. Works on whole columns, without per-row loops over valid data.
    """

    for relation in RELATIONS:
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
                **{relation.child_key: row[relation.child_key]},
            )
        mismatches = _lineage_mismatches(child, parent, relation)
        for row in mismatches.to_pylist():
            report.add(
                "error",
                "lineage",
                relation.child,
                f"{relation.name}: lineage differs from the parent row",
                row[relation.child_id],
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


def check_keys(
    data: Mapping[str, pa.Table],
    unique_keys: Mapping[str, Sequence[str]],
    report: SilverAuditReport,
) -> None:
    """Each unique column of each table holds every value once.

    `unique_keys` maps table identifiers to their unique columns; every
    repeated value is an error.
    """

    for identifier, keys in unique_keys.items():
        for key in keys:
            name = f"{identifier.split('.', 1)[1]}.{key}"
            report.ran(f"unique:{name}")
            counts = pc.value_counts(data[identifier][key])
            repeated = counts.filter(pc.greater(counts.field("counts"), 1))
            report.duplicate_counts[name] = sum(
                count - 1 for count in repeated.field("counts").to_pylist()
            )
            for value, count in zip(
                repeated.field("values").to_pylist(),
                repeated.field("counts").to_pylist(),
                strict=True,
            ):
                report.add(
                    "error",
                    "duplicate_key",
                    identifier,
                    f"{key} appears {count} times",
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

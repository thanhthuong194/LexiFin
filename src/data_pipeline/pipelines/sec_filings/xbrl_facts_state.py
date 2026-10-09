from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.xbrl_concepts import XbrlConceptRecord
from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlInstanceRecord
from src.data_pipeline.contracts.sec_filings.xbrl_facts import XbrlFactRecord
from src.data_pipeline.quality.sec_filings.xbrl_core import instance_problem
from src.data_pipeline.quality.sec_filings.xbrl_facts import facts_problem
from src.data_pipeline.storage.iceberg import (
    GroupSummary,
    filing_dates_filter,
    read_rows_in,
    verify_append,
    verify_written_values,
)
from src.data_pipeline.storage.sec_filings.content_blocks import (
    open_content_blocks_table,
)
from src.data_pipeline.storage.sec_filings.document_table_cells import (
    open_document_table_cells_table,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)
# A document's stored cell IDs and block IDs (non-table blocks), by XPath.
ContentMaps = tuple[Mapping[str, str], Mapping[str, Sequence[str]]]

StoredRows = Mapping[str, Sequence[dict[str, Any]]]


@dataclass(frozen=True)
class InstanceRefs:
    """Stored context and unit refs of every instance, by instance_id."""

    contexts: Mapping[str, Collection[str]]
    units: Mapping[str, Collection[str]]

    def of(self, instance_id: str) -> tuple[Collection[str], Collection[str]]:
        """The (context refs, unit refs) of one instance."""

        return self.contexts.get(instance_id, ()), self.units.get(instance_id, ())


@dataclass
class FactsPlan:
    """How each instance relates to its stored facts, decided before loading XBRL.

    `candidates` have no stored fact (and expect some); complete groups,
    including empty ones of instances without facts, are `skipped`.
    `conflicts` hold (context, reason) pairs.
    """

    candidates: list[XbrlInstanceRecord] = field(default_factory=list)
    skipped: int = 0
    conflicts: list[tuple[dict[str, str], str]] = field(default_factory=list)


def classify_instances(
    instances: Sequence[XbrlInstanceRecord],
    documents: Mapping[str, DocumentRecord],
    stored_facts: Mapping[str, GroupSummary],
) -> FactsPlan:
    """Classify every instance against `xbrl_facts` before reading Bronze.

    `stored_facts` summarizes the fact rows of each instance and `documents`
    holds at least the instances' source documents. A stored group is
    complete when its fact orders are exactly 1..fact_count; facts of
    unknown instances, partial groups and instances whose source documents
    changed are conflicts. The integrity audit checks the fact rows.
    """

    plan = FactsPlan()
    known = {instance.instance_id for instance in instances}
    for instance_id in sorted(stored_facts.keys() - known):
        plan.conflicts.append(({"instance_id": instance_id}, "facts have no instance"))
    for instance in instances:
        context = {"filing_id": instance.filing_id, "instance_id": instance.instance_id}
        problem = instance_problem(instance, documents)
        group = stored_facts.get(instance.instance_id)
        if problem is None and group is not None:
            problem = (
                None
                if group.complete and group.rows == instance.fact_count
                else f"{group.rows} stored facts are not orders 1..{instance.fact_count}"
            )
        if problem is not None:
            plan.conflicts.append((context, problem))
        elif group is not None or instance.fact_count == 0:
            plan.skipped += 1
        else:
            plan.candidates.append(instance)
    return plan


def read_content(
    catalog: Catalog, data_dir: Path, documents: Sequence[DocumentRecord]
) -> Callable[[str], ContentMaps]:
    """Look up the stored cells and non-table blocks of `documents`, by XPath.

    Only those documents' rows are read, from their partitions.
    """

    ids = {document.document_id for document in documents}
    files = filing_dates_filter({document.filing_date for document in documents})
    cells: dict[str, dict[str, str]] = defaultdict(dict)
    for row in read_rows_in(
        open_document_table_cells_table(catalog, data_dir)[0],
        "document_id",
        ids,
        columns=("cell_id", "source_xpath"),
        files=files,
    ):
        cells[row["document_id"]][row["source_xpath"]] = row["cell_id"]
    blocks: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in read_rows_in(
        open_content_blocks_table(catalog, data_dir)[0],
        "document_id",
        ids,
        columns=("block_id", "block_type", "source_xpath"),
        files=files,
    ):
        if row["block_type"] != "table":
            blocks[row["document_id"]][row["source_xpath"]].append(row["block_id"])
    return lambda document_id: (cells.get(document_id, {}), blocks.get(document_id, {}))


def validate_built(
    groups: Mapping[str, Sequence[XbrlFactRecord]],
    instances: Mapping[str, XbrlInstanceRecord],
    concept_ids: Collection[str],
    refs: InstanceRefs,
) -> int:
    """Check every built fact group (by instance_id); return how many are invalid."""

    invalid = 0
    for instance_id, facts in groups.items():
        problem = facts_problem(
            instances[instance_id], facts, concept_ids, refs.of(instance_id)
        )
        if problem is not None:
            invalid += 1
            logger.error(
                "xbrl_instance_facts_conflict", instance_id=instance_id, reason=problem
            )
    return invalid


def verify_concepts(
    table: Table, written: Sequence[XbrlConceptRecord], stored: int, before: int | None
) -> int | None:
    """Re-read concepts: each new concept stored once, with its definition hash."""

    snapshot = verify_append(
        table,
        key_column="concept_id",
        expected_rows=stored + len(written),
        inserted_keys=[concept.concept_id for concept in written],
        snapshot_before=before,
    )
    verify_written_values(
        table, written, key_column="concept_id", columns=("concept_definition_sha256",)
    )
    return snapshot

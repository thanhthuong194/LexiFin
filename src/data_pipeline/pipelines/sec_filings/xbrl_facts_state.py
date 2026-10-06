from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from pydantic import ValidationError
from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.xbrl_concepts import XbrlConceptRecord
from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlInstanceRecord
from src.data_pipeline.contracts.sec_filings.xbrl_facts import XbrlFactRecord
from src.data_pipeline.quality.sec_filings.xbrl_core import instance_problem
from src.data_pipeline.quality.sec_filings.xbrl_facts import facts_problem
from src.data_pipeline.storage.iceberg import verify_append
from src.data_pipeline.storage.sec_filings.content_blocks import (
    open_content_blocks_table,
)
from src.data_pipeline.storage.sec_filings.document_table_cells import (
    open_document_table_cells_table,
)
from src.data_pipeline.storage.sec_filings.xbrl_facts import read_concept_hashes
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


def _stored_problem(
    instance: XbrlInstanceRecord,
    rows: Sequence[dict[str, Any]],
    concept_ids: Collection[str],
    refs: InstanceRefs,
) -> str | None:
    try:
        facts = [XbrlFactRecord.model_validate(row) for row in rows]
    except ValidationError as error:
        return f"stored fact is invalid: {error.errors()[0]['msg']}"
    return facts_problem(instance, facts, concept_ids, refs.of(instance.instance_id))


def classify_instances(
    instances: Sequence[XbrlInstanceRecord],
    documents: Mapping[str, DocumentRecord],
    stored_facts: StoredRows,
    concept_ids: Collection[str],
    refs: InstanceRefs,
) -> FactsPlan:
    """Classify every instance against `xbrl_facts` before reading Bronze.

    `stored_facts` groups fact rows by instance_id. Facts of unknown
    instances, partial or invalid groups, and instances whose source
    documents changed are conflicts.
    """

    plan = FactsPlan()
    known = {instance.instance_id for instance in instances}
    for instance_id in sorted(stored_facts.keys() - known):
        plan.conflicts.append(({"instance_id": instance_id}, "facts have no instance"))
    for instance in instances:
        context = {"filing_id": instance.filing_id, "instance_id": instance.instance_id}
        problem = instance_problem(instance, documents)
        rows = stored_facts.get(instance.instance_id, [])
        if problem is None and (rows or instance.fact_count == 0):
            problem = _stored_problem(instance, rows, concept_ids, refs)
            if problem is None:
                plan.skipped += 1
                continue
        if problem is not None:
            plan.conflicts.append((context, problem))
        else:
            plan.candidates.append(instance)
    return plan


class ContentIndex:
    """Stored narrative blocks and table cells of a document, by XPath.

    Both tables are read once, on the first lookup, so runs that skip every
    instance never read them.
    """

    def __init__(self, catalog: Catalog, data_dir: Path) -> None:
        self._catalog, self._data_dir = catalog, data_dir
        self._rows: tuple[pa.Table, pa.Table] | None = None

    def __call__(self, document_id: str) -> ContentMaps:
        """Cell IDs and block IDs (non-table blocks) of a document, by XPath."""

        if self._rows is None:
            blocks = open_content_blocks_table(self._catalog, self._data_dir)[0]
            cells = open_document_table_cells_table(self._catalog, self._data_dir)[0]
            self._rows = (
                blocks.scan(
                    row_filter="block_type != 'table'",
                    selected_fields=("document_id", "block_id", "source_xpath"),
                ).to_arrow(),
                cells.scan(
                    selected_fields=("document_id", "cell_id", "source_xpath")
                ).to_arrow(),
            )
        blocks_rows, cells_rows = (
            rows.filter(pc.equal(rows["document_id"], document_id)).to_pylist()
            for rows in self._rows
        )
        block_ids: dict[str, list[str]] = defaultdict(list)
        for row in blocks_rows:
            block_ids[row["source_xpath"]].append(row["block_id"])
        return {row["source_xpath"]: row["cell_id"] for row in cells_rows}, block_ids


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
    """Re-read concepts: one row per ID, and each new concept stored exactly."""

    snapshot = verify_append(
        table,
        key_column="concept_id",
        expected_rows=stored + len(written),
        inserted_keys=[concept.concept_id for concept in written],
        snapshot_before=before,
    )
    hashes = read_concept_hashes(table)
    if any(hashes[c.concept_id] != c.concept_definition_sha256 for c in written):
        raise ValueError("xbrl_concepts rows differ after commit")
    return snapshot

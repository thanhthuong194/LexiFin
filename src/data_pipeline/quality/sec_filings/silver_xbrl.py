from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterator, Mapping
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BaseModel

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.xbrl_concepts import XbrlConceptRecord
from src.data_pipeline.contracts.sec_filings.xbrl_core import (
    XbrlContextRecord,
    XbrlInstanceRecord,
    XbrlUnitRecord,
)
from src.data_pipeline.contracts.sec_filings.xbrl_dimensions import XbrlDimensionRecord
from src.data_pipeline.contracts.sec_filings.xbrl_fact_footnotes import (
    XbrlFactFootnoteRecord,
)
from src.data_pipeline.contracts.sec_filings.xbrl_facts import XbrlFactRecord
from src.data_pipeline.contracts.sec_filings.xbrl_labels import XbrlLabelRecord
from src.data_pipeline.contracts.sec_filings.xbrl_networks import (
    XbrlNetworkRecord,
    XbrlRelationshipRecord,
)
from src.data_pipeline.quality.sec_filings.narrative import declared_count_problem
from src.data_pipeline.quality.sec_filings.silver_groups import (
    sorted_groups,
    validated_records,
)
from src.data_pipeline.quality.sec_filings.silver_report import SilverAuditReport
from src.data_pipeline.quality.sec_filings.xbrl_core import (
    contexts_problem,
    instance_problem,
    units_problem,
)
from src.data_pipeline.quality.sec_filings.xbrl_enrichment import (
    dimension_group_problem,
    footnote_group_problem,
    footnote_resources_problem,
)
from src.data_pipeline.quality.sec_filings.xbrl_facts import facts_problem
from src.data_pipeline.quality.sec_filings.xbrl_linkbases import (
    network_problem,
    relationship_attributes_problem,
)
from src.data_pipeline.storage.sec_filings.documents import DOCUMENTS_TABLE
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

# Rows of the dictionary tables validated per slice, to bound memory.
_SLICE_ROWS = 50_000
# Context and unit refs of each instance.
InstanceRefs = dict[str, tuple[set[str], set[str]]]


def _slices(table: pa.Table) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, table.num_rows, _SLICE_ROWS):
        yield table.slice(start, _SLICE_ROWS).to_pylist()


def _validate_rows[Record: BaseModel](
    report: SilverAuditReport, table: pa.Table, identifier: str, model: type[Record]
) -> None:
    """Check every row of a table that has no groups against its contract."""

    key = table.column_names[0]
    for rows in _slices(table):
        validated_records(report, identifier, rows, model, key)


def _instances(
    data: Mapping[str, pa.Table],
    documents: Mapping[str, DocumentRecord],
    report: SilverAuditReport,
) -> dict[str, XbrlInstanceRecord]:
    """Validated instances by ID; each must fit its filing's documents."""

    rows = data[XBRL_INSTANCES_TABLE].to_pylist()
    for row in rows:  # Arrow returns a map column as (key, value) pairs
        row["namespace_map"] = dict(row["namespace_map"])
    instances = validated_records(
        report, XBRL_INSTANCES_TABLE, rows, XbrlInstanceRecord, "instance_id"
    )
    for instance in instances:
        problem = instance_problem(instance, documents)
        if problem is not None:
            report.group_violation(XBRL_INSTANCES_TABLE, instance.instance_id, problem)
    return {instance.instance_id: instance for instance in instances}


def _check_contexts(
    data: Mapping[str, pa.Table],
    instances: Mapping[str, XbrlInstanceRecord],
    report: SilverAuditReport,
) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Each instance's complete context group.

    Returns the context refs of each instance and, for each context of a
    complete group, how many dimension members its segment and scenario hold.
    """

    refs: dict[str, set[str]] = {}
    members: dict[str, int] = {}
    for instance_id, rows in sorted_groups(data[XBRL_CONTEXTS_TABLE], "instance_id"):
        contexts = validated_records(
            report, XBRL_CONTEXTS_TABLE, rows, XbrlContextRecord, "context_key"
        )
        refs[instance_id] = {context.context_ref for context in contexts}
        instance = instances.get(instance_id)
        problem = None if instance is None else contexts_problem(instance, contexts)
        if problem is not None:
            report.group_violation(XBRL_CONTEXTS_TABLE, instance_id, problem)
        elif instance is not None:
            members.update(
                (c.context_key, expected_dimension_count(c.segment_xml, c.scenario_xml))
                for c in contexts
            )
    for instance_id in sorted(instances.keys() - refs.keys()):
        problem = contexts_problem(instances[instance_id], [])
        if problem is not None:
            report.group_violation(XBRL_CONTEXTS_TABLE, instance_id, problem)
    return refs, members


def _check_units(
    data: Mapping[str, pa.Table],
    instances: Mapping[str, XbrlInstanceRecord],
    report: SilverAuditReport,
) -> dict[str, set[str]]:
    """Each instance's complete unit group; return the unit refs of each instance."""

    refs: dict[str, set[str]] = {}
    for instance_id, rows in sorted_groups(data[XBRL_UNITS_TABLE], "instance_id"):
        units = validated_records(
            report, XBRL_UNITS_TABLE, rows, XbrlUnitRecord, "unit_key"
        )
        refs[instance_id] = {unit.unit_ref for unit in units}
        instance = instances.get(instance_id)
        problem = None if instance is None else units_problem(instance, units)
        if problem is not None:
            report.group_violation(XBRL_UNITS_TABLE, instance_id, problem)
    for instance_id in sorted(instances.keys() - refs.keys()):
        problem = units_problem(instances[instance_id], [])
        if problem is not None:
            report.group_violation(XBRL_UNITS_TABLE, instance_id, problem)
    return refs


def _check_facts(
    data: Mapping[str, pa.Table],
    instances: Mapping[str, XbrlInstanceRecord],
    concept_ids: Collection[str],
    refs: InstanceRefs,
    report: SilverAuditReport,
) -> None:
    """Each instance's facts: `fact_count` facts with known concepts, own contexts and units."""

    seen = set()
    for instance_id, rows in sorted_groups(data[XBRL_FACTS_TABLE], "instance_id"):
        seen.add(instance_id)
        facts = validated_records(
            report, XBRL_FACTS_TABLE, rows, XbrlFactRecord, "fact_id"
        )
        instance = instances.get(instance_id)
        if instance is None:
            continue  # orphan facts, reported by the reference check
        own = refs.get(instance_id, (set(), set()))
        problem = facts_problem(instance, facts, concept_ids, own)
        if problem is not None:
            report.group_violation(XBRL_FACTS_TABLE, instance_id, problem)
    for instance_id in sorted(instances.keys() - seen):
        problem = facts_problem(instances[instance_id], [], concept_ids, (set(), set()))
        if problem is not None:
            report.group_violation(XBRL_FACTS_TABLE, instance_id, problem)


def _check_dimensions(
    data: Mapping[str, pa.Table],
    members: Mapping[str, int],
    concept_ids: Collection[str],
    report: SilverAuditReport,
) -> None:
    """Each checked context has exactly the dimension rows its XML reports."""

    seen = set()
    for context_key, rows in sorted_groups(data[XBRL_DIMENSIONS_TABLE], "context_key"):
        seen.add(context_key)
        dimensions = validated_records(
            report, XBRL_DIMENSIONS_TABLE, rows, XbrlDimensionRecord, "dimension_id"
        )
        if context_key not in members:
            continue  # orphan, or a context whose own group is broken
        problem = dimension_group_problem(dimensions, members[context_key], concept_ids)
        if problem is not None:
            report.group_violation(XBRL_DIMENSIONS_TABLE, context_key, problem)
    for context_key in sorted(k for k, n in members.items() if n and k not in seen):
        problem = dimension_group_problem([], members[context_key], concept_ids)
        report.group_violation(XBRL_DIMENSIONS_TABLE, context_key, str(problem))


def _check_footnotes(data: Mapping[str, pa.Table], report: SilverAuditReport) -> None:
    """Each fact's footnote rows are complete; an instance's shared resources agree."""

    footnotes = data[XBRL_FACT_FOOTNOTES_TABLE]
    facts = data[XBRL_FACTS_TABLE].select(["fact_id", "instance_id"])
    noted = facts.filter(
        pc.is_in(facts["fact_id"], value_set=footnotes["fact_id"].combine_chunks())
    )
    instance_of = dict(
        zip(
            noted["fact_id"].to_pylist(),
            noted["instance_id"].to_pylist(),
            strict=True,
        )
    )
    by_instance: dict[str, list[XbrlFactFootnoteRecord]] = defaultdict(list)
    for fact_id, rows in sorted_groups(footnotes, "fact_id"):
        records = validated_records(
            report,
            XBRL_FACT_FOOTNOTES_TABLE,
            rows,
            XbrlFactFootnoteRecord,
            "footnote_relation_id",
        )
        problem = footnote_group_problem(records)
        if problem is not None:
            report.group_violation(XBRL_FACT_FOOTNOTES_TABLE, fact_id, problem)
        if fact_id in instance_of:
            by_instance[instance_of[fact_id]].extend(records)
    for instance_id, records in sorted(by_instance.items()):
        problem = footnote_resources_problem(records)
        if problem is not None:
            report.group_violation(XBRL_FACT_FOOTNOTES_TABLE, instance_id, problem)


def _check_networks(data: Mapping[str, pa.Table], report: SilverAuditReport) -> None:
    """Networks numbered 1..N per instance; each with its complete relationships."""

    networks: dict[str, XbrlNetworkRecord] = {}
    for instance_id, rows in sorted_groups(data[XBRL_NETWORKS_TABLE], "instance_id"):
        records = validated_records(
            report, XBRL_NETWORKS_TABLE, rows, XbrlNetworkRecord, "network_id"
        )
        networks.update((network.network_id, network) for network in records)
        problem = declared_count_problem(
            [network.network_ordinal for network in records],
            {network.instance_network_count for network in records},
            "networks",
        )
        if problem is not None:
            report.group_violation(XBRL_NETWORKS_TABLE, instance_id, problem)
    seen = set()
    relationships = data[XBRL_RELATIONSHIPS_TABLE]
    for network_id, rows in sorted_groups(relationships, "network_id"):
        seen.add(network_id)
        edges = validated_records(
            report,
            XBRL_RELATIONSHIPS_TABLE,
            rows,
            XbrlRelationshipRecord,
            "relationship_id",
        )
        network = networks.get(network_id)
        if network is None:
            continue  # orphan, reported by the reference check
        problem = network_problem(network, edges) or next(
            (
                found
                for edge in edges
                if (found := relationship_attributes_problem(network, edge))
            ),
            None,
        )
        if problem is not None:
            report.group_violation(XBRL_RELATIONSHIPS_TABLE, network_id, problem)
    for network_id in sorted(networks.keys() - seen):
        report.group_violation(
            XBRL_RELATIONSHIPS_TABLE, network_id, "network has no relationships"
        )


def _check_dts_sources(data: Mapping[str, pa.Table], report: SilverAuditReport) -> None:
    """A label or relationship names its DTS file by URL or by document_id.

    A document_id must be a stored document; for a relationship, one of its
    own instance's filing.
    """

    documents = data[DOCUMENTS_TABLE].select(["document_id", "filing_id"])
    for identifier, key in (
        (XBRL_LABELS_TABLE, "label_id"),
        (XBRL_RELATIONSHIPS_TABLE, "relationship_id"),
    ):
        report.ran(f"dts_source:{identifier.split('.', 1)[1]}")
        rows = data[identifier].select([key, "source_document_id"])
        local = rows.filter(
            pc.invert(pc.match_substring(rows["source_document_id"], "://"))
        )
        missing = local.filter(
            pc.invert(
                pc.is_in(
                    local["source_document_id"],
                    value_set=documents["document_id"].combine_chunks(),
                )
            )
        )
        for row in missing.to_pylist():
            report.add(
                "error",
                "orphan",
                identifier,
                "source_document_id: no stored document",
                row[key],
                parent_table=DOCUMENTS_TABLE,
                source_document_id=row["source_document_id"],
            )
    _check_relationship_filings(data, documents, report)


def _check_relationship_filings(
    data: Mapping[str, pa.Table], documents: pa.Table, report: SilverAuditReport
) -> None:
    owners = (
        data[XBRL_RELATIONSHIPS_TABLE]
        .select(["relationship_id", "network_id", "source_document_id"])
        .join(
            data[XBRL_NETWORKS_TABLE].select(["network_id", "instance_id"]),
            "network_id",
        )
        .join(
            data[XBRL_INSTANCES_TABLE].select(["instance_id", "filing_id"]),
            "instance_id",
        )
        .join(
            documents.rename_columns(["source_document_id", "document_filing_id"]),
            "source_document_id",
            join_type="inner",
        )
    )
    foreign = owners.filter(
        pc.not_equal(owners["filing_id"], owners["document_filing_id"])
    )
    for row in foreign.to_pylist():
        report.add(
            "error",
            "lineage",
            XBRL_RELATIONSHIPS_TABLE,
            "source document belongs to another filing than the instance",
            row["relationship_id"],
            parent_table=DOCUMENTS_TABLE,
            source_document_id=row["source_document_id"],
        )


def check_xbrl_groups(
    data: Mapping[str, pa.Table],
    documents: Mapping[str, DocumentRecord],
    report: SilverAuditReport,
) -> None:
    """Validate every XBRL row against its contract and every group's completeness.

    Reuses the stage checks: instance sources, context, unit and fact groups
    per instance, dimensions per context (as many as its stored XML reports),
    footnotes per fact, networks per instance and relationships per network
    (a stored network needs at least one). `data` holds the XBRL tables and
    the documents' `document_id` and `filing_id`; `documents` the validated
    documents by ID.
    """

    report.ran("xbrl.groups")
    instances = _instances(data, documents, report)
    concepts = data[XBRL_CONCEPTS_TABLE]
    _validate_rows(report, concepts, XBRL_CONCEPTS_TABLE, XbrlConceptRecord)
    _validate_rows(report, data[XBRL_LABELS_TABLE], XBRL_LABELS_TABLE, XbrlLabelRecord)
    concept_ids = set(concepts["concept_id"].to_pylist())
    context_refs, members = _check_contexts(data, instances, report)
    unit_refs = _check_units(data, instances, report)
    refs = {
        instance_id: (
            context_refs.get(instance_id, set()),
            unit_refs.get(instance_id, set()),
        )
        for instance_id in instances
    }
    _check_facts(data, instances, concept_ids, refs, report)
    _check_dimensions(data, members, concept_ids, report)
    _check_footnotes(data, report)
    _check_networks(data, report)
    _check_dts_sources(data, report)

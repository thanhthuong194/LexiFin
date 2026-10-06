from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from pydantic import ValidationError
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlInstanceRecord
from src.data_pipeline.contracts.sec_filings.xbrl_labels import (
    XBRL_LABELS_ARROW_SCHEMA,
    XbrlLabelRecord,
)
from src.data_pipeline.contracts.sec_filings.xbrl_networks import XbrlNetworkRecord
from src.data_pipeline.quality.sec_filings.narrative import declared_count_problem
from src.data_pipeline.quality.sec_filings.xbrl_core import instance_problem
from src.data_pipeline.quality.sec_filings.xbrl_linkbases import network_problem
from src.data_pipeline.storage.iceberg import read_rows_by_key

StoredRows = Mapping[str, Sequence[dict[str, Any]]]
# Label columns compared with a stored label of the same ID.
LABEL_PAYLOAD = tuple(
    name
    for name in XBRL_LABELS_ARROW_SCHEMA.names
    if name not in {"label_id", "processed_at"}
)


class LinkbaseConflict(ValueError):
    """Stored rows, or the source, disagree with the parent or sibling rows."""


class RelationshipRow(NamedTuple):
    """The identity and place of a stored relationship in its network."""

    relationship_id: str
    relationship_ordinal: int
    network_relationship_count: int
    relationship_hash: str


@dataclass(frozen=True)
class InstanceState:
    """An instance to load: with no stored network, or with `networks` stored
    without any relationship, as a failed relationships commit leaves them."""

    instance: XbrlInstanceRecord
    networks: list[XbrlNetworkRecord]


@dataclass
class LinkbasesPlan:
    """Instances to load, complete ones skipped, and (context, reason) conflicts."""

    candidates: list[InstanceState] = field(default_factory=list)
    skipped: int = 0
    conflicts: list[tuple[dict[str, str], str]] = field(default_factory=list)


def _stored_networks(
    rows: Sequence[dict[str, Any]], heads: StoredRows
) -> list[XbrlNetworkRecord] | None:
    """An instance's stored networks if none has relationships yet; None if complete."""

    try:
        networks = [XbrlNetworkRecord.model_validate(row) for row in rows]
    except ValidationError as error:
        raise LinkbaseConflict(
            f"stored network is invalid: {error.errors()[0]['msg']}"
        ) from error
    problem = declared_count_problem(
        [n.network_ordinal for n in networks],
        {n.instance_network_count for n in networks},
        "networks",
    )
    if problem is not None:
        raise LinkbaseConflict(problem)
    filled = [n for n in networks if n.network_id in heads]
    if not filled:
        return networks
    if len(filled) != len(networks):
        raise LinkbaseConflict(
            f"relationships are stored for {len(filled)} of {len(networks)} networks"
        )
    for network in networks:
        relationships = [RelationshipRow(**row) for row in heads[network.network_id]]
        problem = network_problem(network, relationships)
        if problem is not None:
            raise LinkbaseConflict(f"network {network.network_id}: {problem}")
    return None


def classify_instances(
    instances: Sequence[XbrlInstanceRecord],
    documents: Mapping[str, DocumentRecord],
    stored_networks: StoredRows,
    relationship_heads: StoredRows,
) -> LinkbasesPlan:
    """Classify every instance against the stored networks before loading XBRL.

    `stored_networks` groups network rows by instance_id and
    `relationship_heads` relationship rows by network_id. Complete instances
    are skipped; networks or relationships without a parent, partial or
    changed groups, and instances whose sources changed are conflicts.
    """

    plan = LinkbasesPlan()
    known = {instance.instance_id for instance in instances}
    for instance_id in sorted(stored_networks.keys() - known):
        plan.conflicts.append(
            ({"instance_id": instance_id}, "networks have no instance")
        )
    network_ids = {
        row["network_id"] for rows in stored_networks.values() for row in rows
    }
    for network_id in sorted(relationship_heads.keys() - network_ids):
        plan.conflicts.append(
            ({"network_id": network_id}, "relationships have no network")
        )
    for instance in instances:
        context = {"filing_id": instance.filing_id, "instance_id": instance.instance_id}
        problem = instance_problem(instance, documents)
        if problem is not None:
            plan.conflicts.append((context, problem))
            continue
        rows = stored_networks.get(instance.instance_id, [])
        try:
            networks = _stored_networks(rows, relationship_heads) if rows else []
        except LinkbaseConflict as error:
            plan.conflicts.append((context, str(error)))
            continue
        if networks is None:
            plan.skipped += 1
        else:
            plan.candidates.append(InstanceState(instance, networks))
    return plan


def new_labels(
    table: Table, labels: Mapping[str, XbrlLabelRecord], stored_ids: set[str]
) -> tuple[list[XbrlLabelRecord], list[str]]:
    """Labels not stored yet, and IDs of stored labels that differ from the source."""

    stored = read_rows_by_key(
        table, labels.keys() & stored_ids, key_column="label_id", columns=LABEL_PAYLOAD
    )
    differ = sorted(
        label_id
        for label_id, row in stored.items()
        if row != labels[label_id].model_dump(include=set(LABEL_PAYLOAD))
    )
    new = [
        label
        for label_id, label in sorted(labels.items())
        if label_id not in stored_ids
    ]
    return new, differ

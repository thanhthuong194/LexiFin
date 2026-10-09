from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlInstanceRecord
from src.data_pipeline.contracts.sec_filings.xbrl_labels import (
    XBRL_LABELS_ARROW_SCHEMA,
    XbrlLabelRecord,
)
from src.data_pipeline.contracts.sec_filings.xbrl_networks import XbrlNetworkRecord
from src.data_pipeline.quality.sec_filings.xbrl_core import instance_problem
from src.data_pipeline.storage.iceberg import GroupSummary, read_rows_in

StoredRows = Mapping[str, Sequence[dict[str, Any]]]
# Label columns compared with a stored label of the same ID.
LABEL_PAYLOAD = tuple(
    name
    for name in XBRL_LABELS_ARROW_SCHEMA.names
    if name not in {"label_id", "processed_at"}
)
# The payload read back from stored labels: the text is pinned by its hash.
LABEL_COMPARED = tuple(name for name in LABEL_PAYLOAD if name != "label_text")


class LinkbaseConflict(ValueError):
    """Stored rows, or the source, disagree with the parent or sibling rows."""


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
    summary: GroupSummary | None,
    networks: Sequence[tuple[str, int]],
    relationships: Mapping[str, GroupSummary],
) -> bool | None:
    """Whether an instance's networks are complete (True), stored without any
    relationship yet (False, as a failed relationships commit leaves them), or
    absent (None). Other states raise LinkbaseConflict.

    `networks` holds (network_id, relationship_count) of the stored networks.
    """

    if summary is None:
        return None
    if not summary.complete:
        raise LinkbaseConflict("network ordinals are not 1..N of their count")
    filled = [(n, count) for n, count in networks if n in relationships]
    if not filled:
        return False
    if len(filled) != len(networks):
        raise LinkbaseConflict(
            f"relationships are stored for {len(filled)} of {len(networks)} networks"
        )
    for network_id, count in networks:
        group = relationships[network_id]
        if not group.complete or group.rows != count:
            raise LinkbaseConflict(
                f"network {network_id}: {group.rows} relationships, declares {count}"
            )
    return True


@dataclass(frozen=True)
class StoredLinkbases:
    """Summaries of the stored networks and relationships, without their rows.

    `networks` summarizes network rows by instance_id and `heads` lists each
    instance's (network_id, relationship_count); `relationships` summarizes
    relationship rows by network_id.
    """

    networks: Mapping[str, GroupSummary]
    heads: Mapping[str, Sequence[tuple[str, int]]]
    relationships: Mapping[str, GroupSummary]


def classify_instances(
    instances: Sequence[XbrlInstanceRecord],
    documents: Mapping[str, DocumentRecord],
    stored: StoredLinkbases,
) -> LinkbasesPlan:
    """Classify every instance against the stored networks before loading XBRL.

    Stored groups are judged by their summaries; `documents` holds at least
    the instances' source documents. Complete instances are skipped;
    networks or relationships without a parent, partial groups, and
    instances whose sources changed are conflicts. Instances to load get
    their stored networks later, in their batch. The integrity audit checks
    the stored rows, and network hashes on its sampled filings.
    """

    plan = LinkbasesPlan()
    known = {instance.instance_id for instance in instances}
    for instance_id in sorted(stored.networks.keys() - known):
        plan.conflicts.append(
            ({"instance_id": instance_id}, "networks have no instance")
        )
    network_ids = {n for heads in stored.heads.values() for n, _ in heads}
    for network_id in sorted(stored.relationships.keys() - network_ids):
        plan.conflicts.append(
            ({"network_id": network_id}, "relationships have no network")
        )
    for instance in instances:
        context = {"filing_id": instance.filing_id, "instance_id": instance.instance_id}
        problem = instance_problem(instance, documents)
        if problem is not None:
            plan.conflicts.append((context, problem))
            continue
        try:
            complete = _stored_networks(
                stored.networks.get(instance.instance_id),
                stored.heads.get(instance.instance_id, ()),
                stored.relationships,
            )
        except LinkbaseConflict as error:
            plan.conflicts.append((context, str(error)))
            continue
        if complete:
            plan.skipped += 1
        else:
            plan.candidates.append(InstanceState(instance, []))
    return plan


def new_labels(
    table: Table, labels: Mapping[str, XbrlLabelRecord], stored_ids: set[str]
) -> tuple[list[XbrlLabelRecord], list[str]]:
    """Labels not stored yet, and IDs of stored labels that differ from the source.

    Only stored labels of the same IDs are read, without their text: the ID
    hashes `label_text_sha256`, which the record contract ties to the text.
    """

    stored = {
        row.pop("label_id"): row
        for row in read_rows_in(
            table, "label_id", labels.keys() & stored_ids, columns=LABEL_COMPARED
        )
    }
    differ = sorted(
        label_id
        for label_id, row in stored.items()
        if row != labels[label_id].model_dump(include=set(LABEL_COMPARED))
    )
    new = [
        label
        for label_id, label in sorted(labels.items())
        if label_id not in stored_ids
    ]
    return new, differ

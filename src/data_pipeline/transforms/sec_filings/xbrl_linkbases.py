from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

from src.data_pipeline.contracts.sec_filings.xbrl_concepts import concept_id_for
from src.data_pipeline.contracts.sec_filings.xbrl_labels import (
    XbrlLabelRecord,
    label_id_for,
    label_resource_id_for,
    normalize_label,
)
from src.data_pipeline.contracts.sec_filings.xbrl_networks import (
    LINKBASE_ORDER,
    XbrlNetworkRecord,
    XbrlRelationshipRecord,
    network_hash,
    network_id_for,
    plain_decimal,
    relationship_hash,
    relationship_id_for,
)
from src.data_pipeline.transforms.sec_filings.xbrl_model_linkbases import (
    LoadedArc,
    LoadedLabel,
)


def label_records(
    labels: Sequence[LoadedLabel], processed_at: datetime
) -> list[XbrlLabelRecord]:
    """The `xbrl_labels` rows of a DTS's concept-label relationships."""

    records = []
    for label in labels:
        text = normalize_label(label.text)
        values: dict[str, Any] = {
            "concept_id": concept_id_for(*label.concept),
            "label_resource_id": label_resource_id_for(
                label.document, label.source_xpath
            ),
            "role_uri": label.role_uri,
            "language": (label.language or "").strip().lower() or None,
            "label_text": text,
            "resource_label": label.label or None,
            "linkrole_uri": label.linkrole_uri,
            "source_document_id": label.document,
            "source_xpath": label.source_xpath,
            "arc_xpath": label.arc_xpath,
            "label_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "processed_at": processed_at,
        }
        values["label_id"] = label_id_for(values)
        records.append(XbrlLabelRecord.model_validate(values))
    return records


def _relationship_values(
    network_id: str, arc: LoadedArc, processed_at: datetime
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "network_id": network_id,
        "source_concept_id": concept_id_for(*arc.source),
        "target_concept_id": concept_id_for(*arc.target),
        "arcrole_uri": arc.arcrole_uri,
        "order_value": plain_decimal(arc.order),
        "weight": None if arc.weight is None else plain_decimal(arc.weight),
        "preferred_label_role": arc.preferred_label_role,
        "target_role_uri": arc.target_role_uri,
        "context_element": arc.context_element,
        "is_closed": arc.is_closed,
        "is_usable": arc.is_usable,
        "source_document_id": arc.document,
        "source_xpath": arc.source_xpath,
        "processed_at": processed_at,
    }
    values["relationship_hash"] = relationship_hash(values)
    values["relationship_id"] = relationship_id_for(
        network_id, values["relationship_hash"]
    )
    return values


def _relationship_order(values: dict[str, Any]) -> tuple[Any, ...]:
    """Source concept, numeric order, target concept, arcrole, arc location."""

    return (
        values["source_concept_id"],
        Decimal(values["order_value"]),
        values["target_concept_id"],
        values["arcrole_uri"],
        values["source_document_id"],
        values["source_xpath"],
    )


def network_records(
    instance_id: str,
    arcs: Sequence[LoadedArc],
    role_definitions: Mapping[str, str | None],
    processed_at: datetime,
) -> tuple[list[XbrlNetworkRecord], list[XbrlRelationshipRecord]]:
    """An instance's networks (one per linkbase type and role) and their relationships.

    Only roles with at least one effective relationship make a network.
    Networks are numbered by linkbase type (presentation, calculation,
    definition), then role URI; relationships in their canonical order.
    """

    groups: dict[tuple[str, str], list[LoadedArc]] = defaultdict(list)
    for arc in arcs:
        groups[arc.linkbase_type, arc.role_uri].append(arc)
    keys = sorted(groups, key=lambda key: (LINKBASE_ORDER[key[0]], key[1]))
    networks: list[XbrlNetworkRecord] = []
    relationships: list[XbrlRelationshipRecord] = []
    for ordinal, (linkbase_type, role_uri) in enumerate(keys, start=1):
        network_id = network_id_for(instance_id, linkbase_type, role_uri)
        rows = sorted(
            (
                _relationship_values(network_id, arc, processed_at)
                for arc in groups[linkbase_type, role_uri]
            ),
            key=_relationship_order,
        )
        relationships.extend(
            XbrlRelationshipRecord.model_validate(
                {
                    **row,
                    "relationship_ordinal": n,
                    "network_relationship_count": len(rows),
                }
            )
            for n, row in enumerate(rows, start=1)
        )
        values: dict[str, Any] = {
            "network_id": network_id,
            "instance_id": instance_id,
            "network_ordinal": ordinal,
            "instance_network_count": len(keys),
            "linkbase_type": linkbase_type,
            "role_uri": role_uri,
            "role_definition": role_definitions.get(role_uri),
            "relationship_count": len(rows),
            "processed_at": processed_at,
        }
        values["network_hash"] = network_hash(
            values, [row["relationship_hash"] for row in rows]
        )
        networks.append(XbrlNetworkRecord.model_validate(values))
    return networks, relationships

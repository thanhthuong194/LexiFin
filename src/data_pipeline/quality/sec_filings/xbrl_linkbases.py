from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.data_pipeline.contracts.sec_filings.xbrl_networks import (
    XbrlNetworkRecord,
    XbrlRelationshipRecord,
    network_hash,
)
from src.data_pipeline.quality.sec_filings.narrative import declared_count_problem

# Relationship attributes, and the linkbase types they apply to.
RELATIONSHIP_ATTRIBUTE_LINKBASES = {
    "weight": "calculation",
    "preferred_label_role": "presentation",
    "target_role_uri": "definition",
    "context_element": "definition",
    "is_closed": "definition",
    "is_usable": "definition",
}


class RelationshipHead(Protocol):
    """The identity and place of a relationship row in its network."""

    @property
    def relationship_id(self) -> str: ...

    @property
    def relationship_ordinal(self) -> int: ...

    @property
    def network_relationship_count(self) -> int: ...

    @property
    def relationship_hash(self) -> str: ...


def network_problem(
    network: XbrlNetworkRecord, relationships: Sequence[RelationshipHead]
) -> str | None:
    """Return why a network's relationship rows are not its complete group, or None.

    The group needs `relationship_count` rows with consistent count and
    ordinals 1..N, distinct IDs, and hashes that, in ordinal order, give
    the network's `network_hash`.
    """

    problem = declared_count_problem(
        [r.relationship_ordinal for r in relationships],
        {r.network_relationship_count for r in relationships},
        "relationships",
    )
    if problem is not None:
        return problem
    if len(relationships) != network.relationship_count:
        return f"{len(relationships)} relationships; the network declares {network.relationship_count}"
    if len({r.relationship_id for r in relationships}) != len(relationships):
        return "a relationship is repeated"
    ordered = sorted(relationships, key=lambda r: r.relationship_ordinal)
    digests = [r.relationship_hash for r in ordered]
    if network_hash(network.model_dump(), digests) != network.network_hash:
        return "network_hash does not match its relationships"
    return None


def relationship_attributes_problem(
    network: XbrlNetworkRecord, relationship: XbrlRelationshipRecord
) -> str | None:
    """Return the attributes a relationship holds that its linkbase type lacks."""

    wrong = sorted(
        name
        for name, linkbase_type in RELATIONSHIP_ATTRIBUTE_LINKBASES.items()
        if getattr(relationship, name) is not None
        and linkbase_type != network.linkbase_type
    )
    if wrong:
        return f"{network.linkbase_type} relationship has {wrong}"
    return None

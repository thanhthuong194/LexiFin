from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from src.data_pipeline.contracts.sec_filings.xbrl_fact_footnotes import (
    XbrlFactFootnoteRecord,
    footnote_content_sha256,
    footnote_relation_id_for,
    footnote_resource_id_for,
)
from src.data_pipeline.contracts.sec_filings.xbrl_facts import fact_id_for
from src.data_pipeline.transforms.sec_filings.html_dom import normalize_text
from src.data_pipeline.transforms.sec_filings.xbrl_model_enrichment import (
    LoadedFootnote,
)


def _order(values: dict[str, Any]) -> tuple[str, ...]:
    return (
        values["footnote_resource_id"],
        values["arcrole_uri"],
        values["linkrole_uri"],
        values["role_uri"],
        values["language"] or "",
        values["footnote_content_sha256"],
        values["arc_xpath"] or "",
    )


def footnote_records(
    instance_id: str, footnotes: Sequence[LoadedFootnote], processed_at: datetime
) -> list[XbrlFactFootnoteRecord]:
    """The `xbrl_fact_footnotes` rows of an instance, by fact, in canonical order.

    A footnote noting several facts yields one row per fact, all with the same
    resource ID; text is normalized and the language lowercased.
    """

    by_fact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for footnote in footnotes:
        values: dict[str, Any] = {
            "fact_id": fact_id_for(instance_id, footnote.fact_index + 1),
            "footnote_resource_id": footnote_resource_id_for(
                instance_id, footnote.document_id, footnote.source_xpath
            ),
            "footnote_text": normalize_text(footnote.text),
            "footnote_html": footnote.html,
            "role_uri": footnote.role_uri,
            "arcrole_uri": footnote.arcrole_uri,
            "linkrole_uri": footnote.linkrole_uri,
            "language": (footnote.language or "").strip().lower() or None,
            "resource_label": footnote.label or None,
            "source_xpath": footnote.source_xpath,
            "arc_xpath": footnote.arc_xpath,
            "processed_at": processed_at,
        }
        values["footnote_content_sha256"] = footnote_content_sha256(values)
        by_fact[values["fact_id"]].append(values)
    records: list[XbrlFactFootnoteRecord] = []
    # Fact IDs end in a zero-padded fact order, so they sort in fact order.
    for fact_id in sorted(by_fact):
        group = sorted(by_fact[fact_id], key=_order)
        records.extend(
            XbrlFactFootnoteRecord.model_validate(
                {
                    **values,
                    "footnote_relation_id": footnote_relation_id_for(fact_id, ordinal),
                    "footnote_ordinal": ordinal,
                    "fact_footnote_count": len(group),
                }
            )
            for ordinal, values in enumerate(group, start=1)
        )
    return records

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pyiceberg.expressions import EqualTo
from pyiceberg.table import Table

from src.data_pipeline.contracts.sec_filings.xbrl_concepts import concept_id_for
from src.data_pipeline.contracts.sec_filings.xbrl_labels import XBRL_LABELS_ARROW_SCHEMA
from src.data_pipeline.contracts.sec_filings.xbrl_networks import (
    XBRL_NETWORKS_ARROW_SCHEMA,
    XBRL_RELATIONSHIPS_ARROW_SCHEMA,
    XbrlNetworkRecord,
    network_id_for,
)
from src.data_pipeline.pipelines.sec_filings import xbrl_linkbases as linkbases_stage
from src.data_pipeline.pipelines.sec_filings import (
    xbrl_linkbases_extraction as linkbases_extraction,
)
from src.data_pipeline.pipelines.sec_filings.bronze_to_silver import (
    run_sec_filings_silver,
)
from src.data_pipeline.storage.iceberg import append_rows, open_silver_catalog
from src.data_pipeline.transforms.sec_filings.xbrl_model_reader import (
    XbrlProcessorError,
)
from src.data_pipeline.transforms.sec_filings.xbrl_processor import (
    ProcessorSettings,
    load_instance,
)
from tests.conftest import STATEMENT_ROLE

pytestmark = pytest.mark.integration

LABELS = "sec_filings.xbrl_labels"
NETWORKS = "sec_filings.xbrl_networks"
RELATIONSHIPS = "sec_filings.xbrl_relationships"
EX = "http://example.com/2024"
FILINGS = [
    ("0000320193", "10-K", "0000320193-24-000123", "20241101"),
    ("0000320193", "10-Q", "0000320193-25-000057", "20250502"),
    ("0000789019", "8-K", "0000789019-25-000010", "20250129"),
]
TEN_K_INSTANCE = "sec:0000320193-24-000123:xbrl:inline:default"
NEW_FILING = ("0000320193", "10-K", "0000320193-26-000002", "20260105")
# Every filing's extension schema embeds three labels of Revenue and one
# statement role with a presentation (2 arcs), a calculation (1) and a
# definition (3) network; each filing's schema is its own document, so its
# labels are its own.
PER_INSTANCE = {"labels": 3, "networks": 3, "relationships": 6}
DIMENSION = "http://xbrl.org/int/dim/arcrole/"


def _write(data_dir: Path, write_sec_filing, filings=FILINGS) -> None:
    for cik, form, accession, filed in filings:
        write_sec_filing(data_dir, cik=cik, form=form, accession=accession, filed=filed)


def _table(data_dir: Path, identifier: str) -> Table:
    with open_silver_catalog(data_dir) as catalog:
        return catalog.load_table(identifier)


def _rows(data_dir: Path, identifier: str) -> list[dict[str, Any]]:
    return _table(data_dir, identifier).scan().to_arrow().to_pylist()


def _snapshots(data_dir: Path) -> tuple[int | None, ...]:
    return tuple(
        None
        if (snapshot := _table(data_dir, t).current_snapshot()) is None
        else snapshot.snapshot_id
        for t in (LABELS, NETWORKS, RELATIONSHIPS)
    )


def _counts(data_dir: Path) -> tuple[int, ...]:
    return tuple(len(_rows(data_dir, t)) for t in (LABELS, NETWORKS, RELATIONSHIPS))


def _settings(data_dir: Path) -> ProcessorSettings:
    return ProcessorSettings(
        cache_dir=data_dir / "_cache" / "arelle",
        allow_network=False,
        http_user_agent=None,
    )


def _delete(data_dir: Path, identifier: str, column: str, value: str) -> None:
    with open_silver_catalog(data_dir) as catalog:
        catalog.load_table(identifier).delete(EqualTo(column, value))


def _fail_on(schema):
    def append(table: Table, rows, arrow_schema) -> None:
        if arrow_schema is schema:
            raise OSError("disk full")
        append_rows(table, rows, arrow_schema)

    return append


def test_linkbases_build_three_tables_with_complete_parents_and_rerun_skips(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, write_sec_filing)

    first = run_sec_filings_silver(tmp_path)

    summary = first.xbrl_linkbases
    assert first.completed
    assert summary is not None
    assert {
        key: asdict(summary)[key] for key in EXPECTED_FIRST_RUN
    } == EXPECTED_FIRST_RUN
    for identifier, schema in (
        (LABELS, XBRL_LABELS_ARROW_SCHEMA),
        (NETWORKS, XBRL_NETWORKS_ARROW_SCHEMA),
        (RELATIONSHIPS, XBRL_RELATIONSHIPS_ARROW_SCHEMA),
    ):
        table = _table(tmp_path, identifier)
        assert table.location() == str(
            tmp_path / "silver" / "sec_filings" / identifier.split(".")[1]
        )
        assert table.scan().to_arrow().schema.equals(schema)
        assert list(table.spec().fields) == []
    _assert_parents_and_counts(tmp_path)
    _assert_ten_k_rows(tmp_path)

    def reload(*_: object, **__: object) -> None:
        raise AssertionError("complete instances must not reload XBRL")

    snapshots, counts = _snapshots(tmp_path), _counts(tmp_path)
    monkeypatch.setattr(linkbases_extraction, "load_instance", reload)
    rerun = run_sec_filings_silver(tmp_path).xbrl_linkbases

    assert rerun is not None and rerun.status == "completed"
    assert (rerun.instances_skipped, rerun.instances_succeeded) == (3, 0)
    assert (_snapshots(tmp_path), _counts(tmp_path)) == (snapshots, counts)


EXPECTED_FIRST_RUN = {
    "status": "completed",
    "instances_discovered": 3,
    "instances_succeeded": 3,
    "instances_skipped": 0,
    "labels_discovered": 9,
    "labels_appended": 9,
    "networks_discovered": 9,
    "networks_appended": 9,
    "relationships_discovered": 18,
    "relationships_appended": 18,
    "presentation_relationships": 6,
    "calculation_relationships": 3,
    "definition_relationships": 9,
}


def _assert_parents_and_counts(data_dir: Path) -> None:
    """Every row has its parents; every network holds exactly its relationships."""

    concepts = {
        row["concept_id"] for row in _rows(data_dir, "sec_filings.xbrl_concepts")
    }
    instances = {
        row["instance_id"] for row in _rows(data_dir, "sec_filings.xbrl_instances")
    }
    labels, networks = _rows(data_dir, LABELS), _rows(data_dir, NETWORKS)
    relationships = _rows(data_dir, RELATIONSHIPS)
    children = Counter(r["network_id"] for r in relationships)
    assert {label["concept_id"] for label in labels} <= concepts
    assert {network["instance_id"] for network in networks} <= instances
    assert set(children) == {network["network_id"] for network in networks}
    assert all(
        children[n["network_id"]] == n["relationship_count"] >= 1 for n in networks
    )
    assert {r["source_concept_id"] for r in relationships} | {
        r["target_concept_id"] for r in relationships
    } <= concepts
    for rows, key in (
        (labels, "label_id"),
        (networks, "network_id"),
        (relationships, "relationship_id"),
    ):
        assert len({row[key] for row in rows}) == len(rows)


def _assert_ten_k_rows(data_dir: Path) -> None:
    networks = sorted(
        (n for n in _rows(data_dir, NETWORKS) if n["instance_id"] == TEN_K_INSTANCE),
        key=lambda n: n["network_ordinal"],
    )
    # One role, three linkbase types: three networks, in type order.
    assert [(n["linkbase_type"], n["role_uri"]) for n in networks] == [
        ("presentation", STATEMENT_ROLE),
        ("calculation", STATEMENT_ROLE),
        ("definition", STATEMENT_ROLE),
    ]
    assert {n["role_definition"] for n in networks} == {"100 - Statement - Revenue"}
    assert [n["instance_network_count"] for n in networks] == [3, 3, 3]
    relationships = {
        (r["arcrole_uri"].rsplit("/", 1)[-1], r["target_concept_id"]): r
        for r in _rows(data_dir, RELATIONSHIPS)
        if r["network_id"].startswith(TEN_K_INSTANCE)
    }
    revenue = relationships["parent-child", concept_id_for(EX, "Revenue")]
    assert (revenue["order_value"], revenue["preferred_label_role"]) == (
        "1",
        "http://www.xbrl.org/2003/role/terseLabel",
    )
    assert relationships["summation-item", concept_id_for(EX, "Cash")]["weight"] == "1"
    hypercube = relationships["all", concept_id_for(EX, "ProductTable")]
    assert (hypercube["context_element"], hypercube["is_closed"]) == ("segment", True)
    assert hypercube["is_usable"] is None
    domain = relationships["dimension-domain", concept_id_for(EX, "WidgetsMember")]
    assert (domain["is_usable"], domain["is_closed"], domain["weight"]) == (
        True,
        None,
        None,
    )
    labels = {
        (label["role_uri"].rsplit("/", 1)[-1], label["language"]): label["label_text"]
        for label in _rows(data_dir, LABELS)
        if label["source_document_id"] == "sec:0000320193-24-000123:document:0003"
    }
    assert labels == {
        ("label", "en-us"): "Revenue",
        ("terseLabel", "en-us"): "Revenues",
        ("label", "fr"): "Chiffre d\u2019affaires",
    }


def test_new_instance_appends_only_its_linkbase_rows(
    tmp_path: Path, write_sec_filing
) -> None:
    _write(tmp_path, write_sec_filing)
    run_sec_filings_silver(tmp_path)
    before = _counts(tmp_path)
    _write(tmp_path, write_sec_filing, [NEW_FILING])

    summary = run_sec_filings_silver(tmp_path).xbrl_linkbases

    assert summary is not None and summary.status == "completed"
    assert (summary.instances_succeeded, summary.instances_skipped) == (1, 3)
    assert (
        summary.labels_appended,
        summary.networks_appended,
        summary.relationships_appended,
    ) == tuple(PER_INSTANCE.values())
    assert _counts(tmp_path) == tuple(
        count + new for count, new in zip(before, PER_INSTANCE.values(), strict=True)
    )


def test_labels_only_commit_resumes_networks_and_relationships(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, write_sec_filing)
    with monkeypatch.context() as patch:
        patch.setattr(
            linkbases_stage, "append_rows", _fail_on(XBRL_NETWORKS_ARROW_SCHEMA)
        )
        with pytest.raises(OSError, match="disk full"):
            run_sec_filings_silver(tmp_path)
    labels_snapshot = _snapshots(tmp_path)[0]
    assert _counts(tmp_path) == (9, 0, 0)

    summary = run_sec_filings_silver(tmp_path).xbrl_linkbases

    assert summary is not None and summary.status == "completed"
    assert (summary.labels_skipped, summary.labels_appended) == (9, 0)
    assert (summary.networks_appended, summary.relationships_appended) == (9, 18)
    assert _snapshots(tmp_path)[0] == labels_snapshot


def test_networks_without_relationships_resume_relationships(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, write_sec_filing)
    with monkeypatch.context() as patch:
        patch.setattr(
            linkbases_stage, "append_rows", _fail_on(XBRL_RELATIONSHIPS_ARROW_SCHEMA)
        )
        with pytest.raises(OSError, match="disk full"):
            run_sec_filings_silver(tmp_path)
    snapshots = _snapshots(tmp_path)
    assert _counts(tmp_path) == (9, 9, 0)

    summary = run_sec_filings_silver(tmp_path).xbrl_linkbases

    assert summary is not None and summary.status == "completed"
    assert (summary.networks_skipped, summary.networks_appended) == (9, 0)
    assert summary.relationships_appended == 18
    assert _snapshots(tmp_path)[:2] == snapshots[:2]
    _assert_parents_and_counts(tmp_path)


def _change_network(data_dir: Path) -> None:
    """Store the 10-K presentation network again with another definition."""

    network_id = network_id_for(TEN_K_INSTANCE, "presentation", STATEMENT_ROLE)
    with open_silver_catalog(data_dir) as catalog:
        table = catalog.load_table(NETWORKS)
        row = next(
            r
            for r in table.scan().to_arrow().to_pylist()
            if r["network_id"] == network_id
        )
        table.delete(EqualTo("network_id", network_id))
        changed = XbrlNetworkRecord.model_validate(
            {**row, "role_definition": "changed"}
        )
        append_rows(table, [changed], XBRL_NETWORKS_ARROW_SCHEMA)


def test_changed_network_content_fails_the_integrity_gate(
    tmp_path: Path, write_sec_filing
) -> None:
    # The stage judges stored networks by their counts and ordinals; only the
    # audit's sampled filings recompute network_hash from the relationships.
    _write(tmp_path, write_sec_filing)
    run_sec_filings_silver(tmp_path)
    _change_network(tmp_path)

    result = run_sec_filings_silver(tmp_path)

    assert result.xbrl_linkbases is not None
    assert result.xbrl_linkbases.status == "completed"
    assert result.failed_stage == "integrity"
    (failure,) = result.summary()["integrity_failures"]
    assert failure["check_name"] == "sample_xbrl"
    assert "network_hash does not match" in failure["message"]


@pytest.mark.parametrize(
    "case",
    [
        ("partial-relationships", "1 relationships, declares 2"),
        ("relationships-without-network", "relationships have no network"),
        ("concept-missing", "stored rows refer to missing concepts"),
    ],
)
def test_damaged_linkbase_state_rejects_the_batch_before_any_write(
    tmp_path: Path,
    write_sec_filing,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    case: tuple[str, str],
) -> None:
    damage, reason = case
    _write(tmp_path, write_sec_filing)
    run_sec_filings_silver(tmp_path)
    presentation = network_id_for(TEN_K_INSTANCE, "presentation", STATEMENT_ROLE)
    if damage == "partial-relationships":
        first = next(
            r["relationship_id"]
            for r in _rows(tmp_path, RELATIONSHIPS)
            if r["network_id"] == presentation and r["relationship_ordinal"] == 1
        )
        _delete(tmp_path, RELATIONSHIPS, "relationship_id", first)
    elif damage == "relationships-without-network":
        _delete(tmp_path, NETWORKS, "network_id", presentation)
    else:
        _delete(
            tmp_path,
            "sec_filings.xbrl_concepts",
            "concept_id",
            concept_id_for(EX, "StatementAbstract"),
        )
    # A new instance in the same batch must not be written either.
    _write(tmp_path, write_sec_filing, [NEW_FILING])
    before, counts = _snapshots(tmp_path), _counts(tmp_path)
    capsys.readouterr()
    caplog.clear()

    summary = linkbases_stage.run_xbrl_linkbases_stage(
        tmp_path, datetime.now(UTC), _settings(tmp_path)
    )

    assert summary.status == "rejected"
    assert summary.instances_conflicted >= 1
    assert (_snapshots(tmp_path), _counts(tmp_path)) == (before, counts)
    # Events print to stdout until a CLI configures logging, then go through
    # the logging module.
    output = capsys.readouterr().out + caplog.text
    assert "xbrl_linkbases_conflict" in output
    assert reason in output


def test_processor_failure_writes_no_linkbase_row(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, write_sec_filing)

    def broken_load(files, **options):
        if files[0].document_id.startswith("sec:0000320193-25-000057"):
            raise XbrlProcessorError("unreadable instance")
        return load_instance(files, **options)

    monkeypatch.setattr(linkbases_extraction, "load_instance", broken_load)
    result = run_sec_filings_silver(tmp_path)

    summary = result.xbrl_linkbases
    assert result.xbrl_enrichment is not None
    assert result.xbrl_enrichment.status == "completed"
    assert summary is not None and summary.status == "rejected"
    assert (summary.instances_failed, summary.instances_succeeded) == (1, 2)
    assert _snapshots(tmp_path) == (None, None, None)

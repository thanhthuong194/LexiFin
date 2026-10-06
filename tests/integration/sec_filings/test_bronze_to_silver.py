from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, fields
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pytest
from pyiceberg.expressions import EqualTo
from pyiceberg.table import Table

from scripts.sec_filings.build_silver import REPOSITORY_ROOT, parse_args
from src.data_pipeline.contracts.sec_filings.content_blocks import (
    CONTENT_BLOCKS_ARROW_SCHEMA,
)
from src.data_pipeline.contracts.sec_filings.document_links import (
    DOCUMENT_LINKS_ARROW_SCHEMA,
    DocumentLinkRecord,
)
from src.data_pipeline.contracts.sec_filings.document_table_cells import (
    DOCUMENT_TABLE_CELLS_ARROW_SCHEMA,
)
from src.data_pipeline.contracts.sec_filings.document_tables import (
    DOCUMENT_TABLES_ARROW_SCHEMA,
    DocumentTableRecord,
)
from src.data_pipeline.contracts.sec_filings.documents import (
    DOCUMENTS_ARROW_SCHEMA,
    DocumentRecord,
)
from src.data_pipeline.contracts.sec_filings.filing_entities import (
    FILING_ENTITIES_ARROW_SCHEMA,
)
from src.data_pipeline.contracts.sec_filings.filings import FILINGS_ARROW_SCHEMA
from src.data_pipeline.contracts.sec_filings.sections import SECTIONS_ARROW_SCHEMA
from src.data_pipeline.contracts.sec_filings.xbrl_concepts import (
    XBRL_CONCEPTS_ARROW_SCHEMA,
    ConceptDefinition,
    XbrlConceptRecord,
    concept_id_for,
)
from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlInstanceRecord
from src.data_pipeline.contracts.sec_filings.xbrl_dimensions import (
    XBRL_DIMENSIONS_ARROW_SCHEMA,
)
from src.data_pipeline.contracts.sec_filings.xbrl_fact_footnotes import (
    XBRL_FACT_FOOTNOTES_ARROW_SCHEMA,
)
from src.data_pipeline.contracts.sec_filings.xbrl_facts import (
    XBRL_FACTS_ARROW_SCHEMA,
    XbrlFactRecord,
)
from src.data_pipeline.pipelines.sec_filings import (
    document_link_extraction,
    document_table_extraction,
    xbrl_core_extraction,
    xbrl_enrichment_extraction,
    xbrl_facts_extraction,
)
from src.data_pipeline.pipelines.sec_filings import (
    document_tables as document_tables_stage,
)
from src.data_pipeline.pipelines.sec_filings import documents as documents_stage
from src.data_pipeline.pipelines.sec_filings import narrative as narrative_stage
from src.data_pipeline.pipelines.sec_filings import xbrl_core as xbrl_core_stage
from src.data_pipeline.pipelines.sec_filings import (
    xbrl_enrichment as xbrl_enrichment_stage,
)
from src.data_pipeline.pipelines.sec_filings import xbrl_facts as xbrl_facts_stage
from src.data_pipeline.pipelines.sec_filings.bronze_to_silver import (
    run_sec_filings_silver,
)
from src.data_pipeline.storage.bronze import sha256_file
from src.data_pipeline.storage.iceberg import append_rows, open_silver_catalog
from src.data_pipeline.storage.sec_filings.filing_entities import (
    read_filing_entity_groups,
)
from src.data_pipeline.storage.sec_filings.xbrl_core import XBRL_CORE_SCHEMAS
from src.data_pipeline.transforms.sec_filings.xbrl_model_reader import (
    XbrlProcessorError,
)
from src.data_pipeline.transforms.sec_filings.xbrl_processor import (
    ProcessorSettings,
    load_instance,
)

pytestmark = pytest.mark.integration

FILINGS = "sec_filings.filings"
DOCUMENTS = "sec_filings.documents"
ENTITIES = "sec_filings.filing_entities"
SECTIONS = "sec_filings.sections"
BLOCKS = "sec_filings.content_blocks"
TEN_K_DOCUMENT = "sec:0000320193-24-000123:document:0001"
TEN_Q_DOCUMENT = "sec:0000320193-25-000057:document:0001"
EIGHT_K_DOCUMENT = "sec:0000789019-25-000010:document:0001"
NEW_YEAR_DOCUMENT = "sec:0000320193-26-000001:document:0001"
# Sections and blocks that the fixture's PRIMARY_HTML yields per form.
EXPECTED_CODES = {
    TEN_K_DOCUMENT: [
        "TABLE OF CONTENTS", "PART I", "ITEM 1", "ITEM 1A", "PART II", "ITEM 7",
        "SIGNATURES",
    ],
    TEN_Q_DOCUMENT: [
        "PART I", "ITEM 1", "ITEM 2", "PART II", "ITEM 1", "ITEM 1A", "SIGNATURES",
    ],
    EIGHT_K_DOCUMENT: ["ITEM 2.02", "ITEM 9.01", "EXHIBIT INDEX", "SIGNATURES"],
}  # fmt: skip
SECTION_ROWS, BLOCK_ROWS = 21, 32
TABLES = "sec_filings.document_tables"
CELLS = "sec_filings.document_table_cells"
# The 10-K holds a linked TOC table (8 cells) and a revenue table (2 cells),
# the 8-K an exhibit table (2 cells); the 10-Q has no table.
TABLE_ROWS, CELL_ROWS = 3, 12
LINKS = "sec_filings.document_links"
# Each filing has three HTML documents (primary, exhibit ex31.htm, R page) of
# which only the 10-K and 8-K primaries hold links: the 10-K five TOC fragments,
# an in-text fragment, its exhibit by SEC URL and an external site; the 8-K its
# exhibit, a missing file and another filing's ex31.htm by SEC URL.
HTML_DOCUMENTS, LINK_ROWS = 9, 11
TEN_K_EXHIBIT = "sec:0000320193-24-000123:document:0002"
INSTANCES = "sec_filings.xbrl_instances"
CONTEXTS = "sec_filings.xbrl_contexts"
UNITS = "sec_filings.xbrl_units"
# (facts, contexts, units) of each primary document's one default Inline
# target, merged with EDGAR's XML rendering (document 4): the 10-K keeps an
# unused EUR unit that the rendering leaves out; the 8-K has no unit.
XBRL_COUNTS = {
    TEN_K_DOCUMENT: (6, 3, 4),
    TEN_Q_DOCUMENT: (2, 2, 1),
    EIGHT_K_DOCUMENT: (2, 1, 0),
}
EIGHT_K_EXHIBIT = "sec:0000789019-25-000010:document:0002"
# Every fake submission from the write_sec_filing fixture holds six documents.
DOCUMENTS_PER_FILING = 6
CIK = "0000320193"
OTHER_CIK = "0000789019"
# The 8-K is filed jointly with a subsidiary, so its header has two FILER blocks.
CO_REGISTRANT_CIK = "0001652044"
EIGHT_K_ID = "sec:0000789019-25-000010"
INITIAL_FILINGS = [
    (CIK, "10-K", "0000320193-24-000123", "20241101"),
    (CIK, "10-Q", "0000320193-25-000057", "20250502"),
    (OTHER_CIK, "8-K", "0000789019-25-000010", "20250129"),
]
MANIFEST_RUN_ID = "3c0c83bd38034ccbbbd9a746a215e6b1"


def _write_bronze(data_dir: Path, write_sec_filing) -> None:
    for cik, form, accession, filed in INITIAL_FILINGS:
        write_sec_filing(
            data_dir,
            cik=cik,
            form=form,
            accession=accession,
            filed=filed,
            co_registrant_cik=CO_REGISTRANT_CIK if form == "8-K" else None,
        )
    # The Bronze manifest records which ingestion run wrote the 10-K artifacts.
    cik, form, accession, _ = INITIAL_FILINGS[0]
    folder = f"bronze/sec_filings/sec-edgar-filings/{cik}/{form}/{accession}"
    artifacts = [
        {
            "role": role,
            "relative_path": f"{folder}/{name}",
            "sha256": sha256_file(data_dir / folder / name),
            "size_bytes": (data_dir / folder / name).stat().st_size,
        }
        for role, name in [
            ("full_submission", "full-submission.txt"),
            ("primary_document", "primary-document.html"),
        ]
    ]
    manifest = (
        data_dir / "bronze" / "sec_filings" / "_metadata" / "filing_manifest.jsonl"
    )
    manifest.parent.mkdir(parents=True)
    entry = {
        "run_id": MANIFEST_RUN_ID,
        "filing_id": f"{cik}:{form}:{accession}",
        "cik": cik,
        "accession_number": accession,
        "form": form,
        "filing_date": "2024-11-01",
        "status": "success",
        "artifacts": artifacts,
        "error": None,
        "recorded_at": "2026-09-23T07:58:19Z",
    }
    manifest.write_text(json.dumps(entry) + "\n")


def _write_new_year_filing(data_dir: Path, write_sec_filing) -> None:
    write_sec_filing(
        data_dir,
        cik=CIK,
        form="8-K",
        accession="0000320193-26-000001",
        filed="20260105",
    )


def _table(data_dir: Path, identifier: str = FILINGS) -> Table:
    with open_silver_catalog(data_dir) as catalog:
        return catalog.load_table(identifier)


def _rows_per_year(table: Table) -> Counter[int]:
    """Count rows per partition year from the table's current data files."""

    counts: Counter[int] = Counter()
    for data_file in table.inspect.files().to_pylist():
        # Iceberg's year transform stores years since 1970.
        counts[1970 + data_file["partition"]["filing_year"]] += data_file[
            "record_count"
        ]
    return counts


def _rows_by_document(data_dir: Path, identifier: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in _table(data_dir, identifier).scan().to_arrow().to_pylist():
        groups.setdefault(row["document_id"], []).append(row)
    return groups


def _snapshot(data_dir: Path, identifier: str) -> int | None:
    snapshot = _table(data_dir, identifier).current_snapshot()
    return None if snapshot is None else snapshot.snapshot_id


def _delete_rows(data_dir: Path, identifier: str, column: str, value: str) -> int:
    """Delete matching rows through PyIceberg and return the new snapshot ID."""

    with open_silver_catalog(data_dir) as catalog:
        table = catalog.load_table(identifier)
        table.delete(EqualTo(column, value))
        return table.current_snapshot().snapshot_id


def _partition_columns(table: Table) -> list[tuple[str, str, str | None]]:
    return [
        (
            field.name,
            str(field.transform),
            table.schema().find_column_name(field.source_id),
        )
        for field in table.spec().fields
    ]


def test_first_run_writes_partitioned_rows_and_rerun_skips_them(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)

    first = run_sec_filings_silver(tmp_path)

    assert first.completed
    assert (first.filings.discovered, first.filings.parsed) == (3, 3)
    assert (first.filings.inserted, first.filings.skipped) == (3, 0)
    assert (first.filings.conflicts, first.filings.failed) == (0, 0)
    assert first.filings.snapshot_before is None
    assert first.filings.snapshot_after is not None

    table = _table(tmp_path)
    assert table.location() == str(tmp_path / "silver" / "sec_filings" / "filings")
    assert _partition_columns(table) == [("filing_year", "year", "filing_date")]
    rows = table.scan().to_arrow()
    assert rows.schema.equals(FILINGS_ARROW_SCHEMA)
    by_id = {row["filing_id"]: row for row in rows.to_pylist()}
    assert sorted(by_id) == sorted(f"sec:{acc}" for _, _, acc, _ in INITIAL_FILINGS)
    ten_k = by_id["sec:0000320193-24-000123"]
    assert ten_k["form"] == "10-K"
    assert ten_k["cik"] == CIK
    assert ten_k["filing_date"] == date(2024, 11, 1)
    assert ten_k["report_date"] == date(2024, 12, 31)
    assert ten_k["accepted_at"] == datetime.fromisoformat("2024-11-01T16:01:02")
    assert ten_k["public_document_count"] == DOCUMENTS_PER_FILING
    assert ten_k["full_submission_sha256"] == sha256_file(
        tmp_path / ten_k["full_submission_path"]
    )
    assert ten_k["ingestion_run_id"] == MANIFEST_RUN_ID
    assert by_id["sec:0000789019-25-000010"]["ingestion_run_id"] is None
    assert len({row["processed_at"] for row in by_id.values()}) == 1
    assert _rows_per_year(table) == {2024: 1, 2025: 2}

    rerun = run_sec_filings_silver(tmp_path)

    assert rerun.completed
    assert (rerun.filings.inserted, rerun.filings.skipped) == (0, 3)
    assert rerun.filings.conflicts == 0
    assert (
        rerun.filings.snapshot_before
        == rerun.filings.snapshot_after
        == first.filings.snapshot_after
    )
    assert _table(tmp_path).scan().to_arrow().num_rows == 3


def test_documents_inventory_every_block_with_exact_offsets_and_reruns_skip(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)

    first = run_sec_filings_silver(tmp_path)

    documents = first.documents
    assert documents is not None
    assert documents.status == "completed"
    assert documents.filings_considered == 3
    assert (documents.filings_inserted, documents.filings_skipped) == (3, 0)
    assert (documents.documents_inserted, documents.documents_skipped) == (18, 0)
    assert (documents.filings_conflicted, documents.documents_failed) == (0, 0)
    assert documents.document_count_mismatches == 0
    assert documents.snapshot_before is None
    assert documents.snapshot_after is not None

    table = _table(tmp_path, DOCUMENTS)
    assert table.location() == str(tmp_path / "silver" / "sec_filings" / "documents")
    assert _partition_columns(table) == [("filing_year", "year", "filing_date")]
    rows = table.scan().to_arrow()
    assert rows.schema.equals(DOCUMENTS_ARROW_SCHEMA)
    assert rows.num_rows == 3 * DOCUMENTS_PER_FILING
    assert _rows_per_year(table) == {2024: 6, 2025: 12}
    parents = {
        row["filing_id"]: row for row in _table(tmp_path).scan().to_arrow().to_pylist()
    }
    by_filing: dict[str, list[dict]] = {}
    for row in rows.to_pylist():
        by_filing.setdefault(row["filing_id"], []).append(row)
    assert set(by_filing) == set(parents)
    for filing_id, group in by_filing.items():
        parent = parents[filing_id]
        assert sorted(row["document_ordinal"] for row in group) == list(range(1, 7))
        assert {row["filing_date"] for row in group} == {parent["filing_date"]}
        assert {row["processed_at"] for row in group} == {parent["processed_at"]}
        assert Counter(row["role"] for row in group) == {
            "primary_document": 1,
            "exhibit": 1,
            "xbrl_schema": 1,
            "xbrl_instance": 1,
            "graphic": 1,
            "generated": 1,
        }
        raw = (tmp_path / parent["full_submission_path"]).read_bytes()
        for row in group:
            block = raw[row["document_start_byte"] : row["document_end_byte"]]
            assert block.startswith(b"<DOCUMENT>")
            assert block.endswith(b"</DOCUMENT>")
            content = raw[row["content_start_byte"] : row["content_end_byte"]]
            assert len(content) == row["content_size_bytes"]
            assert hashlib.sha256(content).hexdigest() == row["content_sha256"]
    primary = next(
        row
        for row in by_filing["sec:0000320193-24-000123"]
        if row["role"] == "primary_document"
    )
    assert primary["document_id"] == "sec:0000320193-24-000123:document:0001"
    assert (primary["document_type"], primary["filename"]) == ("10-K", "primary.htm")
    raw = (tmp_path / primary["source_path"]).read_bytes()
    content = raw[primary["content_start_byte"] : primary["content_end_byte"]]
    assert content.startswith(b"\n<XBRL>\n")
    assert content.endswith(b"</html>\n</XBRL>\n")

    rerun = run_sec_filings_silver(tmp_path)

    assert rerun.completed
    assert rerun.documents is not None
    assert (rerun.documents.filings_skipped, rerun.documents.documents_skipped) == (
        3,
        18,
    )
    assert rerun.documents.documents_inserted == 0
    assert (
        rerun.documents.snapshot_before
        == rerun.documents.snapshot_after
        == documents.snapshot_after
    )
    assert _table(tmp_path, DOCUMENTS).scan().to_arrow().num_rows == 18


def test_filing_entities_record_every_entity_block_and_reruns_skip(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)

    first = run_sec_filings_silver(tmp_path)

    entities = first.filing_entities
    assert first.completed
    assert entities is not None
    assert entities.filings_considered == 3
    assert (entities.filings_inserted, entities.filings_skipped) == (3, 0)
    assert (entities.entities_inserted, entities.entities_skipped) == (4, 0)
    assert (entities.filings_conflicted, entities.entities_failed) == (0, 0)
    assert entities.snapshot_before is None
    assert entities.snapshot_after is not None

    table = _table(tmp_path, ENTITIES)
    assert table.location() == str(
        tmp_path / "silver" / "sec_filings" / "filing_entities"
    )
    assert table.spec().is_unpartitioned()
    rows = table.scan().to_arrow()
    assert rows.schema.equals(FILING_ENTITIES_ARROW_SCHEMA)
    assert rows.num_rows == 4
    parents = {
        row["filing_id"]: row for row in _table(tmp_path).scan().to_arrow().to_pylist()
    }
    by_filing: dict[str, list[dict]] = {}
    for row in rows.to_pylist():
        by_filing.setdefault(row["filing_id"], []).append(row)
    assert set(by_filing) == set(parents)
    for filing_id, group in by_filing.items():
        parent = parents[filing_id]
        assert any(
            row["entity_role"] == "filer" and row["cik"] == parent["cik"]
            for row in group
        )
        assert {row["form_type"] for row in group} == {parent["form"]}
        assert {row["filing_date"] for row in group} == {parent["filing_date"]}
        raw = (tmp_path / parent["full_submission_path"]).read_bytes()
        for row in group:
            span = raw[row["entity_start_byte"] : row["entity_end_byte"]]
            assert span.startswith(b"FILER:")
            assert hashlib.sha256(span).hexdigest() == row["entity_block_sha256"]
            assert row["source_submission_sha256"] == parent["full_submission_sha256"]
    eight_k = sorted(by_filing[EIGHT_K_ID], key=lambda row: row["entity_ordinal"])
    assert [(row["filing_entity_id"], row["cik"]) for row in eight_k] == [
        (f"{EIGHT_K_ID}:entity:0001", OTHER_CIK),
        (f"{EIGHT_K_ID}:entity:0002", CO_REGISTRANT_CIK),
    ]
    assert eight_k[0]["entity_end_byte"] == eight_k[1]["entity_start_byte"]
    assert (eight_k[0]["sic"], eight_k[0]["sic_description"]) == (
        "3571",
        "ELECTRONIC COMPUTERS",
    )

    rerun = run_sec_filings_silver(tmp_path)

    assert rerun.completed
    assert rerun.filing_entities is not None
    assert (
        rerun.filing_entities.filings_skipped,
        rerun.filing_entities.entities_skipped,
    ) == (3, 4)
    assert rerun.filing_entities.entities_inserted == 0
    assert (
        rerun.filing_entities.snapshot_before
        == rerun.filing_entities.snapshot_after
        == entities.snapshot_after
    )
    assert _table(tmp_path, ENTITIES).scan().to_arrow().num_rows == 4


def test_new_year_and_backfill_append_into_their_year_partitions(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    first = run_sec_filings_silver(tmp_path)

    _write_new_year_filing(tmp_path, write_sec_filing)
    new_year = run_sec_filings_silver(tmp_path)

    assert (new_year.filings.inserted, new_year.filings.skipped) == (1, 3)
    assert new_year.filings.snapshot_before == first.filings.snapshot_after
    assert new_year.filings.snapshot_after != new_year.filings.snapshot_before
    assert _rows_per_year(_table(tmp_path)) == {2024: 1, 2025: 2, 2026: 1}
    assert new_year.documents is not None
    assert (
        new_year.documents.filings_inserted,
        new_year.documents.filings_skipped,
    ) == (
        1,
        3,
    )
    assert new_year.documents.documents_inserted == DOCUMENTS_PER_FILING
    document_filings = Counter(
        _table(tmp_path, DOCUMENTS)
        .scan(selected_fields=("filing_id",))
        .to_arrow()["filing_id"]
        .to_pylist()
    )
    assert len(document_filings) == 4
    assert document_filings["sec:0000320193-26-000001"] == DOCUMENTS_PER_FILING
    assert _rows_per_year(_table(tmp_path, DOCUMENTS)) == {2024: 6, 2025: 12, 2026: 6}
    assert new_year.filing_entities is not None
    assert (
        new_year.filing_entities.filings_inserted,
        new_year.filing_entities.entities_inserted,
        new_year.filing_entities.entities_skipped,
    ) == (1, 1, 4)
    new_entities = read_filing_entity_groups(_table(tmp_path, ENTITIES))
    assert len(new_entities["sec:0000320193-26-000001"]) == 1
    assert new_year.narrative is not None
    assert (
        new_year.narrative.documents_inserted,
        new_year.narrative.documents_skipped,
        new_year.narrative.sections_inserted,
        new_year.narrative.blocks_inserted,
    ) == (1, 3, 5, 7)
    assert new_year.document_tables is not None
    assert (
        new_year.document_tables.documents_inserted,
        new_year.document_tables.documents_skipped,
        new_year.document_tables.documents_without_tables,
        new_year.document_tables.tables_written,
        new_year.document_tables.cells_written,
    ) == (1, 2, 1, 1, 2)
    assert new_year.document_links is not None
    # Documents without links are parsed again: the first run's seven plus the
    # new filing's exhibit and R page.
    assert (
        new_year.document_links.documents_inserted,
        new_year.document_links.documents_skipped,
        new_year.document_links.documents_without_links,
        new_year.document_links.links_inserted,
    ) == (1, 2, 7 + 2, 3)
    assert _rows_per_year(_table(tmp_path, LINKS)) == {2024: 8, 2025: 3, 2026: 3}
    assert new_year.xbrl_core is not None
    assert (
        new_year.xbrl_core.filings_inserted,
        new_year.xbrl_core.filings_skipped,
        new_year.xbrl_core.instances_inserted,
    ) == (1, 3, 1)
    sections = _rows_by_document(tmp_path, SECTIONS)
    assert len(sections) == 4
    assert len(sections[NEW_YEAR_DOCUMENT]) == 5

    write_sec_filing(
        tmp_path,
        cik=OTHER_CIK,
        form="10-K",
        accession="0000789019-24-000099",
        filed="20240730",
    )
    backfill = run_sec_filings_silver(tmp_path)

    assert (backfill.filings.inserted, backfill.filings.skipped) == (1, 4)
    assert _rows_per_year(_table(tmp_path)) == {2024: 2, 2025: 2, 2026: 1}
    assert _table(tmp_path).scan().to_arrow().num_rows == 5
    assert _rows_per_year(_table(tmp_path, DOCUMENTS)) == {
        2024: 12,
        2025: 12,
        2026: 6,
    }


def test_filing_checksum_conflict_rejects_both_stages(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    first = run_sec_filings_silver(tmp_path)
    cik, form, accession, _ = INITIAL_FILINGS[1]
    changed = (
        tmp_path / "bronze/sec_filings/sec-edgar-filings" / cik / form / accession
    ) / "primary-document.html"
    changed.write_bytes(b"<html>different bytes</html>")
    _write_new_year_filing(tmp_path, write_sec_filing)

    result = run_sec_filings_silver(tmp_path)

    assert not result.completed
    assert result.filings.status == "rejected"
    assert (result.filings.inserted, result.filings.skipped) == (0, 2)
    assert result.filings.conflicts == 1
    assert (
        result.filings.snapshot_after
        == result.filings.snapshot_before
        == first.filings.snapshot_after
    )
    assert result.documents is None
    table = _table(tmp_path)
    assert table.current_snapshot().snapshot_id == first.filings.snapshot_after
    assert table.scan().to_arrow().num_rows == 3
    assert first.documents is not None
    documents = _table(tmp_path, DOCUMENTS)
    assert documents.current_snapshot().snapshot_id == first.documents.snapshot_after
    assert documents.scan().to_arrow().num_rows == 18
    assert result.filing_entities is None
    assert result.narrative is None
    assert _table(tmp_path, ENTITIES).scan().to_arrow().num_rows == 4


def test_partial_document_group_is_a_conflict_without_partial_append(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    run_sec_filings_silver(tmp_path)
    with open_silver_catalog(tmp_path) as catalog:
        documents = catalog.load_table(DOCUMENTS)
        documents.delete(
            EqualTo("document_id", "sec:0000320193-25-000057:document:0003")
        )
        damaged_snapshot = documents.current_snapshot().snapshot_id
    _write_new_year_filing(tmp_path, write_sec_filing)

    result = run_sec_filings_silver(tmp_path)

    assert not result.completed
    assert (result.filings.status, result.filings.inserted) == ("completed", 1)
    assert result.documents is not None
    assert result.documents.status == "rejected"
    assert result.documents.filings_conflicted == 1
    assert result.documents.filings_skipped == 2
    assert result.documents.documents_inserted == 0
    assert (result.filing_entities, result.narrative) == (None, None)
    table = _table(tmp_path, DOCUMENTS)
    assert table.current_snapshot().snapshot_id == damaged_snapshot
    assert table.scan().to_arrow().num_rows == 17
    assert "sec:0000320193-26-000001" not in set(
        table.scan(selected_fields=("filing_id",)).to_arrow()["filing_id"].to_pylist()
    )


def test_rerun_builds_documents_missing_after_a_failed_documents_stage(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bronze(tmp_path, write_sec_filing)

    def fail_append(*_args: object) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(documents_stage, "append_rows", fail_append)
        with pytest.raises(OSError, match="disk full"):
            run_sec_filings_silver(tmp_path)
    assert _table(tmp_path).scan().to_arrow().num_rows == 3
    assert _table(tmp_path, DOCUMENTS).current_snapshot() is None

    recovered = run_sec_filings_silver(tmp_path)

    assert recovered.completed
    assert (recovered.filings.inserted, recovered.filings.skipped) == (0, 3)
    assert recovered.documents is not None
    assert recovered.documents.filings_inserted == 3
    assert recovered.documents.documents_inserted == 18
    assert recovered.filing_entities is not None
    assert recovered.filing_entities.entities_inserted == 4
    assert recovered.narrative is not None
    assert recovered.narrative.documents_inserted == 3
    assert recovered.document_tables is not None
    assert recovered.document_tables.documents_inserted == 2


def test_invalid_filing_rejects_batch_before_any_write(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    write_sec_filing(
        tmp_path,
        cik=CIK,
        form="8-K",
        accession="0000320193-26-000001",
        filed="20260105",
        header_form="8-K/A",
    )

    result = run_sec_filings_silver(tmp_path)

    assert result.filings.status == "rejected"
    assert (result.filings.discovered, result.filings.parsed) == (4, 3)
    assert (result.filings.failed, result.filings.inserted) == (1, 0)
    assert result.documents is None
    assert (result.filing_entities, result.narrative) == (None, None)
    for table_name in (
        "filings",
        "documents",
        "filing_entities",
        "sections",
        "content_blocks",
    ):
        assert not (tmp_path / "silver" / "sec_filings" / table_name).exists()


def test_missing_tail_entity_is_a_conflict_without_partial_append(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    run_sec_filings_silver(tmp_path)
    with open_silver_catalog(tmp_path) as catalog:
        entities = catalog.load_table(ENTITIES)
        entities.delete(EqualTo("filing_entity_id", f"{EIGHT_K_ID}:entity:0002"))
        damaged_snapshot = entities.current_snapshot().snapshot_id
    _write_new_year_filing(tmp_path, write_sec_filing)

    result = run_sec_filings_silver(tmp_path)

    assert not result.completed
    assert (result.filings.inserted, result.filings.status) == (1, "completed")
    assert result.documents is not None
    assert result.documents.documents_inserted == DOCUMENTS_PER_FILING
    assert result.filing_entities is not None
    assert result.filing_entities.status == "rejected"
    assert result.filing_entities.filings_conflicted == 1
    assert result.filing_entities.entities_inserted == 0
    assert (result.narrative, result.document_tables, result.document_links) == (
        None,
        None,
        None,
    )
    table = _table(tmp_path, ENTITIES)
    assert table.current_snapshot().snapshot_id == damaged_snapshot
    assert table.scan().to_arrow().num_rows == 3
    assert "sec:0000320193-26-000001" not in read_filing_entity_groups(table)


def test_changed_entity_with_the_same_id_is_a_conflict(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    run_sec_filings_silver(tmp_path)
    ten_k_entity = "sec:0000320193-24-000123:entity:0001"
    with open_silver_catalog(tmp_path) as catalog:
        entities = catalog.load_table(ENTITIES)
        stored = next(
            record
            for group in read_filing_entity_groups(entities).values()
            for record in group
            if record.filing_entity_id == ten_k_entity
        )
        entities.delete(EqualTo("filing_entity_id", ten_k_entity))
        renamed = stored.model_copy(update={"company_name": "Renamed Corp"})
        append_rows(entities, [renamed], FILING_ENTITIES_ARROW_SCHEMA)
        changed_snapshot = entities.current_snapshot().snapshot_id

    result = run_sec_filings_silver(tmp_path)

    assert result.filing_entities is not None
    assert result.filing_entities.status == "rejected"
    assert result.filing_entities.filings_conflicted == 1
    assert result.filing_entities.filings_skipped == 2
    assert result.filing_entities.entities_inserted == 0
    table = _table(tmp_path, ENTITIES)
    assert table.current_snapshot().snapshot_id == changed_snapshot
    assert table.scan().to_arrow().num_rows == 4


def test_cli_defaults_to_the_repository_data_dir_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["build_silver.py"])

    args = parse_args()

    assert args.data_dir == REPOSITORY_ROOT / "data"
    # The taxonomy cache defaults to <data-dir>/_cache/arelle, never downloaded.
    assert (args.taxonomy_cache_dir, args.allow_taxonomy_network) == (None, False)


def test_narrative_builds_sections_and_blocks_and_reruns_skip(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)

    first = run_sec_filings_silver(tmp_path)

    narrative = first.narrative
    assert first.completed
    assert narrative is not None
    assert (narrative.documents_considered, narrative.documents_inserted) == (3, 3)
    assert (narrative.documents_skipped, narrative.documents_resumed) == (0, 0)
    assert (narrative.documents_conflicted, narrative.documents_failed) == (0, 0)
    assert (narrative.sections_inserted, narrative.blocks_inserted) == (
        SECTION_ROWS,
        BLOCK_ROWS,
    )
    assert (narrative.root_only_documents, narrative.duplicate_section_codes) == (0, 0)
    assert narrative.sections_snapshot_before is None
    assert narrative.content_blocks_snapshot_before is None
    for identifier, schema in (
        (SECTIONS, SECTIONS_ARROW_SCHEMA),
        (BLOCKS, CONTENT_BLOCKS_ARROW_SCHEMA),
    ):
        table = _table(tmp_path, identifier)
        assert table.location() == str(
            tmp_path / "silver" / "sec_filings" / identifier.split(".")[1]
        )
        assert _partition_columns(table) == [("filing_year", "year", "filing_date")]
        assert table.scan().to_arrow().schema.equals(schema)

    sections = _rows_by_document(tmp_path, SECTIONS)
    blocks = _rows_by_document(tmp_path, BLOCKS)
    primaries = {
        row["document_id"]: row
        for row in _table(tmp_path, DOCUMENTS).scan().to_arrow().to_pylist()
        if row["role"] == "primary_document"
    }
    assert set(sections) == set(blocks) == set(primaries)
    for document_id, codes in EXPECTED_CODES.items():
        ordered = sorted(sections[document_id], key=lambda row: row["section_ordinal"])
        assert [row["section_code"] for row in ordered[1:]] == codes
        assert ordered[0]["section_kind"] == "document"
        section_ids = {row["section_id"]: row for row in ordered}
        for block in blocks[document_id]:
            assert block["section_id"] in section_ids
            assert block["filing_id"] == primaries[document_id]["filing_id"]
            assert (
                block["source_document_sha256"]
                == primaries[document_id]["content_sha256"]
            )
    all_blocks = [block for group in blocks.values() for block in group]
    all_sections = [section for group in sections.values() for section in group]
    assert len({b["block_id"] for b in all_blocks}) == len(all_blocks) == BLOCK_ROWS
    assert (
        len({s["section_id"] for s in all_sections})
        == len(all_sections)
        == SECTION_ROWS
    )
    texts = [block["text"] for block in all_blocks]
    assert "We make widgets." in texts
    assert "false" not in texts and "example" not in texts
    revenue = next(b for b in blocks[TEN_K_DOCUMENT] if b["text"] == "Revenue\n$1,000")
    assert (revenue["block_type"], revenue["source_tag"]) == ("table", "table")
    assert not revenue["inside_table"] and revenue["source_table_xpath"] is None
    assert not any(b["block_type"] == "table_cell" for b in all_blocks)
    eight_k_root = f"{EIGHT_K_DOCUMENT}:section:0001"
    assert not any(b["section_id"] == eight_k_root for b in blocks[EIGHT_K_DOCUMENT])

    rerun = run_sec_filings_silver(tmp_path)

    assert rerun.completed
    assert rerun.narrative is not None
    assert (rerun.narrative.documents_skipped, rerun.narrative.documents_inserted) == (
        3,
        0,
    )
    assert (
        rerun.narrative.sections_snapshot_before,
        rerun.narrative.content_blocks_snapshot_before,
    ) == (narrative.sections_snapshot_after, narrative.content_blocks_snapshot_after)
    assert (
        rerun.narrative.sections_snapshot_after,
        rerun.narrative.content_blocks_snapshot_after,
    ) == (narrative.sections_snapshot_after, narrative.content_blocks_snapshot_after)
    assert _table(tmp_path, SECTIONS).scan().to_arrow().num_rows == SECTION_ROWS
    assert _table(tmp_path, BLOCKS).scan().to_arrow().num_rows == BLOCK_ROWS


def test_narrative_resumes_blocks_after_a_failed_blocks_commit(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bronze(tmp_path, write_sec_filing)

    def append_sections_only(table: Table, rows, arrow_schema) -> None:
        if arrow_schema is CONTENT_BLOCKS_ARROW_SCHEMA:
            raise OSError("disk full")
        append_rows(table, rows, arrow_schema)

    with monkeypatch.context() as patch:
        patch.setattr(narrative_stage, "append_rows", append_sections_only)
        with pytest.raises(OSError, match="disk full"):
            run_sec_filings_silver(tmp_path)
    sections_snapshot = _snapshot(tmp_path, SECTIONS)
    assert _table(tmp_path, SECTIONS).scan().to_arrow().num_rows == SECTION_ROWS
    assert _snapshot(tmp_path, BLOCKS) is None

    resumed = run_sec_filings_silver(tmp_path)

    assert resumed.completed
    assert resumed.narrative is not None
    assert (
        resumed.narrative.documents_resumed,
        resumed.narrative.documents_inserted,
    ) == (
        3,
        0,
    )
    assert (resumed.narrative.sections_inserted, resumed.narrative.blocks_inserted) == (
        0,
        BLOCK_ROWS,
    )
    assert _snapshot(tmp_path, SECTIONS) == sections_snapshot
    assert _table(tmp_path, SECTIONS).scan().to_arrow().num_rows == SECTION_ROWS
    assert _table(tmp_path, BLOCKS).scan().to_arrow().num_rows == BLOCK_ROWS


@pytest.mark.parametrize(
    "deleted",
    [
        (SECTIONS, "document_id", TEN_Q_DOCUMENT),
        (BLOCKS, "block_id", f"{TEN_K_DOCUMENT}:block:000013"),
        (SECTIONS, "section_id", f"{TEN_K_DOCUMENT}:section:0008"),
    ],
    ids=["blocks-without-sections", "truncated-blocks", "truncated-sections"],
)
def test_incomplete_narrative_groups_are_conflicts_before_any_write(
    tmp_path: Path, write_sec_filing, deleted: tuple[str, str, str]
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    run_sec_filings_silver(tmp_path)
    _delete_rows(tmp_path, *deleted)
    before = (_snapshot(tmp_path, SECTIONS), _snapshot(tmp_path, BLOCKS))
    _write_new_year_filing(tmp_path, write_sec_filing)

    result = run_sec_filings_silver(tmp_path)

    assert result.filings.inserted == 1
    assert result.narrative is not None
    assert result.narrative.status == "rejected"
    assert result.narrative.documents_conflicted == 1
    assert result.narrative.documents_inserted == 0
    assert (_snapshot(tmp_path, SECTIONS), _snapshot(tmp_path, BLOCKS)) == before
    assert NEW_YEAR_DOCUMENT not in _rows_by_document(tmp_path, SECTIONS)
    assert NEW_YEAR_DOCUMENT not in _rows_by_document(tmp_path, BLOCKS)


def test_changed_parent_checksum_is_a_narrative_conflict(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    run_sec_filings_silver(tmp_path)
    with open_silver_catalog(tmp_path) as catalog:
        documents = catalog.load_table(DOCUMENTS)
        row = next(
            row
            for row in documents.scan().to_arrow().to_pylist()
            if row["document_id"] == TEN_K_DOCUMENT
        )
        documents.delete(EqualTo("document_id", TEN_K_DOCUMENT))
        changed = DocumentRecord.model_validate({**row, "content_sha256": "0" * 64})
        append_rows(documents, [changed], DOCUMENTS_ARROW_SCHEMA)
    before = (_snapshot(tmp_path, SECTIONS), _snapshot(tmp_path, BLOCKS))

    result = run_sec_filings_silver(tmp_path)

    assert result.documents is not None
    assert result.documents.status == "completed"
    assert result.narrative is not None
    assert result.narrative.status == "rejected"
    assert result.narrative.documents_conflicted == 1
    assert (_snapshot(tmp_path, SECTIONS), _snapshot(tmp_path, BLOCKS)) == before


def test_one_invalid_candidate_blocks_every_new_narrative(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    first = run_sec_filings_silver(tmp_path)
    assert first.narrative is not None
    _write_new_year_filing(tmp_path, write_sec_filing)
    write_sec_filing(
        tmp_path,
        cik=OTHER_CIK,
        form="8-K",
        accession="0000789019-26-000002",
        filed="20260210",
        primary_text='\n<html><body><div style="display:none">x</div></body></html>\n',
    )

    result = run_sec_filings_silver(tmp_path)

    assert result.filings.inserted == 2
    assert result.narrative is not None
    assert result.narrative.status == "rejected"
    assert (result.narrative.documents_failed, result.narrative.documents_inserted) == (
        1,
        0,
    )
    assert _snapshot(tmp_path, SECTIONS) == first.narrative.sections_snapshot_after
    assert _snapshot(tmp_path, BLOCKS) == first.narrative.content_blocks_snapshot_after
    assert set(_rows_by_document(tmp_path, SECTIONS)) == set(EXPECTED_CODES)


def test_document_tables_link_each_table_block_and_reruns_skip(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bronze(tmp_path, write_sec_filing)

    first = run_sec_filings_silver(tmp_path)

    summary = first.document_tables
    assert first.completed
    assert summary is not None
    assert (summary.documents_considered, summary.documents_without_tables) == (3, 1)
    assert (summary.documents_inserted, summary.table_blocks) == (2, TABLE_ROWS)
    assert (summary.tables_written, summary.cells_written) == (TABLE_ROWS, CELL_ROWS)
    assert (summary.documents_conflicted, summary.documents_failed) == (0, 0)
    for identifier, schema in (
        (TABLES, DOCUMENT_TABLES_ARROW_SCHEMA),
        (CELLS, DOCUMENT_TABLE_CELLS_ARROW_SCHEMA),
    ):
        table = _table(tmp_path, identifier)
        assert table.location() == str(
            tmp_path / "silver" / "sec_filings" / identifier.split(".")[1]
        )
        assert _partition_columns(table) == [("filing_year", "year", "filing_date")]
        assert table.scan().to_arrow().schema.equals(schema)

    blocks = _table(tmp_path, BLOCKS).scan().to_arrow().to_pylist()
    tables = _table(tmp_path, TABLES).scan().to_arrow().to_pylist()
    cells = _table(tmp_path, CELLS).scan().to_arrow().to_pylist()
    table_blocks = {b["block_id"]: b for b in blocks if b["block_type"] == "table"}
    # content_blocks 1 -> 0..1 document_tables: exactly the table blocks have one.
    assert sorted(t["block_id"] for t in tables) == sorted(table_blocks)
    for table in tables:
        block = table_blocks[table["block_id"]]
        assert table["source_xpath"] == block["source_xpath"]
        own_cells = sorted(
            (c for c in cells if c["table_id"] == table["table_id"]),
            key=lambda c: c["cell_ordinal"],
        )
        assert len(own_cells) == table["cell_count"] >= 1
        assert block["text"] == "\n".join(c["text"] for c in own_cells if c["text"])
    assert len({c["cell_id"] for c in cells}) == len(cells) == CELL_ROWS
    assert TEN_Q_DOCUMENT not in {t["document_id"] for t in tables}

    def bronze_read(*_: object) -> bytes:
        raise AssertionError("complete table groups must not re-read Bronze")

    monkeypatch.setattr(document_table_extraction, "read_document_content", bronze_read)
    rerun = run_sec_filings_silver(tmp_path)

    assert rerun.completed
    assert rerun.document_tables is not None
    assert (
        rerun.document_tables.documents_skipped,
        rerun.document_tables.documents_without_tables,
        rerun.document_tables.documents_inserted,
    ) == (2, 1, 0)
    assert (_snapshot(tmp_path, TABLES), _snapshot(tmp_path, CELLS)) == (
        summary.tables_snapshot_after,
        summary.cells_snapshot_after,
    )
    assert _table(tmp_path, TABLES).scan().to_arrow().num_rows == TABLE_ROWS
    assert _table(tmp_path, CELLS).scan().to_arrow().num_rows == CELL_ROWS


def test_tables_only_state_resumes_cells_without_new_tables(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bronze(tmp_path, write_sec_filing)

    def append_tables_only(table: Table, rows, arrow_schema) -> None:
        if arrow_schema is DOCUMENT_TABLE_CELLS_ARROW_SCHEMA:
            raise OSError("disk full")
        append_rows(table, rows, arrow_schema)

    with monkeypatch.context() as patch:
        patch.setattr(document_tables_stage, "append_rows", append_tables_only)
        with pytest.raises(OSError, match="disk full"):
            run_sec_filings_silver(tmp_path)
    tables_snapshot = _snapshot(tmp_path, TABLES)
    assert _table(tmp_path, TABLES).scan().to_arrow().num_rows == TABLE_ROWS
    assert _snapshot(tmp_path, CELLS) is None

    resumed = run_sec_filings_silver(tmp_path)

    assert resumed.completed
    assert resumed.document_tables is not None
    assert (
        resumed.document_tables.documents_resumed,
        resumed.document_tables.documents_inserted,
        resumed.document_tables.tables_written,
        resumed.document_tables.cells_written,
    ) == (2, 0, 0, CELL_ROWS)
    assert _snapshot(tmp_path, TABLES) == tables_snapshot
    assert _table(tmp_path, CELLS).scan().to_arrow().num_rows == CELL_ROWS


@pytest.mark.parametrize(
    "deleted",
    [
        (CELLS, "cell_id", f"{TEN_K_DOCUMENT}:block:000003:table:cell:000008"),
        (TABLES, "table_id", f"{TEN_K_DOCUMENT}:block:000011:table"),
        (TABLES, "document_id", EIGHT_K_DOCUMENT),
    ],
    ids=["missing-cell", "missing-table", "cells-without-tables"],
)
def test_incomplete_table_groups_are_conflicts_before_any_write(
    tmp_path: Path, write_sec_filing, deleted: tuple[str, str, str]
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    run_sec_filings_silver(tmp_path)
    _delete_rows(tmp_path, *deleted)
    before = (_snapshot(tmp_path, TABLES), _snapshot(tmp_path, CELLS))
    _write_new_year_filing(tmp_path, write_sec_filing)

    result = run_sec_filings_silver(tmp_path)

    assert result.narrative is not None
    assert result.narrative.documents_inserted == 1
    assert result.document_tables is not None
    assert result.document_tables.status == "rejected"
    assert result.document_tables.documents_conflicted == 1
    assert (_snapshot(tmp_path, TABLES), _snapshot(tmp_path, CELLS)) == before
    assert NEW_YEAR_DOCUMENT not in _rows_by_document(tmp_path, TABLES)


def test_stored_table_source_hash_mismatch_is_a_conflict(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    run_sec_filings_silver(tmp_path)
    with open_silver_catalog(tmp_path) as catalog:
        tables = catalog.load_table(TABLES)
        row = next(
            row
            for row in tables.scan().to_arrow().to_pylist()
            if row["document_id"] == EIGHT_K_DOCUMENT
        )
        tables.delete(EqualTo("table_id", row["table_id"]))
        changed = DocumentTableRecord.model_validate(
            {**row, "source_document_sha256": "0" * 64}
        )
        append_rows(tables, [changed], DOCUMENT_TABLES_ARROW_SCHEMA)
    before = (_snapshot(tmp_path, TABLES), _snapshot(tmp_path, CELLS))

    result = run_sec_filings_silver(tmp_path)

    assert result.document_tables is not None
    assert result.document_tables.status == "rejected"
    assert result.document_tables.documents_conflicted == 1
    assert (_snapshot(tmp_path, TABLES), _snapshot(tmp_path, CELLS)) == before


def test_persisted_table_cell_blocks_stop_the_pipeline_without_writes(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    run_sec_filings_silver(tmp_path)
    with open_silver_catalog(tmp_path) as catalog:
        blocks = catalog.load_table(BLOCKS)
        row = blocks.scan().to_arrow().to_pylist()[0]
        # A row in the retired one-block-per-cell layout, written as raw Arrow
        # because the current contract no longer allows that block type.
        legacy = {**row, "block_id": f"{row['block_id']}x", "block_type": "table_cell"}
        blocks.append(
            pa.Table.from_pylist([legacy], schema=CONTENT_BLOCKS_ARROW_SCHEMA)
        )
    before = {
        name: _snapshot(tmp_path, name) for name in (SECTIONS, BLOCKS, TABLES, CELLS)
    }
    _write_new_year_filing(tmp_path, write_sec_filing)

    result = run_sec_filings_silver(tmp_path)

    assert not result.completed
    assert result.narrative is not None
    assert result.narrative.status == "rejected"
    assert result.narrative.legacy_table_cell_blocks == 1
    assert result.narrative.documents_inserted == 0
    assert result.document_tables is None
    assert {name: _snapshot(tmp_path, name) for name in before} == before


def test_document_links_reference_source_and_target_documents_and_reruns_skip(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bronze(tmp_path, write_sec_filing)

    first = run_sec_filings_silver(tmp_path)

    summary = first.document_links
    assert first.completed
    assert summary is not None
    assert (
        summary.documents_considered,
        summary.documents_inserted,
        summary.documents_without_links,
        summary.links_inserted,
    ) == (HTML_DOCUMENTS, 2, 7, LINK_ROWS)
    assert (
        summary.links_resolved,
        summary.links_unresolved,
        summary.links_ambiguous,
        summary.links_not_applicable,
        summary.links_external,
    ) == (8, 2, 0, 1, 1)
    table = _table(tmp_path, LINKS)
    assert table.location() == str(
        tmp_path / "silver" / "sec_filings" / "document_links"
    )
    assert _partition_columns(table) == [("filing_year", "year", "filing_date")]
    arrow = table.scan().to_arrow()
    assert arrow.schema.equals(DOCUMENT_LINKS_ARROW_SCHEMA)
    assert "target_exhibit_id" not in arrow.schema.names
    links = arrow.to_pylist()
    documents = {
        row["document_id"]: row
        for row in _table(tmp_path, DOCUMENTS).scan().to_arrow().to_pylist()
    }
    assert len({link["link_id"] for link in links}) == len(links) == LINK_ROWS
    # Both references point at documents: every link has one HTML source, and a
    # resolved target is a document of the same filing.
    for link in links:
        source = documents[link["source_document_id"]]
        assert source["content_kind"] == "html"
        if link["target_document_id"] is not None:
            target = documents[link["target_document_id"]]
            assert target["filing_id"] == source["filing_id"] == link["filing_id"]
    assert Counter(link["source_document_id"] for link in links) == {
        TEN_K_DOCUMENT: 8,
        EIGHT_K_DOCUMENT: 3,
    }
    assert Counter(link["target_document_id"] for link in links) == {
        TEN_K_DOCUMENT: 6,
        TEN_K_EXHIBIT: 1,
        EIGHT_K_EXHIBIT: 1,
        None: 3,
    }
    by_href = {link["href"]: link for link in links}
    expected = {
        "#i1a": ("fragment", "resolved", TEN_K_DOCUMENT, "Item 1A."),
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/ex31.htm": (
            "sec_url", "resolved", TEN_K_EXHIBIT, "Example Corp",
        ),
        "https://www.example.com/products": (
            "external_url", "not_applicable", None, "widgets",
        ),
        "ex31.htm": ("relative_document", "resolved", EIGHT_K_EXHIBIT, "99.1"),
        "press-release.htm": (
            "relative_document", "unresolved", None, "Press release",
        ),
        "https://www.sec.gov/Archives/edgar/data/789019/000078901924000001/ex31.htm": (
            "sec_url", "unresolved", None, "results",
        ),
    }  # fmt: skip
    assert {
        href: (
            by_href[href]["link_kind"],
            by_href[href]["resolution_status"],
            by_href[href]["target_document_id"],
            by_href[href]["link_text"],
        )
        for href in expected
    } == expected
    assert {documents[TEN_K_EXHIBIT]["role"], documents[EIGHT_K_EXHIBIT]["role"]} == {
        "exhibit"
    }

    read: list[str] = []
    read_content = document_link_extraction.read_document_content

    def recording_read(parent: DocumentRecord, data_dir: Path) -> bytes:
        read.append(parent.document_id)
        return read_content(parent, data_dir)

    monkeypatch.setattr(
        document_link_extraction, "read_document_content", recording_read
    )
    rerun = run_sec_filings_silver(tmp_path)

    assert rerun.document_links is not None
    assert (
        rerun.document_links.documents_skipped,
        rerun.document_links.documents_without_links,
        rerun.document_links.documents_inserted,
    ) == (2, 7, 0)
    # Complete groups are not re-read; documents without links are parsed again.
    assert len(read) == 7
    assert {TEN_K_DOCUMENT, EIGHT_K_DOCUMENT}.isdisjoint(read)
    assert _snapshot(tmp_path, LINKS) == summary.snapshot_after
    assert _table(tmp_path, LINKS).scan().to_arrow().num_rows == LINK_ROWS


@pytest.mark.parametrize(
    "changes",
    [None, {"source_document_sha256": "0" * 64}],
    ids=["missing-link", "changed-source-hash"],
)
def test_incomplete_or_changed_link_groups_are_conflicts_before_any_write(
    tmp_path: Path, write_sec_filing, changes: dict[str, str] | None
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    run_sec_filings_silver(tmp_path)
    with open_silver_catalog(tmp_path) as catalog:
        links = catalog.load_table(LINKS)
        row = next(
            row
            for row in links.scan().to_arrow().to_pylist()
            if row["source_document_id"] == EIGHT_K_DOCUMENT
        )
        links.delete(EqualTo("link_id", row["link_id"]))
        if changes is not None:
            changed = DocumentLinkRecord.model_validate({**row, **changes})
            append_rows(links, [changed], DOCUMENT_LINKS_ARROW_SCHEMA)
    before = _snapshot(tmp_path, LINKS)
    _write_new_year_filing(tmp_path, write_sec_filing)

    result = run_sec_filings_silver(tmp_path)

    assert result.document_tables is not None
    assert result.document_tables.documents_inserted == 1
    assert result.document_links is not None
    assert result.document_links.status == "rejected"
    assert result.document_links.documents_conflicted == 1
    assert result.document_links.links_inserted == 0
    assert _snapshot(tmp_path, LINKS) == before
    sources = {
        row["source_document_id"]
        for row in _table(tmp_path, LINKS).scan().to_arrow().to_pylist()
    }
    assert NEW_YEAR_DOCUMENT not in sources


def _instance_id(document_id: str) -> str:
    return f"{document_id.rsplit(':document:', 1)[0]}:xbrl:inline:default"


def _xbrl_snapshots(data_dir: Path) -> tuple[int | None, ...]:
    return tuple(_snapshot(data_dir, table) for table in (INSTANCES, CONTEXTS, UNITS))


def test_xbrl_core_builds_instances_contexts_and_units_and_reruns_skip(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bronze(tmp_path, write_sec_filing)

    first = run_sec_filings_silver(tmp_path)

    summary = first.xbrl_core
    assert first.completed
    assert summary is not None
    assert (summary.filings_considered, summary.filings_inserted) == (3, 3)
    assert (summary.instances_inserted, summary.contexts_inserted) == (3, 6)
    assert (summary.units_inserted, summary.inline_instances) == (5, 3)
    assert (summary.default_targets, summary.xml_only_instances) == (3, 0)
    assert (summary.instant_contexts, summary.duration_contexts) == (2, 4)
    assert (summary.simple_units, summary.divide_units) == (4, 1)
    for identifier, schema in XBRL_CORE_SCHEMAS.items():
        table = _table(tmp_path, identifier)
        assert table.location() == str(
            tmp_path / "silver" / "sec_filings" / identifier.split(".")[1]
        )
        assert _partition_columns(table) == []
        assert table.scan().to_arrow().schema.equals(schema)

    instances = {
        row["instance_id"]: row
        for row in _table(tmp_path, INSTANCES).scan().to_arrow().to_pylist()
    }
    contexts = Counter(
        _table(tmp_path, CONTEXTS).scan().to_arrow()["instance_id"].to_pylist()
    )
    units = Counter(
        _table(tmp_path, UNITS).scan().to_arrow()["instance_id"].to_pylist()
    )
    assert set(instances) == {_instance_id(document) for document in XBRL_COUNTS}
    assert set(contexts) | set(units) <= set(instances)
    for document, (facts, context_count, unit_count) in XBRL_COUNTS.items():
        instance = instances[_instance_id(document)]
        generated_xml = f"{document[:-4]}0004"
        assert instance["source_document_ids"] == [document, generated_xml]
        assert instance["schema_refs"] == ["ex-2024.xsd"]
        assert (instance["fact_count"], instance["context_count"]) == (
            facts,
            context_count,
        )
        assert instance["unit_count"] == unit_count
        assert (contexts[instance["instance_id"]], units[instance["instance_id"]]) == (
            context_count,
            unit_count,
        )

    def bronze_read(*_: object) -> bytes:
        raise AssertionError("complete XBRL groups must not re-read Bronze")

    monkeypatch.setattr(xbrl_core_extraction, "read_document_content", bronze_read)
    rerun = run_sec_filings_silver(tmp_path)

    assert rerun.xbrl_core is not None
    assert (rerun.xbrl_core.filings_skipped, rerun.xbrl_core.instances_inserted) == (
        3,
        0,
    )
    assert _xbrl_snapshots(tmp_path) == (
        summary.instances_snapshot_after,
        summary.contexts_snapshot_after,
        summary.units_snapshot_after,
    )


@pytest.mark.parametrize("failing", [CONTEXTS, UNITS])
def test_xbrl_core_recovers_child_groups_after_a_failed_commit(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    _write_bronze(tmp_path, write_sec_filing)

    def fail_on(table: Table, rows, arrow_schema) -> None:
        if arrow_schema is XBRL_CORE_SCHEMAS[failing]:
            raise OSError("disk full")
        append_rows(table, rows, arrow_schema)

    with monkeypatch.context() as patch:
        patch.setattr(xbrl_core_stage, "append_rows", fail_on)
        with pytest.raises(OSError, match="disk full"):
            run_sec_filings_silver(tmp_path)
    committed = _xbrl_snapshots(tmp_path)

    resumed = run_sec_filings_silver(tmp_path)

    summary = resumed.xbrl_core
    assert resumed.completed
    assert summary is not None
    assert (summary.instances_inserted, summary.units_inserted) == (0, 5)
    if failing == CONTEXTS:
        # Every instance has contexts; the 8-K has no unit, so nothing to recover.
        assert (summary.filings_recovered, summary.contexts_inserted) == (3, 6)
        assert _xbrl_snapshots(tmp_path)[0] == committed[0]
    else:
        assert (summary.filings_recovered, summary.filings_skipped) == (2, 1)
        assert summary.contexts_inserted == 0
        assert _xbrl_snapshots(tmp_path)[:2] == committed[:2]
    assert _table(tmp_path, CONTEXTS).scan().to_arrow().num_rows == 6
    assert _table(tmp_path, UNITS).scan().to_arrow().num_rows == 5


def _replace_instance(data_dir: Path, document: str, **changes: object) -> None:
    with open_silver_catalog(data_dir) as catalog:
        table = catalog.load_table(INSTANCES)
        instance_id = _instance_id(document)
        row = next(
            row
            for row in table.scan().to_arrow().to_pylist()
            if row["instance_id"] == instance_id
        )
        table.delete(EqualTo("instance_id", instance_id))
        if changes:
            changed = XbrlInstanceRecord.model_validate(
                {**row, "namespace_map": dict(row["namespace_map"]), **changes}
            )
            append_rows(table, [changed], XBRL_CORE_SCHEMAS[INSTANCES])


@pytest.mark.parametrize(
    "damage",
    [
        "partial-contexts",
        "children-without-instance",
        "source-set-changed",
        "bad-new-filing",
    ],
)
def test_xbrl_core_rejects_the_batch_before_any_write(
    tmp_path: Path, write_sec_filing, damage: str
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    run_sec_filings_silver(tmp_path)
    if damage == "partial-contexts":
        context = next(iter(_table(tmp_path, CONTEXTS).scan().to_arrow().to_pylist()))
        _delete_rows(tmp_path, CONTEXTS, "context_key", context["context_key"])
    elif damage == "children-without-instance":
        _replace_instance(tmp_path, EIGHT_K_DOCUMENT)
    elif damage == "source-set-changed":
        _replace_instance(tmp_path, TEN_Q_DOCUMENT, source_set_sha256="0" * 64)
    before = _xbrl_snapshots(tmp_path)
    write_sec_filing(
        tmp_path,
        cik=CIK,
        form="8-K",
        accession="0000320193-26-000001",
        filed="20260105",
        # Names the Inline XBRL namespace but is not well-formed XML.
        primary_text=None
        if damage != "bad-new-filing"
        else '\n<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><body>'
        "<div>Item 2.02 Results.</div><p>open</body></html>\n",
    )

    result = run_sec_filings_silver(tmp_path)

    assert result.xbrl_core is not None
    assert result.xbrl_core.status == "rejected"
    assert result.xbrl_core.instances_inserted == 0
    if damage == "bad-new-filing":
        assert result.xbrl_core.filings_failed == 1
    else:
        assert result.xbrl_core.filings_conflicted >= 1
    assert _xbrl_snapshots(tmp_path) == before
    stored = _table(tmp_path, INSTANCES).scan().to_arrow()["instance_id"].to_pylist()
    assert _instance_id(NEW_YEAR_DOCUMENT) not in stored


def test_filing_without_xbrl_gets_no_instance_and_reruns_quietly(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    write_sec_filing(
        tmp_path,
        cik=CIK,
        form="8-K",
        accession="0000320193-26-000001",
        filed="20260105",
        primary_text="\n<html><body><div>Item 2.02 Results.</div></body></html>\n",
        xbrl_instance=False,
    )

    first = run_sec_filings_silver(tmp_path)
    rerun = run_sec_filings_silver(tmp_path)

    assert first.completed and rerun.completed
    assert first.xbrl_core is not None and rerun.xbrl_core is not None
    assert (first.xbrl_core.filings_without_xbrl, first.xbrl_core.filings_inserted) == (
        1,
        3,
    )
    assert (rerun.xbrl_core.filings_without_xbrl, rerun.xbrl_core.filings_skipped) == (
        1,
        3,
    )
    assert _xbrl_snapshots(tmp_path)[0] == first.xbrl_core.instances_snapshot_after
    stored = _table(tmp_path, INSTANCES).scan().to_arrow()["filing_id"].to_pylist()
    assert "sec:0000320193-26-000001" not in stored


CONCEPTS = "sec_filings.xbrl_concepts"
FACTS = "sec_filings.xbrl_facts"
EX = "http://example.com/2024"
# The fixture's extension schema defines ten items and the typed domain
# ex:Region, and imports the dimension schema, whose two abstract items are
# standard concepts. The 10-K has four hidden facts, its product in a
# paragraph and its revenue again in a table cell; the 10-Q and 8-K have two
# hidden facts each.
EXTENSION_CONCEPTS, STANDARD_CONCEPTS, FACT_ROWS = 11, 2, 10
DEI_SCHEMA_URL = "https://xbrl.sec.gov/dei/2024/dei-2024.xsd"
# A standard taxonomy schema as the processor caches it after a download.
DEI_SCHEMA = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" '
    'xmlns:xbrli="http://www.xbrl.org/2003/instance" '
    'targetNamespace="http://xbrl.sec.gov/dei/2024" elementFormDefault="qualified">'
    '<xs:import namespace="http://www.xbrl.org/2003/instance" '
    'schemaLocation="http://www.xbrl.org/2003/xbrl-instance-2003-12-31.xsd"/>'
    '<xs:element id="dei_DocumentType" name="DocumentType" '
    'type="xbrli:stringItemType" substitutionGroup="xbrli:item" '
    'xbrli:periodType="duration" nillable="true"/></xs:schema>\n'
)


def _inline_primary(schema_href: str, body: str) -> str:
    """An 8-K primary document with one context and no unit."""

    return (
        "\n<XBRL>\n<?xml version='1.0' encoding='ASCII'?>\n"
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:ix="http://www.xbrl.org/2013/inlineXBRL" '
        'xmlns:xbrli="http://www.xbrl.org/2003/instance" '
        'xmlns:link="http://www.xbrl.org/2003/linkbase" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xmlns:dei="http://xbrl.sec.gov/dei/2024"><head><title>8-K</title></head>'
        '<body><div style="display: none"><ix:header><ix:references>'
        f'<link:schemaRef xlink:type="simple" xlink:href="{schema_href}"/>'
        '</ix:references><ix:resources><xbrli:context id="D"><xbrli:entity>'
        '<xbrli:identifier scheme="http://www.sec.gov/CIK">0000320193'
        "</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:startDate>"
        "2026-01-05</xbrli:startDate><xbrli:endDate>2026-01-05</xbrli:endDate>"
        "</xbrli:period></xbrli:context></ix:resources></ix:header></div>"
        f"<div>Item 2.02 Results of Operations.</div>{body}</body></html>\n</XBRL>\n"
    )


def _write_dei_filing(data_dir: Path, write_sec_filing) -> None:
    """A new 8-K whose one fact is defined only by the standard dei taxonomy."""

    write_sec_filing(
        data_dir,
        cik=CIK,
        form="8-K",
        accession="0000320193-26-000001",
        filed="20260105",
        primary_text=_inline_primary(
            DEI_SCHEMA_URL,
            '<div>Form <ix:nonNumeric name="dei:DocumentType" contextRef="D">'
            "8-K</ix:nonNumeric>.</div>",
        ),
        xbrl_instance=False,
    )


def _warm_dei_cache(data_dir: Path) -> None:
    cached = data_dir / "_cache" / "arelle" / "https" / "xbrl.sec.gov" / "dei"
    cached.mkdir(parents=True)
    (cached / "2024" / "dei-2024.xsd").parent.mkdir()
    (cached / "2024" / "dei-2024.xsd").write_text(DEI_SCHEMA)


def _facts_snapshots(data_dir: Path) -> tuple[int | None, int | None]:
    return _snapshot(data_dir, CONCEPTS), _snapshot(data_dir, FACTS)


def _rows(data_dir: Path, identifier: str) -> list[dict]:
    return _table(data_dir, identifier).scan().to_arrow().to_pylist()


def test_xbrl_facts_build_concepts_and_fact_occurrences_and_reruns_skip(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bronze(tmp_path, write_sec_filing)

    first = run_sec_filings_silver(tmp_path)

    summary = first.xbrl_facts
    assert first.completed
    assert summary is not None
    assert {key: asdict(summary)[key] for key in EXPECTED_FIRST_FACTS_RUN} == (
        EXPECTED_FIRST_FACTS_RUN
    )
    concepts_table, facts_table = _table(tmp_path, CONCEPTS), _table(tmp_path, FACTS)
    for table, schema in (
        (concepts_table, XBRL_CONCEPTS_ARROW_SCHEMA),
        (facts_table, XBRL_FACTS_ARROW_SCHEMA),
    ):
        assert table.location() == str(
            tmp_path / "silver" / "sec_filings" / table.name()[-1]
        )
        assert table.scan().to_arrow().schema.equals(schema)
    assert _partition_columns(concepts_table) == []
    assert _partition_columns(facts_table) == [
        ("instance_bucket", "bucket[32]", "instance_id")
    ]

    instances = {row["instance_id"]: row for row in _rows(tmp_path, INSTANCES)}
    concepts = {row["concept_id"]: row for row in _rows(tmp_path, CONCEPTS)}
    facts = _rows(tmp_path, FACTS)
    contexts = {row["context_key"] for row in _rows(tmp_path, CONTEXTS)}
    units = {row["unit_key"] for row in _rows(tmp_path, UNITS)}
    # Every instance loaded the same extension schema; each concept is stored once.
    assert len(concepts) == len(concepts_table.scan().to_arrow()) == 13
    assert Counter(fact["instance_id"] for fact in facts) == {
        instance_id: row["fact_count"] for instance_id, row in instances.items()
    }
    assert len({fact["fact_id"] for fact in facts}) == len(facts) == FACT_ROWS
    for fact in facts:
        XbrlFactRecord.model_validate(fact)  # recomputes the fact hash
        assert fact["concept_id"] in concepts
        assert fact["context_key"] in contexts
        if fact["fact_kind"] == "non_numeric":
            assert fact["unit_key"] is None
        else:
            assert fact["unit_key"] in units
    revenue = concepts[concept_id_for(EX, "Revenue")]
    assert (revenue["is_extension"], revenue["source_schema_uri"]) == (
        True,
        "ex-2024.xsd",
    )
    assert (revenue["data_type"], revenue["balance"]) == (
        "{http://www.xbrl.org/2003/instance}monetaryItemType",
        "credit",
    )

    # The hidden revenue fact and its formatted repetition in a table cell stay
    # two occurrences; the visible facts point at their cell and block.
    ten_k = sorted(
        (f for f in facts if f["instance_id"] == _instance_id(TEN_K_DOCUMENT)),
        key=lambda f: f["fact_order"],
    )
    revenues = [f for f in ten_k if f["concept_id"] == revenue["concept_id"]]
    assert [
        (f["is_hidden"], f["raw_value"], f["numeric_value"], f["block_id"])
        for f in revenues
    ] == [(True, "1000", "1000", None), (False, "1,000", "1000", None)]
    cell = next(
        row
        for row in _rows(tmp_path, CELLS)
        if row["cell_id"] == revenues[1]["cell_id"]
    )
    assert (cell["document_id"], cell["text"]) == (TEN_K_DOCUMENT, "$1,000")
    product = next(f for f in ten_k if f["concept_id"] == concept_id_for(EX, "Product"))
    block = next(
        row for row in _rows(tmp_path, BLOCKS) if row["block_id"] == product["block_id"]
    )
    assert (block["document_id"], block["text"]) == (TEN_K_DOCUMENT, "We make widgets.")
    assert product["cell_id"] is None

    def reload(*_: object) -> None:
        raise AssertionError("complete fact groups must not reload XBRL")

    monkeypatch.setattr(xbrl_facts_extraction, "load_instance", reload)
    rerun = run_sec_filings_silver(tmp_path)

    assert rerun.completed
    assert rerun.xbrl_facts is not None
    assert (rerun.xbrl_facts.instances_skipped, rerun.xbrl_facts.facts_inserted) == (
        3,
        0,
    )
    assert _facts_snapshots(tmp_path) == (
        summary.concepts_snapshot_after,
        summary.facts_snapshot_after,
    )
    assert len(_rows(tmp_path, FACTS)) == FACT_ROWS


EXPECTED_FIRST_FACTS_RUN = {
    "status": "completed",
    "instances_considered": 3,
    "instances_inserted": 3,
    "instances_skipped": 0,
    "instances_recovered": 0,
    "concepts_discovered": EXTENSION_CONCEPTS + STANDARD_CONCEPTS,
    "concepts_inserted": EXTENSION_CONCEPTS + STANDARD_CONCEPTS,
    "standard_concepts": STANDARD_CONCEPTS,
    "extension_concepts": EXTENSION_CONCEPTS,
    "facts_inserted": FACT_ROWS,
    "numeric_facts": 5,
    "nonnumeric_facts": 5,
    "fraction_facts": 0,
    "nil_facts": 0,
    "hidden_facts": 8,
    "continued_facts": 0,
    "aligned_block_facts": 1,
    "aligned_cell_facts": 1,
    "unaligned_facts": 8,
    "normalization_warnings": 0,
}


def test_new_instance_appends_only_its_new_concepts_and_fact_group(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    first = run_sec_filings_silver(tmp_path).xbrl_facts
    stored = {row["fact_id"]: row for row in _rows(tmp_path, FACTS)}
    _write_dei_filing(tmp_path, write_sec_filing)
    _warm_dei_cache(tmp_path)

    result = run_sec_filings_silver(tmp_path)

    summary = result.xbrl_facts
    assert result.completed
    assert first is not None and summary is not None
    assert (summary.instances_inserted, summary.instances_skipped) == (1, 3)
    assert (summary.concepts_inserted, summary.standard_concepts) == (1, 1)
    assert summary.facts_inserted == 1
    for table, before in (
        (CONCEPTS, first.concepts_snapshot_after),
        (FACTS, first.facts_snapshot_after),
    ):
        assert _table(tmp_path, table).current_snapshot().parent_snapshot_id == before
    facts = {row["fact_id"]: row for row in _rows(tmp_path, FACTS)}
    assert {i: facts[i] for i in stored} == stored
    [new_fact] = [row for i, row in facts.items() if i not in stored]
    dei = next(
        row
        for row in _rows(tmp_path, CONCEPTS)
        if row["concept_id"] == new_fact["concept_id"]
    )
    assert (dei["namespace_uri"], dei["local_name"]) == (
        "http://xbrl.sec.gov/dei/2024",
        "DocumentType",
    )
    assert (dei["is_extension"], dei["source_schema_uri"]) == (False, DEI_SCHEMA_URL)
    assert (new_fact["raw_value"], new_fact["is_hidden"]) == ("8-K", False)
    assert new_fact["block_id"] is not None


def test_taxonomy_cache_miss_rejects_the_batch_until_the_cache_is_warm(
    tmp_path: Path, write_sec_filing, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    _write_dei_filing(tmp_path, write_sec_filing)

    offline = run_sec_filings_silver(tmp_path)

    assert offline.xbrl_facts is not None
    assert offline.xbrl_facts.status == "rejected"
    assert (offline.xbrl_facts.instances_failed, offline.xbrl_facts.facts_inserted) == (
        1,
        0,
    )
    assert _facts_snapshots(tmp_path) == (None, None)
    output = capsys.readouterr().out
    assert "xbrl_taxonomy_cache_miss" in output
    assert DEI_SCHEMA_URL in output
    assert not any((tmp_path / "_cache" / "arelle").rglob("*.xsd"))

    _warm_dei_cache(tmp_path)
    warm = run_sec_filings_silver(tmp_path)

    assert warm.completed
    assert warm.xbrl_facts is not None
    assert (warm.xbrl_facts.instances_inserted, warm.xbrl_facts.facts_inserted) == (
        4,
        FACT_ROWS + 1,
    )


def test_concepts_only_commit_is_recovered_without_new_concepts(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bronze(tmp_path, write_sec_filing)

    def fail_facts(table: Table, rows, arrow_schema) -> None:
        if arrow_schema is XBRL_FACTS_ARROW_SCHEMA:
            raise OSError("disk full")
        append_rows(table, rows, arrow_schema)

    with monkeypatch.context() as patch:
        patch.setattr(xbrl_facts_stage, "append_rows", fail_facts)
        with pytest.raises(OSError, match="disk full"):
            run_sec_filings_silver(tmp_path)
    concepts_snapshot, facts_snapshot = _facts_snapshots(tmp_path)
    assert concepts_snapshot is not None and facts_snapshot is None

    resumed = run_sec_filings_silver(tmp_path)

    summary = resumed.xbrl_facts
    assert resumed.completed
    assert summary is not None
    assert (summary.instances_recovered, summary.instances_inserted) == (3, 0)
    assert (summary.concepts_inserted, summary.concepts_skipped) == (0, 13)
    assert summary.facts_inserted == FACT_ROWS
    assert _snapshot(tmp_path, CONCEPTS) == concepts_snapshot
    assert len(_rows(tmp_path, CONCEPTS)) == 13

    # A later filing of the same extension taxonomy is new, not recovered.
    _write_new_year_filing(tmp_path, write_sec_filing)
    later = run_sec_filings_silver(tmp_path).xbrl_facts

    assert later is not None
    assert (later.instances_inserted, later.instances_recovered) == (1, 0)


def _change_stored_concept(data_dir: Path, local_name: str) -> None:
    """Store another, self-consistent definition under the concept's ID."""

    with open_silver_catalog(data_dir) as catalog:
        table = catalog.load_table(CONCEPTS)
        concept_id = concept_id_for(EX, local_name)
        row = next(
            r
            for r in table.scan().to_arrow().to_pylist()
            if r["concept_id"] == concept_id
        )
        table.delete(EqualTo("concept_id", concept_id))
        changed = {**row, "balance": "debit"}
        definition = ConceptDefinition(
            **{f.name: changed[f.name] for f in fields(ConceptDefinition)}
        )
        changed["concept_definition_sha256"] = definition.sha256()
        append_rows(
            table,
            [XbrlConceptRecord.model_validate(changed)],
            XBRL_CONCEPTS_ARROW_SCHEMA,
        )


@pytest.mark.parametrize(
    ("damage", "event"),
    [
        ("partial-fact-group", "xbrl_instance_facts_conflict"),
        ("concept-definition-changed", "xbrl_instance_facts_conflict"),
        ("source-set-changed", "xbrl_instance_facts_conflict"),
        ("bronze-slice-changed", "xbrl_facts_instance_failed"),
        ("xbrl-core-fact-count", "xbrl_core_semantics_incompatible"),
    ],
)
def test_xbrl_facts_reject_the_batch_before_any_write(
    tmp_path: Path,
    write_sec_filing,
    capsys: pytest.CaptureFixture[str],
    damage: str,
    event: str,
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    run_sec_filings_silver(tmp_path)
    ten_q = _instance_id(TEN_Q_DOCUMENT)
    if damage == "partial-fact-group":
        _delete_rows(tmp_path, FACTS, "fact_id", f"{ten_q}:fact:00000002")
    else:
        # The 10-Q must be extracted again to meet the damage.
        _delete_rows(tmp_path, FACTS, "instance_id", ten_q)
    if damage == "concept-definition-changed":
        _change_stored_concept(tmp_path, "Revenue")
    elif damage == "source-set-changed":
        _replace_instance(tmp_path, TEN_Q_DOCUMENT, source_set_sha256="0" * 64)
    elif damage == "bronze-slice-changed":
        submission = next(
            tmp_path.glob("bronze/**/0000320193-25-000057/full-submission.txt")
        )
        text = submission.read_text()
        submission.write_text(text.replace("Net income rose.", "Net income fell."))
    elif damage == "xbrl-core-fact-count":
        _replace_instance(tmp_path, TEN_Q_DOCUMENT, fact_count=3)
    before = _facts_snapshots(tmp_path)
    capsys.readouterr()

    summary = xbrl_facts_stage.run_xbrl_facts_stage(
        tmp_path,
        datetime.now(UTC),
        ProcessorSettings(
            cache_dir=tmp_path / "_cache" / "arelle",
            allow_network=False,
            http_user_agent=None,
        ),
    )

    assert summary.status == "rejected"
    assert (summary.instances_conflicted + summary.instances_failed) == 1
    assert (summary.concepts_inserted, summary.facts_inserted) == (0, 0)
    assert _facts_snapshots(tmp_path) == before
    assert event in capsys.readouterr().out


def test_instance_without_facts_is_a_complete_empty_group(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    write_sec_filing(
        tmp_path,
        cik=CIK,
        form="8-K",
        accession="0000320193-26-000001",
        filed="20260105",
        primary_text=_inline_primary("ex-2024.xsd", ""),
        xbrl_instance=False,
    )

    first = run_sec_filings_silver(tmp_path)

    assert first.completed
    assert first.xbrl_facts is not None
    assert (
        first.xbrl_facts.instances_skipped,
        first.xbrl_facts.instances_inserted,
    ) == (
        1,
        3,
    )
    instance = _instance_id(NEW_YEAR_DOCUMENT)
    assert {
        row["instance_id"]: row["fact_count"] for row in _rows(tmp_path, INSTANCES)
    }[instance] == 0
    assert instance not in {row["instance_id"] for row in _rows(tmp_path, FACTS)}


DIMENSIONS = "sec_filings.xbrl_dimensions"
FOOTNOTES = "sec_filings.xbrl_fact_footnotes"
TEN_K_INSTANCE = _instance_id(TEN_K_DOCUMENT)
# The 10-K's FY_Widgets context reports an explicit ProductAxis and a typed
# RegionAxis; its hidden revenue (fact 2) and table-cell revenue (fact 6)
# share a footnote with markup, and fact 6 has a second, plain footnote.
HIDDEN_REVENUE = f"{TEN_K_INSTANCE}:fact:00000002"
CELL_REVENUE = f"{TEN_K_INSTANCE}:fact:00000006"
FOOTNOTE_ROLE = "http://www.xbrl.org/2003/role/footnote"
FACT_FOOTNOTE = "http://www.xbrl.org/2003/arcrole/fact-footnote"
STANDARD_LINK = "http://www.xbrl.org/2003/role/link"


def _enrichment_snapshots(data_dir: Path) -> tuple[int | None, int | None]:
    return _snapshot(data_dir, DIMENSIONS), _snapshot(data_dir, FOOTNOTES)


def _settings(data_dir: Path) -> ProcessorSettings:
    return ProcessorSettings(
        cache_dir=data_dir / "_cache" / "arelle",
        allow_network=False,
        http_user_agent=None,
    )


def _assert_fixture_dimensions(data_dir: Path) -> None:
    """The 10-K FY_Widgets context: explicit ProductAxis, typed RegionAxis."""

    contexts = {row["context_key"]: row for row in _rows(data_dir, CONTEXTS)}
    concepts = {row["concept_id"] for row in _rows(data_dir, CONCEPTS)}
    dimensions = sorted(_rows(data_dir, DIMENSIONS), key=lambda d: d["dimension_id"])
    assert len({d["dimension_id"] for d in dimensions}) == len(dimensions) == 2
    explicit, typed = dimensions
    assert contexts[explicit["context_key"]]["context_ref"] == "FY_Widgets"
    assert {explicit["context_key"], typed["context_key"]} == {explicit["context_key"]}
    assert [
        (d["dimension_ordinal"], d["context_dimension_count"]) for d in dimensions
    ] == [(1, 2), (2, 2)]
    assert (explicit["member_type"], explicit["context_element"]) == (
        "explicit",
        "segment",
    )
    assert (explicit["dimension_concept_id"], explicit["member_concept_id"]) == (
        concept_id_for(EX, "ProductAxis"),
        concept_id_for(EX, "WidgetsMember"),
    )
    assert (typed["member_type"], typed["member_concept_id"]) == ("typed", None)
    assert typed["dimension_concept_id"] == concept_id_for(EX, "RegionAxis")
    assert (typed["typed_member_qname"], typed["typed_member_value"]) == (
        f"{{{EX}}}Region",
        "North",
    )
    assert typed["typed_member_xml"] == (
        '<ns0:Region xmlns:ns0="http://example.com/2024">North</ns0:Region>'
    )
    assert {explicit["dimension_concept_id"], explicit["member_concept_id"]} <= concepts


def _assert_fixture_footnotes(data_dir: Path) -> None:
    """The 10-K revenues share a footnote; the cell revenue has a second one."""

    footnotes = _rows(data_dir, FOOTNOTES)
    facts = {row["fact_id"] for row in _rows(data_dir, FACTS)}
    assert {f["fact_id"] for f in footnotes} <= facts
    assert len({f["footnote_relation_id"] for f in footnotes}) == len(footnotes) == 3
    by_fact = {
        fid: sorted(
            (f for f in footnotes if f["fact_id"] == fid),
            key=lambda f: f["footnote_ordinal"],
        )
        for fid in (HIDDEN_REVENUE, CELL_REVENUE)
    }
    assert [f["fact_footnote_count"] for f in by_fact[CELL_REVENUE]] == [2, 2]
    assert [f["footnote_ordinal"] for f in by_fact[CELL_REVENUE]] == [1, 2]
    [shared] = by_fact[HIDDEN_REVENUE]
    cell_shared = next(
        f for f in by_fact[CELL_REVENUE] if f["role_uri"] == FOOTNOTE_ROLE
    )
    estimate = next(f for f in by_fact[CELL_REVENUE] if f is not cell_shared)
    # One resource noting two facts keeps one ID and one content hash.
    assert shared["footnote_resource_id"] == cell_shared["footnote_resource_id"]
    assert shared["footnote_content_sha256"] == cell_shared["footnote_content_sha256"]
    assert (shared["footnote_text"], shared["footnote_html"]) == (
        "Excludes discontinued operations.",
        "Excludes <b>discontinued</b>\u00a0operations.",
    )
    assert (shared["language"], shared["resource_label"]) == ("en-us", "fn-revenue")
    assert (shared["arcrole_uri"], shared["linkrole_uri"]) == (
        FACT_FOOTNOTE,
        STANDARD_LINK,
    )
    assert (
        estimate["footnote_text"],
        estimate["footnote_html"],
        estimate["language"],
    ) == (
        "Estimated.",
        None,
        None,
    )
    assert (estimate["role_uri"], estimate["linkrole_uri"]) == (
        "http://example.com/role/estimate",
        "http://example.com/role/notes",
    )


def test_xbrl_enrichment_builds_dimensions_and_footnotes_and_reruns_skip(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    loads: list[str] = []

    def counted_load(files, **options):
        loads.append(files[0].document_id)
        return load_instance(files, **options)

    with monkeypatch.context() as patch:
        patch.setattr(xbrl_enrichment_extraction, "load_instance", counted_load)
        first = run_sec_filings_silver(tmp_path)

    summary = first.xbrl_enrichment
    assert first.completed
    assert summary is not None
    # Only the 10-K reports dimensions or footnotes; the others are complete
    # without a load and get no row.
    assert loads == [TEN_K_DOCUMENT]
    assert {key: asdict(summary)[key] for key in EXPECTED_FIRST_ENRICHMENT} == (
        EXPECTED_FIRST_ENRICHMENT
    )
    for identifier, schema in (
        (DIMENSIONS, XBRL_DIMENSIONS_ARROW_SCHEMA),
        (FOOTNOTES, XBRL_FACT_FOOTNOTES_ARROW_SCHEMA),
    ):
        table = _table(tmp_path, identifier)
        assert table.location() == str(
            tmp_path / "silver" / "sec_filings" / identifier.split(".")[1]
        )
        assert table.scan().to_arrow().schema.equals(schema)
        assert _partition_columns(table) == []

    _assert_fixture_dimensions(tmp_path)
    _assert_fixture_footnotes(tmp_path)

    def reload(*_: object, **__: object) -> None:
        raise AssertionError("complete instances must not reload XBRL")

    monkeypatch.setattr(xbrl_enrichment_extraction, "load_instance", reload)
    rerun = run_sec_filings_silver(tmp_path)

    assert rerun.completed
    assert rerun.xbrl_enrichment is not None
    assert (
        rerun.xbrl_enrichment.instances_skipped,
        rerun.xbrl_enrichment.instances_inserted,
    ) == (3, 0)
    assert _enrichment_snapshots(tmp_path) == (
        summary.dimensions_snapshot_after,
        summary.footnotes_snapshot_after,
    )
    assert (len(_rows(tmp_path, DIMENSIONS)), len(_rows(tmp_path, FOOTNOTES))) == (2, 3)


EXPECTED_FIRST_ENRICHMENT = {
    "status": "completed",
    "instances_considered": 3,
    "instances_inserted": 1,
    "instances_skipped": 2,
    "instances_resumed": 0,
    "contexts_considered": 3,
    "contexts_with_dimensions": 1,
    "contexts_without_dimensions": 2,
    "dimensions_inserted": 2,
    "explicit_dimensions": 1,
    "typed_dimensions": 1,
    "facts_considered": 6,
    "facts_with_footnotes": 2,
    "facts_without_footnotes": 4,
    "footnote_relationships_inserted": 3,
    "unique_footnote_resources": 2,
    "empty_footnotes": 0,
    "dimensions_snapshot_before": None,
    "footnotes_snapshot_before": None,
}


def _write_second_ten_k(data_dir: Path, write_sec_filing) -> str:
    """A new 10-K like the first: its own instance with dimensions and footnotes."""

    write_sec_filing(
        data_dir,
        cik=CIK,
        form="10-K",
        accession="0000320193-26-000002",
        filed="20260105",
    )
    return "sec:0000320193-26-000002:xbrl:inline:default"


def test_new_instance_appends_only_its_dimensions_and_footnotes(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    first = run_sec_filings_silver(tmp_path).xbrl_enrichment
    stored = (_rows(tmp_path, DIMENSIONS), _rows(tmp_path, FOOTNOTES))
    new_instance = _write_second_ten_k(tmp_path, write_sec_filing)

    result = run_sec_filings_silver(tmp_path)

    summary = result.xbrl_enrichment
    assert result.completed
    assert first is not None and summary is not None
    assert (summary.instances_inserted, summary.instances_skipped) == (1, 3)
    assert (summary.dimensions_inserted, summary.footnote_relationships_inserted) == (
        2,
        3,
    )
    for table, before in (
        (DIMENSIONS, first.dimensions_snapshot_after),
        (FOOTNOTES, first.footnotes_snapshot_after),
    ):
        assert _table(tmp_path, table).current_snapshot().parent_snapshot_id == before
    dimensions, footnotes = _rows(tmp_path, DIMENSIONS), _rows(tmp_path, FOOTNOTES)
    assert all(row in dimensions for row in stored[0])
    assert all(row in footnotes for row in stored[1])
    assert {d["context_key"].split(":context:")[0] for d in dimensions} == {
        TEN_K_INSTANCE,
        new_instance,
    }
    assert sum(f["fact_id"].startswith(new_instance) for f in footnotes) == 3


def _fail_footnotes(table: Table, rows, arrow_schema) -> None:
    if arrow_schema is XBRL_FACT_FOOTNOTES_ARROW_SCHEMA:
        raise OSError("disk full")
    append_rows(table, rows, arrow_schema)


def test_dimensions_only_commit_resumes_footnotes(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    with monkeypatch.context() as patch:
        patch.setattr(xbrl_enrichment_stage, "append_rows", _fail_footnotes)
        with pytest.raises(OSError, match="disk full"):
            run_sec_filings_silver(tmp_path)
    dimensions_snapshot, footnotes_snapshot = _enrichment_snapshots(tmp_path)
    assert dimensions_snapshot is not None and footnotes_snapshot is None

    resumed = run_sec_filings_silver(tmp_path).xbrl_enrichment

    assert resumed is not None and resumed.status == "completed"
    assert (resumed.instances_resumed, resumed.instances_inserted) == (1, 0)
    assert (resumed.dimensions_inserted, resumed.footnote_relationships_inserted) == (
        0,
        3,
    )
    assert _snapshot(tmp_path, DIMENSIONS) == dimensions_snapshot
    assert (len(_rows(tmp_path, DIMENSIONS)), len(_rows(tmp_path, FOOTNOTES))) == (2, 3)


def test_footnotes_without_dimensions_resume_dimensions(
    tmp_path: Path, write_sec_filing
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    run_sec_filings_silver(tmp_path)
    [context_key] = {row["context_key"] for row in _rows(tmp_path, DIMENSIONS)}
    _delete_rows(tmp_path, DIMENSIONS, "context_key", context_key)
    footnotes_snapshot = _snapshot(tmp_path, FOOTNOTES)

    resumed = run_sec_filings_silver(tmp_path).xbrl_enrichment

    assert resumed is not None and resumed.status == "completed"
    assert (resumed.instances_resumed, resumed.dimensions_inserted) == (1, 2)
    assert resumed.footnote_relationships_inserted == 0
    assert _snapshot(tmp_path, FOOTNOTES) == footnotes_snapshot
    assert len(_rows(tmp_path, DIMENSIONS)) == 2


def _drop_concept(data_dir: Path, local_name: str) -> None:
    with open_silver_catalog(data_dir) as catalog:
        catalog.load_table(CONCEPTS).delete(
            EqualTo("concept_id", concept_id_for(EX, local_name))
        )


@pytest.mark.parametrize(
    ("damage", "event"),
    [
        ("partial-dimension-group", "xbrl_enrichment_instance_conflict"),
        ("partial-footnote-group", "xbrl_enrichment_instance_conflict"),
        ("member-concept-missing", "xbrl_dimension_parent_missing"),
        ("context-missing", "dimensions have no context"),
        ("fact-missing", "xbrl_facts holds 5 of 6 facts"),
    ],
)
def test_xbrl_enrichment_rejects_the_batch_before_any_write(
    tmp_path: Path,
    write_sec_filing,
    capsys: pytest.CaptureFixture[str],
    damage: str,
    event: str,
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    run_sec_filings_silver(tmp_path)
    [context_key] = {row["context_key"] for row in _rows(tmp_path, DIMENSIONS)}
    if damage == "partial-dimension-group":
        _delete_rows(
            tmp_path, DIMENSIONS, "dimension_id", f"{context_key}:dimension:0002"
        )
    elif damage == "partial-footnote-group":
        _delete_rows(
            tmp_path, FOOTNOTES, "footnote_relation_id", f"{CELL_REVENUE}:footnote:0002"
        )
    elif damage == "member-concept-missing":
        # The dimensions are extracted again and meet the missing parent.
        _delete_rows(tmp_path, DIMENSIONS, "context_key", context_key)
        _drop_concept(tmp_path, "WidgetsMember")
    elif damage == "context-missing":
        _delete_rows(tmp_path, CONTEXTS, "context_key", context_key)
    elif damage == "fact-missing":
        _delete_rows(tmp_path, FACTS, "fact_id", f"{TEN_K_INSTANCE}:fact:00000005")
    # A second instance in the same batch must not be written either.
    _write_second_ten_k(tmp_path, write_sec_filing)
    before = _enrichment_snapshots(tmp_path)
    capsys.readouterr()

    summary = xbrl_enrichment_stage.run_xbrl_enrichment_stage(
        tmp_path, datetime.now(UTC), _settings(tmp_path)
    )

    assert summary.status == "rejected"
    assert summary.instances_conflicted >= 1
    assert (summary.dimensions_inserted, summary.footnote_relationships_inserted) == (
        0,
        0,
    )
    assert _enrichment_snapshots(tmp_path) == before
    assert event in capsys.readouterr().out


def test_parser_error_in_one_instance_writes_no_instance(
    tmp_path: Path, write_sec_filing, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bronze(tmp_path, write_sec_filing)
    new_instance = _write_second_ten_k(tmp_path, write_sec_filing)

    def broken_load(files, **options):
        if files[0].document_id.startswith(new_instance.split(":xbrl:")[0]):
            raise XbrlProcessorError("unreadable instance")
        return load_instance(files, **options)

    monkeypatch.setattr(xbrl_enrichment_extraction, "load_instance", broken_load)
    result = run_sec_filings_silver(tmp_path)

    summary = result.xbrl_enrichment
    assert result.xbrl_facts is not None and result.xbrl_facts.status == "completed"
    assert summary is not None and summary.status == "rejected"
    assert (summary.instances_failed, summary.dimensions_inserted) == (1, 0)
    assert _enrichment_snapshots(tmp_path) == (None, None)

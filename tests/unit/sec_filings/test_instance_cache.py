from __future__ import annotations

from pathlib import Path

import pytest

from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlInstanceRecord
from src.data_pipeline.pipelines.sec_filings.xbrl_instance_loading import (
    InstanceCache,
)
from src.data_pipeline.transforms.sec_filings.xbrl_processor import LoadedInstance

pytestmark = pytest.mark.unit

FILING = "sec:0000320193-24-000123"


def _instance(source_set: str = "a" * 64) -> XbrlInstanceRecord:
    return XbrlInstanceRecord.model_construct(
        instance_id=f"{FILING}:xbrl:inline:default",
        filing_id=FILING,
        is_inline=True,
        target_name=None,
        source_set_sha256=source_set,
    )


def _loaded() -> LoadedInstance:
    return LoadedInstance(
        concepts=[],
        facts=[],
        inline_fact_totals={"d": 1},
        context_hashes={"c1": "h"},
        dimensions=[],
        footnotes=[],
        labels=[],
        arcs=[],
        role_definitions={"r": None},
        downloaded_uris=["https://xbrl.fasb.org/us-gaap.xsd"],
    )


def test_a_cached_load_is_served_without_its_downloads(tmp_path: Path) -> None:
    cache = InstanceCache(tmp_path / "cache")

    assert cache.get(_instance()) is None
    cache.put(_instance(), _loaded())

    served = cache.get(_instance())
    assert served is not None
    assert served.context_hashes == {"c1": "h"}
    assert served.downloaded_uris == []
    assert [p.suffix for p in (tmp_path / "cache").iterdir()] == [".zst"]


def test_a_changed_source_set_misses(tmp_path: Path) -> None:
    cache = InstanceCache(tmp_path)
    cache.put(_instance(), _loaded())

    assert cache.get(_instance("b" * 64)) is None


def test_an_unreadable_entry_misses_and_clear_removes_every_entry(
    tmp_path: Path,
) -> None:
    cache = InstanceCache(tmp_path / "cache")
    cache.put(_instance(), _loaded())
    (entry,) = (tmp_path / "cache").iterdir()
    entry.write_bytes(entry.read_bytes()[:10])  # cut by a power loss

    assert cache.get(_instance()) is None
    cache.clear()
    assert not (tmp_path / "cache").exists()

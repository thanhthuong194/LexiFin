from __future__ import annotations

import hashlib
import os
import pickle
import shutil
from collections import defaultdict, deque
from collections.abc import Callable, Generator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import replace
from functools import cache, partial
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from typing import NoReturn

import zstandard

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlInstanceRecord
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.storage.sec_filings.documents import read_document_content
from src.data_pipeline.transforms.sec_filings import xbrl_processor
from src.data_pipeline.transforms.sec_filings.xbrl_processor import (
    LoadedInstance,
    ProcessorSettings,
    SourceFile,
    load_instance,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# The filing's own taxonomy: the processor resolves the instance's DTS from it.
_DTS_ROLES = frozenset(
    {
        "xbrl_schema",
        "xbrl_calculation",
        "xbrl_definition",
        "xbrl_label",
        "xbrl_presentation",
    }
)

# A call returning an instance's verified files and its processor load, or
# raising what reading or loading them raised.
InstanceLoad = Callable[[], tuple[list[SourceFile], LoadedInstance]]


@cache
def _loader_version() -> str:
    """The processor version and the code that turns its model into records:
    a load cached by other code is not reused."""

    digest = hashlib.sha256(version("arelle-release").encode())
    for path in sorted(Path(xbrl_processor.__file__).parent.glob("xbrl_*.py")):
        digest.update(path.name.encode() + b"\x00" + path.read_bytes())
    return digest.hexdigest()


class InstanceCache:
    """Processor loads kept on disk between the XBRL stages of a build.

    The facts, enrichment and linkbases stages each need the same load of an
    instance; the first stage that loads it stores it (pickled, zstd), the
    next ones read it instead of loading the DTS again. An entry is keyed by
    the instance, its target, the hash of its source set and the loader
    version, so a changed source or loader is never served from the cache.
    The files themselves are still read and verified by every stage. An
    unreadable entry is a miss; entries are written atomically.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, instance: XbrlInstanceRecord) -> Path:
        key = "\x00".join(
            (
                instance.instance_id,
                str(instance.is_inline),
                instance.target_name or "",
                instance.source_set_sha256,
                _loader_version(),
            )
        )
        return self.directory / f"{hashlib.sha256(key.encode()).hexdigest()}.pkl.zst"

    def get(self, instance: XbrlInstanceRecord) -> LoadedInstance | None:
        path = self._path(instance)
        try:
            data = zstandard.ZstdDecompressor().decompress(path.read_bytes())
            loaded = pickle.loads(data)
        except FileNotFoundError:
            return None
        except (
            OSError,
            zstandard.ZstdError,
            pickle.UnpicklingError,
            EOFError,
        ) as error:
            logger.warning(
                "xbrl_instance_cache_unreadable",
                instance_id=instance.instance_id,
                error=str(error),
            )
            return None
        if not isinstance(loaded, LoadedInstance):
            return None
        # Nothing was downloaded to serve it.
        return replace(loaded, downloaded_uris=[])

    def put(self, instance: XbrlInstanceRecord, loaded: LoadedInstance) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(instance)
        partial_path = path.with_name(f"{path.name}.{os.getpid()}.partial")
        partial_path.write_bytes(
            zstandard.ZstdCompressor(level=3).compress(
                pickle.dumps(loaded, protocol=pickle.HIGHEST_PROTOCOL)
            )
        )
        partial_path.replace(path)

    def clear(self) -> None:
        """Remove every entry: no later stage needs them."""

        shutil.rmtree(self.directory, ignore_errors=True)


def instance_cache(settings: ProcessorSettings) -> InstanceCache | None:
    """The cache the settings name, if any."""

    directory = settings.instance_cache_dir
    return None if directory is None else InstanceCache(directory)


def entry_documents(
    instance: XbrlInstanceRecord, documents: Mapping[str, DocumentRecord]
) -> list[DocumentRecord]:
    """The instance's own documents: its Inline documents, or its XML instance."""

    return [
        documents[document_id]
        for document_id in instance.source_document_ids
        if (documents[document_id].content_kind == "html") == instance.is_inline
    ]


def source_files(
    instance: XbrlInstanceRecord,
    documents: Mapping[str, DocumentRecord],
    filing_documents: Sequence[DocumentRecord],
    bronze: SecBronze,
) -> list[SourceFile]:
    """The instance's entry documents and its filing's taxonomy, read and verified."""

    entries = entry_documents(instance, documents)
    taxonomy = [
        d for d in filing_documents if d.role in _DTS_ROLES and d.content_sha256
    ]
    return [
        SourceFile(
            d.document_id,
            d.filename or "",
            read_document_content(d, bronze),
            is_entry=d in entries,
        )
        for d in (*entries, *taxonomy)
    ]


def load_ahead(
    instances: Sequence[XbrlInstanceRecord],
    documents: Mapping[str, DocumentRecord],
    bronze: SecBronze,
    settings: ProcessorSettings,
) -> Generator[InstanceLoad]:
    """One load per instance, in instance order.

    `documents` must hold every document of the instances' filings. With one
    worker each load reads and loads when called. With more, the files are
    read here and the processor runs in `settings.workers` spawned processes,
    up to two instances per worker ahead of the caller; loads are still
    handed out in instance order, so the rows built from them do not depend
    on the number of workers. A failure belongs to its instance: calling its
    load raises it. Close the iterator to stop the workers early. With an
    instance cache in `settings`, a cached load is served instead of running
    the processor, and a new load is cached.
    """

    by_filing: dict[str, list[DocumentRecord]] = defaultdict(list)
    for document in documents.values():
        by_filing[document.filing_id].append(document)

    def read(instance: XbrlInstanceRecord) -> list[SourceFile]:
        return source_files(instance, documents, by_filing[instance.filing_id], bronze)

    cache = instance_cache(settings)
    if settings.workers == 1:
        return (
            partial(_load_here, read, instance, settings, cache)
            for instance in instances
        )
    return _load_in_workers(instances, read, settings, cache)


def _load_here(
    read: Callable[[XbrlInstanceRecord], list[SourceFile]],
    instance: XbrlInstanceRecord,
    settings: ProcessorSettings,
    cache: InstanceCache | None,
) -> tuple[list[SourceFile], LoadedInstance]:
    files = read(instance)
    cached = None if cache is None else cache.get(instance)
    if cached is not None:
        return files, cached
    loaded = load_instance(
        files,
        is_inline=instance.is_inline,
        target_name=instance.target_name,
        settings=settings,
    )
    if cache is not None:
        cache.put(instance, loaded)
    return files, loaded


def _raise(error: Exception) -> NoReturn:
    raise error


def _ready(
    files: list[SourceFile], loaded: LoadedInstance
) -> tuple[list[SourceFile], LoadedInstance]:
    return files, loaded


def _wait(
    files: list[SourceFile],
    future: Future[LoadedInstance],
    store: Callable[[LoadedInstance], None] | None,
) -> tuple[list[SourceFile], LoadedInstance]:
    loaded = future.result()
    if store is not None:
        store(loaded)
    return files, loaded


def _load_in_workers(
    instances: Sequence[XbrlInstanceRecord],
    read: Callable[[XbrlInstanceRecord], list[SourceFile]],
    settings: ProcessorSettings,
    cache: InstanceCache | None,
) -> Generator[InstanceLoad]:
    pools: list[ProcessPoolExecutor] = []

    def pool() -> ProcessPoolExecutor:
        # Started at the first load the cache cannot serve. Spawned workers
        # start clean instead of copying this process's memory.
        if not pools:
            pools.append(
                ProcessPoolExecutor(settings.workers, mp_context=get_context("spawn"))
            )
        return pools[0]

    pending: deque[InstanceLoad] = deque()
    upcoming = iter(instances)

    def submit_next() -> None:
        instance = next(upcoming, None)
        if instance is None:
            return
        try:
            files = read(instance)
        except (OSError, ValueError) as error:  # raised when the load is used
            pending.append(partial(_raise, error))
            return
        cached = None if cache is None else cache.get(instance)
        if cached is not None:
            pending.append(partial(_ready, files, cached))
            return
        future = pool().submit(
            load_instance,
            files,
            is_inline=instance.is_inline,
            target_name=instance.target_name,
            settings=settings,
        )
        store = None if cache is None else partial(cache.put, instance)
        pending.append(partial(_wait, files, future, store))

    try:
        for _ in range(2 * settings.workers):
            submit_next()
        while pending:
            load = pending.popleft()
            submit_next()
            yield load
    finally:
        for started in pools:
            started.shutdown(cancel_futures=True)

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Generator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from typing import NoReturn

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.xbrl_core import XbrlInstanceRecord
from src.data_pipeline.storage.bronze import SecBronze
from src.data_pipeline.storage.sec_filings.documents import read_document_content
from src.data_pipeline.transforms.sec_filings.xbrl_processor import (
    LoadedInstance,
    ProcessorSettings,
    SourceFile,
    load_instance,
)

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
    load raises it. Close the iterator to stop the workers early.
    """

    by_filing: dict[str, list[DocumentRecord]] = defaultdict(list)
    for document in documents.values():
        by_filing[document.filing_id].append(document)

    def read(instance: XbrlInstanceRecord) -> list[SourceFile]:
        return source_files(instance, documents, by_filing[instance.filing_id], bronze)

    if settings.workers == 1:
        return (partial(_load_here, read, instance, settings) for instance in instances)
    return _load_in_workers(instances, read, settings)


def _load_here(
    read: Callable[[XbrlInstanceRecord], list[SourceFile]],
    instance: XbrlInstanceRecord,
    settings: ProcessorSettings,
) -> tuple[list[SourceFile], LoadedInstance]:
    files = read(instance)
    return files, load_instance(
        files,
        is_inline=instance.is_inline,
        target_name=instance.target_name,
        settings=settings,
    )


def _raise(error: Exception) -> NoReturn:
    raise error


def _wait(
    files: list[SourceFile], future: Future[LoadedInstance]
) -> tuple[list[SourceFile], LoadedInstance]:
    return files, future.result()


def _load_in_workers(
    instances: Sequence[XbrlInstanceRecord],
    read: Callable[[XbrlInstanceRecord], list[SourceFile]],
    settings: ProcessorSettings,
) -> Generator[InstanceLoad]:
    # Spawned workers start clean instead of copying this process's memory.
    pool = ProcessPoolExecutor(settings.workers, mp_context=get_context("spawn"))
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
        future = pool.submit(
            load_instance,
            files,
            is_inline=instance.is_inline,
            target_name=instance.target_name,
            settings=settings,
        )
        pending.append(partial(_wait, files, future))

    try:
        for _ in range(2 * settings.workers):
            submit_next()
        while pending:
            load = pending.popleft()
            submit_next()
            yield load
    finally:
        pool.shutdown(cancel_futures=True)

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from arelle.api.Session import Session
from arelle.RuntimeOptions import RuntimeOptions

from src.data_pipeline.transforms.sec_filings.xbrl_model_reader import (
    LoadedConcept,
    LoadedFact,
    XbrlProcessorError,
    read_concepts,
    read_facts,
)
from src.data_pipeline.transforms.sec_filings.xbrl_sources import strip_edgar_envelope

# Seconds a taxonomy download may stall before the file counts as unavailable.
_DOWNLOAD_TIMEOUT_SECONDS = 60


class TaxonomyUnavailable(XbrlProcessorError):
    """Files of the DTS are missing from the cache and were not downloaded.

    Either downloads are off, or a download failed or timed out.
    """

    def __init__(self, uris: Sequence[str]) -> None:
        super().__init__(f"taxonomy files unavailable: {sorted(uris)[:10]}")
        self.uris = tuple(sorted(uris))


@dataclass(frozen=True)
class SourceFile:
    """A verified document of the filing to place next to the instance.

    `is_entry` marks the instance's own documents (Inline documents, or the
    XML instance); the others are the filing's schema and linkbases.
    """

    document_id: str
    filename: str
    content: bytes
    is_entry: bool


@dataclass(frozen=True)
class ProcessorSettings:
    """Where the taxonomy cache lives and whether it may be filled online."""

    cache_dir: Path
    allow_network: bool
    http_user_agent: str | None


@dataclass(frozen=True)
class LoadedInstance:
    """Concepts of the DTS and fact occurrences of one logical instance.

    `inline_fact_totals` counts every Inline fact element of each entry
    document, whatever its target, so positions can be checked elsewhere.
    """

    concepts: list[LoadedConcept]
    facts: list[LoadedFact]
    inline_fact_totals: dict[str, int]
    downloaded_uris: list[str]


def _materialize(files: Sequence[SourceFile], directory: Path) -> dict[str, Path]:
    paths = {}
    for file in files:
        name = file.filename
        if PurePosixPath(name).name != name or name in {"", ".", ".."} or "\\" in name:
            raise XbrlProcessorError(f"{file.document_id}: unsafe filename {name!r}")
        path = directory / name
        if path.exists():
            raise XbrlProcessorError(f"two documents are named {name!r}")
        path.write_bytes(strip_edgar_envelope(file.content))
        paths[file.document_id] = path
    return paths


@contextmanager
def _session(entry: str, settings: ProcessorSettings) -> Iterator[Session]:
    options = RuntimeOptions(
        entrypointFile=entry,
        plugins="inlineXbrlDocumentSet",
        keepOpen=True,
        cacheDirectory=str(settings.cache_dir),
        internetConnectivity="online" if settings.allow_network else "offline",
        internetLogDownloads=True,
        # Cached files are reused as they are; only missing ones are fetched.
        internetRecheck="never",
        # Without a timeout a stalled download blocks the run forever.
        internetTimeout=_DOWNLOAD_TIMEOUT_SECONDS,
        httpUserAgent=settings.http_user_agent,
        logFile="logToStructuredMessage",
        # The session returns the processor's records as structured messages;
        # they stay out of the application's own log.
        logPropagate=False,
    )
    with Session() as session:
        session.run(options)
        yield session


def load_instance(
    files: Sequence[SourceFile],
    *,
    is_inline: bool,
    target_name: str | None,
    settings: ProcessorSettings,
) -> LoadedInstance:
    """Load one logical instance and its DTS with Arelle; return neutral records.

    Files are written to a temporary directory removed afterwards. An Inline
    instance is the document set of its entry files restricted to
    `target_name` (None is the default target); an XML instance is its one
    entry file. Raises TaxonomyUnavailable when DTS files are missing offline
    and XbrlProcessorError for anything else the processor cannot load.
    """

    with TemporaryDirectory(prefix="lexifin-xbrl-") as tmp:
        directory = Path(tmp).resolve()
        paths = _materialize(files, directory)
        entries = [(f.document_id, paths[f.document_id]) for f in files if f.is_entry]
        if not entries or (not is_inline and len(entries) != 1):
            raise XbrlProcessorError(f"{len(entries)} entry documents")
        target = "(default)" if target_name is None else target_name
        entry = (
            json.dumps(
                [{"ixds": [{"file": str(p)} for _, p in entries], "ixdsTarget": target}]
            )
            if is_inline
            else str(entries[0][1])
        )
        with _session(entry, settings) as session:
            messages = session.get_log_messages()
            missing = [
                str(m.get("msg", "")).rpartition("file: ")[2]
                for m in messages
                if m.get("messageCode") == "IOerror"
            ]
            if missing:
                raise TaxonomyUnavailable(missing)
            models = session.get_models()
            if len(models) != 1:
                raise XbrlProcessorError(f"the processor loaded {len(models)} models")
            model = models[0]
            try:
                facts, totals = read_facts(model, entries, inline=is_inline)
                return LoadedInstance(
                    concepts=read_concepts(model, directory),
                    facts=facts,
                    inline_fact_totals=totals if is_inline else {},
                    downloaded_uris=[
                        str(m.get("msg", "")).removeprefix("Downloaded ")
                        for m in messages
                        if m.get("messageCode") == "webCache:download"
                    ],
                )
            finally:
                model.close()

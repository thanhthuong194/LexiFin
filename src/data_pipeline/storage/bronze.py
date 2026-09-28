from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from src.data_pipeline.contracts.ingestion import RawArtifact


def sha256_bytes(contents: bytes) -> str:
    """Return the SHA-256 digest of bytes."""

    return hashlib.sha256(contents).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_root(path: Path, root: Path) -> str:
    """Return a POSIX relative path and reject paths outside root."""

    resolved_root = root.resolve()
    try:
        relative = path.resolve().relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"Path escapes data directory: {path}") from error
    return relative.as_posix()


def resolve_relative_path(relative_path: str, root: Path) -> Path:
    """Resolve a stored relative path while preventing directory traversal."""

    candidate = (root / relative_path).resolve()
    relative_to_root(candidate, root)
    return candidate


def atomic_write_bytes(
    path: Path,
    contents: bytes,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write bytes, allowing an identical existing immutable file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        if path.is_file() and path.read_bytes() == contents:
            return
        raise FileExistsError(f"Refusing to overwrite different contents: {path}")

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as file_handle:
            file_handle.write(contents)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_text(
    path: Path,
    contents: str,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write UTF-8 text."""

    atomic_write_bytes(path, contents.encode(), overwrite=overwrite)


def _json_value(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def atomic_write_json(
    path: Path,
    value: BaseModel | dict[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write deterministic, human-readable JSON."""

    contents = json.dumps(
        _json_value(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    atomic_write_text(path, f"{contents}\n", overwrite=overwrite)


def atomic_write_jsonl(path: Path, values: Iterable[BaseModel]) -> None:
    """Atomically write one compact JSON object per line."""

    lines = [
        json.dumps(value.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for value in values
    ]
    atomic_write_text(path, "".join(f"{line}\n" for line in lines))


def append_jsonl(path: Path, value: BaseModel) -> None:
    """Append and fsync one JSONL model record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
    )
    with path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(f"{line}\n")
        file_handle.flush()
        os.fsync(file_handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL objects, returning an empty list when absent."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def artifact_for_path(
    role: Literal["full_submission", "primary_document"],
    path: Path,
    data_dir: Path,
) -> RawArtifact:
    """Create verified artifact metadata for a file under data_dir."""

    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Artifact is missing or empty: {path}")
    return RawArtifact(
        role=role,
        relative_path=relative_to_root(path, data_dir),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )

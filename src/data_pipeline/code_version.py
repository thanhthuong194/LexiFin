from __future__ import annotations

import ast
import hashlib
from functools import cache
from pathlib import Path

# Imports followed: the repository's own code; libraries are pinned by uv.lock.
_OWN_PACKAGE = "src"
# The directory holding the `src` package (this file is src/data_pipeline/...).
_ROOT = Path(__file__).resolve().parents[2]


def _source(module: str) -> Path | None:
    """The source file of one of our modules, or None for a name that is not
    a module (an attribute imported from a module). Found by path, so nothing
    is imported."""

    base = _ROOT.joinpath(*module.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def _imported_modules(module: str, source: Path) -> set[str]:
    package = module if source.name == "__init__.py" else module.rpartition(".")[0]
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source.read_bytes(), str(source))):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = package.rsplit(".", node.level - 1)[0]
                base = f"{parent}.{base}" if base else parent
            imported.add(base)
            imported.update(f"{base}.{alias.name}" for alias in node.names)
    return {
        name
        for name in imported
        if name == _OWN_PACKAGE or name.startswith(f"{_OWN_PACKAGE}.")
    }


@cache
def code_sha256(module: str) -> str:
    """SHA-256 of the source of `module` and of every module of ours it
    imports, directly or not (read statically, so whatever runs first).

    For results kept on disk between runs, such as cached XBRL loads or a
    stopped audit: one computed by other code is never reused, while edits to
    code the module does not import keep them valid. Raises
    ModuleNotFoundError when `module` has no source file.
    """

    root = _source(module)
    if root is None:
        raise ModuleNotFoundError(f"no source file for {module}")
    sources = {module: root}
    pending = [module]
    while pending:
        name = pending.pop()
        for imported in _imported_modules(name, sources[name]):
            if imported not in sources and (path := _source(imported)) is not None:
                sources[imported] = path
                pending.append(imported)
    digest = hashlib.sha256()
    for name in sorted(sources):
        digest.update(name.encode() + b"\x00" + sources[name].read_bytes() + b"\x00")
    return digest.hexdigest()

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from src.data_pipeline import code_version

pytestmark = pytest.mark.unit


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A `src` package: `a` imports `b` (absolutely) and `c` (relatively);
    `d` is imported by nothing."""

    root = tmp_path / "src"
    (root / "pkg").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "a.py").write_text(
        "import json\nfrom src.pkg.b import VALUE\nfrom . import c\n"
    )
    (root / "pkg" / "b.py").write_text("VALUE = 1\n")
    (root / "pkg" / "c.py").write_text("")
    (root / "pkg" / "d.py").write_text("")
    monkeypatch.setattr(code_version, "_ROOT", tmp_path)
    code_version.code_sha256.cache_clear()
    yield root / "pkg"
    code_version.code_sha256.cache_clear()


def _changed(package: Path, module: str) -> bool:
    before = code_version.code_sha256("src.pkg.a")
    (package / module).write_text("# edited\n", encoding="utf-8")
    code_version.code_sha256.cache_clear()
    return code_version.code_sha256("src.pkg.a") != before


@pytest.mark.parametrize("module", ["a.py", "b.py", "c.py"])
def test_an_edit_to_the_module_or_an_import_changes_the_hash(
    package: Path, module: str
) -> None:
    assert _changed(package, module)


def test_an_edit_to_code_the_module_does_not_import_keeps_the_hash(
    package: Path,
) -> None:
    assert not _changed(package, "d.py")


def test_a_module_without_source_is_an_error(package: Path) -> None:
    with pytest.raises(ModuleNotFoundError):
        code_version.code_sha256("src.pkg.missing")

"""The section boundary: a section module imports paperlib and third parties only."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _section_of(module: Path) -> str:
    """The nearest numbered folder above a module, the section it belongs to."""
    folders = [p.name for p in module.relative_to(ROOT).parents if p.name[:2].isdigit()]
    return folders[0]


def _imported_names(module: Path) -> set[str]:
    """Every dotted part and every from-imported name the module imports."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(part for alias in node.names for part in alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            names.update((node.module or "").split("."))
            names.update(alias.name for alias in node.names)
    return names


def _crossings() -> set[tuple[str, str]]:
    modules = [path for path in ROOT.glob("[0-9][0-9]_*/**/*.py") if path.name != "conftest.py"]
    section_of = {path.stem: _section_of(path) for path in modules}
    return {
        (str(path.relative_to(ROOT)), name)
        for path in modules
        for name in _imported_names(path)
        if name in section_of and section_of[name] != _section_of(path)
    }


@pytest.mark.functional
def test_no_section_imports_another() -> None:
    assert _crossings() == set()

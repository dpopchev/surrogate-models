"""One assert per test function: the suite refuses to run while a test checks more than one thing.

Several facts about one behavior are a test class with a shared fixture and one single-assert test
per fact, or separate test functions (rules/python.md, Tests). Written by add-python as
tests/conftest.py (package level) or scripts/conftest.py (helper level).
"""

import ast
import inspect
import textwrap
from collections.abc import Callable

import pytest


def _asserts(function: Callable[..., object]) -> list[ast.Assert]:
    """The asserts of the test's own body; those of nested helper functions are not counted."""
    body = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    found: list[ast.Assert] = []
    stack: list[ast.AST] = list(ast.iter_child_nodes(body))
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef):
            continue
        if isinstance(node, ast.Assert):
            found.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return found


def _offence(function: Callable[..., object]) -> str | None:
    asserts = _asserts(function)
    if len(asserts) > 1:
        return f"{len(asserts)} asserts"
    if asserts and isinstance(asserts[0].test, ast.BoolOp):
        return "a compound assert (`and` / `or`)"
    return None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    offenders: dict[str, str] = {}
    for item in items:
        if isinstance(item, pytest.Function) and (why := _offence(item.function)):
            offenders[item.nodeid.split("[")[0]] = why  # a parametrized test is one function
    if offenders:
        lines = "\n".join(f"  {nodeid}: {why}" for nodeid, why in sorted(offenders.items()))
        raise pytest.UsageError(
            f"one assert per test -- split into a test class or separate tests:\n{lines}"
        )

"""One assert per test function: the suite refuses to run while a test checks more than one thing.

Several facts about one behavior are a test class with a shared fixture and one single-assert test
per fact, or separate test functions (rules/python.md, Tests). Once the project registers the
category markers, every test function carries exactly one (W-116). Every run also appends each
test's duration to the `stats_file` ini path, warns over `suite_budget_s`, and `--suite-stats`
reports the recorded history (W-115). Written by add-python as tests/conftest.py (package
level) or scripts/conftest.py (helper level).
"""

import ast
import inspect
import json
import subprocess
import textwrap
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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


_CATEGORIES = frozenset({"edge", "model", "command", "view", "functional", "integration"})


def _registered_categories(config: pytest.Config) -> frozenset[str]:
    """The categories the project registered in the `markers` ini; none keeps the check off."""
    lines = config.getini("markers")
    names = {line.split(":")[0].split("(")[0].strip() for line in lines}
    return _CATEGORIES & names


def _category_offence(item: pytest.Item, categories: frozenset[str]) -> str | None:
    own = sorted({mark.name for mark in item.iter_markers() if mark.name in categories})
    if not own:
        return "no category marker"
    if len(own) > 1:
        return f"{len(own)} category markers ({', '.join(own)})"
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    offenders: dict[str, str] = {}
    for item in items:
        if isinstance(item, pytest.Function) and (why := _offence(item.function)):
            offenders[item.nodeid.split("[")[0]] = why  # a parametrized test is one function
    if offenders:
        lines = "\n".join(f"  {nodeid}: {why}" for nodeid, why in sorted(offenders.items()))
        raise pytest.UsageError(
            f"one assert per test -- split into a test class or separate tests:\n{lines}"
        )
    categories = _registered_categories(config)
    unsorted = {
        item.nodeid.split("[")[0]: why
        for item in items  # a doctest carries no marker; only test functions are sorted
        if categories
        and isinstance(item, pytest.Function)
        and (why := _category_offence(item, categories))
    }
    if unsorted:
        lines = "\n".join(f"  {nodeid}: {why}" for nodeid, why in sorted(unsorted.items()))
        raise pytest.UsageError(
            f"one category marker per test -- one of {', '.join(sorted(categories))}:\n{lines}"
        )


_BUILTIN_MARKERS = frozenset(
    {"parametrize", "skip", "skipif", "xfail", "usefixtures", "filterwarnings"}
)


def _category(item: pytest.Item) -> str:
    """The test's first own marker; W-116 makes it the one category marker."""
    names = (mark.name for mark in item.iter_markers() if mark.name not in _BUILTIN_MARKERS)
    return next(names, "unmarked")


@dataclass(frozen=True)
class _Recorder:
    """Appends one JSON line per test call to the stats file, so every run is kept."""

    path: Path
    run: str
    commit: str

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(
        self, item: pytest.Item, call: pytest.CallInfo[None]
    ) -> Generator[None, pytest.TestReport, pytest.TestReport]:
        report = yield
        if call.when == "call":
            record = {
                "nodeid": report.nodeid,
                "marker": _category(item),
                "outcome": report.outcome,
                "duration": report.duration,
                "run": self.run,
                "commit": self.commit,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stats:
                stats.write(json.dumps(record) + "\n")
        return report


@dataclass(frozen=True)
class _Record:
    """One recorded test call, as a line of the stats file holds it."""

    nodeid: str
    marker: str
    duration: float
    run: str
    commit: str


def _parse(line: str) -> _Record:
    fields = json.loads(line)
    return _Record(
        nodeid=str(fields["nodeid"]),
        marker=str(fields.get("marker", "unmarked")),
        duration=float(fields["duration"]),
        run=str(fields.get("run", "")),
        commit=str(fields.get("commit", "")),
    )


def _report(records: list[_Record]) -> list[str]:
    """The suite stats over the recorded history: the 10 slowest tests at their latest duration,
    the latest run's total per marker and the last 5 runs' totals, oldest first."""
    latest = {r.nodeid: r for r in records}  # a later record replaces an earlier one
    slowest = sorted(latest.values(), key=lambda record: record.duration, reverse=True)[:10]
    last_run = [r for r in records if r.run == records[-1].run] if records else []
    markers = {
        r.marker: sum(s.duration for s in last_run if s.marker == r.marker) for r in last_run
    }
    by_total = sorted(markers.items(), key=lambda item: item[1], reverse=True)
    commits = {r.run: r.commit for r in records}  # insertion order is the recorded order
    last_runs = list(commits.items())[-5:]
    return [
        "slowest 10 tests:",
        *(f"  {r.duration:8.2f}s  {r.nodeid}" for r in slowest),
        "",
        "total per marker:",
        *(f"  {marker:<12}  {total:8.2f}s" for marker, total in by_total),
        "",
        "last 5 runs:",
        *(
            f"  {run}  {commit or '-':<9}  {sum(r.duration for r in records if r.run == run):8.2f}s"
            for run, commit in last_runs
        ),
    ]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--suite-stats", action="store_true", help="show the recorded test durations and exit"
    )
    parser.addini(
        "stats_file", "where every run appends its test durations", default="local/test-stats.jsonl"
    )
    parser.addini(
        "suite_budget_s", "warn when a run's test calls take longer, in seconds", default=""
    )
    parser.addini(
        "suite_budget_fail", "fail the run over the suite budget", type="bool", default=False
    )


def _commit(root: Path) -> str:
    """The short HEAD commit of the repository at root; empty outside git."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return ""
    return head.stdout.strip() if head.returncode == 0 else ""


def pytest_cmdline_main(config: pytest.Config) -> int | None:
    """With --suite-stats, print the report over the stats file and exit without running tests."""
    if not config.getoption("suite_stats"):
        return None
    path = config.rootpath / str(config.getini("stats_file"))
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    print("\n".join(_report([_parse(line) for line in lines if line.strip()])))
    return 0


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(
        _Recorder(
            path=config.rootpath / str(config.getini("stats_file")),
            run=datetime.now().astimezone().isoformat(timespec="seconds"),
            commit=_commit(config.rootpath),
        )
    )


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Warn when the run's test calls exceed `suite_budget_s`; fail too with `suite_budget_fail`."""
    budget = str(session.config.getini("suite_budget_s"))
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if not budget or reporter is None:
        return
    calls = [
        report
        for reports in reporter.stats.values()
        for report in reports
        if isinstance(report, pytest.TestReport) and report.when == "call"
    ]
    total = sum(report.duration for report in calls)
    if not calls or total <= float(budget):
        return
    slowest = max(calls, key=lambda report: report.duration)
    reporter.write_line(
        f"WARNING: tests took {total:.2f}s, over the suite budget of {float(budget)}s,"
        f" slowest {slowest.nodeid} ({slowest.duration:.2f}s)"
    )
    if session.config.getini("suite_budget_fail"):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED

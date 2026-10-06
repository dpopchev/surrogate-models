"""The baseline runs at a glance (W-057): make runs lists them, make run shows one run's files.

Every folder under <state dir>/51_algorithms/ is a run (fit_baseline.py names it); its run.json
gives the line. A folder without a readable run.json is listed with the reason -- still running,
failed, or written before the record had its current fields -- never skipped.

Run as `uv run python <this file> <state dir> [--run <run folder> | --run latest]`.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from shared.runs import RunRecord, from_json
from shared.surrogate import duration_text

SECTION = "51_algorithms"

# --- vocabulary and types ---------------------------------------------------------------------


@dataclass(frozen=True)
class Finished:
    """A run folder with a readable run record."""

    name: str
    record: RunRecord


@dataclass(frozen=True)
class Unscored:
    """A run folder without a readable run record, and why."""

    name: str
    why: str


Listed = Finished | Unscored

# --- pure functions ---------------------------------------------------------------------------


def runs_table(runs: list[Listed]) -> list[str]:
    """One line per run, newest first."""
    return [_line(run) for run in sorted(runs, key=lambda run: run.name, reverse=True)]


def _line(run: Listed) -> str:
    """One run's line: its scores, epochs, fit time and commit."""
    match run:
        case Finished(name, record):
            return (
                f"{name}  MARE {record.mare:.2e}  RMSE {record.rmse:.2e}  "
                f"epochs {record.epochs} (best {record.best_epoch})  "
                f"fit {duration_text(record.seconds)}  {record.commit}"
                + (" (uncommitted changes)" if record.dirty else "")
            )
        case Unscored(name, why):
            return f"{name}  no scores: {why}"


# --- shell ------------------------------------------------------------------------------------


def read_runs(folder: Path) -> list[Listed]:
    """Every run folder under folder, finished or not; latest.log and other files are skipped."""
    runs: list[Listed] = []
    for run in sorted(path for path in folder.iterdir() if path.is_dir()):
        record = run / "run.json"
        if not record.is_file():
            runs.append(Unscored(run.name, "no run.json yet -- running or failed"))
            continue
        try:
            runs.append(Finished(run.name, from_json(record.read_text())))
        except TypeError, KeyError:
            runs.append(Unscored(run.name, "run.json in an older format"))
    return runs


def run_files(run: Path) -> list[str]:
    """Each file of one run folder with its size, by name."""
    return [f"  {path.name}  {_size(path.stat().st_size)}" for path in sorted(run.iterdir())]


def _size(size: int) -> str:
    """A byte count as "3 B", "2 KB" or "1.4 MB"."""
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.0f} KB"
    return f"{size / 1024**2:.1f} MB"


def main(argv: list[str]) -> None:
    """Print the runs, or one run's files."""
    args = _arguments(argv)
    folder = args.state / SECTION
    runs = read_runs(folder) if folder.is_dir() else []
    if not runs:
        print(f"no baseline runs yet -- make baseline starts one (looked in {folder})")
        return
    if args.run is None:
        for line in runs_table(runs):
            print(line)
        return
    names = sorted(run.name for run in runs)
    name = names[-1] if args.run == "latest" else args.run
    if name not in names:
        raise SystemExit(f"no run {name} under {folder}; make runs lists them")
    print(f"{folder / name}:")
    for line in run_files(folder / name):
        print(line)


def _arguments(argv: list[str]) -> argparse.Namespace:
    """Parse the state dir and the optional run to show."""
    parser = argparse.ArgumentParser(prog="list_runs.py")
    parser.add_argument("state", type=Path)
    parser.add_argument("--run", help="a run folder name, or latest")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])

"""The run shell every runnable shares: a batch's identity, its files and the wall of a fit (W-091).

Each runnable (the baseline, the representation and family screens, the tuning) is one make run,
a batch: a uuid7 id, the commit it ran from and when, its ledger entries and its trial journal.
The helpers here read the clock, git and the filesystem, so they belong to the imperative shell;
the runnables inject them into their pure parts.
"""

import logging
import signal
import subprocess
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from shared.harness import FitRows, Fitter, Predictor, Run
from shared.progress import duration_text
from shared.runs import (
    LedgerEntry,
    RunMetadata,
    entry_from_json,
    entry_from_run,
    entry_name,
    entry_to_json,
)

# --- vocabulary and types ---------------------------------------------------------------------


@dataclass(frozen=True)
class Batch:
    """One make run: its id, where its ledger entries and its trial journal go, the code it ran
    from and when it started."""

    id: str
    ledger: Path
    journal: Path
    commit: str
    dirty: bool
    started: datetime


class WriteCheckError(OSError):
    """A written file does not read back as the text written."""


class WallError(Exception):
    """A fit ran past its wall."""


# --- pure functions ---------------------------------------------------------------------------


def outcomes_path(state: Path, section: str, batch: str) -> Path:
    """Where a section's batch keeps its outcomes."""
    return state / section / f"{batch}.json"


def progress_tail(done: int, total: int, elapsed: float) -> str:
    """The elapsed time and, from the mean time per job, the time left: the tail of a job line."""
    left = elapsed / done * (total - done)
    return f"elapsed {duration_text(elapsed)}, about {duration_text(left)} left"


def make_batch(
    state: Path,
    journal: Callable[[str], Path],
    commit: Callable[[], str],
    dirty: Callable[[], bool],
    now: Callable[[], datetime],
    new_id: Callable[[], str],
) -> Batch:
    """A new batch under the state folder; journal names the trial journal of a batch id."""
    batch = new_id()
    return Batch(batch, state / "ledger", journal(batch), commit(), dirty(), now())


# --- shell ------------------------------------------------------------------------------------


def write_checked(path: Path, text: str, read: Callable[[Path], str] = Path.read_text) -> None:
    """Write the text to path; raise WriteCheckError when it does not read back."""
    path.write_text(text)
    if read(path) != text:
        raise WriteCheckError(f"{path} does not read back")


def write_entry(ledger: Path, run: Run, meta: RunMetadata) -> LedgerEntry:
    """Write the run's ledger entry to <ledger>/<id>.json and return it; raise WriteCheckError
    when the file does not read back to the same text (NaN spreads compare unequal as values,
    so the texts are compared)."""
    entry = entry_from_run(run, meta)
    write_checked(
        ledger / entry_name(entry),
        entry_to_json(entry),
        read=lambda path: entry_to_json(entry_from_json(path.read_text())),
    )
    return entry


def read_ledger(folder: Path) -> tuple[LedgerEntry, ...]:
    """Every entry in the ledger folder, in the order of their file names (uuid7: of creation)."""
    return tuple(entry_from_json(path.read_text()) for path in sorted(folder.glob("*.json")))


@contextmanager
def at_level(level: str) -> Iterator[None]:
    """While the block runs, let the root logger pass records at level; then restore it."""
    root = logging.getLogger()
    previous = root.level
    root.setLevel(level)
    try:
        yield
    finally:
        root.setLevel(previous)


def _git(root: Path | None, *args: str) -> str:
    """The output of one git command in the repository at root (the working folder if None)."""
    folder = [] if root is None else ["-C", str(root)]
    done = subprocess.run(["git", *folder, *args], capture_output=True, text=True, check=True)
    return done.stdout


def git_commit(root: Path | None = None) -> str:
    """The short hash of the checked-out commit."""
    return _git(root, "rev-parse", "--short", "HEAD").strip()


def git_dirty(root: Path | None = None) -> bool:
    """Whether the work tree holds uncommitted changes."""
    return bool(_git(root, "status", "--porcelain").strip())


def new_id() -> str:
    """A new time-sortable id (uuid7), for a batch or a ledger entry."""
    return str(uuid.uuid7())


def utc_now() -> datetime:
    """The current time in UTC."""
    return datetime.now(UTC)


def wall(fitter: Fitter, seconds: float) -> Fitter:
    """The fitter, each of its fits stopped by WallError once it computes past `seconds` of the
    process's CPU time: a wait or a suspend is not charged, and a fit pinned to one thread is
    charged its compute. A timer's signal, so it runs in a process's main thread, as a worker's
    job does, and it stops a fit when the interpreter next runs, not inside one library call."""

    def stop(signum: int, frame: object) -> None:
        raise WallError(f"past {seconds / 60:g} min")

    def fit(rows: FitRows, seed: int) -> Predictor:
        previous = signal.signal(signal.SIGPROF, stop)
        signal.setitimer(signal.ITIMER_PROF, seconds)
        try:
            return fitter(rows, seed)
        finally:
            signal.setitimer(signal.ITIMER_PROF, 0)
            signal.signal(signal.SIGPROF, previous)

    return fit


def walled(fitter: Fitter, seconds: float) -> Fitter:
    """The fitter behind its wall: a fit past `seconds` of compute predicts nan, and so does every
    later fit of the same fitter, so a walled trial scores nan (failed) without waiting again."""
    timed = wall(fitter, seconds)
    met = False

    def fit(rows: FitRows, seed: int) -> Predictor:
        nonlocal met
        if met:
            return _nan
        try:
            return timed(rows, seed)
        except WallError:
            met = True
            return _nan

    return fit


def _nan(x: np.ndarray) -> np.ndarray:
    """The prediction of a walled fit: nan for every row."""
    return np.full(len(x), np.nan)

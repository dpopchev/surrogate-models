"""Regenerate the surrogates rung by rung and print the time of each (W-172).

Run as `uv run python paperlib/tools/regenerate.py <state folder> [--fresh]` (make surrogates):
the family screen, the tuning, the pipeline, the extrapolation and the precision regime run as
make calls in order, each with its batch variable empty so it fits again, and each fresh batch is
handed to the rungs that read it. After each rung the record <state>/surrogates/run.json gains
its batch id and elapsed seconds; a rung already in it is skipped, so an interrupted run resumes.
--fresh starts a new record. A rung is forced with make -W on its script: its outputs exist, so
make would answer "assets current" otherwise.
"""

import json
import logging
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from paperlib.batches import write_checked
from paperlib.progress import duration_text

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------


@dataclass(frozen=True)
class Rung:
    """One make call of the chain: its name, its make arguments, the batch variables it takes
    from earlier rungs (variable, rung name) and the variable its own fresh batch is pinned as."""

    name: str
    args: tuple[str, ...]
    takes: tuple[tuple[str, str], ...]
    yields: str | None
    folder: str = ""
    suffix: str = ".json"


@dataclass(frozen=True)
class Done:
    """A finished rung: the batch it wrote, when it wrote one, and its elapsed seconds."""

    batch: str | None
    seconds: float


Record = MutableMapping[str, Done]

RUNGS = (
    Rung(
        "families",
        (
            "families",
            "FAMILIES_BATCH=",
            "-W",
            "90_appendix/98_families/screen_families.py",
        ),
        (),
        "FAMILIES_BATCH",
        "98_families",
    ),
    Rung(
        "tuning",
        ("tuning", "TUNING_BATCH=", "-W", "90_appendix/99_precision/tune_survivors.py"),
        (("FAMILIES_BATCH", "families"),),
        "TUNING_BATCH",
        "99_precision",
    ),
    Rung("pipeline", ("pipeline",), (), None),
    Rung(
        "extrapolation",
        ("extrapolation", "EXTRAPOLATION_BATCH="),
        (("TUNING_BATCH", "tuning"),),
        "EXTRAPOLATION_BATCH",
        "96_optimization",
        "-extrapolation.json",
    ),
    Rung(
        "precision",
        (
            "precision",
            "PRECISION_BATCH=",
            "-W",
            "90_appendix/99_precision/precision_regime.py",
        ),
        (),
        "PRECISION_BATCH",
        "99_precision",
        "-precision.json",
    ),
)

_ID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"

# --- pure functions ---------------------------------------------------------------------------


def argv_of(rung: Rung, record: Record) -> tuple[str, ...]:
    """The make command of a rung, handing it the recorded batches of the rungs it takes from."""
    handed = []
    for variable, source in rung.takes:
        done = record[source]
        assert done.batch is not None, f"rung {source} recorded no batch"
        handed.append(f"{variable}={done.batch}")
    return ("make", *rung.args, *handed)


def rung_line(rung: Rung, done: Done, skipped: bool) -> str:
    """One line of a rung: its batch and elapsed time."""
    note = " (recorded, skipped)" if skipped else ""
    return f"{rung.name}: {done.batch or '-'} in {duration_text(done.seconds)}{note}"


def pin_lines(rungs: Sequence[Rung], record: Record) -> list[str]:
    """One `pin: <VARIABLE> ?= <batch>` line per recorded batch, in rung order."""
    return [
        f"pin: {rung.yields} ?= {record[rung.name].batch}"
        for rung in rungs
        if rung.yields and rung.name in record
    ]


def winner_lines(summary: str) -> list[str]:
    """The winners of summary.json, one line each: the test figure beside its ceiling."""
    return [
        f"winner: {row['dataset']} {row['target']}: {row['family']} {row['unit']} "
        f"{row['test']:.2f} vs ceiling {row['ceiling']:.2f}"
        for row in json.loads(summary)
    ]


# --- the chain --------------------------------------------------------------------------------


def run(
    rungs: Sequence[Rung],
    record: Record,
    execute: Callable[[tuple[str, ...]], int],
    clock: Callable[[], float],
    fresh: Callable[[Rung, float], str | None],
    say: Callable[[str], None],
) -> int:
    """Run the rungs in order, skipping a recorded one, and return 0 or the exit status of the
    rung that stopped the chain; each finished rung is stored in the record before the next."""
    for rung in rungs:
        if rung.name in record:
            say(rung_line(rung, record[rung.name], skipped=True))
            continue
        started = clock()
        status = execute(argv_of(rung, record))
        if status:
            say(f"{rung.name} failed with exit status {status}")
            return status
        batch = fresh(rung, started) if rung.yields else None
        if rung.yields and batch is None:
            say(f"{rung.name} wrote no batch under {rung.folder}")
            return 1
        record[rung.name] = Done(batch, clock() - started)
        say(rung_line(rung, record[rung.name], skipped=False))
    return 0


# --- shell ------------------------------------------------------------------------------------


class FileRecord(Record):
    """The record file: every store is written whole and read back."""

    def __init__(self, path: Path, fresh: bool) -> None:
        self._path = path
        self._items: dict[str, Done] = {}
        if not fresh and path.is_file():
            self._items = {k: Done(**v) for k, v in json.loads(path.read_text()).items()}
        else:
            self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        write_checked(
            self._path,
            json.dumps({k: vars(v) for k, v in self._items.items()}, indent=1),
        )

    def __getitem__(self, key: str) -> Done:
        return self._items[key]

    def __setitem__(self, key: str, value: Done) -> None:
        self._items[key] = value
        self._save()

    def __delitem__(self, key: str) -> None:
        del self._items[key]
        self._save()

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


def newest_batch(state: Path, rung: Rung, since: float) -> str | None:
    """The id of the newest batch file the rung wrote under its state folder since the time."""
    found = sorted(
        path.name.removesuffix(rung.suffix)
        for path in (state / rung.folder).glob(f"*{rung.suffix}")
        if re.fullmatch(_ID + re.escape(rung.suffix), path.name) and path.stat().st_mtime >= since
    )
    return found[-1] if found else None


def main(argv: list[str]) -> int:
    """Run the chain over the state folder argv[0]; --fresh starts a new record."""
    assert argv and argv[0] != "--fresh", "usage: regenerate.py <state folder> [--fresh]"
    state = Path(argv[0])
    record = FileRecord(state / "surrogates" / "run.json", "--fresh" in argv[1:])
    status = run(
        RUNGS,
        record,
        lambda command: subprocess.run(command, check=False).returncode,
        time.time,
        lambda rung, since: newest_batch(state, rung, since),
        logger.info,
    )
    if status == 0:
        for line in winner_lines((state / "41_algorithm" / "summary.json").read_text()):
            logger.info(line)
        for line in pin_lines(RUNGS, record):
            logger.info(line)
    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main(sys.argv[1:]))

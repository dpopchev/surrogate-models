"""The extrapolation of every tuned candidate (W-170): each fitted once at its tuned params
without the edge curves and the test curves, and scored on the edge curves it never saw beside
the interior test curves, at the 95th percentile and at the worst row.

The jobs are the tuned outcomes of a recorded tuning batch, one fit each, on worker processes;
the outcomes are saved after every job under <state>/96_optimization/<batch>-extrapolation.json.
A fit past its wall is saved as a walled outcome. --batch fits nothing and reprints a saved batch.
A section never imports another: the tuning outcomes are read from the state folder.
"""

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from paperlib.batches import WallError, new_id, progress_tail, wall, write_checked
from paperlib.candidates import Candidate, Family, Knobs, Unit, tuned_fitter
from paperlib.config import load_config
from paperlib.design import BLACK_HOLES, NEUTRON_STARS, SPECS, TARGETS, Pair, pairs
from paperlib.harness import Fitter, one_thread
from paperlib.surrogate import Training, make_training
from paperlib.workers import run_jobs
from paperlib.zones import ZoneFigures, edge_held_out

logger = logging.getLogger(__name__)

SECTION = "96_optimization"
TUNING_SECTION = "99_precision"

# --- vocabulary and types ---------------------------------------------------------------------

# The fitter of a candidate on a pair at a tuned trial's params.
MakeFitter = Callable[[Pair, Candidate, Mapping[str, Any]], Fitter]


@dataclass(frozen=True)
class Tuned:
    """One tuned candidate to measure: its pair, its candidate and its tuned params."""

    dataset: str
    target: str
    candidate: Candidate
    params: Mapping[str, Any]


@dataclass(frozen=True)
class Measured:
    """One candidate's figures on the edge and interior zones, and the seconds of its fit."""

    dataset: str
    target: str
    family: Family
    unit: Unit
    params: Mapping[str, Any]
    zones: Mapping[str, ZoneFigures | None]
    fit_seconds: float
    predict_seconds: float
    kind: str = "measured"


@dataclass(frozen=True)
class Walled:
    """One candidate whose fit ran past its wall."""

    dataset: str
    target: str
    family: Family
    unit: Unit
    params: Mapping[str, Any]
    reason: str
    kind: str = "walled"


Outcome = Measured | Walled

# --- pure functions ---------------------------------------------------------------------------


def outcomes_file(state: Path, batch: str) -> Path:
    """Where a batch keeps its extrapolation outcomes."""
    return state / SECTION / f"{batch}-extrapolation.json"


def figures_text(found: ZoneFigures | None, worst: bool) -> str:
    """A zone's figures as "p95 <x> worst <y>" (p95 alone unless worst), or n/a when empty."""
    if found is None:
        return "n/a"
    return f"p95 {found.p95:.2f} worst {found.worst:.2f}" if worst else f"p95 {found.p95:.2f}"


def line(outcome: Outcome) -> str:
    """The printed line of one pair and candidate."""
    who = f"{outcome.dataset} {outcome.target} {outcome.family} {outcome.unit}"
    if isinstance(outcome, Walled):
        return f"{who}: walled, {outcome.reason}"
    return (
        f"{who}: edge {figures_text(outcome.zones['edge'], True)}, "
        f"interior {figures_text(outcome.zones['interior'], False)}, "
        f"fit {outcome.fit_seconds:.0f} s"
    )


def outcomes_text(outcomes: Sequence[Outcome]) -> str:
    """The outcomes as indented JSON."""
    return json.dumps([asdict(outcome) for outcome in outcomes], indent=2) + "\n"


def outcomes_from_text(text: str) -> tuple[Outcome, ...]:
    """The outcomes an outcomes_text holds."""
    found: list[Outcome] = []
    for fields in json.loads(text):
        if fields["kind"] == "walled":
            found.append(Walled(**fields))
            continue
        zones = {n: None if z is None else ZoneFigures(**z) for n, z in fields["zones"].items()}
        found.append(Measured(**{**fields, "zones": zones}))
    return tuple(found)


def tuned_from_text(text: str, wanted: Sequence[tuple[str, str]]) -> tuple[Tuned, ...]:
    """The tuned outcomes of a tuning batch, only the wanted dataset-target pairs when given;
    a walled tuning outcome has no params to measure."""
    found = []
    for fields in json.loads(text):
        job = fields["job"]
        if fields["kind"] != "tuned" or (wanted and (job["dataset"], job["target"]) not in wanted):
            continue
        found.append(
            Tuned(job["dataset"], job["target"], Candidate(**job["candidate"]), fields["params"])
        )
    return tuple(found)


# --- shell: fits and reads the state ----------------------------------------------------------


def measure(
    tuned: Tuned,
    pair: Pair,
    make_fitter: MakeFitter,
    seed: int,
    valid_fraction: float,
    seconds: float,
) -> Outcome:
    """Fit the tuned candidate without the edge and score it; a fit past `seconds` is walled."""
    candidate = tuned.candidate
    identity = (tuned.dataset, tuned.target, candidate.family, candidate.unit, tuned.params)
    fitter = wall(make_fitter(pair, candidate, tuned.params), seconds)
    try:
        with one_thread():
            found = edge_held_out(
                pair.design, pair.target, fitter, seed, valid_fraction, time.perf_counter
            )
    except WallError as error:
        return Walled(*identity, str(error))
    return Measured(*identity, found.zones, found.fit_seconds, found.predict_seconds)


def run(
    args: argparse.Namespace,
    say: Callable[[str], None],
    make_fitter: MakeFitter,
    valid_fraction: float,
    batch_id: Callable[[], str] = new_id,
) -> tuple[Outcome, ...]:
    """Fit every tuned candidate of the tuning batch without the edge, saving the outcomes after
    each job and printing one line per candidate; with --batch print the saved batch instead."""
    if args.batch:
        saved = outcomes_from_text(outcomes_file(args.state, args.batch).read_text())
        for outcome in saved:
            say(line(outcome))
        return saved
    tables = {
        NEUTRON_STARS.dataset: pd.read_parquet(args.ns),
        BLACK_HOLES.dataset: pd.read_parquet(args.bh),
    }
    by_key = {
        (p.dataset, p.target): p for p in pairs(tables, pd.read_parquet(args.split), SPECS, TARGETS)
    }
    text = (args.state / TUNING_SECTION / f"{args.tuning_batch}.json").read_text()
    works = tuned_from_text(text, args.pair)
    batch = batch_id()
    path = outcomes_file(args.state, batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    jobs = [
        partial(
            measure,
            work,
            by_key[work.dataset, work.target],
            make_fitter,
            args.seed,
            valid_fraction,
            args.minutes * 60.0,
        )
        for work in works
    ]
    done: dict[int, Outcome] = {}
    started = time.monotonic()

    def finished(index: int, outcome: Outcome) -> None:
        done[index] = outcome
        write_checked(path, outcomes_text([done[i] for i in sorted(done)]))
        say(line(outcome))
        logger.info(
            "%d of %d done, %s",
            len(done),
            len(jobs),
            progress_tail(len(done), len(jobs), time.monotonic() - started),
        )

    results = run_jobs(jobs, args.workers, on_done=finished)
    write_checked(path, outcomes_text(results))
    logger.info("extrapolation batch %s saved to %s", batch, path)
    return tuple(results)


def _pair(text: str) -> tuple[str, str]:
    """A --pair value dataset:target as a tuple."""
    dataset, _, target = text.partition(":")
    if not dataset or not target:
        raise argparse.ArgumentTypeError(f"{text!r} is not dataset:target")
    return dataset, target


def arguments(argv: Sequence[str]) -> argparse.Namespace:
    """The inputs, the state folder, the base knobs, the tuning batch and the run's limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ns", "bh", "split", "state"):
        parser.add_argument(name, type=Path)
    for name in ("k", "neighbours", "knots", "seed", "workers"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--minutes", type=float, required=True, help="the wall of one fit")
    parser.add_argument("--tuning-batch", required=True, help="the recorded tuning batch")
    parser.add_argument("--pair", type=_pair, action="append", default=[], help="dataset:target")
    parser.add_argument("--batch", default="", help="reprint this saved batch, fit nothing")
    return parser.parse_args(argv)


def _tuned_fitter(knobs: Knobs, training: Training, pair: Pair, c: Candidate, p: Any) -> Fitter:
    """The tuned fitter of the candidate, built as a winner's is."""
    return tuned_fitter(c, pair.space, knobs, training, p)


def main(argv: Sequence[str], say: Callable[[str], None] = print) -> None:
    """Wire the real fitters: the paper's network settings under the tuning's base knobs."""
    args = arguments(argv)
    training = make_training(load_config().network)
    knobs = Knobs(args.k, args.neighbours, args.knots, per_curve=0)
    run(args, say, partial(_tuned_fitter, knobs, training), training.valid_fraction)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main(sys.argv[1:])

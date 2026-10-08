"""The family screen behind Section 5.2 (W-067): every model family on W-066's representation,
pointwise and curve-wise, eliminated by successive halving over the training rows.

Every candidate -- a family in one unit of prediction -- is scored through shared/harness.py on
each pair (dataset and target) at the first budget of training rows, thinned along each curve;
the best share per pair, ranked on the frozen folds' p95 significant figures, goes on to the
next budget. A fit past the time budget is stopped in its worker, and a pointwise GPR whose
kernel matrix would pass the memory budget is not run: both are the family's wall at that
budget. One line per finished job tells how far the screen is and how long is left.
"""

import argparse
import json
import logging
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal, assert_never

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline

from shared import surrogate
from shared.config import PaperConfig, load_config
from shared.curvewise import curvewise_fitter
from shared.design import BLACK_HOLES, NEUTRON_STARS, Design, DesignSpec, Target, design
from shared.eda import CurveSpace, booktabs
from shared.families import make_gpr, make_knn, make_rbf, make_xgboost
from shared.harness import Fitter, Predictor, Run, harness
from shared.runs import RunMetadata, Setting, entry_from_run, entry_name, entry_to_json
from shared.scorecard import significant_figures
from shared.surrogate import (
    ResMLP,
    Training,
    duration_text,
    make_estimator,
    network_fitter,
    sklearn_fitter,
)
from shared.workers import run_jobs

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

SECTION = "62_families"

PAIRS: tuple[tuple[DesignSpec, Target], ...] = (
    (NEUTRON_STARS, "mass"),
    (NEUTRON_STARS, "charge"),
    (BLACK_HOLES, "mass"),
    (BLACK_HOLES, "charge"),
)

DATASET_TEX = {"neutron_stars": "NS", "black_holes": "BH"}
TARGET_TEX = {"mass": "$M$", "charge": "$\\Dch$"}

Family = Literal["k-NN", "local RBF", "GPR", "XGBoost", "MLP", "ResNet"]
FAMILIES: tuple[Family, ...] = ("k-NN", "local RBF", "GPR", "XGBoost", "MLP", "ResNet")
FAMILY_TEX = {
    "k-NN": "$k$-NN",
    "local RBF": "local RBF",
    "GPR": "GPR",
    "XGBoost": "XGBoost",
    "MLP": "MLP",
    "ResNet": "ResNet",
}
# The unit of prediction: one row, or a whole curve along its normalized coordinate.
Unit = Literal["pointwise", "curve-wise"]
UNITS: tuple[Unit, ...] = ("pointwise", "curve-wise")


@dataclass(frozen=True)
class Candidate:
    """One family in one unit of prediction."""

    family: Family
    unit: Unit


@dataclass(frozen=True)
class Job:
    """One candidate on one pair at one budget of training rows, in one round of the halving."""

    dataset: str
    target: Target
    candidate: Candidate
    rows: int
    round: int


@dataclass(frozen=True)
class Scored:
    """A job that ran: the mean p95 significant figures over its folds and on its test curves,
    its ripple per curve parameter, its final fit's seconds and its ledger entry."""

    job: Job
    folds: float
    test: float
    ripple: tuple[float, ...]
    fit_seconds: float
    entry: str


@dataclass(frozen=True)
class Wall:
    """A job past the time or the memory budget: why, in words."""

    job: Job
    reason: str


Outcome = Scored | Wall


@dataclass(frozen=True)
class Knobs:
    """The families' settings: the k-NN's k, the local RBF's neighbours, the curve-wise
    spline's knots (ends included), and the fewest rows a thinned curve keeps."""

    k: int
    neighbours: int
    knots: int
    per_curve: int


class FitTimeoutError(Exception):
    """A fit ran past its time budget."""


# --- pure functions ---------------------------------------------------------------------------


def candidates() -> tuple[Candidate, ...]:
    """Every family in every unit."""
    return tuple(Candidate(family, unit) for family in FAMILIES for unit in UNITS)


def thin(design: Design, rows: int, per_curve: int, seed: int) -> Design:
    """The design with its training rows cut to about `rows`: every test row is kept; each
    training curve keeps rows evenly spaced along its curve, at least per_curve of them, and
    when there are too few rows for that on every curve, a seeded choice of rows // per_curve
    whole curves does."""
    train = ~design.test
    total = int(train.sum())
    keep = design.test.copy()
    curves = np.unique(design.groups[train])
    if rows < len(curves) * per_curve:
        curves = np.random.default_rng(seed).choice(curves, rows // per_curve, replace=False)
    for curve in curves:
        mine = np.flatnonzero(train & (design.groups == curve))
        mine = mine[np.argsort(design.X[mine, 0])]
        share = min(len(mine), max(per_curve, round(len(mine) * rows / total)))
        keep[mine[np.unique(np.linspace(0, len(mine) - 1, share).round().astype(int))]] = True
    return _rows(design, keep)


def _rows(design: Design, keep: np.ndarray) -> Design:
    """The design's rows where keep is set."""
    return Design(
        X=design.X[keep],
        y=design.y[keep],
        groups=design.groups[keep],
        test=design.test[keep],
        fold=design.fold[keep],
        ablation=design.ablation[keep],
    )


def survivors(outcomes: Sequence[Outcome], keep: int) -> tuple[Candidate, ...]:
    """The best `keep` candidates of one pair's round on the folds' figures; a wall never
    survives."""
    ran: list[Scored] = []
    for outcome in outcomes:
        match outcome:
            case Scored():
                ran.append(outcome)
            case Wall():
                pass
            case _:
                assert_never(outcome)
    ranked = sorted(ran, key=lambda outcome: -outcome.folds)
    return tuple(outcome.job.candidate for outcome in ranked[:keep])


def next_jobs(
    outcomes: Sequence[Outcome], budgets: Sequence[int], keeps: Sequence[int]
) -> tuple[Job, ...]:
    """The jobs of the round after the outcomes' round: each pair's best keeps[round]
    candidates at the next budget of rows, pairs in the order they first appear."""
    by_pair: dict[tuple[str, Target], list[Outcome]] = {}
    for outcome in outcomes:
        by_pair.setdefault((outcome.job.dataset, outcome.job.target), []).append(outcome)
    jobs = []
    for (dataset, target), mine in by_pair.items():
        done = mine[0].job.round
        for candidate in survivors(mine, keeps[done]):
            jobs.append(Job(dataset, target, candidate, budgets[done + 1], done + 1))
    return tuple(jobs)


def family_table(outcomes: Sequence[Outcome]) -> str:
    """The Section 5.2 table: one row per candidate, one column per pair, each cell the
    candidate's folds' p95 significant figures at the most rows it ran on (or its wall there),
    the pair's leader at the most rows in bold."""
    last: dict[tuple[str, Target, Candidate], Outcome] = {}
    for outcome in sorted(outcomes, key=lambda found: found.job.rows):
        job = outcome.job
        last[(job.dataset, job.target, job.candidate)] = outcome
    leaders = {
        (spec.dataset, target, leader)
        for spec, target in PAIRS
        for leader in survivors(_at_most_rows(outcomes, spec.dataset, target), keep=1)
    }
    rows = [
        f"{FAMILY_TEX[candidate.family]} & {candidate.unit} & "
        + " & ".join(
            _cell(
                last.get((spec.dataset, target, candidate)),
                (spec.dataset, target, candidate) in leaders,
            )
            for spec, target in PAIRS
        )
        for candidate in candidates()
    ]
    pairs = " & ".join(f"{DATASET_TEX[s.dataset]} {TARGET_TEX[target]}" for s, target in PAIRS)
    return booktabs(
        "families",
        "Each family in each unit of prediction, eliminated by successive halving over the "
        "training rows: the significant figures kept at the 95th percentile of the error "
        "($-\\log_{10}$) on the frozen folds, at the most training rows the candidate ran on "
        "(the subscript); a wall is a fit past the time or the memory budget there. The pair's "
        "best at the most rows is in bold. The errors are measured in $M$ or $\\Dch$.",
        "@{}ll*{4}{r}@{}",
        f"Family & Unit & {pairs}",
        rows,
    )


def _at_most_rows(outcomes: Sequence[Outcome], dataset: str, target: Target) -> list[Outcome]:
    """The pair's outcomes at the most rows any of them ran on."""
    mine = [o for o in outcomes if (o.job.dataset, o.job.target) == (dataset, target)]
    most = max((o.job.rows for o in mine), default=0)
    return [o for o in mine if o.job.rows == most]


def _cell(outcome: Outcome | None, bold: bool) -> str:
    """One candidate's table cell on one pair: its folds' figures with its rows as the
    subscript, in bold for the pair's leader; dashes for a candidate the pair never ran."""
    if outcome is None:
        return "--"
    rows = f"_{{10^{{{round(np.log10(outcome.job.rows))}}}}}"
    match outcome:
        case Scored():
            figures = f"\\mathbf{{{outcome.folds:.2f}}}" if bold else f"{outcome.folds:.2f}"
            return f"${figures}{rows}$"
        case Wall():
            return f"wall${rows}$"
        case _:
            assert_never(outcome)


def outcomes_to_json(outcomes: Sequence[Outcome]) -> str:
    """The outcomes as indented JSON, each tagged scored or wall."""
    tagged = []
    for outcome in outcomes:
        match outcome:
            case Scored():
                tagged.append({"kind": "scored", **asdict(outcome)})
            case Wall():
                tagged.append({"kind": "wall", **asdict(outcome)})
            case _:
                assert_never(outcome)
    return json.dumps(tagged, indent=2) + "\n"


def outcomes_from_json(text: str) -> tuple[Outcome, ...]:
    """The outcomes an outcomes_to_json text holds."""
    found: list[Outcome] = []
    for fields in json.loads(text):
        raw = fields["job"]
        job = Job(
            raw["dataset"],
            raw["target"],
            Candidate(**raw["candidate"]),
            raw["rows"],
            raw["round"],
        )
        if fields["kind"] == "wall":
            found.append(Wall(job, fields["reason"]))
        else:
            found.append(
                Scored(
                    job,
                    fields["folds"],
                    fields["test"],
                    tuple(fields["ripple"]),
                    fields["fit_seconds"],
                    fields["entry"],
                )
            )
    return tuple(found)


def over_memory(candidate: Candidate, rows: int, memory_bytes: float) -> bool:
    """Whether the candidate's fit is known to pass the memory budget unrun: a pointwise GPR's
    kernel matrix of rows x rows doubles."""
    return candidate == Candidate("GPR", "pointwise") and rows * rows * 8 > memory_bytes


def fitter_of(candidate: Candidate, space: CurveSpace, knobs: Knobs, training: Training) -> Fitter:
    """The harness fitter of the candidate on a design of the curve space: the family fitted
    row by row, or as the base that predicts each curve's spline from its curve keys."""
    family = candidate.family
    match candidate.unit:
        case "pointwise":
            if family == "MLP":
                return network_fitter(training)
            if family == "ResNet":
                return network_fitter(training, before_fit=_with_skips)
            return sklearn_fitter(_factory(family, knobs))
        case "curve-wise":
            keys, along = curve_columns(space)
            if family in ("MLP", "ResNet"):
                outputs = knobs.knots + 4  # the spline's knots + 2 coefficients, start, end
                base = partial(CurveNet, training, len(keys), outputs, family == "ResNet")
                return curvewise_fitter(base, keys, along, knobs.knots)
            return curvewise_fitter(_factory(family, knobs), keys, along, knobs.knots)
        case _:
            assert_never(candidate.unit)


def _factory(family: Family, knobs: Knobs) -> Callable[[], Any]:
    """The scikit-learn estimator of a family that is not a network."""
    match family:
        case "k-NN":
            return partial(make_knn, knobs.k)
        case "local RBF":
            return partial(make_rbf, knobs.neighbours)
        case "GPR":
            return make_gpr
        case "XGBoost":
            return make_xgboost
        case "MLP" | "ResNet":
            raise ValueError(f"{family} is a network, fitted by network_fitter")
        case _:
            assert_never(family)


def _with_skips(pipeline: Pipeline) -> None:
    """Turn a built network into the residual network of the same size (identity skips)."""
    pipeline.named_steps["net"].set_params(module=ResMLP)


def curve_columns(space: CurveSpace) -> tuple[tuple[int, ...], int]:
    """The design columns of the curve keys and of the one input along a curve."""
    names = [name for name, _ in space.inputs]
    along = next(i for i, name in enumerate(names) if name not in space.curve)
    return tuple(names.index(name) for name in space.curve), along


class CurveNet(RegressorMixin, BaseEstimator):
    """A network as the curve-wise base: one row per curve, so each row is its own group and
    the network validates on whole curves."""

    def __init__(self, training: Training, n_inputs: int, n_outputs: int, skips: bool) -> None:
        self.training = training
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.skips = skips

    def fit(self, x: np.ndarray, y: np.ndarray) -> CurveNet:
        """Fit the network on the curves' keys and outputs."""
        self.pipeline_ = make_estimator(
            self.training, self.n_inputs, n_outputs=self.n_outputs, skips=self.skips
        )
        rows = np.asarray(x, dtype=np.float32)
        self.pipeline_.fit(rows, np.asarray(y, dtype=np.float32), net__groups=np.arange(len(x)))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Every output of each curve."""
        return self.pipeline_.predict(np.asarray(x, dtype=np.float32))


def run_job(
    job: Job,
    data: Design,
    fitter: Fitter,
    seed: int,
    valid_fraction: float,
    per_curve: int,
    seconds: float,
    memory_bytes: float,
    record: Callable[[Run], str],
) -> Outcome:
    """One job through the harness on the design thinned to the job's rows, each fit within
    the time budget; record keeps the run (its ledger entry) and names it. A wall when the
    candidate is known to pass the memory budget, or a fit passes the time or the memory."""
    if over_memory(job.candidate, job.rows, memory_bytes):
        return Wall(job, f"kernel matrix over {memory_bytes / 1e9:g} GB")
    thinned = thin(data, job.rows, per_curve, seed)
    entries: list[str] = []
    try:
        (run,) = harness(
            thinned,
            job.target,
            timed(fitter, seconds),
            [seed],
            valid_fraction,
            record=lambda ran: entries.append(record(ran)),
        )
    except FitTimeoutError as stopped:
        return Wall(job, str(stopped))
    except MemoryError:
        return Wall(job, "out of memory")
    folds = float(np.mean([significant_figures(fold.p95) for fold in run.folds]))
    test = significant_figures(run.test.zones["test"].p95)
    return Scored(job, folds, test, run.ripple, run.timing.fit, entries[0])


def timed(fitter: Fitter, seconds: float) -> Fitter:
    """The fitter, each of its fits stopped by FitTimeoutError once it runs past `seconds`: a real
    timer's signal, so it runs in a process's main thread, as a worker's job does, and it stops
    a fit when the interpreter next runs, not inside one long library call."""

    def stop(signum: int, frame: object) -> None:
        raise FitTimeoutError(f"past {seconds / 60:g} min")

    def fit(
        x_fit: np.ndarray, y_fit: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, seed: int
    ) -> Predictor:
        previous = signal.signal(signal.SIGALRM, stop)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return fitter(x_fit, y_fit, x_valid, y_valid, seed)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    return fit


def job_line(outcome: Outcome, done: int, total: int, elapsed: float) -> str:
    """One finished job: its pair, candidate, rows and round, its fold and test figures or its
    wall, then how many of the round's jobs are done, the elapsed time and the time left."""
    job = outcome.job
    left = elapsed / done * (total - done)
    match outcome:
        case Scored():
            result = f"folds {outcome.folds:.2f}, test {outcome.test:.2f} figures"
        case Wall():
            result = f"wall, {outcome.reason}"
        case _:
            assert_never(outcome)
    return (
        f"job {done}/{total} of round {job.round + 1} done: {job.dataset} {job.target}, "
        f"{job.candidate.family} {job.candidate.unit}, {job.rows} rows: {result}; "
        f"elapsed {duration_text(elapsed)}, about {duration_text(left)} left"
    )


# --- shell ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Context:
    """What every job of one screen shares: the batch, where its ledger entries go, the code
    it ran from and when, the families' settings and the budgets of one fit."""

    batch: str
    ledger: Path
    commit: str
    dirty: bool
    started: datetime
    knobs: Knobs
    training: Training
    seconds: float
    memory_bytes: float
    log_level: str


@dataclass(frozen=True)
class Work:
    """One job with its pair's design and curve space, run in a worker process."""

    job: Job
    data: Design
    space: CurveSpace
    context: Context


def main(
    argv: list[str],
    config: PaperConfig | None = None,
    commit: Callable[[], str] | None = None,
    dirty: Callable[[], bool] | None = None,
    now: Callable[[], datetime] | None = None,
) -> None:
    """Run the screen round by round on the workers, saving every round's outcomes under the
    state folder, each run leaving its ledger entry -- or, given --batch, fit nothing and take
    that saved batch."""
    args = _arguments(argv)
    for path in (args.ns, args.bh, args.split):
        assert path.is_file(), f"input not found: {path}"
    assert len(args.keeps) == len(args.budgets) - 1, "one keep per round after the first"
    config = config or load_config()
    settings = config.methodology.algorithms
    training = Training(**settings.model_dump(exclude={"log_level", "threads", "workers"}))
    with _at_level(settings.log_level):
        if args.batch is None:
            batch = _run_screen(args, training, settings.log_level, commit, dirty, now)
        else:
            batch = args.batch
            logger.info("rebuilding the assets from batch %s; nothing is fitted", batch)
        saved = _outcomes_path(args.state, batch)
        outcomes = outcomes_from_json(saved.read_text())
        out = args.assets / SECTION
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{SECTION}_tab_families.tex"
        _write_checked(path, family_table(outcomes))
        logger.info("done: family table of batch %s, %d outcomes -> %s", batch, len(outcomes), path)


def _run_screen(
    args: argparse.Namespace,
    training: Training,
    log_level: str,
    commit: Callable[[], str] | None,
    dirty: Callable[[], bool] | None,
    now: Callable[[], datetime] | None,
) -> str:
    """Run every round of the screen as a new batch, saving the outcomes after each round;
    return the batch id."""
    commit, dirty, now = commit or _git_commit, dirty or _git_dirty, now or _utc_now
    tables = {
        NEUTRON_STARS.dataset: pd.read_parquet(args.ns),
        BLACK_HOLES.dataset: pd.read_parquet(args.bh),
    }
    split = pd.read_parquet(args.split)
    designs: dict[tuple[str, Target], tuple[Design, CurveSpace]] = {
        (spec.dataset, target): (design(tables[spec.dataset], split, spec, target), spec.space)
        for spec, target in PAIRS
    }
    batch = _new_id()
    context = Context(
        batch=batch,
        ledger=args.state / "ledger",
        commit=commit(),
        dirty=dirty(),
        started=now(),
        knobs=Knobs(args.k, args.neighbours, args.knots, args.per_curve),
        training=training,
        seconds=args.minutes * 60.0,
        memory_bytes=args.memory_gb * 1e9,
        log_level=log_level,
    )
    context.ledger.mkdir(parents=True, exist_ok=True)
    saved = _outcomes_path(args.state, batch)
    saved.parent.mkdir(parents=True, exist_ok=True)
    jobs = tuple(
        Job(spec.dataset, target, candidate, args.budgets[0], 0)
        for spec, target in PAIRS
        for candidate in candidates()
    )
    outcomes: list[Outcome] = []
    for round_ in range(len(args.budgets)):
        logger.info(
            "round %d of %d: %d jobs at %d rows on %d workers, batch %s",
            round_ + 1,
            len(args.budgets),
            len(jobs),
            args.budgets[round_],
            args.workers,
            batch,
        )
        found = _run_round(jobs, designs, context, args.workers)
        outcomes += found
        _write_checked(saved, outcomes_to_json(outcomes))
        if round_ + 1 < len(args.budgets):
            jobs = next_jobs(found, args.budgets, args.keeps)
    return batch


def _run_round(
    jobs: Sequence[Job],
    designs: dict[tuple[str, Target], tuple[Design, CurveSpace]],
    context: Context,
    workers: int,
) -> list[Outcome]:
    """One round's jobs on the workers, a line logged as each ends; outcomes in job order."""
    works = [
        partial(screen_job, Work(job, *designs[(job.dataset, job.target)], context)) for job in jobs
    ]
    started = time.monotonic()
    done = 0

    def report(index: int, outcome: Outcome) -> None:
        nonlocal done
        done += 1
        logger.info(job_line(outcome, done, len(works), time.monotonic() - started))

    return run_jobs(works, workers, on_done=report)


def screen_job(work: Work) -> Outcome:
    """Run one job at the screen's log level with the networks' epoch tables silenced: a
    worker started by forkserver inherits neither the level nor the console handler."""
    if not logging.getLogger().handlers:
        logging.basicConfig(format="%(message)s")
    logging.getLogger(surrogate.__name__).setLevel(logging.WARNING)
    job, context = work.job, work.context

    def record(run: Run) -> str:
        """Write the run's ledger entry, check it reads back, and name it."""
        settings: dict[str, Setting] = {
            "unit": job.candidate.unit,
            "rows": job.rows,
            "round": job.round,
            **asdict(context.knobs),
        }
        meta = RunMetadata(
            id=_new_id(),
            batch=context.batch,
            dataset=job.dataset,
            target=job.target,
            candidate=job.candidate.family,
            settings=settings,
            commit=context.commit,
            dirty=context.dirty,
            started=context.started,
            epochs=None,
            best_epoch=None,
        )
        entry = entry_from_run(run, meta)
        _write_checked(context.ledger / entry_name(entry), entry_to_json(entry))
        return meta.id

    with _at_level(context.log_level):
        fitter = fitter_of(job.candidate, work.space, context.knobs, context.training)
        return run_job(
            job,
            work.data,
            fitter,
            context.training.seed,
            context.training.valid_fraction,
            context.knobs.per_curve,
            context.seconds,
            context.memory_bytes,
            record,
        )


def _outcomes_path(state: Path, batch: str) -> Path:
    """Where a batch's outcomes are saved."""
    return state / SECTION / f"{batch}.json"


def _write_checked(path: Path, text: str) -> None:
    """Write the text; it must read back unchanged."""
    path.write_text(text)
    assert path.read_text() == text, f"{path} does not read back"


def _arguments(argv: list[str]) -> argparse.Namespace:
    """The inputs, the asset and state folders, and the knobs; mk/paper.mk passes every knob,
    so it is the one place their values are written."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ns", "bh", "split", "assets", "state"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--k", type=int, required=True, help="the k-NN's neighbours")
    parser.add_argument("--neighbours", type=int, required=True, help="the local RBF's")
    parser.add_argument("--knots", type=int, required=True, help="the curve-wise spline's")
    parser.add_argument("--per-curve", type=int, required=True, help="fewest rows per curve")
    parser.add_argument("--budgets", type=int, nargs="+", required=True, help="rows per round")
    parser.add_argument("--keeps", type=int, nargs="*", required=True, help="kept per round")
    parser.add_argument("--minutes", type=float, required=True, help="time budget of a fit")
    parser.add_argument("--memory-gb", type=float, required=True, help="memory budget of a fit")
    parser.add_argument("--workers", type=int, required=True, help="fits at once")
    parser.add_argument("--batch", help="fit nothing; rebuild the assets from this saved batch")
    return parser.parse_args(argv)


@contextmanager
def _at_level(level: str) -> Iterator[None]:
    """While the block runs, let the root logger pass records at level; then restore it."""
    root = logging.getLogger()
    previous = root.level
    root.setLevel(level)
    try:
        yield
    finally:
        root.setLevel(previous)


def _git_commit() -> str:
    """The short hash of the checked-out commit."""
    done = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def _git_dirty() -> bool:
    """Whether the work tree holds uncommitted changes."""
    done = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    return bool(done.stdout.strip())


def _new_id() -> str:
    """A new time-sortable id (uuid7), for a batch or a ledger entry."""
    return str(uuid.uuid7())


def _utc_now() -> datetime:
    """The current time in UTC."""
    return datetime.now(UTC)


if __name__ == "__main__":
    # The console handler only; main sets the level from the config for the run's duration.
    logging.basicConfig(format="%(message)s")
    main(sys.argv[1:])

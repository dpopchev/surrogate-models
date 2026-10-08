"""The representation screen behind Section 5.1 (W-066): one factor at a time from the baseline
representation, each scored at equal model.

The baseline representation is the one of Section 4.1: NS log10 rho_c, the target M or the
charge target Y = log10(D/M), one row at a time. Each other cell changes one factor -- NS rho_c
raw, the mass as log10 M, the charge as linear D, or the curve-wise unit (shared/curvewise.py) --
and is scored through shared/harness.py with the weighted k-NN, the local RBF (shared/families.py)
and the baseline MLP.
"""

import argparse
import logging
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Literal, assert_never

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from shared.config import PaperConfig, load_config
from shared.curvewise import curvewise_fitter
from shared.design import (
    BLACK_HOLES,
    NEUTRON_STARS,
    Design,
    DesignSpec,
    Target,
    TargetForm,
    design,
)
from shared.eda import CurveSpace, booktabs, make_curve_space
from shared.families import make_knn, make_rbf
from shared.harness import Fitter, Run, harness
from shared.runs import (
    LedgerEntry,
    RunMetadata,
    Setting,
    entry_from_json,
    entry_from_run,
    entry_name,
    entry_to_json,
)
from shared.scorecard import significant_figures
from shared.surrogate import Training, network_fitter, sklearn_fitter
from shared.trials import ReportEpochs, open_study, tell_run
from shared.workers import run_jobs

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

SECTION = "61_representation"

PAIRS: tuple[tuple[DesignSpec, Target], ...] = (
    (NEUTRON_STARS, "mass"),
    (NEUTRON_STARS, "charge"),
    (BLACK_HOLES, "mass"),
    (BLACK_HOLES, "charge"),
)
DATASET_TEX = {"neutron_stars": "NS", "black_holes": "BH"}
TARGET_TEX = {"mass": "$M$", "charge": "$\\Dch$"}
REPRESENTATION_TEX = {
    "baseline": "baseline",
    "raw rho_c": "raw $\\rhoc$",
    "log10 M": "$\\log_{10} M$",
    "linear D": "linear $\\Dch$",
    "curve-wise": "curve-wise",
}
# The NS design with rho_c on its raw scale, the one factor the raw rho_c cells change.
NS_RAW_RHO = DesignSpec(
    NEUTRON_STARS.dataset,
    make_curve_space({"rho_c": "raw", "beta": "raw", "lambda": "raw"}, ("beta", "lambda")),
)

Representation = Literal["baseline", "raw rho_c", "log10 M", "linear D", "curve-wise"]
# The unit of prediction: one row, or a whole curve along its normalized coordinate.
Unit = Literal["pointwise", "curve-wise"]


@dataclass(frozen=True)
class Cell:
    """One representation of one dataset and target: the design it reads, the target form
    the model sees and the unit it predicts."""

    dataset: str
    target: Target
    representation: Representation
    spec: DesignSpec
    form: TargetForm
    unit: Unit


Model = Literal["k-NN", "local RBF", "MLP"]
MODELS: tuple[Model, ...] = ("k-NN", "local RBF", "MLP")


@dataclass(frozen=True)
class Job:
    """One model on one cell for one seed: a harness run, a ledger entry and a trial."""

    cell: Cell
    model: Model
    seed: int


@dataclass(frozen=True)
class Score:
    """One model on one representation of a pair, over its seeds: the mean p95 significant
    figures on the test curves, and their mean and standard deviation over folds and seeds."""

    dataset: str
    target: str
    representation: str
    model: str
    test: float
    folds: float
    spread: float


@dataclass(frozen=True)
class Knobs:
    """The references' settings: the k-NN's k, the local RBF's neighbours and the curve-wise
    spline's knots (ends included)."""

    k: int
    neighbours: int
    knots: int


# --- pure functions ---------------------------------------------------------------------------


def cells() -> tuple[Cell, ...]:
    """Every representation cell of the screen, the baseline first within each pair."""
    found = []
    for spec, target in PAIRS:
        base: TargetForm = target
        found.append(Cell(spec.dataset, target, "baseline", spec, base, "pointwise"))
        if spec is NEUTRON_STARS:
            found.append(Cell(spec.dataset, target, "raw rho_c", NS_RAW_RHO, base, "pointwise"))
        changed: tuple[Representation, TargetForm] = (
            ("log10 M", "log_mass") if target == "mass" else ("linear D", "linear_charge")
        )
        found.append(Cell(spec.dataset, target, changed[0], spec, changed[1], "pointwise"))
        found.append(Cell(spec.dataset, target, "curve-wise", spec, base, "curve-wise"))
    return tuple(found)


def curve_columns(space: CurveSpace) -> tuple[tuple[int, ...], int]:
    """The design columns of the curve keys and of the one input along a curve."""
    names = [name for name, _ in space.inputs]
    along = next(i for i, name in enumerate(names) if name not in space.curve)
    return tuple(names.index(name) for name in space.curve), along


def plan(screened: Sequence[Cell], seeds: Sequence[int]) -> tuple[Job, ...]:
    """Every (cell, model, seed) job of the screen: the deterministic references on the first
    seed, the MLP on every seed; the MLP fits one output, so it is no curve-wise base (T-193)."""
    jobs = []
    for cell in screened:
        for model in MODELS:
            if model != "MLP":
                jobs.append(Job(cell, model, seeds[0]))
            elif cell.unit == "pointwise":
                jobs += [Job(cell, model, seed) for seed in seeds]
    return tuple(jobs)


def run_job(
    job: Job,
    data: Design,
    knobs: Knobs,
    training: Training,
    record: Callable[[dict[str, Setting], Run], None],
    before_fit: Callable[[Pipeline], None] = lambda _: None,
) -> Run:
    """Score one job through the harness on the cell's design; record is handed the job's
    settings and its run as it ends (the ledger entry); before_fit wraps every MLP fit.

    Raises ValueError for the MLP on a curve-wise cell.
    """
    cell = job.cell
    settings: dict[str, Setting] = {
        "representation": cell.representation,
        "form": cell.form,
        "unit": cell.unit,
        "model": job.model,
    }
    fitter: Fitter
    match job.model:
        case "k-NN":
            settings["k"] = knobs.k
            fitter = _reference_fitter(cell, partial(make_knn, knobs.k), knobs, settings)
        case "local RBF":
            settings["neighbours"] = knobs.neighbours
            fitter = _reference_fitter(cell, partial(make_rbf, knobs.neighbours), knobs, settings)
        case "MLP":
            if cell.unit != "pointwise":
                raise ValueError(
                    f"{cell.dataset} {cell.target}: the MLP fits one output, so it is no "
                    f"{cell.unit} base (T-193)"
                )
            settings |= {key: value for key, value in asdict(training).items() if key != "seed"}
            fitter = network_fitter(training, before_fit=before_fit)
        case _:
            assert_never(job.model)
    (run,) = harness(
        data,
        cell.form,
        fitter,
        [job.seed],
        training.valid_fraction,
        record=lambda ran: record(settings, ran),
    )
    return run


def scores(entries: Sequence[LedgerEntry]) -> tuple[Score, ...]:
    """The screen's ledger entries as one score per pair, representation and model, in the
    order each first appears."""
    groups: dict[tuple[str, str, str, str], list[LedgerEntry]] = {}
    for entry in entries:
        meta = entry.meta
        key = (meta.dataset, meta.target, str(meta.settings["representation"]), meta.candidate)
        groups.setdefault(key, []).append(entry)
    found = []
    for (dataset, target, representation, model), mine in groups.items():
        test = [significant_figures(entry.test.zones["test"].p95) for entry in mine]
        folds = [significant_figures(fold.p95) for entry in mine for fold in entry.folds]
        found.append(
            Score(
                dataset,
                target,
                representation,
                model,
                test=float(np.mean(test)),
                folds=float(np.mean(folds)),
                spread=float(np.std(folds)),
            )
        )
    return tuple(found)


def beats(score: Score, baseline: Score) -> bool:
    """Whether a representation's fold figures gain on the baseline's, at the same model, by
    more than the larger of their spreads over folds and seeds."""
    return score.folds - baseline.folds > max(score.spread, baseline.spread)


def representation_table(found: Sequence[Score]) -> str:
    """The Section 5.1 table: one row per pair and representation, each model's test figures
    and fold figures (mean +/- standard deviation over folds and seeds), a fold entry in bold
    where it beats the pair's baseline at that model."""
    by_key = {(s.dataset, s.target, s.representation, s.model): s for s in found}
    rows = []
    for dataset, target, representation in dict.fromkeys(key[:3] for key in by_key):
        entries = [
            _entry(
                by_key.get((dataset, target, representation, model)),
                by_key.get((dataset, target, "baseline", model)),
            )
            for model in MODELS
        ]
        rows.append(
            f"{DATASET_TEX[dataset]} & {TARGET_TEX[target]} & "
            f"{REPRESENTATION_TEX[representation]} & {' & '.join(entries)}"
        )
    return booktabs(
        "representation",
        "Each representation at equal model, one factor changed from the baseline: the "
        "significant figures kept at the 95th percentile of the error ($-\\log_{10}$) on the test "
        "curves, and on the frozen folds as mean $\\pm$ standard deviation over folds and seeds "
        "(the MLP on three seeds, the deterministic references on one). A fold entry in bold "
        "gains on the pair's baseline at that model by more than the larger of the two "
        "deviations. The errors of every target form are measured in $M$ or $\\Dch$.",
        "lllrrrrrr",
        "Data & Target & Representation & \\multicolumn{2}{c}{$k$-NN} & "
        "\\multicolumn{2}{c}{local RBF} & \\multicolumn{2}{c}{MLP} \\\\\n"
        " & & & Test & Folds & Test & Folds & Test & Folds",
        rows,
    )


def _entry(score: Score | None, baseline: Score | None) -> str:
    """One model's two columns of a table row: test figures, then fold mean +/- deviation, in
    bold when it beats the baseline; dashes for a model the cell does not run (the MLP on
    curve-wise)."""
    if score is None:
        return "-- & --"
    folds = f"{score.folds:.2f} \\pm {score.spread:.2f}"
    if baseline is not None and beats(score, baseline):
        folds = f"\\mathbf{{{folds}}}"
    return f"{score.test:.2f} & ${folds}$"


def _reference_fitter(
    cell: Cell, make_base: Callable[[], Pipeline], knobs: Knobs, settings: dict[str, Setting]
) -> Fitter:
    """A reference on the cell's unit: fitted row by row, or as the base of the curve-wise
    spline, whose knots then join the settings."""
    match cell.unit:
        case "pointwise":
            return sklearn_fitter(make_base)
        case "curve-wise":
            keys, along = curve_columns(cell.spec.space)
            settings["knots"] = knobs.knots
            return curvewise_fitter(make_base, keys, along, knobs.knots)
        case _:
            assert_never(cell.unit)


# --- shell ------------------------------------------------------------------------------------


def main(
    argv: list[str],
    config: PaperConfig | None = None,
    commit: Callable[[], str] | None = None,
    dirty: Callable[[], bool] | None = None,
    now: Callable[[], datetime] | None = None,
) -> None:
    """Run every job of the screen as a trial of one study, each leaving its ledger entry;
    write the representation table from the batch's entries into the section's asset folder,
    after emptying that folder of an earlier run."""
    args = _arguments(argv)
    for path in (args.ns, args.bh, args.split):
        assert path.is_file(), f"input not found: {path}"
    config = config or load_config()
    commit, dirty, now = commit or _git_commit, dirty or _git_dirty, now or _utc_now
    settings = config.methodology.algorithms
    training = Training(**settings.model_dump(exclude={"log_level", "threads", "workers"}))
    tables = {
        NEUTRON_STARS.dataset: pd.read_parquet(args.ns),
        BLACK_HOLES.dataset: pd.read_parquet(args.bh),
    }
    split = pd.read_parquet(args.split)
    screened = cells()
    designs = {cell: design(tables[cell.dataset], split, cell.spec, cell.form) for cell in screened}
    batch, ledger = _new_id(), args.state / "ledger"
    ledger.mkdir(parents=True, exist_ok=True)
    context = _Context(
        batch=batch,
        ledger=ledger,
        journal=args.state / "optuna" / "journal.log",
        commit=commit(),
        dirty=dirty(),
        started=now(),
        knobs=Knobs(args.k, args.neighbours, args.knots),
        training=training,
    )
    # The batch's study is made once here; every job joins it as its trial runs.
    open_study(batch, context.journal)
    seeds = tuple(training.seed + i for i in range(args.seeds))
    jobs = [
        partial(_screen_job, _Work(job, designs[job.cell], context))
        for job in plan(screened, seeds)
    ]
    out = args.assets / SECTION
    with _at_level(settings.log_level):
        logger.info(
            "%d jobs on %d workers, batch %s; follow them with make dashboard",
            len(jobs),
            settings.workers,
            batch,
        )
        run_jobs(jobs, settings.workers)
        entries = [entry for entry in _read_ledger(ledger) if entry.meta.batch == batch]
        assert len(entries) == len(jobs), (
            f"batch {batch}: {len(entries)} ledger entries for {len(jobs)} jobs"
        )
        path = _write_table(out, representation_table(scores(entries)))
        logger.info("done: representation table of batch %s -> %s", batch, path)


@dataclass(frozen=True)
class _Context:
    """What every job of one run shares: the batch, where its ledger entries and its study
    journal go, the code it ran from and when, and the settings of its models."""

    batch: str
    ledger: Path
    journal: Path
    commit: str
    dirty: bool
    started: datetime
    knobs: Knobs
    training: Training


@dataclass(frozen=True)
class _Work:
    """One job with its cell's design, run in a worker process."""

    job: Job
    data: Design
    context: _Context


def _screen_job(work: _Work) -> None:
    """Run one job as a trial of the batch's study: its ledger entry is written and told to the
    trial as the run ends; an MLP reports each epoch to the trial."""
    job, context = work.job, work.context
    cell = job.cell
    study = open_study(context.batch, context.journal)
    trial = study.ask()
    attributes = {
        "dataset": cell.dataset,
        "target": cell.target,
        "candidate": job.model,
        "representation": cell.representation,
    }
    for key, value in attributes.items():
        trial.set_user_attr(key, value)

    def record(settings: dict[str, Setting], ran: Run) -> None:
        """Write the run's ledger entry, check it reads back, then tell the trial."""
        meta = RunMetadata(
            id=_new_id(),
            batch=context.batch,
            dataset=cell.dataset,
            target=cell.target,
            candidate=job.model,
            settings=settings,
            commit=context.commit,
            dirty=context.dirty,
            started=context.started,
            epochs=None,
            best_epoch=None,
        )
        entry = entry_from_run(ran, meta)
        path = context.ledger / entry_name(entry)
        text = entry_to_json(entry)
        path.write_text(text)
        assert entry_to_json(entry_from_json(path.read_text())) == text, (
            f"{path} does not round-trip"
        )
        tell_run(study, trial, entry)

    built: list[Pipeline] = []

    def before_fit(pipeline: Pipeline) -> None:
        """Report this fit's epochs to the trial after the epochs of its earlier fits."""
        offset = sum(len(p.named_steps["net"].history) for p in built)
        built.append(pipeline)
        net = pipeline.named_steps["net"]
        net.callbacks = [*net.callbacks, ("report", ReportEpochs(trial, offset))]

    run_job(job, work.data, context.knobs, context.training, record, before_fit)
    logger.info(
        "job done: %s %s, %s, %s, seed %d",
        cell.dataset,
        cell.target,
        cell.representation,
        job.model,
        job.seed,
    )


def _read_ledger(folder: Path) -> list[LedgerEntry]:
    """Every entry in the ledger folder."""
    return [entry_from_json(path.read_text()) for path in sorted(folder.glob("*.json"))]


def _write_table(out: Path, table: str) -> Path:
    """Empty the section's asset folder, then write the table; it must read back unchanged."""
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.iterdir():
        stale.unlink()
    path = out / f"{SECTION}_tab_representation.tex"
    path.write_text(table)
    assert path.read_text() == table, f"{path} does not read back"
    return path


def _arguments(argv: list[str]) -> argparse.Namespace:
    """The inputs, the asset and state folders, and the knobs; mk/paper.mk passes every knob,
    so it is the one place their values are written."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ns", "bh", "split", "assets", "state"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--k", type=int, required=True, help="the k-NN's neighbours")
    parser.add_argument("--neighbours", type=int, required=True, help="the local RBF's")
    parser.add_argument("--knots", type=int, required=True, help="the curve-wise spline's")
    parser.add_argument("--seeds", type=int, required=True, help="MLP seeds from paper.toml's")
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

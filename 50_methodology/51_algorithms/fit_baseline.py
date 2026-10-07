"""The baseline surrogate behind Section 4.1 (W-034, W-036): a plain MLP on NS and BH, mass
and charge, against the mean and nearest-curve references.

Inputs: the prepared NS and BH tables and the frozen curve split (local/state/). Per pair the
net is scored on each frozen fold and, trained on every non-test curve through shared/design.py
and shared/surrogate.py with early stopping on held-out training curves, once on the test
curves. Outputs, in the section's asset folder: the baseline table, the \\baseNsMass... macros
(test MARE, RMSE, fit seconds, epochs) and the NS mass parity figure; in
<state dir>/51_algorithms/<run>/, one run record and its diagnostics per pair (W-035); in
<state dir>/ledger/, one ledger entry per predictor, pair and seed, written as each ends (W-077).

Run as `uv run python <this file> <ns.parquet> <bh.parquet> <split.parquet> <asset dir>
<state dir>` (mk/paper.mk does).
"""

import logging
import multiprocessing
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, assert_never

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from sklearn.pipeline import Pipeline

from shared.ceilings import Spread
from shared.config import AlgorithmsSection, PaperConfig, load_config
from shared.design import BLACK_HOLES, NEUTRON_STARS, Design, DesignSpec, Target, design
from shared.diagnostics import curve_overlay, error_cdf, loss_curve
from shared.eda import booktabs, curve_ids, render_macros, sci_tex
from shared.harness import Fitter, Predictor, Run, Timing, harness
from shared.plots import PlotStyle, anchor_color, apply_style
from shared.runs import (
    LedgerEntry,
    RunMetadata,
    RunRecord,
    Setting,
    entry_from_json,
    entry_from_run,
    entry_name,
    entry_to_json,
    from_json,
    run_name,
    run_stem,
    to_json,
)
from shared.scorecard import significant_figures
from shared.surrogate import (
    LIVE_EVERY,
    Training,
    duration_text,
    make_mean_reference,
    make_nearest_reference,
    mare,
    mare_in_d,
    network_fitter,
    rebuild_charge,
    rmse,
    sklearn_fitter,
)
from shared.trials import ReportEpochs, open_study, tell_run
from shared.workers import run_jobs

logger = logging.getLogger(__name__)

SECTION = "51_algorithms"
# The diagnostics are for the screen, not print: a lower resolution than the paper's figures.
DIAGNOSTICS_DPI = 150

# --- vocabulary and types ---------------------------------------------------------------------

# A score of predictions against the truth on the target's rows: mare, or mare_in_d for the
# charge target, whose error is measured in D rebuilt with the true M (W-073).
Metric = Callable[[np.ndarray, np.ndarray], float]
# record(candidate, settings, run, (epochs, best epoch) or None): keeps one harness run.
Recorder = Callable[[str, dict[str, Setting], Run, tuple[int, int] | None], None]


@dataclass(frozen=True)
class Fit:
    """One fitted baseline scored on the test curves, with its per-epoch history (batches
    dropped) and the test rows' inputs and curves for the diagnostics."""

    y_true: np.ndarray
    y_pred: np.ndarray
    mare: float
    rmse: float
    seconds: float
    epochs: int
    best_epoch: int
    history: tuple[dict[str, Any], ...]
    x_test: np.ndarray
    groups_test: np.ndarray


@dataclass(frozen=True)
class Scored:
    """One predictor's scores on a pair through the harness: the error spread on the test
    curves and on each frozen fold, and the timing of its final fit and calls."""

    predictor: str
    test: Spread
    folds: tuple[Spread, ...]
    timing: Timing


@dataclass(frozen=True)
class PairScores:
    """The scores of every predictor on one dataset and target."""

    dataset: str
    target: str
    scored: tuple[Scored, ...]


PAIRS: tuple[tuple[DesignSpec, Target], ...] = (
    (NEUTRON_STARS, "mass"),
    (NEUTRON_STARS, "charge"),
    (BLACK_HOLES, "mass"),
    (BLACK_HOLES, "charge"),
)
DATASET_TEX = {"neutron_stars": "NS", "black_holes": "BH"}
TARGET_TEX = {"mass": "$M$", "charge": "$\\Dch$"}
X_LABELS = {"neutron_stars": "$\\log_{10}\\rho_c$", "black_holes": "$r_h$"}


# --- pure functions ---------------------------------------------------------------------------


def final_fit(design: Design, run: Run, estimator: Pipeline, score: Metric) -> Fit:
    """The MLP's final fit from its harness run and its fitted network: the test rows, the
    run's predictions on them, the timing and the per-epoch history (batches dropped)."""
    test = design.test
    y_true, y_pred = design.y[test], run.predictions
    history = tuple(
        {key: value for key, value in row.items() if key != "batches"}
        for row in estimator.named_steps["net"].history
    )
    _, best_epoch = _epochs(estimator)
    return Fit(
        y_true=y_true,
        y_pred=y_pred,
        mare=score(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        seconds=run.timing.fit,
        epochs=len(history),
        best_epoch=best_epoch,
        history=history,
        x_test=design.X[test],
        groups_test=design.groups[test],
    )


def _counted(fitter: Fitter, after_fit: Callable[[], None]) -> Fitter:
    """The fitter, calling after_fit once each of its fits is done."""

    def fit(
        x_fit: np.ndarray, y_fit: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, seed: int
    ) -> Predictor:
        predict = fitter(x_fit, y_fit, x_valid, y_valid, seed)
        after_fit()
        return predict

    return fit


def _epochs(network: Pipeline) -> tuple[int, int]:
    """The epochs a fitted network ran and the epoch of its lowest validation loss."""
    history = network.named_steps["net"].history
    best = min(history, key=lambda row: row["valid_loss"])
    return len(history), int(best["epoch"])


def _recording(
    record: Recorder,
    name: str,
    settings: dict[str, Setting],
    epochs: Callable[[], tuple[int, int] | None],
) -> Callable[[Run], None]:
    """The harness's record of one candidate: its run, with its name, settings and epochs."""
    return lambda run: record(name, settings, run, epochs())


# The predictors of the baseline table, in its row order.
CANDIDATES = ("MLP", "mean", "nearest curve")


def score_candidate(
    design: Design,
    target: Target,
    training: Training,
    name: str,
    live_plot: Path | None = None,
    after_fit: Callable[[], None] = lambda: None,
    clock: Callable[[], float] | None = None,
    record: Recorder = lambda name, settings, run, epochs: None,
    before_fit: Callable[[Pipeline], None] = lambda _: None,
) -> tuple[Scored, Run, Pipeline | None]:
    """Score one candidate of CANDIDATES through the harness; after_fit and before_fit wrap
    every MLP fit, record is handed its run as it ends. Returns its scores, its run and, for
    the MLP, its last fitted network."""
    networks: list[Pipeline] = []
    fitter: Fitter
    settings: dict[str, Setting]
    match name:
        case "MLP":
            network = network_fitter(training, live_plot, networks.append, before_fit)
            fitter, settings = _counted(network, after_fit), asdict(training)
        case "mean":
            fitter, settings = sklearn_fitter(make_mean_reference), {"estimator": name}
        case "nearest curve":
            fitter, settings = sklearn_fitter(make_nearest_reference), {"estimator": name}
        case _:
            raise ValueError(f"no candidate {name!r}; the candidates are {CANDIDATES}")
    epochs = (lambda: _epochs(networks[-1])) if name == "MLP" else (lambda: None)
    (run,) = harness(
        design,
        target,
        fitter,
        [training.seed],
        training.valid_fraction,
        clock=clock,
        record=_recording(record, name, settings, epochs),
    )
    scored = Scored(name, run.test.zones["test"], run.folds, run.timing)
    return scored, run, networks[-1] if networks else None


def score_pair(
    design: Design,
    dataset: str,
    target: Target,
    training: Training,
    live_plot: Path | None = None,
    after_fit: Callable[[], None] = lambda: None,
    clock: Callable[[], float] | None = None,
    record: Recorder = lambda name, settings, run, epochs: None,
) -> tuple[PairScores, Fit]:
    """Score one dataset and target through the harness: the MLP, then the mean and the
    nearest-curve references, each on the same folds, validation curves and test curves.
    after_fit is called after every MLP fit; record is handed each candidate's run as it ends.
    Returns the scores and the MLP's final fit, for the figures and the run record."""
    found = [
        score_candidate(design, target, training, name, live_plot, after_fit, clock, record)
        for name in CANDIDATES
    ]
    _, mlp_run, network = found[0]
    assert network is not None, "the MLP left no fitted network"
    fit = final_fit(design, mlp_run, network, metric_for(target))
    return PairScores(dataset, target, tuple(scored for scored, _, _ in found)), fit


def metric_for(target: Target) -> Metric:
    """The score of a target: the MARE of M, or of D rebuilt with the true M for the charge."""
    match target:
        case "mass":
            return mare
        case "charge":
            return mare_in_d
        case _:
            assert_never(target)


def _row(pair: PairScores, scored: Scored) -> str:
    """One table row: the pair, the predictor, its test MARE, fold mean +/- std, the
    significant figures at the 95th percentile, the fit seconds and the one-row call seconds."""
    folds = np.array([fold.mean for fold in scored.folds])
    return (
        f"{DATASET_TEX[pair.dataset]} & {TARGET_TEX[pair.target]} & {scored.predictor} & "
        f"${sci_tex(scored.test.mean)}$ & "
        f"${sci_tex(float(folds.mean()))} \\pm {sci_tex(float(folds.std()))}$ & "
        f"{significant_figures(scored.test.p95):.1f} & {scored.timing.fit:.0f} & "
        f"${sci_tex(scored.timing.predict_one)}$"
    )


def baseline_table(pairs: list[PairScores]) -> str:
    """The Section 4.1 table: every predictor on every dataset and target."""
    return booktabs(
        "baseline",
        "The baseline MLP against the mean and nearest-curve references, every predictor "
        "fitted on the same curves and timed on one thread: MARE on the test curves and on the "
        "frozen folds (mean $\\pm$ standard deviation), the significant figures kept at the "
        "95th percentile of the test error ($-\\log_{10}$), the fit time and the time of a "
        "one-row call in seconds. The $\\Dch$ rows give the error of the charge rebuilt with "
        "the true $M$ from the predicted $Y = \\log_{10}(\\Dch/M)$.",
        "lllllrrr",
        "Data & Target & Predictor & Test MARE & Fold MARE & Figures & Fit s & Call s",
        [_row(pair, scored) for pair in pairs for scored in pair.scored],
    )


def numbers(fit: Fit) -> dict[str, str]:
    """The \\baseNsMass... macros of one fit."""
    return {
        "baseNsMassMare": sci_tex(fit.mare),
        "baseNsMassRmse": sci_tex(fit.rmse),
        "baseNsMassFitSeconds": f"{fit.seconds:.0f}",
        "baseNsMassEpochs": str(fit.epochs),
    }


def charge_mares(mass: Fit, charge: Fit) -> tuple[float, float]:
    """The MARE of D rebuilt from the charge model's Y with the true M, and with the mass
    model's predicted M; both fits are of one dataset, on the same test rows in the same order."""
    d_true = rebuild_charge(charge.y_true, mass.y_true)
    with_true_m = rebuild_charge(charge.y_pred, mass.y_true)
    with_pred_m = rebuild_charge(charge.y_pred, mass.y_pred)
    return mare(d_true, with_true_m), mare(d_true, with_pred_m)


def charge_numbers(mares: dict[str, tuple[float, float]]) -> dict[str, str]:
    """The \\base<Ns|Bh>ChargeMareTrueM and ...MarePredM macros, keyed by dataset."""
    found = {}
    for dataset, (true_m, pred_m) in mares.items():
        tag = DATASET_TEX[dataset].capitalize()
        found[f"base{tag}ChargeMareTrueM"] = sci_tex(true_m)
        found[f"base{tag}ChargeMarePredM"] = sci_tex(pred_m)
    return found


def start_banner(
    run: Path,
    out: Path,
    training: Training,
    rows: tuple[int, int],
    curves: tuple[int, int],
    threads: int,
) -> list[str]:
    """The lines a run's log opens with: the stop criterion first, then what it fits on, the
    files to watch while it runs and those it writes at the end; rows and curves are
    (training, test)."""
    data = (
        f"data: {rows[0]} training rows on {curves[0]} curves, "
        f"{rows[1]} test rows on {curves[1]} curves"
    )
    return [
        "stop criterion:",
        f"  - early stop after {training.patience} epochs in a row without a validation loss "
        "0.01% below the best",
        "  - the best epoch is then restored",
        f"  - at most {training.max_epochs} epochs",
        f"folder: {run}",
        data,
        f"validation: {training.valid_fraction:.0%} of the training curves",
        f"threads: {threads} torch threads per fit",
        "live while it runs:",
        f"  - {run / 'train.log'} (this log)",
        f"  - {run / 'loss_curve.png'} (redrawn every {LIVE_EVERY} epochs)",
        "at the end:",
        f"  - {run}/ (run.json, error_cdf.png, curves.png)",
        f"  - {out}/ (Section 4.1 numbers, parity figure)",
        "follow live: make follow",
    ]


def end_banner(fit: Fit, training: Training, files: list[Path]) -> list[str]:
    """The lines a run's log closes with: why it stopped, how it scored, what it wrote."""
    stopped = (
        f"stopped: max_epochs {training.max_epochs} reached"
        if fit.epochs >= training.max_epochs
        else (
            f"stopped: early -- no better validation loss for {training.patience} epochs "
            f"after epoch {fit.best_epoch}"
        )
    )
    scores = (
        f"best epoch {fit.best_epoch} restored; test MARE {fit.mare:.2e}, RMSE {fit.rmse:.2e}; "
        f"fit {duration_text(fit.seconds)}"
    )
    return [
        stopped,
        scores,
        "wrote:",
        *(f"  {path}" for path in files),
        "list runs: make runs; show this run: make run",
    ]


def progress_line(done: int, total: int, elapsed: float) -> str:
    """How far a series of fits has come and, from the mean time per fit, how long is left."""
    left = elapsed / done * (total - done)
    return (
        f"fit {done}/{total} done, elapsed {duration_text(elapsed)}, "
        f"about {duration_text(left)} left"
    )


SYMBOLS = {"beta": "$\\beta$", "lambda": "$\\lambda$"}


def curve_names(table: pd.DataFrame, spec: DesignSpec) -> dict[int, str]:
    """Each curve id of the table named by its key, e.g. "$\\beta$ = 1, $\\lambda$ = 2"."""
    keys = table[list(spec.space.curve)].assign(curve=curve_ids(table, spec.space))
    return {
        int(row["curve"]): ", ".join(
            f"{SYMBOLS.get(column, column)} = {row[column]:g}" for column in spec.space.curve
        )
        for _, row in keys.drop_duplicates("curve").iterrows()
    }


def parity(fit: Fit, style: PlotStyle) -> Figure:
    """Predicted against true mass on the test curves, with the y = x line."""
    side = 0.5 * float(plt.rcParams["figure.figsize"][0])
    figure, axes = plt.subplots(figsize=(side, side), layout="constrained")
    axes.scatter(fit.y_true, fit.y_pred, s=2, alpha=0.3, color=anchor_color(style, "neutron_stars"))
    span = [
        float(min(fit.y_true.min(), fit.y_pred.min())),
        float(max(fit.y_true.max(), fit.y_pred.max())),
    ]
    axes.plot(span, span, color="black", linestyle="--", linewidth=0.8)
    axes.set_xlabel("true $M$ ($M_\\odot$)")
    axes.set_ylabel("predicted $M$ ($M_\\odot$)")
    axes.set_aspect("equal")
    return figure


PARITY = f"{SECTION}_fig_parity"


def figure_tex() -> str:
    """The figure environment of the parity figure, placed here or at the top of a page."""
    return (
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=0.5\\textwidth]{{{PARITY}}}\n"
        "\\caption{Neutron stars: the baseline MLP's predicted mass against the true mass on "
        "the test curves; the dashed line is $y = x$.}\n"
        "\\label{fig:alg-parity}\n\\end{figure}\n"
    )


# --- shell ------------------------------------------------------------------------------------


def main(
    argv: list[str],
    config: PaperConfig | None = None,
    commit: Callable[[], str] | None = None,
    dirty: Callable[[], bool] | None = None,
    now: Callable[[], datetime] | None = None,
) -> None:
    """Score the baseline and its references on every dataset and target; write the baseline
    table, the NS mass numbers and parity figure into the section's folder of the asset dir,
    after emptying that folder of an earlier run; leave each pair's run record and diagnostics
    in a folder named after its run under the state dir."""
    assert len(argv) == 5, (
        "usage: fit_baseline.py <ns.parquet> <bh.parquet> <split.parquet> <asset dir> "
        f"<state dir>, got {argv}"
    )
    ns, bh, split_file = Path(argv[0]), Path(argv[1]), Path(argv[2])
    out, state = Path(argv[3]) / SECTION, Path(argv[4]) / SECTION
    for path in (ns, bh, split_file):
        assert path.is_file(), f"input not found: {path}"
    config = config or load_config()
    commit, dirty, now = commit or _git_commit, dirty or _git_dirty, now or _utc_now
    settings = config.methodology.algorithms
    apply_style(config.plot)
    torch.set_num_threads(settings.threads)
    training = Training(**settings.model_dump(exclude={"log_level", "threads", "workers"}))
    tables = {NEUTRON_STARS.dataset: pd.read_parquet(ns), BLACK_HOLES.dataset: pd.read_parquet(bh)}
    split = pd.read_parquet(split_file)
    designs: list[tuple[DesignSpec, Target, Design]] = [
        (spec, target, design(tables[spec.dataset], split, spec, target)) for spec, target in PAIRS
    ]
    total = sum(len(set(data.fold[data.fold >= 0].tolist())) + 1 for _, _, data in designs)
    first = time.time()
    head, tail = commit(), dirty()
    batch, ledger = _new_id(), Path(argv[4]) / "ledger"
    journal = Path(argv[4]) / "optuna" / "journal.log"
    ledger.mkdir(parents=True, exist_ok=True)
    # The batch's study is made once here; every job joins it as its trials run.
    open_study(batch, journal)
    planned: list[tuple[_Pair, _Run]] = []
    with multiprocessing.Manager() as manager:
        progress = _Progress(manager.Value("i", 0), manager.Lock(), total, first)
        jobs = []
        for spec, target, data in designs:
            started = now()
            folder = state / run_stem(spec.dataset, target, started)
            folder.mkdir(parents=True, exist_ok=True)
            # latest.log is what make follow tails: repointed at each run's log as it is made.
            latest = state / "latest.log"
            latest.unlink(missing_ok=True)
            latest.symlink_to(Path(folder.name) / "train.log")
            pair = _Pair(spec, target, data, tables[spec.dataset])
            run = _Run(folder, out, started, head, tail, batch, ledger, journal)
            planned.append((pair, run))
            jobs += [
                partial(_score_job, _Job(pair, run, name, training, settings, progress))
                for name in CANDIDATES
            ]
        logger.info(
            "%d jobs on %d workers; follow them with make dashboard", len(jobs), settings.workers
        )
        found = run_jobs(jobs, settings.workers)
    pairs, fits = [], {}
    for i, (pair, run) in enumerate(planned):
        mine = found[i * len(CANDIDATES) : (i + 1) * len(CANDIDATES)]
        pairs.append(PairScores(pair.spec.dataset, pair.target, tuple(s for s, _ in mine)))
        mlp_fit = mine[0][1]
        assert mlp_fit is not None, f"{run.folder.name}: the MLP job returned no fit"
        fits[pair.spec.dataset, pair.target] = mlp_fit
    run = planned[-1][1].folder
    assert (NEUTRON_STARS.dataset, "mass") in fits, "the NS mass pair did not run"
    mares = {}
    for dataset in (NEUTRON_STARS.dataset, BLACK_HOLES.dataset):
        mass, charge = fits[dataset, "mass"], fits[dataset, "charge"]
        assert np.array_equal(mass.groups_test, charge.groups_test), (
            f"{dataset}: the mass and charge fits are not scored on the same test rows"
        )
        mares[dataset] = charge_mares(mass, charge)
    # The section's assets are written last, into the last run's log that make follow shows.
    with _logging_into(run / "train.log", settings.log_level):
        _write_section(out, pairs, fits[NEUTRON_STARS.dataset, "mass"], config, mares)
        logger.info("all %d fits done in %s", total, duration_text(time.time() - first))
        logger.info("done: baseline table, NS mass numbers and parity figure -> %s", out)


@dataclass(frozen=True)
class _Pair:
    """One dataset and target to score, with the table its curve names come from."""

    spec: DesignSpec
    target: Target
    data: Design
    table: pd.DataFrame


@dataclass(frozen=True)
class _Run:
    """Where and from what one pair's run is recorded; batch names the make baseline run it
    belongs to, ledger the folder its ledger entries go to."""

    folder: Path
    out: Path
    started: datetime
    commit: str
    dirty: bool
    batch: str
    ledger: Path
    journal: Path


@dataclass(frozen=True)
class _Progress:
    """The fit count every worker shares for the progress line: a counter and its lock from a
    multiprocessing manager, the fits in all and the wall-clock start."""

    count: Any
    lock: Any
    total: int
    first: float


@dataclass(frozen=True)
class _Job:
    """One candidate on one pair, run in a worker process."""

    pair: _Pair
    run: _Run
    candidate: str
    training: Training
    settings: AlgorithmsSection
    progress: _Progress


def _tick(progress: _Progress) -> None:
    """Count one more MLP fit across every worker and log how far the run has come."""
    with progress.lock:
        progress.count.value += 1
        done = progress.count.value
    logger.info(progress_line(done, progress.total, time.time() - progress.first))


def _score_job(job: _Job) -> tuple[Scored, Fit | None]:
    """Score one candidate on one pair as a trial of the batch's study, logging into the
    pair's train.log; every fit leaves its ledger entry, which the trial is told. The MLP job
    reports each epoch to the trial and leaves the pair's run record and diagnostics."""
    pair, run = job.pair, job.run
    with _logging_into(run.folder / "train.log", job.settings.log_level):
        study = open_study(run.batch, run.journal)
        trial = study.ask()
        record = _ledger_writer(pair, run, lambda entry: tell_run(study, trial, entry))
        if job.candidate != "MLP":
            scored, _, _ = score_candidate(
                pair.data, pair.target, job.training, job.candidate, record=record
            )
            return scored, None
        built: list[Pipeline] = []

        def before_fit(pipeline: Pipeline) -> None:
            """Report this fit's epochs to the trial after the epochs of its earlier fits."""
            offset = sum(len(p.named_steps["net"].history) for p in built)
            built.append(pipeline)
            net = pipeline.named_steps["net"]
            net.callbacks = [*net.callbacks, ("report", ReportEpochs(trial, offset))]

        return _score_mlp(job, record, before_fit)


def _score_mlp(
    job: _Job, record: Recorder, before_fit: Callable[[Pipeline], None]
) -> tuple[Scored, Fit]:
    """Score the MLP on one pair, its log lines going to the run's train.log; leave its run
    record and diagnostics in the run's folder."""
    pair, run, training = job.pair, job.run, job.training
    if run.dirty:
        logger.info("the code holds uncommitted changes: run.json marks this run dirty")
    data, train = pair.data, ~pair.data.test
    rows = (int(train.sum()), int(data.test.sum()))
    curves = (len(np.unique(data.groups[train])), len(np.unique(data.groups[data.test])))
    logger.info("== baseline run: %s / %s ==", pair.spec.dataset, pair.target)
    for line in start_banner(run.folder, run.out, training, rows, curves, job.settings.threads):
        logger.info(line)
    scored, ran, network = score_candidate(
        data,
        pair.target,
        training,
        "MLP",
        live_plot=run.folder / "loss_curve.png",
        after_fit=partial(_tick, job.progress),
        record=record,
        before_fit=before_fit,
    )
    assert network is not None, "the MLP left no fitted network"
    fit = final_fit(data, ran, network, metric_for(pair.target))
    summary = RunRecord(
        dataset=pair.spec.dataset,
        target=pair.target,
        training=training,
        commit=run.commit,
        dirty=run.dirty,
        started=run.started,
        seconds=fit.seconds,
        epochs=fit.epochs,
        best_epoch=fit.best_epoch,
        mare=fit.mare,
        rmse=fit.rmse,
    )
    folder = run.folder
    assert folder.name == run_name(summary), f"run folder {folder.name} is not {run_name(summary)}"
    names = curve_names(pair.table, pair.spec)
    write_diagnostics(fit, summary, folder, names, X_LABELS[pair.spec.dataset])
    assert from_json((folder / "run.json").read_text()) == summary, (
        f"{folder}/run.json does not round-trip"
    )
    logger.info("== baseline run finished ==")
    for line in end_banner(fit, training, sorted(folder.iterdir())):
        logger.info(line)
    return scored, fit


def _ledger_writer(
    pair: _Pair, run: _Run, on_entry: Callable[[LedgerEntry], None] = lambda _: None
) -> Recorder:
    """The recorder writing each harness run of the pair to <ledger>/<id>.json, a new uuid7
    id per entry, then handing the entry to on_entry (the trial is told it); the text must
    read back to itself (NaN spreads compare unequal as values)."""

    def record(
        candidate: str, settings: dict[str, Setting], ran: Run, epochs: tuple[int, int] | None
    ) -> None:
        meta = RunMetadata(
            id=_new_id(),
            batch=run.batch,
            dataset=pair.spec.dataset,
            target=pair.target,
            candidate=candidate,
            settings=settings,
            commit=run.commit,
            dirty=run.dirty,
            started=run.started,
            epochs=None if epochs is None else epochs[0],
            best_epoch=None if epochs is None else epochs[1],
        )
        entry = entry_from_run(ran, meta)
        path = run.ledger / entry_name(entry)
        text = entry_to_json(entry)
        path.write_text(text)
        assert entry_to_json(entry_from_json(path.read_text())) == text, (
            f"{path} does not round-trip"
        )
        on_entry(entry)

    return record


def _write_section(
    out: Path,
    pairs: list[PairScores],
    ns_mass: Fit,
    config: PaperConfig,
    mares: dict[str, tuple[float, float]],
) -> None:
    """Empty the section's asset folder, then write the baseline table, the NS mass numbers
    with the charge MAREs of each dataset, and the NS mass parity figure."""
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.iterdir():
        stale.unlink()
    (out / f"{SECTION}_tab_baseline.tex").write_text(baseline_table(pairs))
    figure = parity(ns_mass, config.plot)
    figure.savefig(out / f"{PARITY}.png", dpi=config.plot.dpi)
    plt.close(figure)
    (out / f"{PARITY}.tex").write_text(figure_tex())
    macros = numbers(ns_mass) | charge_numbers(mares)
    (out / f"{SECTION}_num.tex").write_text(render_macros(macros))


def write_diagnostics(
    fit: Fit, record: RunRecord, run: Path, names: dict[int, str], x_label: str
) -> None:
    """Write the run record, the loss curve, the error CDF and the curve overlay (curves
    labelled by names, the first input on the x axis as x_label) into run."""
    run.mkdir(parents=True, exist_ok=True)
    (run / "run.json").write_text(to_json(record))
    overlay = curve_overlay(
        fit.x_test[:, 0], fit.y_true, fit.y_pred, fit.groups_test, k=3, names=names
    )
    overlay.axes[0].set_xlabel(x_label)
    figures = {
        "loss_curve": loss_curve(fit.history),
        "error_cdf": error_cdf(np.abs((fit.y_pred - fit.y_true) / fit.y_true)),
        "curves": overlay,
    }
    for name, figure in figures.items():
        figure.savefig(run / f"{name}.png", dpi=DIAGNOSTICS_DPI)
        plt.close(figure)


@contextmanager
def _logging_into(log: Path, level: str) -> Iterator[None]:
    """While the block runs, log at level to the console handlers and to the file log; then
    detach the file and restore the level, also when the block raises."""
    root = logging.getLogger()
    handler = logging.FileHandler(log)
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous = root.level
    root.setLevel(level)
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(previous)


def _git_commit() -> str:
    """The short hash of the checked-out commit."""
    done = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def _git_dirty() -> bool:
    """Whether the work tree holds uncommitted changes (the query of 00_metadata/build_stamp.py)."""
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

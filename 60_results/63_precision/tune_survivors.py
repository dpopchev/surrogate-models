"""The tuning behind Section 5.3 (W-068): every candidate on the short list of the family
screen (Section 5.2) searched over its own hyperparameters with the same budget of trials.

One job per pair and short-listed candidate runs shared/search.py's search on the full design,
its trials scored on the frozen folds only; the best trial's params are then refitted once
through the harness into the run ledger, so its test figures stand beside its folds'. A fit past
the time budget fails its trial -- the trial's wall -- and the job goes on.
"""

import argparse
import json
import logging
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, assert_never

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
from optuna.distributions import IntDistribution
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

from shared import surrogate
from shared.candidates import FAMILIES, UNITS, Candidate, Knobs, tuned_fitter
from shared.config import NOTATION_TEX, PaperConfig, load_config
from shared.design import BLACK_HOLES, NEUTRON_STARS, Design, DesignSpec, Target, design
from shared.eda import CurveSpace, booktabs
from shared.harness import Fitter, Predictor, Run, harness
from shared.plots import SYMBOLS, PlotStyle, apply_style, colormap, figure_size
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
from shared.search import KNOTS, SPACES, Family, Space, search, space_of
from shared.surrogate import Training, duration_text
from shared.trials import open_study
from shared.workers import run_jobs

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

SECTION = "63_precision"
HISTORY = f"{SECTION}_fig_history"
RANGES = f"{SECTION}_fig_ranges"
TARGETS: tuple[Target, ...] = ("mass", "charge")
DATASET_TEX = {"neutron_stars": "NS", "black_holes": "BH"}
TARGET_TEX = {"mass": "$M$", "charge": "$\\Dch$"}
# A target's symbol in shared/plots.py's SYMBOLS, for the figures.
TARGET_SYMBOL = {"mass": "M", "charge": "D"}
# The data ceilings of Sections 4.1 and 4.2 (W-064) in significant figures, as their macros.
CEILING_TEX: Mapping[tuple[str, Target], str] = {
    ("neutron_stars", "mass"): "\\nsEdaCeilMFigures",
    ("neutron_stars", "charge"): "\\nsEdaCeilDFigures",
    ("black_holes", "mass"): "\\bhEdaCeilMFigures",
    ("black_holes", "charge"): "\\bhEdaCeilDFigures",
}

PAIRS: tuple[tuple[DesignSpec, Target], ...] = (
    (NEUTRON_STARS, "mass"),
    (NEUTRON_STARS, "charge"),
    (BLACK_HOLES, "mass"),
    (BLACK_HOLES, "charge"),
)

_RBF = (Candidate("local RBF", "pointwise"), Candidate("local RBF", "curve-wise"))
_BH = (*_RBF, Candidate("GPR", "curve-wise"), Candidate("XGBoost", "pointwise"))

# The short list per pair, as Section 5.2 states it: the candidates within one significant
# figure of the pair's best on the screen's last round.
SHORT_LIST: Mapping[tuple[str, Target], tuple[Candidate, ...]] = {
    ("neutron_stars", "mass"): _RBF,
    ("neutron_stars", "charge"): (
        *_RBF,
        Candidate("ResNet", "pointwise"),
        Candidate("k-NN", "curve-wise"),
    ),
    ("black_holes", "mass"): _BH,
    ("black_holes", "charge"): _BH,
}


@dataclass(frozen=True)
class Job:
    """One short-listed candidate on one pair."""

    dataset: str
    target: Target
    candidate: Candidate


@dataclass(frozen=True)
class Tuned:
    """A job whose search ended: the best trial's params, its mean fold figures, the test
    figures of its refit, the trials run and how many of them met the wall, and the refit's
    ledger entry."""

    job: Job
    params: dict[str, Any]
    folds: float
    test: float
    trials: int
    walls: int
    entry: str


@dataclass(frozen=True)
class Walled:
    """A job every trial of which met the wall."""

    job: Job
    reason: str


Outcome = Tuned | Walled


@dataclass(frozen=True)
class TrialPoint:
    """One trial of a job's search: its number, its fold figures (None for a walled or failed
    trial) and its params."""

    job: Job
    number: int
    figures: float | None
    params: Mapping[str, Any]


# --- pure functions ---------------------------------------------------------------------------


def jobs() -> tuple[Job, ...]:
    """One job per pair and short-listed candidate."""
    return tuple(
        Job(spec.dataset, target, candidate)
        for spec, target in PAIRS
        for candidate in SHORT_LIST[(spec.dataset, target)]
    )


def run_job(
    job: Job,
    data: Design,
    family: Family,
    space: Space,
    trials: int,
    seed: int,
    valid_fraction: float,
    seconds: float,
    study: optuna.Study,
    record: Callable[[Run], str],
) -> Outcome:
    """Search the job's space with `trials` trials, each fit within `seconds`, then refit the
    best params once through the harness; record keeps that run and names its entry. A Walled
    when no trial completes."""
    try:
        found = search(
            lambda params: walled(family(params), seconds),
            space,
            trials,
            data,
            job.target,
            seed,
            valid_fraction,
            study,
        )
    except ValueError:
        # Optuna's best_params has no trial to name; any other ValueError is not a wall.
        if any(trial.state == TrialState.COMPLETE for trial in study.trials):
            raise
        return Walled(job, f"all {trials} trials past {seconds / 60:g} min")
    walls = sum(trial.state == TrialState.FAIL for trial in study.trials)
    entries: list[str] = []
    (run,) = harness(
        data,
        job.target,
        walled(family(found.params), seconds),
        [seed],
        valid_fraction,
        record=lambda ran: entries.append(record(ran)),
    )
    test = significant_figures(run.test.zones["test"].p95)
    return Tuned(job, dict(found.params), found.figures, test, found.trials, walls, entries[0])


def walled(fitter: Fitter, seconds: float) -> Fitter:
    """The fitter with a wall: a fit past `seconds` predicts nan, and so does every later fit
    of the same fitter, so a walled trial scores nan (failed) without waiting again. The wall
    is a real timer's signal, so it runs in a process's main thread, as a worker's job does."""

    met = False

    def stop(signum: int, frame: object) -> None:
        raise _WallError

    def fit(
        x_fit: np.ndarray, y_fit: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, seed: int
    ) -> Predictor:
        nonlocal met
        if met:
            return _nan
        previous = signal.signal(signal.SIGALRM, stop)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return fitter(x_fit, y_fit, x_valid, y_valid, seed)
        except _WallError:
            met = True
            return _nan
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    return fit


class _WallError(Exception):
    """A fit ran past its wall."""


def _nan(x: np.ndarray) -> np.ndarray:
    """The prediction of a walled fit: nan for every row."""
    return np.full(len(x), np.nan)


def screen_figures(entries: Sequence[LedgerEntry], batch: str) -> Mapping[Job, float]:
    """Each candidate's mean fold figures on the screen's last round it ran, from the screen
    batch's ledger entries."""
    last: dict[Job, tuple[int, float]] = {}
    for entry in entries:
        meta = entry.meta
        if meta.batch != batch:
            continue
        family, unit, target = meta.candidate, meta.settings["unit"], meta.target
        if family not in FAMILIES or unit not in UNITS or target not in TARGETS:
            raise ValueError(f"ledger entry {meta.id}: unknown candidate {family} {unit} {target}")
        rows = meta.settings["rows"]
        if not isinstance(rows, int):
            raise ValueError(f"ledger entry {meta.id}: rows {rows!r} is not a count")
        job = Job(meta.dataset, target, Candidate(family, unit))
        figures = float(np.mean([significant_figures(fold.p95) for fold in entry.folds]))
        if job not in last or rows > last[job][0]:
            last[job] = (rows, figures)
    return {job: figures for job, (_, figures) in last.items()}


def study_name(batch: str, job: Job) -> str:
    """The name of the job's study on the batch's journal."""
    candidate = job.candidate
    return f"{batch} {job.dataset} {job.target} {candidate.family} {candidate.unit}"


def recovered(
    batch: str,
    studies: Mapping[str, optuna.Study],
    entries: Sequence[LedgerEntry],
    minutes: float,
) -> tuple[Outcome, ...]:
    """The batch's outcomes rebuilt from the records that survive a dying parent: per job,
    its study on the journal (by study_name) and its refit's ledger entry. A study with no
    COMPLETE trial is a wall over its finished trials; one with a best trial but no entry
    did not finish its refit and is left out with a warning; a job without a study did not
    start."""
    found: list[Outcome] = []
    for job in jobs():
        name = study_name(batch, job)
        if name not in studies:
            continue
        trials = studies[name].trials
        finished = [t for t in trials if t.state in (TrialState.COMPLETE, TrialState.FAIL)]
        walls = sum(t.state == TrialState.FAIL for t in finished)
        if walls == len(finished):
            found.append(Walled(job, f"all {len(finished)} trials past {minutes:g} min"))
            continue
        best = studies[name].best_trial
        refits = [
            entry
            for entry in entries
            if entry.meta.batch == batch
            and entry.meta.dataset == job.dataset
            and entry.meta.target == job.target
            and entry.meta.candidate == job.candidate.family
            and entry.meta.settings["unit"] == job.candidate.unit
        ]
        if not refits:
            logger.warning("%s: a best trial but no refit in the ledger; left out", name)
            continue
        entry = refits[-1]
        test = significant_figures(entry.test.zones["test"].p95)
        found.append(
            Tuned(job, dict(best.params), best.values[0], test, len(finished), walls, entry.meta.id)
        )
    return tuple(found)


def outcomes_to_json(outcomes: Sequence[Outcome]) -> str:
    """The outcomes as indented JSON, each tagged tuned or walled."""
    tagged = []
    for outcome in outcomes:
        match outcome:
            case Tuned():
                tagged.append({"kind": "tuned", **asdict(outcome)})
            case Walled():
                tagged.append({"kind": "walled", **asdict(outcome)})
            case _:
                assert_never(outcome)
    return json.dumps(tagged, indent=2) + "\n"


def outcomes_from_json(text: str) -> tuple[Outcome, ...]:
    """The outcomes an outcomes_to_json text holds."""
    found: list[Outcome] = []
    for fields in json.loads(text):
        raw = fields["job"]
        job = Job(raw["dataset"], raw["target"], Candidate(**raw["candidate"]))
        if fields["kind"] == "walled":
            found.append(Walled(job, fields["reason"]))
        else:
            found.append(
                Tuned(
                    job,
                    fields["params"],
                    fields["folds"],
                    fields["test"],
                    fields["trials"],
                    fields["walls"],
                    fields["entry"],
                )
            )
    return tuple(found)


def tuning_table(outcomes: Sequence[Outcome], screen: Mapping[Job, float]) -> str:
    """The tuning table: per pair and candidate the screen's figures, the tuned fold and test
    figures, and the pair's data ceiling."""
    rows = []
    for outcome in outcomes:
        job = outcome.job
        before = screen.get(job)
        match outcome:
            case Tuned():
                tuned = f"${outcome.folds:.2f}$ & ${outcome.test:.2f}$"
            case Walled():
                tuned = "\\multicolumn{2}{c}{wall}"
            case _:
                assert_never(outcome)
        rows.append(
            f"{DATASET_TEX[job.dataset]} {TARGET_TEX[job.target]} & "
            f"{job.candidate.family} & {job.candidate.unit} & "
            f"{'--' if before is None else f'${before:.2f}$'} & {tuned} & "
            f"${CEILING_TEX[(job.dataset, job.target)]}$"
        )
    return booktabs(
        "tuning",
        "Each short-listed candidate of \\cref{tab:families} after the same budget of trials "
        "over its own hyperparameters: the significant figures kept at the 95th percentile of "
        "the error ($-\\log_{10}$), on the frozen folds at the screen and after the tuning, on "
        "the test curves for the best trial refitted, against the pair's data ceiling. A wall: "
        "every trial ran past the time budget. The errors are measured in $M$ or $\\Dch$.",
        "@{}lll*{4}{r}@{}",
        "Pair & Family & Unit & Screen & Tuned & Test & Ceiling",
        rows,
    )


def trial_points(study: optuna.Study, job: Job) -> tuple[TrialPoint, ...]:
    """The finished trials of a job's study as points, a failed one without figures."""
    finished = (TrialState.COMPLETE, TrialState.FAIL)
    return tuple(
        TrialPoint(
            job,
            trial.number,
            trial.value if trial.state == TrialState.COMPLETE else None,
            trial.params,
        )
        for trial in study.get_trials(deepcopy=False, states=finished)
    )


def history_figure(points: Sequence[TrialPoint], style: PlotStyle) -> Figure:
    """Per pair a panel: each candidate's trial figures against the trial number as dots and
    its best so far as a line, family by color and unit by line style (curve-wise dashed)."""
    pairs = list(dict.fromkeys((p.job.dataset, p.job.target) for p in points))
    columns = min(2, len(pairs))
    rows = -(-len(pairs) // columns)
    figure, grid = plt.subplots(
        rows,
        columns,
        figsize=figure_size(style, rows, 0.36),
        layout="constrained",
        squeeze=False,
    )
    palette: Mapping[str, Any] = dict(
        zip(FAMILIES, sns.color_palette(style.palette, len(FAMILIES)), strict=True)
    )
    for axes, (dataset, target) in zip(grid.flat, pairs, strict=False):
        for job in dict.fromkeys(
            p.job for p in points if (p.job.dataset, p.job.target) == (dataset, target)
        ):
            trials = sorted((p for p in points if p.job == job), key=lambda p: p.number)
            color = palette[job.candidate.family]
            numbers = [p.number for p in trials if p.figures is not None]
            figures = [p.figures for p in trials if p.figures is not None]
            axes.plot(
                numbers, figures, linestyle="", marker="o", markersize=2.5, color=color, alpha=0.5
            )
            axes.plot(
                numbers,
                np.maximum.accumulate(figures),
                color=color,
                linestyle="--" if job.candidate.unit == "curve-wise" else "-",
            )
            walls = [p.number for p in trials if p.figures is None]
            if walls:
                # at the panel's bottom whatever its figures' range: the x in data, the y in axes
                axes.plot(
                    walls,
                    [0.03] * len(walls),
                    linestyle="",
                    marker="x",
                    markersize=5,
                    color=color,
                    transform=axes.get_xaxis_transform(),
                )
        axes.set_title(f"{DATASET_TEX[dataset]} {SYMBOLS[TARGET_SYMBOL[target]]}", fontsize="small")
        axes.set_ylabel("folds' figures")
    for axes in grid.flat[len(pairs) :]:
        axes.set_visible(False)
    for axes in grid[-1]:
        axes.set_xlabel("trial")
    shown = [family for family in FAMILIES if any(p.job.candidate.family == family for p in points)]
    handles = [
        *(Line2D([], [], color=palette[family]) for family in shown),
        Line2D([], [], color="black"),
        Line2D([], [], color="black", linestyle="--"),
        Line2D([], [], color="black", marker="x", linestyle=""),
    ]
    labels = [*shown, *UNITS, "wall"]
    figure.legend(handles, labels, loc="outside lower center", ncols=4, fontsize="x-small")
    return figure


def ranges_figure(points: Sequence[TrialPoint], style: PlotStyle) -> Figure:
    """The trials against the ranges they searched, the ends of each range dashed: the folds'
    figures against the knots of every curve-wise trial and against the neighbours of every
    local RBF trial (a color per pair, pointwise dots and curve-wise squares), and the
    network's trials in width and learning rate colored by their figures, walled ones crosses."""
    figure, (knots_axes, neighbours_axes, network_axes) = plt.subplots(
        1, 3, figsize=figure_size(style, 1, 0.45), layout="constrained"
    )
    pairs = [(spec.dataset, target) for spec, target in PAIRS]
    # the paper's dataset colors, cool for neutron stars and warm for black holes, the mass
    # the darker shade
    shades = {
        "neutron_stars": colormap(style.neutron_stars.beta_cmap),
        "black_holes": colormap(style.black_holes.beta_cmap),
    }
    tone = {"mass": 0.25, "charge": 0.7}
    colors = {(dataset, target): shades[dataset](tone[target]) for dataset, target in pairs}
    marker = {"pointwise": "o", "curve-wise": "s"}
    scored = [p for p in points if p.figures is not None]
    for axes, name, chosen in (
        (knots_axes, "knots", [p for p in scored if p.job.candidate.unit == "curve-wise"]),
        (
            neighbours_axes,
            "neighbours",
            [p for p in scored if p.job.candidate.family == "local RBF"],
        ),
    ):
        for p in chosen:
            axes.plot(
                [p.params[name]],
                [p.figures],
                linestyle="",
                marker=marker[p.job.candidate.unit],
                markersize=2.5,
                alpha=0.6,
                color=colors[(p.job.dataset, p.job.target)],
            )
        axes.set_xlabel(name)
        axes.set_ylabel("folds' figures")
    neighbours_axes.set_xscale("log")
    _plain_log_ticks(neighbours_axes.xaxis, (50, 100, 200, 400))
    for axes, distribution in (
        (knots_axes, KNOTS),
        (neighbours_axes, SPACES["local RBF"]["neighbours"]),
        (network_axes, SPACES["ResNet"]["width"]),
    ):
        assert isinstance(distribution, IntDistribution), f"{distribution} is no integer range"
        for end in (distribution.low, distribution.high):
            axes.axvline(end, color="grey", linestyle="--", linewidth=0.8)
    network = [p for p in points if p.job.candidate.family == "ResNet"]
    ran = [p for p in network if p.figures is not None]
    shown = network_axes.scatter(
        [p.params["width"] for p in ran],
        [p.params["lr"] for p in ran],
        c=[p.figures for p in ran],
        cmap="viridis",
        s=10,
    )
    walls = [p for p in network if p.figures is None]
    network_axes.plot(
        [p.params["width"] for p in walls],
        [p.params["lr"] for p in walls],
        linestyle="",
        marker="x",
        markersize=4,
        color="black",
    )
    network_axes.set_xscale("log")
    _plain_log_ticks(network_axes.xaxis, (32, 128, 512))
    network_axes.set_yscale("log")
    network_axes.set_xlabel("width")
    network_axes.set_ylabel("learning rate")
    figure.colorbar(shown, ax=network_axes, label="folds' figures")
    handles = [
        *(Line2D([], [], color=colors[pair], marker="o", linestyle="") for pair in pairs),
        Line2D([], [], color="black", marker="o", linestyle=""),
        Line2D([], [], color="black", marker="s", linestyle=""),
        Line2D([], [], color="black", marker="x", linestyle=""),
    ]
    labels = [
        *(f"{DATASET_TEX[dataset]} {SYMBOLS[TARGET_SYMBOL[target]]}" for dataset, target in pairs),
        *UNITS,
        "wall",
    ]
    figure.legend(handles, labels, loc="outside lower center", ncols=4, fontsize="x-small")
    return figure


def _plain_log_ticks(axis: Any, ticks: Sequence[float]) -> None:
    """Label a log axis at the given ticks only, as plain numbers."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(NullFormatter())


def history_tex(minutes: float) -> str:
    """The figure environment of the history figure, its wall stated in the caption."""
    return (
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{HISTORY}}}\n"
        "\\caption{The searches of \\cref{tab:tuning}, per dataset and target: each "
        "short-listed candidate's trials scored on the frozen folds (dots) and its best so far "
        "(line, curve-wise dashed) against the trial number. The first ten trials are drawn at "
        "random, the rest by the tree-structured Parzen estimator. A cross at the bottom is a "
        f"trial past the wall of ${minutes:g}$ minutes a fit.}}\n"
        "\\label{fig:tuning-history}\n\\end{figure}\n"
    )


def ranges_tex() -> str:
    """The figure environment of the ranges figure, placed here or at the top of a page."""
    return (
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{RANGES}}}\n"
        "\\caption{The trials of \\cref{tab:tuning} against the ranges they searched, the ends "
        "of each range dashed: the significant figures on the folds against the knots of every "
        "curve-wise trial (left) and against the neighbours of every trial of the local "
        "interpolant (middle), a color per dataset and target, pointwise dots and curve-wise "
        "squares; and the residual network's trials on the neutron-star charge in width and "
        "learning rate (right), colored by their figures, a cross a trial past the wall. A best "
        "trial on the end of its range marks the range to widen.}\n"
        "\\label{fig:tuning-ranges}\n\\end{figure}\n"
    )


def job_line(outcome: Outcome, done: int, total: int, elapsed: float) -> str:
    """One finished job: its pair and candidate, its tuned figures or its wall, then how many
    jobs are done, the elapsed time and the time left."""
    job = outcome.job
    left = elapsed / done * (total - done)
    match outcome:
        case Tuned():
            result = (
                f"folds {outcome.folds:.2f}, test {outcome.test:.2f} figures, "
                f"{outcome.walls} of {outcome.trials} trials walled"
            )
        case Walled():
            result = f"wall, {outcome.reason}"
        case _:
            assert_never(outcome)
    return (
        f"job {done}/{total} done: {job.dataset} {job.target}, "
        f"{job.candidate.family} {job.candidate.unit}: {result}; "
        f"elapsed {duration_text(elapsed)}, about {duration_text(left)} left"
    )


# --- shell ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Context:
    """What every job of one tuning shares: the batch, where its ledger entries and its trials'
    journal go, the code it ran from and when, the settings a trial does not search, the
    trials per job and the wall of one fit."""

    batch: str
    ledger: Path
    journal: Path
    commit: str
    dirty: bool
    started: datetime
    knobs: Knobs
    training: Training
    trials: int
    seconds: float
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
    """Tune every short-listed candidate on the workers, saving the outcomes under the state
    folder, and write the tuning table -- or, given --batch, fit nothing and take that batch."""
    args = _arguments(argv)
    for path in (args.ns, args.bh, args.split):
        assert path.is_file(), f"input not found: {path}"
    config = config or load_config()
    settings = config.methodology.algorithms
    training = Training(**settings.model_dump(exclude={"log_level", "threads", "workers"}))
    tables = {
        NEUTRON_STARS.dataset: pd.read_parquet(args.ns),
        BLACK_HOLES.dataset: pd.read_parquet(args.bh),
    }
    split = pd.read_parquet(args.split)
    designs: dict[tuple[str, Target], tuple[Design, CurveSpace]] = {
        (spec.dataset, target): (design(tables[spec.dataset], split, spec, target), spec.space)
        for spec, target in PAIRS
    }
    with _at_level(settings.log_level):
        if args.recover is not None:
            batch = args.recover
            _recover(args, batch)
        elif args.batch is None:
            batch = _run_tuning(args, designs, training, settings.log_level, commit, dirty, now)
        else:
            batch = args.batch
            logger.info("rebuilding the tuning table from batch %s; nothing is fitted", batch)
        outcomes = outcomes_from_json(_outcomes_path(args.state, batch).read_text())
        screen: Mapping[Job, float] = {}
        if args.screen_batch is not None:
            screen = screen_figures(_ledger_entries(args.state), args.screen_batch)
        out = args.assets / SECTION
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{SECTION}_tab_tuning.tex"
        _write_checked(path, tuning_table(outcomes, screen))
        logger.info("done: tuning table of batch %s, %d jobs -> %s", batch, len(outcomes), path)
        points = _journal_points(_outcomes_path(args.state, batch).with_suffix(".journal"), batch)
        apply_style(config.plot, NOTATION_TEX.read_text())
        drawn = (
            (HISTORY, history_figure(points, config.plot), history_tex(args.minutes)),
            (RANGES, ranges_figure(points, config.plot), ranges_tex()),
        )
        for name, figure, tex in drawn:
            figure.savefig(out / f"{name}.png", dpi=config.plot.dpi)
            plt.close(figure)
            _write_checked(out / f"{name}.tex", tex)
            logger.info("done: %s of batch %s -> %s", name, batch, out / f"{name}.png")


def _run_tuning(
    args: argparse.Namespace,
    designs: dict[tuple[str, Target], tuple[Design, CurveSpace]],
    training: Training,
    log_level: str,
    commit: Callable[[], str] | None,
    dirty: Callable[[], bool] | None,
    now: Callable[[], datetime] | None,
) -> str:
    """Run every job of the tuning as a new batch on the workers, a line logged as each ends,
    and save the outcomes; return the batch id."""
    commit, dirty, now = commit or _git_commit, dirty or _git_dirty, now or _utc_now
    batch = _new_id()
    saved = _outcomes_path(args.state, batch)
    saved.parent.mkdir(parents=True, exist_ok=True)
    context = Context(
        batch=batch,
        ledger=args.state / "ledger",
        journal=saved.with_suffix(".journal"),
        commit=commit(),
        dirty=dirty(),
        started=now(),
        # The tuning never thins a design, so no rows per curve.
        knobs=Knobs(args.k, args.neighbours, args.knots, per_curve=0),
        training=training,
        trials=args.trials,
        seconds=args.minutes * 60.0,
        log_level=log_level,
    )
    context.ledger.mkdir(parents=True, exist_ok=True)
    found = jobs()
    logger.info(
        "tuning %d jobs, %d trials each, on %d workers, batch %s; journal %s",
        len(found),
        args.trials,
        args.workers,
        batch,
        context.journal,
    )
    works = [
        partial(tune_job, Work(job, *designs[(job.dataset, job.target)], context)) for job in found
    ]
    started = time.monotonic()
    finished: dict[int, Outcome] = {}

    def report(index: int, outcome: Outcome) -> None:
        # Saved after every job, in job order, so a parent that dies leaves the finished ones.
        finished[index] = outcome
        logger.info(job_line(outcome, len(finished), len(works), time.monotonic() - started))
        _write_checked(saved, outcomes_to_json([finished[i] for i in sorted(finished)]))

    outcomes = run_jobs(works, args.workers, on_done=report)
    _write_checked(saved, outcomes_to_json(outcomes))
    return batch


def tune_job(work: Work) -> Outcome:
    """Run one job at the tuning's log level with the networks' epoch tables silenced: a worker
    started by forkserver inherits neither the level nor the console handler."""
    if not logging.getLogger().handlers:
        logging.basicConfig(format="%(message)s")
    logging.getLogger(surrogate.__name__).setLevel(logging.WARNING)
    job, context = work.job, work.context
    candidate = job.candidate
    study = open_study(study_name(context.batch, job), context.journal)
    study.sampler = optuna.samplers.TPESampler(seed=context.training.seed)

    def record(run: Run) -> str:
        """Write the best params' run as a ledger entry, check it reads back, and name it."""
        best = study.best_params
        settings: dict[str, Setting] = {
            "unit": candidate.unit,
            "trials": context.trials,
            **asdict(context.knobs),
            **best,
        }
        meta = RunMetadata(
            id=_new_id(),
            batch=context.batch,
            dataset=job.dataset,
            target=job.target,
            candidate=candidate.family,
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
        return run_job(
            job,
            work.data,
            lambda params: tuned_fitter(
                candidate, work.space, context.knobs, context.training, params
            ),
            space_of(candidate.family, candidate.unit),
            context.trials,
            context.training.seed,
            context.training.valid_fraction,
            context.seconds,
            study,
            record,
        )


def _journal_points(journal: Path, batch: str) -> tuple[TrialPoint, ...]:
    """Every job's trials, read from the batch's study journal (nothing is created there)."""
    storage = JournalStorage(JournalFileBackend(str(journal)))
    return tuple(
        point
        for job in jobs()
        for point in trial_points(
            optuna.load_study(study_name=study_name(batch, job), storage=storage), job
        )
    )


def _recover(args: argparse.Namespace, batch: str) -> None:
    """Rebuild the batch's outcomes file from its journal and the ledger; nothing is fitted."""
    saved = _outcomes_path(args.state, batch)
    journal = saved.with_suffix(".journal")
    assert journal.is_file(), f"no journal of batch {batch}: {journal}"
    storage = JournalStorage(JournalFileBackend(str(journal)))
    studies = {
        name: optuna.load_study(study_name=name, storage=storage)
        for name in optuna.get_all_study_names(storage)
        if name.startswith(f"{batch} ")
    }
    outcomes = recovered(batch, studies, _ledger_entries(args.state), args.minutes)
    _write_checked(saved, outcomes_to_json(outcomes))
    logger.info("recovered %d outcomes of batch %s -> %s", len(outcomes), batch, saved)


def _ledger_entries(state: Path) -> list[LedgerEntry]:
    """Every entry of the run ledger under the state folder, oldest first."""
    ledger = sorted((state / "ledger").glob("*.json"))
    return [entry_from_json(path.read_text()) for path in ledger]


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
    parser.add_argument("--k", type=int, required=True, help="the k-NN's k before a trial's")
    parser.add_argument("--neighbours", type=int, required=True, help="the local RBF's, same")
    parser.add_argument("--knots", type=int, required=True, help="the curve-wise spline's, same")
    parser.add_argument("--trials", type=int, required=True, help="trials per job")
    parser.add_argument("--minutes", type=float, required=True, help="the wall of one fit")
    parser.add_argument("--workers", type=int, required=True, help="jobs at once")
    parser.add_argument("--batch", help="fit nothing; rebuild the table from this saved batch")
    parser.add_argument(
        "--recover",
        help="fit nothing; rebuild this batch's outcomes from its journal and the ledger, "
        "then the table",
    )
    parser.add_argument("--screen-batch", help="the family screen's batch, for its figures")
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

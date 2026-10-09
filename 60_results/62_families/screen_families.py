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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline

from shared import surrogate
from shared.ceilings import profile
from shared.config import NOTATION_TEX, PaperConfig, load_config
from shared.curvewise import curvewise_fitter
from shared.design import BLACK_HOLES, NEUTRON_STARS, Design, DesignSpec, Target, design
from shared.eda import CurveSpace, booktabs
from shared.families import make_gpr, make_knn, make_rbf, make_xgboost
from shared.harness import Fitter, Predictor, Run, harness
from shared.plots import CoolCmap, PlotStyle, WarmCmap, apply_style, colormap, figure_size
from shared.runs import (
    RunMetadata,
    Setting,
    entry_from_json,
    entry_from_run,
    entry_name,
    entry_to_json,
)
from shared.scorecard import relative_errors, significant_figures
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
# The parity figure's titles and axis labels, the target as the model sees it; matplotlib
# knows no \Dch.
PLOT_TARGET = {"mass": "$M$", "charge": "$D$"}
AXIS_TEX = {"mass": "$M$ ($M_\\odot$)", "charge": "$Y = \\log_{10}(D/M)$"}
PARITY = f"{SECTION}_fig_parity"
ERRORS = f"{SECTION}_fig_errors"
SCALING = f"{SECTION}_fig_scaling"
# Equal-count bins of the true target for the error panel's median and p95 lines.
PROFILE_BINS = 20

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


@dataclass(frozen=True)
class Series:
    """One candidate's predictions against the truth on the test rows and their relative
    errors in M or D, labelled for a legend."""

    label: str
    y_true: np.ndarray
    y_pred: np.ndarray
    errors: np.ndarray


@dataclass(frozen=True)
class Panel:
    """One pair's panel of the parity figure: its dataset (for the colors), its title, the
    target's axis label and the candidates shown."""

    dataset: str
    title: str
    target: str
    series: tuple[Series, ...]


@dataclass(frozen=True)
class Trajectory:
    """One candidate's way through the rounds on one pair: the rows, folds' figures and final
    fit seconds of each round it was scored in, and the rows of any wall it met."""

    dataset: str
    target: Target
    candidate: Candidate
    rows: tuple[int, ...]
    folds: tuple[float, ...]
    seconds: tuple[float, ...]
    walls: tuple[int, ...]


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
    return tuple(outcome.job.candidate for outcome in _ranked(outcomes)[:keep])


def _ranked(outcomes: Sequence[Outcome]) -> list[Scored]:
    """The scored outcomes, best on the folds' figures first; walls left out."""
    ran: list[Scored] = []
    for outcome in outcomes:
        match outcome:
            case Scored():
                ran.append(outcome)
            case Wall():
                pass
            case _:
                assert_never(outcome)
    return sorted(ran, key=lambda outcome: -outcome.folds)


def next_jobs(
    outcomes: Sequence[Outcome], budgets: Sequence[int], keeps: Sequence[int]
) -> tuple[Job, ...]:
    """The jobs of the round after the outcomes' round: each pair's best keeps[round]
    candidates of each unit at the next budget of rows, pairs and then units in the order they
    first appear. Each unit is ranked apart: a thinned design keeps few curves, which starves
    only the curve-wise unit (W-067)."""
    by_group: dict[tuple[str, Target, Unit], list[Outcome]] = {}
    for outcome in outcomes:
        job = outcome.job
        by_group.setdefault((job.dataset, job.target, job.candidate.unit), []).append(outcome)
    jobs = []
    for (dataset, target, _), mine in by_group.items():
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


def parity_picks(outcomes: Sequence[Outcome]) -> tuple[Scored, ...]:
    """Per pair, in the order the pairs first appear, the best scored candidate of each unit
    at the most rows the pair ran on: the candidates the parity figure shows."""
    picks: list[Scored] = []
    pairs: dict[tuple[str, Target], None] = dict.fromkeys(
        (o.job.dataset, o.job.target) for o in outcomes
    )
    for dataset, target in pairs:
        at_most = _at_most_rows(outcomes, dataset, target)
        for unit in UNITS:
            ranked = _ranked([o for o in at_most if o.job.candidate.unit == unit])
            picks += ranked[:1]
    return tuple(picks)


def parity_figure(panels: Sequence[Panel], style: PlotStyle) -> Figure:
    """Predicted against true target on the test curves, one panel per pair in a grid of two
    columns, with the y = x line; each candidate in one end of its dataset's colormap.

    Raises ValueError for a panel of a dataset with no colormap.
    """
    rows = (len(panels) + 1) // 2
    figure, grid = plt.subplots(
        rows, 2, figsize=figure_size(style, rows, 0.5), layout="constrained", squeeze=False
    )
    for axes, panel in zip(grid.flat, panels, strict=False):
        colors = _colors(panel, style)
        for color, series in zip(colors, panel.series, strict=True):
            axes.scatter(
                series.y_true, series.y_pred, s=2, alpha=0.4, color=color, label=series.label
            )
        values = np.concatenate([np.r_[s.y_true, s.y_pred] for s in panel.series])
        span = [float(values.min()), float(values.max())]
        axes.plot(span, span, color="black", linestyle="--", linewidth=0.8)
        axes.set_title(panel.title)
        axes.set_xlabel(f"true {panel.target}")
        axes.set_ylabel(f"predicted {panel.target}")
        # The points follow the diagonal, so the lower right corner stays empty.
        axes.legend(fontsize="x-small", markerscale=3, loc="lower right")
    return figure


def error_figure(panels: Sequence[Panel], style: PlotStyle) -> Figure:
    """The relative error in M or D against the true target on a log scale, a row per pair
    and a column per candidate, the y axis shared along a row: each panel the candidate's
    scatter in its color, with its median (solid) and 95th percentile (dashed) in black.

    Raises ValueError for a panel of a dataset with no colormap.
    """
    columns = max(len(panel.series) for panel in panels)
    figure, grid = plt.subplots(
        len(panels),
        columns,
        figsize=figure_size(style, len(panels), 0.38),
        layout="constrained",
        squeeze=False,
        sharey="row",
    )
    for row, panel in enumerate(panels):
        for column, (color, series) in enumerate(
            zip(_colors(panel, style), panel.series, strict=True)
        ):
            axes = grid[row, column]
            axes.scatter(
                series.y_true, series.errors, s=1, alpha=0.15, color=color, rasterized=True
            )
            found = profile(series.y_true, series.errors, PROFILE_BINS)
            axes.plot(found.centers, found.median, color="black", linewidth=1.0)
            axes.plot(found.centers, found.p95, color="black", linewidth=1.0, linestyle="--")
            axes.set_yscale("log")
            axes.set_title(f"{panel.title}: {series.label}", fontsize="small")
            axes.set_xlabel(f"true {panel.target}")
            axes.set_ylabel("relative error")
    return figure


def scaling_figure(
    paths: Sequence[Trajectory], style: PlotStyle, wall_seconds: float = 1800.0
) -> Figure:
    """Per pair a row: the folds' p95 significant figures and the final fit's seconds against
    the training rows on a log scale, a line per candidate up to the round it left, its family
    by color and its unit by line style (curve-wise dashed), a wall a cross at its rows and at
    wall_seconds, the time budget of a fit."""
    pairs = list(dict.fromkeys((p.dataset, p.target) for p in paths))
    figure, grid = plt.subplots(
        len(pairs),
        2,
        figsize=figure_size(style, len(pairs), 0.42),
        layout="constrained",
        squeeze=False,
        sharex=True,
    )
    palette = dict(zip(FAMILIES, sns.color_palette(style.palette, len(FAMILIES)), strict=True))
    for row, (dataset, target) in enumerate(pairs):
        figures_axes, seconds_axes = grid[row]
        for path in (p for p in paths if (p.dataset, p.target) == (dataset, target)):
            line = {
                "color": palette[path.candidate.family],
                "linestyle": "--" if path.candidate.unit == "curve-wise" else "-",
                "marker": "o",
                "markersize": 3,
            }
            figures_axes.plot(path.rows, path.folds, **line)
            seconds_axes.plot(path.rows, path.seconds, **line)
            for rows in path.walls:
                seconds_axes.plot(
                    [rows], [wall_seconds], marker="x", markersize=7, color=line["color"]
                )
        title = f"{DATASET_TEX[dataset]} {PLOT_TARGET[target]}"
        figures_axes.set_ylabel(f"{title}: folds' figures")
        seconds_axes.set_ylabel("fit seconds")
        seconds_axes.set_yscale("log")
        for axes in (figures_axes, seconds_axes):
            axes.set_xscale("log")
    for axes in grid[-1]:
        axes.set_xlabel("training rows")
    handles = [
        *(Line2D([], [], color=palette[family]) for family in FAMILIES),
        Line2D([], [], color="black"),
        Line2D([], [], color="black", linestyle="--"),
        Line2D([], [], color="black", marker="x", linestyle=""),
    ]
    labels = [*FAMILIES, *UNITS, "wall"]
    figure.legend(handles, labels, loc="outside lower center", ncols=5, fontsize="x-small")
    return figure


def _colors(panel: Panel, style: PlotStyle) -> list[Any]:
    """One color per candidate of the panel, spread to the two ends of the dataset's colormap
    so that two candidates never blend."""
    shades = colormap(_cmap(panel.dataset, style))
    return [shades(p) for p in np.linspace(0.15, 0.85, len(panel.series))]


def parity_tex() -> str:
    """The figure environment of the parity figure, placed here or at the top of a page."""
    return (
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{PARITY}}}\n"
        "\\caption{Predicted against true target on the test curves, per dataset and target, for "
        "the best candidate of each unit of prediction on the most training rows; the legend "
        "gives the significant figures kept on the folds, and the dashed line is $y = x$. At "
        "these precisions every candidate lies on the line; \\cref{fig:families-errors} shows "
        "where they differ. The charge is shown as the target the models see, $Y = "
        "\\log_{10}(\\Dch/M)$.}\n"
        "\\label{fig:families-parity}\n\\end{figure}\n"
    )


def errors_tex() -> str:
    """The figure environment of the error figure, placed here or at the top of a page."""
    return (
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{ERRORS}}}\n"
        "\\caption{The relative error in $M$ or $\\Dch$ on the test curves against the true "
        "target, for the candidates of \\cref{fig:families-parity}: a row per dataset and "
        "target, a column per unit of prediction, the error axis shared along a row. The black "
        "lines are each candidate's median (solid) and 95th percentile (dashed) in equal-count "
        "bins of the true target.}\n"
        "\\label{fig:families-errors}\n\\end{figure}\n"
    )


def scaling_tex(minutes: float, memory_gb: float) -> str:
    """The figure environment of the scaling figure, its budgets stated in the caption."""
    return (
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{SCALING}}}\n"
        "\\caption{Successive halving over the training rows, per dataset and target: the "
        "significant figures kept on the folds (left) and the seconds of the final fit at one "
        "thread (right) against the training rows, a line per candidate up to the round it "
        "was eliminated in, curve-wise dashed. A cross is a wall, a fit past "
        f"${minutes:g}$ minutes or ${memory_gb:g}$\\,GB at those rows, drawn at the time "
        "budget.}\n"
        "\\label{fig:families-scaling}\n\\end{figure}\n"
    )


def _cmap(dataset: str, style: PlotStyle) -> CoolCmap | WarmCmap:
    """The dataset's colormap of the paper style.

    Raises ValueError for any other dataset.
    """
    if dataset == NEUTRON_STARS.dataset:
        return style.neutron_stars.beta_cmap
    if dataset == BLACK_HOLES.dataset:
        return style.black_holes.beta_cmap
    raise ValueError(f"no colormap for dataset {dataset}")


def trajectories(outcomes: Sequence[Outcome]) -> tuple[Trajectory, ...]:
    """Every candidate's way through the rounds, per pair in the order the pairs first
    appear, candidates in the screen's order: the scaling figure's lines and walls."""
    found: dict[tuple[str, Target, Candidate], list[Outcome]] = {}
    for outcome in sorted(outcomes, key=lambda o: o.job.rows):
        job = outcome.job
        found.setdefault((job.dataset, job.target, job.candidate), []).append(outcome)
    pairs: dict[tuple[str, Target], None] = dict.fromkeys(
        (dataset, target) for dataset, target, _ in found
    )
    paths = []
    for dataset, target in pairs:
        for candidate in candidates():
            mine = found.get((dataset, target, candidate), [])
            scored = _ranked_by_rows(mine)
            walls = tuple(o.job.rows for o in mine if o not in scored)
            if mine:
                paths.append(
                    Trajectory(
                        dataset,
                        target,
                        candidate,
                        tuple(o.job.rows for o in scored),
                        tuple(o.folds for o in scored),
                        tuple(o.fit_seconds for o in scored),
                        walls,
                    )
                )
    return tuple(paths)


def _ranked_by_rows(outcomes: Sequence[Outcome]) -> list[Scored]:
    """The scored outcomes in the order of their rows; walls left out."""
    return sorted(_ranked(outcomes), key=lambda o: o.job.rows)


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
        if args.batch is None:
            batch = _run_screen(args, designs, training, settings.log_level, commit, dirty, now)
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
        panels = _parity_panels(parity_picks(outcomes), designs, args.state / "ledger")
        apply_style(config.plot, NOTATION_TEX.read_text())
        drawn = (
            (PARITY, parity_figure(panels, config.plot), parity_tex()),
            (ERRORS, error_figure(panels, config.plot), errors_tex()),
            (
                SCALING,
                scaling_figure(trajectories(outcomes), config.plot, args.minutes * 60.0),
                scaling_tex(args.minutes, args.memory_gb),
            ),
        )
        for name, figure, tex in drawn:
            figure.savefig(out / f"{name}.png", dpi=config.plot.dpi)
            plt.close(figure)
            _write_checked(out / f"{name}.tex", tex)
            logger.info("done: %s of batch %s -> %s", name, batch, out / f"{name}.png")


def _parity_panels(
    picks: Sequence[Scored],
    designs: dict[tuple[str, Target], tuple[Design, CurveSpace]],
    ledger: Path,
) -> list[Panel]:
    """One panel per pair of the picks: each pick's test predictions, read from its ledger
    entry, against the true target of the pair's test rows."""
    series: dict[tuple[str, Target], list[Series]] = {}
    for pick in picks:
        job = pick.job
        truth = designs[(job.dataset, job.target)][0]
        truth = truth.y[truth.test]
        entry = entry_from_json((ledger / f"{pick.entry}.json").read_text())
        predicted = np.array(entry.predictions)
        assert len(predicted) == len(truth), (
            f"entry {pick.entry}: {len(predicted)} predictions for {len(truth)} test rows"
        )
        label = f"{job.candidate.family} {job.candidate.unit}, {pick.folds:.2f}"
        errors = relative_errors(truth, predicted, job.target)
        found = Series(label, truth, predicted, errors)
        series.setdefault((job.dataset, job.target), []).append(found)
    return [
        Panel(dataset, f"{DATASET_TEX[dataset]} {PLOT_TARGET[target]}", AXIS_TEX[target], tuple(s))
        for (dataset, target), s in series.items()
    ]


def _run_screen(
    args: argparse.Namespace,
    designs: dict[tuple[str, Target], tuple[Design, CurveSpace]],
    training: Training,
    log_level: str,
    commit: Callable[[], str] | None,
    dirty: Callable[[], bool] | None,
    now: Callable[[], datetime] | None,
) -> str:
    """Run every round of the screen as a new batch, saving the outcomes after each round;
    return the batch id."""
    commit, dirty, now = commit or _git_commit, dirty or _git_dirty, now or _utc_now
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

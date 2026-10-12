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
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, assert_never, get_args

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
from optuna.distributions import BaseDistribution, IntDistribution
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

from shared import surrogate
from shared.batches import (
    Batch,
    at_level,
    git_commit,
    git_dirty,
    make_batch,
    new_id,
    outcomes_path,
    progress_tail,
    read_ledger,
    utc_now,
    walled,
    write_checked,
    write_entry,
)
from shared.candidates import FAMILIES, UNITS, Candidate, Knobs, tuned_fitter
from shared.config import (
    NOTATION_TEX,
    PaperConfig,
    PrecisionHistoryVariant,
    PrecisionRangesVariant,
    TableLayout,
    load_config,
)
from shared.design import (
    BLACK_HOLES,
    DATASET_TEX,
    NEUTRON_STARS,
    SPECS,
    TARGETS,
    Design,
    Pair,
    Target,
    pair_tex,
    pairs,
    target_symbol,
)
from shared.eda import booktabs
from shared.harness import Run, harness
from shared.plots import (
    SYMBOLS,
    PlotStyle,
    apply_style,
    colormap,
    figure_size,
    line_width,
    mark_alpha,
    mark_size,
    write_variant_sheet,
)
from shared.runs import LedgerEntry, RunMetadata, Setting
from shared.scorecard import significant_figures
from shared.search import (
    KNOTS,
    SPACES,
    NoCompletedTrial,
    Searched,
    Space,
    TrialFitter,
    search,
    space_of,
)
from shared.surrogate import Training, make_training
from shared.trials import open_study
from shared.workers import run_jobs

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

SECTION = "63_precision"
HISTORY = f"{SECTION}_fig_history"
RANGES = f"{SECTION}_fig_ranges"
# The data ceilings of Sections 4.1 and 4.2 (W-064) in significant figures, as their macros.
CEILING_TEX: Mapping[tuple[str, Target], str] = {
    ("neutron_stars", "mass"): "\\nsEdaCeilMFigures",
    ("neutron_stars", "charge"): "\\nsEdaCeilDFigures",
    ("black_holes", "mass"): "\\bhEdaCeilMFigures",
    ("black_holes", "charge"): "\\bhEdaCeilDFigures",
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
    params: Mapping[str, Setting]
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
# Per pair, the candidates the family screen short-listed (short_list, Section 5.2).
ShortList = Mapping[tuple[str, Target], tuple[Candidate, ...]]


@dataclass(frozen=True)
class TrialPoint:
    """One trial of a job's search: its number, its fold figures (None for a walled or failed
    trial) and its params."""

    job: Job
    number: int
    figures: float | None
    params: Mapping[str, Setting]


# --- pure functions ---------------------------------------------------------------------------


def jobs(short: ShortList) -> tuple[Job, ...]:
    """One job per pair and short-listed candidate, the pairs in the paper's order."""
    return tuple(
        Job(spec.dataset, target, candidate)
        for spec, target in product(SPECS, TARGETS)
        for candidate in short.get((spec.dataset, target), ())
    )


def run_job(
    job: Job,
    data: Design,
    family: TrialFitter,
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
    match found:
        case NoCompletedTrial():
            return Walled(job, f"all {trials} trials past {seconds / 60:g} min")
        case Searched():
            pass
        case _:
            assert_never(found)
    walls = sum(trial.state == TrialState.FAIL for trial in study.trials)
    entries: list[str] = []
    (run,) = harness(
        data,
        job.target,
        walled(family(dict(found.params)), seconds),
        [seed],
        valid_fraction,
        record=lambda ran: entries.append(record(ran)),
    )
    test = significant_figures(run.test.zones["test"].p95)
    return Tuned(job, found.params, found.figures, test, found.trials, walls, entries[0])


@dataclass(frozen=True)
class RangeEnds:
    """The two ends of an integer search range, drawn on the ranges figure."""

    low: int
    high: int


def range_ends(distribution: BaseDistribution) -> RangeEnds:
    """The ends of an integer distribution of the search spaces.

    Raises TypeError on any other distribution, which has no integer range.
    """
    match distribution:
        case IntDistribution(low=low, high=high):
            return RangeEnds(low, high)
        case _:
            raise TypeError(f"{distribution} has no integer range")


def short_list(entries: Sequence[LedgerEntry], batch: str, within: float) -> ShortList:
    """Per pair, the candidates of the screen batch that ran the most rows the pair ran and
    whose mean fold figures there lie within `within` figures of the best (Section 5.2), in
    FAMILIES x UNITS order."""
    by_pair: dict[tuple[str, Target], dict[Job, tuple[int, float]]] = {}
    for job, last_round in _last_rounds(entries, batch).items():
        by_pair.setdefault((job.dataset, job.target), {})[job] = last_round
    found: dict[tuple[str, Target], tuple[Candidate, ...]] = {}
    for pair, mine in by_pair.items():
        most = max(rows for rows, _ in mine.values())
        at = {job.candidate: figures for job, (rows, figures) in mine.items() if rows == most}
        best = max((f for f in at.values() if np.isfinite(f)), default=float("nan"))
        found[pair] = tuple(
            sorted(
                (c for c, f in at.items() if np.isfinite(f) and f >= best - within),
                key=lambda c: (FAMILIES.index(c.family), UNITS.index(c.unit)),
            )
        )
    return found


def screen_figures(entries: Sequence[LedgerEntry], batch: str) -> Mapping[Job, float]:
    """Each candidate's mean fold figures on the screen's last round it ran, from the screen
    batch's ledger entries."""
    return {job: figures for job, (_, figures) in _last_rounds(entries, batch).items()}


def _last_rounds(entries: Sequence[LedgerEntry], batch: str) -> dict[Job, tuple[int, float]]:
    """Each candidate of the screen batch with the most rows it ran and its mean fold figures
    there.

    Raises ValueError on an entry naming an unknown candidate or rows that are no count.
    """
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
    return last


def study_name(batch: str, job: Job) -> str:
    """The name of the job's study on the batch's journal."""
    candidate = job.candidate
    return f"{batch} {job.dataset} {job.target} {candidate.family} {candidate.unit}"


def recovered(
    batch: str,
    studies: Mapping[str, optuna.Study],
    entries: Sequence[LedgerEntry],
    minutes: float,
    wanted: Sequence[Job],
) -> tuple[Outcome, ...]:
    """The batch's outcomes rebuilt from the records that survive a dying parent: per job it
    ran (wanted), its study on the journal (by study_name) and its refit's ledger entry. A
    study with no COMPLETE trial is a wall over its finished trials; one with a best trial but
    no entry did not finish its refit and is left out with a warning; a job without a study
    did not start."""
    found: list[Outcome] = []
    for job in wanted:
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


def tuning_table(
    outcomes: Sequence[Outcome], screen: Mapping[Job, float], layout: TableLayout = "small"
) -> str:
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
            f"{pair_tex(job.dataset, job.target)} & "
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
        layout,
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


# The regret of a trial that set a new best, on the log axis of the regret history.
REGRET_FLOOR = 1e-3


def history_figure(
    points: Sequence[TrialPoint],
    style: PlotStyle,
    variant: PrecisionHistoryVariant = "dots_and_best",
) -> Figure:
    """Per pair a panel: each candidate's trial figures against the trial number as dots and
    its best so far as a line (dots_and_best), the best so far alone (best_only), or the best
    so far minus each trial as dots on a log scale (regret); family by color and unit by line
    style (curve-wise dashed)."""
    pairs: list[tuple[str, Target]] = list(
        dict.fromkeys((p.job.dataset, p.job.target) for p in points)
    )
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
            dots = {
                "linestyle": "",
                "marker": "o",
                "markersize": mark_size(style, 2.5),
                "color": color,
            }
            dots["alpha"] = 0.5
            best = {"color": color}
            best["linestyle"] = "--" if job.candidate.unit == "curve-wise" else "-"
            match variant:
                case "dots_and_best":
                    axes.plot(numbers, figures, **dots)
                    axes.plot(numbers, np.maximum.accumulate(figures), **best)
                case "best_only":
                    axes.plot(numbers, np.maximum.accumulate(figures), **best)
                case "regret":
                    # A trial that sets a new best has no regret: it is drawn at the floor.
                    regret = np.maximum.accumulate(figures) - np.asarray(figures)
                    axes.plot(numbers, np.maximum(regret, REGRET_FLOOR), **dots)
                case _:
                    assert_never(variant)
            walls = [p.number for p in trials if p.figures is None]
            if walls:
                # at the panel's bottom whatever its figures' range: the x in data, the y in axes
                axes.plot(
                    walls,
                    [0.03] * len(walls),
                    linestyle="",
                    marker="x",
                    markersize=mark_size(style, 5),
                    color=color,
                    transform=axes.get_xaxis_transform(),
                )
        axes.set_title(f"{DATASET_TEX[dataset]} {SYMBOLS[target_symbol(target)]}", fontsize="small")
        if variant == "regret":
            axes.set_yscale("log")
            axes.set_ylabel("figures behind the best")
        else:
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


def ranges_figure(
    points: Sequence[TrialPoint], style: PlotStyle, variant: PrecisionRangesVariant = "row"
) -> Figure:
    """The trials against the ranges they searched, the ends of each range dashed: the folds'
    figures against the knots of every curve-wise trial and against the neighbours of every
    local RBF trial (a color per pair, pointwise dots and curve-wise squares), and the
    network's trials in width and learning rate colored by their figures, walled ones crosses:
    side by side (row), 2 x 2 with the knots split by dataset (grid), or without the network
    (no_network)."""
    scored = [p for p in points if p.figures is not None]
    curve_wise = [p for p in scored if p.job.candidate.unit == "curve-wise"]
    match variant:
        case "row":
            figure, (knots_axes, neighbours_axes, network_axes) = plt.subplots(
                1, 3, figsize=figure_size(style, 1, 0.45), layout="constrained"
            )
            knots = [(knots_axes, curve_wise)]
        case "grid":
            figure, grid = plt.subplots(
                2, 2, figsize=figure_size(style, 2, 0.4), layout="constrained"
            )
            knots = [
                (grid[0, column], [p for p in curve_wise if p.job.dataset == spec.dataset])
                for column, spec in enumerate(SPECS)
            ]
            for (axes, _), spec in zip(knots, SPECS, strict=True):
                axes.set_title(DATASET_TEX[spec.dataset], fontsize="small")
            neighbours_axes, network_axes = grid[1, 0], grid[1, 1]
        case "no_network":
            figure, (knots_axes, neighbours_axes) = plt.subplots(
                1, 2, figsize=figure_size(style, 1, 0.45), layout="constrained"
            )
            knots, network_axes = [(knots_axes, curve_wise)], None
        case _:
            assert_never(variant)
    pairs: list[tuple[str, Target]] = [
        (spec.dataset, target) for spec, target in product(SPECS, TARGETS)
    ]
    # the paper's dataset colors, cool for neutron stars and warm for black holes, the mass
    # the darker shade
    shades = {
        "neutron_stars": colormap(style.neutron_stars.beta_cmap),
        "black_holes": colormap(style.black_holes.beta_cmap),
    }
    tone = {"mass": 0.25, "charge": 0.7}
    colors = {(dataset, target): shades[dataset](tone[target]) for dataset, target in pairs}
    marker = {"pointwise": "o", "curve-wise": "s"}
    for axes, name, chosen in (
        *((axes, "knots", chosen) for axes, chosen in knots),
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
                markersize=mark_size(style, 2.5),
                alpha=mark_alpha(style, 0.6),
                color=colors[(p.job.dataset, p.job.target)],
            )
        axes.set_xlabel(name)
        axes.set_ylabel("folds' figures")
    neighbours_axes.set_xscale("log")
    _plain_log_ticks(neighbours_axes.xaxis, (50, 100, 200, 400))
    ranged: list[tuple[Axes, BaseDistribution]] = [
        *((axes, KNOTS) for axes, _ in knots),
        (neighbours_axes, SPACES["local RBF"]["neighbours"]),
    ]
    if network_axes is not None:
        ranged.append((network_axes, SPACES["ResNet"]["width"]))
    for axes, distribution in ranged:
        ends = range_ends(distribution)
        for end in (ends.low, ends.high):
            axes.axvline(end, color="grey", linestyle="--", linewidth=line_width(style, 0.8))
    if network_axes is None:
        _ranges_legend(figure, pairs, colors, wall=False)
        return figure
    network = [p for p in points if p.job.candidate.family == "ResNet"]
    ran = [p for p in network if p.figures is not None]
    shown = network_axes.scatter(
        [p.params["width"] for p in ran],
        [p.params["lr"] for p in ran],
        c=[p.figures for p in ran],
        cmap="viridis",
        s=mark_size(style, 10),
    )
    walls = [p for p in network if p.figures is None]
    network_axes.plot(
        [p.params["width"] for p in walls],
        [p.params["lr"] for p in walls],
        linestyle="",
        marker="x",
        markersize=mark_size(style, 4),
        color="black",
    )
    network_axes.set_xscale("log")
    _plain_log_ticks(network_axes.xaxis, (32, 128, 512))
    network_axes.set_yscale("log")
    network_axes.set_xlabel("width")
    network_axes.set_ylabel("learning rate")
    figure.colorbar(shown, ax=network_axes, label="folds' figures")
    _ranges_legend(figure, pairs, colors, wall=True)
    return figure


def _ranges_legend(
    figure: Figure,
    pairs: Sequence[tuple[str, Target]],
    colors: Mapping[tuple[str, str], Any],
    wall: bool,
) -> None:
    """The pairs by colour, the units by marker, and the wall cross if the network is drawn,
    below the panels."""
    handles = [
        *(Line2D([], [], color=colors[pair], marker="o", linestyle="") for pair in pairs),
        Line2D([], [], color="black", marker="o", linestyle=""),
        Line2D([], [], color="black", marker="s", linestyle=""),
    ]
    labels = [
        *(f"{DATASET_TEX[dataset]} {SYMBOLS[target_symbol(target)]}" for dataset, target in pairs),
        *UNITS,
    ]
    if wall:
        handles.append(Line2D([], [], color="black", marker="x", linestyle=""))
        labels.append("wall")
    figure.legend(handles, labels, loc="outside lower center", ncols=4, fontsize="x-small")


def _plain_log_ticks(axis: Any, ticks: Sequence[float]) -> None:
    """Label a log axis at the given ticks only, as plain numbers."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(NullFormatter())


def _history_caption(minutes: float, variant: PrecisionHistoryVariant) -> str:
    """The caption of the history figure in one variant, its wall stated."""
    searches = "The searches of \\cref{tab:tuning}, per dataset and target: each short-listed"
    drawn = (
        "The first ten trials are drawn at random, the rest by the tree-structured Parzen "
        "estimator. A cross at the bottom is a trial past the wall of "
        f"${minutes:g}$ minutes a fit."
    )
    match variant:
        case "dots_and_best":
            return (
                f"{searches} candidate's trials scored on the frozen folds (dots) and its best "
                f"so far (line, curve-wise dashed) against the trial number. {drawn}"
            )
        case "best_only":
            return (
                f"{searches} candidate's best significant figures on the frozen folds so far "
                f"(curve-wise dashed) against the trial number. {drawn}"
            )
        case "regret":
            return (
                f"{searches} candidate's trials as the figures they fall behind its best so far, "
                f"on a log scale, against the trial number; a new best sits at the bottom. {drawn}"
            )
        case _:
            assert_never(variant)


def history_tex(minutes: float, variant: PrecisionHistoryVariant = "dots_and_best") -> str:
    """The figure environment of the history figure, its wall stated in the caption."""
    return (
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{HISTORY}}}\n"
        f"\\caption{{{_history_caption(minutes, variant)}}}\n"
        "\\label{fig:tuning-history}\n\\end{figure}\n"
    )


def _ranges_caption(variant: PrecisionRangesVariant) -> str:
    """The caption of the ranges figure in one variant."""
    trials = (
        "The trials of \\cref{tab:tuning} against the ranges they searched, the ends of each "
        "range dashed: the significant figures on the folds against the knots of every "
        "curve-wise trial"
    )
    marks = "a color per dataset and target, pointwise dots and curve-wise squares"
    network = "the residual network's trials on the neutron-star charge in width and learning rate"
    widen = "A best trial on the end of its range marks the range to widen."
    match variant:
        case "row":
            return (
                f"{trials} (left) and against the neighbours of every trial of the local "
                f"interpolant (middle), {marks}; and {network} (right), colored by their "
                f"figures, a cross a trial past the wall. {widen}"
            )
        case "grid":
            return (
                f"{trials}, the neutron stars and the black holes apart (top), and against the "
                f"neighbours of every trial of the local interpolant (bottom left), {marks}; "
                f"and {network} (bottom right), colored by their figures, a cross a trial past "
                f"the wall. {widen}"
            )
        case "no_network":
            return (
                f"{trials} (left) and against the neighbours of every trial of the local "
                f"interpolant (right), {marks}. {widen}"
            )
        case _:
            assert_never(variant)


def ranges_tex(variant: PrecisionRangesVariant = "row") -> str:
    """The figure environment of the ranges figure, placed here or at the top of a page."""
    return (
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{RANGES}}}\n"
        f"\\caption{{{_ranges_caption(variant)}}}\n"
        "\\label{fig:tuning-ranges}\n\\end{figure}\n"
    )


def job_line(outcome: Outcome, done: int, total: int, elapsed: float) -> str:
    """One finished job: its pair and candidate, its tuned figures or its wall, then how many
    jobs are done, the elapsed time and the time left."""
    job = outcome.job
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
        f"{progress_tail(done, total, elapsed)}"
    )


# --- shell ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Context:
    """What every job of one tuning shares: the batch (its ledger, its trials' journal, its code
    and its start), the settings a trial does not search, the trials per job and the wall of one
    fit."""

    batch: Batch
    knobs: Knobs
    training: Training
    trials: int
    seconds: float
    log_level: str


@dataclass(frozen=True)
class Work:
    """One job with its pair, run in a worker process."""

    job: Job
    pair: Pair
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
    training = make_training(settings)
    tables = {
        NEUTRON_STARS.dataset: pd.read_parquet(args.ns),
        BLACK_HOLES.dataset: pd.read_parquet(args.bh),
    }
    split = pd.read_parquet(args.split)
    by_key = {(p.dataset, p.target): p for p in pairs(tables, split, SPECS, TARGETS)}
    with at_level(settings.log_level):
        if args.recover is not None:
            batch = args.recover
            _recover(args, batch)
        elif args.batch is None:
            batch = _run_tuning(args, by_key, training, settings.log_level, commit, dirty, now)
        else:
            batch = args.batch
            logger.info("rebuilding the tuning table from batch %s; nothing is fitted", batch)
        saved = outcomes_path(args.state, SECTION, batch)
        outcomes = outcomes_from_json(saved.read_text())
        if args.variants is not None:
            ran = tuple(outcome.job for outcome in outcomes)
            points = journal_points(saved.with_suffix(".journal"), batch, ran)
            apply_style(config.plot, NOTATION_TEX.read_text())
            _write_variants(args.variants / SECTION, points, config.plot)
            return
        screen = screen_figures(read_ledger(args.state / "ledger"), args.screen_batch)
        out = args.assets / SECTION
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{SECTION}_tab_tuning.tex"
        write_checked(path, tuning_table(outcomes, screen, config.tables.layout))
        logger.info("done: tuning table of batch %s, %d jobs -> %s", batch, len(outcomes), path)
        # A batch's figures show its own jobs, in the order it ran them.
        ran = tuple(outcome.job for outcome in outcomes)
        points = journal_points(saved.with_suffix(".journal"), batch, ran)
        apply_style(config.plot, NOTATION_TEX.read_text())
        picks = config.results.precision.variants
        drawn = (
            (
                HISTORY,
                history_figure(points, config.plot, picks.history),
                history_tex(args.minutes, picks.history),
            ),
            (
                RANGES,
                ranges_figure(points, config.plot, picks.ranges),
                ranges_tex(picks.ranges),
            ),
        )
        for name, figure, tex in drawn:
            figure.savefig(out / f"{name}.png", dpi=config.plot.dpi)
            plt.close(figure)
            write_checked(out / f"{name}.tex", tex)
            logger.info("done: %s of batch %s -> %s", name, batch, out / f"{name}.png")


def _run_tuning(
    args: argparse.Namespace,
    by_key: Mapping[tuple[str, str], Pair],
    training: Training,
    log_level: str,
    commit: Callable[[], str] | None,
    dirty: Callable[[], bool] | None,
    now: Callable[[], datetime] | None,
) -> str:
    """Run every job of the tuning as a new batch on the workers, a line logged as each ends,
    and save the outcomes; return the batch id."""
    batch = make_batch(
        args.state,
        journal=lambda found: outcomes_path(args.state, SECTION, found).with_suffix(".journal"),
        commit=commit or git_commit,
        dirty=dirty or git_dirty,
        now=now or utc_now,
        new_id=new_id,
    )
    saved = outcomes_path(args.state, SECTION, batch.id)
    saved.parent.mkdir(parents=True, exist_ok=True)
    context = Context(
        batch=batch,
        # The tuning never thins a design, so no rows per curve.
        knobs=Knobs(args.k, args.neighbours, args.knots, per_curve=0),
        training=training,
        trials=args.trials,
        seconds=args.minutes * 60.0,
        log_level=log_level,
    )
    batch.ledger.mkdir(parents=True, exist_ok=True)
    found = jobs(short_list(read_ledger(batch.ledger), args.screen_batch, args.within))
    logger.info(
        "tuning %d jobs, %d trials each, on %d workers, batch %s; journal %s",
        len(found),
        args.trials,
        args.workers,
        batch.id,
        batch.journal,
    )
    works = [
        partial(tune_job, Work(job, by_key[(job.dataset, job.target)], context)) for job in found
    ]
    started = time.monotonic()
    finished: dict[int, Outcome] = {}

    def report(index: int, outcome: Outcome) -> None:
        # Saved after every job, in job order, so a parent that dies leaves the finished ones.
        finished[index] = outcome
        logger.info(job_line(outcome, len(finished), len(works), time.monotonic() - started))
        write_checked(saved, outcomes_to_json([finished[i] for i in sorted(finished)]))

    # Each worker logs at the tuning's level with the networks' epoch tables silenced.
    outcomes = run_jobs(
        works,
        args.workers,
        log_level=context.log_level,
        quiet=(surrogate.__name__,),
        on_done=report,
    )
    write_checked(saved, outcomes_to_json(outcomes))
    return batch.id


def tune_job(work: Work) -> Outcome:
    """Run one job: search its candidate's space and record the best params' refit."""
    job, context = work.job, work.context
    candidate = job.candidate
    batch = context.batch
    study = open_study(study_name(batch.id, job), batch.journal)
    study.sampler = optuna.samplers.TPESampler(seed=context.training.seed)

    def record(run: Run) -> str:
        """Write the best params' run as a ledger entry and name it."""
        best = study.best_params
        settings: dict[str, Setting] = {
            "unit": candidate.unit,
            "trials": context.trials,
            **asdict(context.knobs),
            **best,
        }
        meta = RunMetadata(
            id=new_id(),
            batch=batch.id,
            dataset=job.dataset,
            target=job.target,
            candidate=candidate.family,
            settings=settings,
            commit=batch.commit,
            dirty=batch.dirty,
            started=batch.started,
            epochs=None,
        )
        return write_entry(batch.ledger, run, meta).meta.id

    return run_job(
        job,
        work.pair.design,
        lambda params: tuned_fitter(
            candidate, work.pair.space, context.knobs, context.training, params
        ),
        space_of(candidate.family, candidate.unit),
        context.trials,
        context.training.seed,
        context.training.valid_fraction,
        context.seconds,
        study,
        record,
    )


def journal_points(journal: Path, batch: str, wanted: Sequence[Job]) -> tuple[TrialPoint, ...]:
    """The trials of each wanted job that started, in that order, read from the batch's study
    journal (nothing is created there); a job without a study did not start and has none."""
    storage = JournalStorage(JournalFileBackend(str(journal)))
    held = set(optuna.get_all_study_names(storage))
    return tuple(
        point
        for job in wanted
        if study_name(batch, job) in held
        for point in trial_points(
            optuna.load_study(study_name=study_name(batch, job), storage=storage), job
        )
    )


def _recover(args: argparse.Namespace, batch: str) -> None:
    """Rebuild the batch's outcomes file from its journal and the ledger; nothing is fitted."""
    saved = outcomes_path(args.state, SECTION, batch)
    journal = saved.with_suffix(".journal")
    assert journal.is_file(), f"no journal of batch {batch}: {journal}"
    storage = JournalStorage(JournalFileBackend(str(journal)))
    studies = {
        name: optuna.load_study(study_name=name, storage=storage)
        for name in optuna.get_all_study_names(storage)
        if name.startswith(f"{batch} ")
    }
    entries = read_ledger(args.state / "ledger")
    wanted = jobs(short_list(entries, args.screen_batch, args.within))
    outcomes = recovered(batch, studies, entries, args.minutes, wanted)
    write_checked(saved, outcomes_to_json(outcomes))
    logger.info("recovered %d outcomes of batch %s -> %s", len(outcomes), batch, saved)


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
    parser.add_argument(
        "--screen-batch",
        required=True,
        help="the family screen's batch: its short list and its figures",
    )
    parser.add_argument(
        "--within",
        type=float,
        required=True,
        help="the short list keeps the candidates this many figures behind the pair's best",
    )
    parser.add_argument(
        "--variants",
        type=Path,
        help="with --batch: draw every figure variant into this dir to pick from; assets untouched",
    )
    args = parser.parse_args(argv)
    if args.variants is not None and args.batch is None:
        parser.error("--variants draws from a recorded batch: give --batch too")
    return args


def _write_variants(out: Path, points: Sequence[TrialPoint], style: PlotStyle) -> None:
    """Every variant of the history and ranges figures as <stem>_<variant>.png in out and per
    figure their contact sheet as <stem>_variants.png, at the screen's resolution."""
    out.mkdir(parents=True, exist_ok=True)
    sheets: dict[str, dict[str, Callable[[], Figure]]] = {
        HISTORY: {
            v: partial(history_figure, points, style, v) for v in get_args(PrecisionHistoryVariant)
        },
        RANGES: {
            v: partial(ranges_figure, points, style, v) for v in get_args(PrecisionRangesVariant)
        },
    }
    for stem, drawers in sheets.items():
        write_variant_sheet(out, stem, drawers, style)


if __name__ == "__main__":
    # The console handler only; main sets the level from the config for the run's duration.
    logging.basicConfig(format="%(message)s")
    main(sys.argv[1:])

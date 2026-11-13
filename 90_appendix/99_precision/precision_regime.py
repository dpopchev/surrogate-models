"""The precision regime of the tuning and precision appendix (W-093): each precision
option laid on the tuned winner of every pair, scored on the frozen folds beside the tuned
figures and the pair's data ceiling.

The winners are the saved surrogates of make pipeline, read as plain JSON from
<state>/41_algorithm/summary.json. One job per pair, applicable option and seed runs the harness
on the pair's design into the run ledger; the scores are saved after every job, so --batch
rebuilds the figure and the macros from a saved batch with nothing fitted. An option is kept for
a pair only when its mean fold figures beat the tuned ones by more than the tuned spread over
folds and seeds; the test curves never decide.
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from precision_options import Option, applicable, option_fitter

from paperlib import surrogate
from paperlib.batches import (
    Batch,
    at_level,
    git_commit,
    git_dirty,
    make_batch,
    new_id,
    progress_tail,
    utc_now,
    walled,
    write_checked,
    write_entry,
)
from paperlib.candidates import Knobs
from paperlib.config import NOTATION_TEX, PaperConfig, load_config
from paperlib.design import (
    BLACK_HOLES,
    DATASET_TEX,
    NEUTRON_STARS,
    SPECS,
    TARGETS,
    Pair,
    Target,
    pairs,
    target_symbol,
)
from paperlib.eda import render_macros
from paperlib.final import Winner
from paperlib.harness import Run, harness
from paperlib.plots import SYMBOLS, PlotStyle, apply_style, figure_size, line_width, mark_size
from paperlib.runs import RunMetadata, Setting
from paperlib.scorecard import significant_figures
from paperlib.surrogate import Training, make_training
from paperlib.workers import run_jobs

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

SECTION = "99_precision"
FIGURE = f"{SECTION}_fig_precision"
MACROS = f"{SECTION}_precision_num"
SUMMARY = Path("41_algorithm") / "summary.json"
OPTIONS: tuple[Option, ...] = get_args(Option)
PairKey = tuple[str, Target]
OptionFitter = Callable[..., Any]


@dataclass(frozen=True)
class Saved:
    """One saved surrogate of make pipeline: its winner, its test figures and its ceiling."""

    winner: Winner
    test: float
    ceiling: float


@dataclass(frozen=True)
class Job:
    """One option on one pair at one seed."""

    dataset: str
    target: Target
    option: Option
    seed: int


@dataclass(frozen=True)
class Result:
    """A finished job: the figures of each frozen fold, the test figures, the ledger entry."""

    job: Job
    folds: tuple[float, ...]
    test: float
    entry: str


@dataclass(frozen=True)
class Scored:
    """One option on one pair: the fold figures of each seed (a tuple per seed) and the test
    figures of each seed, as significant figures of the p95."""

    dataset: str
    target: Target
    option: Option
    folds: tuple[tuple[float, ...], ...]
    test: tuple[float, ...]


@dataclass(frozen=True)
class Context:
    """What every job shares: the batch, the knobs the winners' params lie over, the training,
    the residual stage's neighbours, the wall of a fit and the option's fitter builder."""

    batch: Batch
    knobs: Knobs
    training: Training
    residual_neighbours: int
    seconds: float
    fitter: OptionFitter


@dataclass(frozen=True)
class Work:
    """One job with its saved winner and pair, run in a worker process."""

    job: Job
    saved: Saved
    pair: Pair
    context: Context


# --- pure functions ---------------------------------------------------------------------------


def key_of(scored: Scored) -> PairKey:
    """The pair a score belongs to."""
    return scored.dataset, scored.target


def fold_figures(run: Run) -> tuple[float, ...]:
    """The p95 figures of each frozen fold of a run."""
    return tuple(significant_figures(fold.p95) for fold in run.folds)


def mean_folds(scored: Scored) -> float:
    """The mean fold figures over folds and seeds."""
    return float(np.mean(scored.folds))


def spread(tuned: Scored) -> float:
    """The population standard deviation of the tuned option's fold figures over folds and
    seeds (5 folds x 3 seeds)."""
    return float(np.std(tuned.folds))


def gain(option: Scored, tuned: Scored) -> float:
    """The option's mean fold figures minus the tuned ones."""
    return mean_folds(option) - mean_folds(tuned)


def _tuned_of(scores: Sequence[Scored]) -> Scored:
    tuned = [s for s in scores if s.option == "tuned"]
    assert tuned, "the scores of a pair hold its tuned option"
    return tuned[0]


def best_gain(scores: Sequence[Scored]) -> Scored | None:
    """The option other than tuned with the largest gain, None when only tuned was scored."""
    tuned = _tuned_of(scores)
    others = [s for s in scores if s.option != "tuned"]
    return max(others, key=lambda s: gain(s, tuned), default=None)


def kept_option(scores: Sequence[Scored]) -> Option:
    """The option of one pair with the largest gain when it exceeds the tuned spread, else
    tuned; the test figures never decide."""
    tuned = _tuned_of(scores)
    best = best_gain(scores)
    if best is not None and gain(best, tuned) > spread(tuned):
        return best.option
    return "tuned"


def by_pair(scores: Sequence[Scored]) -> dict[PairKey, list[Scored]]:
    """The scores grouped by pair, in the paper's order of pairs and of options."""
    grouped: dict[PairKey, list[Scored]] = defaultdict(list)
    for scored in sorted(scores, key=lambda s: OPTIONS.index(s.option)):
        grouped[key_of(scored)].append(scored)
    order: list[PairKey] = [(spec.dataset, target) for spec in SPECS for target in TARGETS]
    return {key: grouped[key] for key in order if key in grouped}


def scores_of(results: Sequence[Result]) -> tuple[Scored, ...]:
    """The finished jobs gathered per pair and option, the seeds in seed order."""
    grouped: dict[tuple[str, Target, Option], list[Result]] = defaultdict(list)
    for result in sorted(results, key=lambda r: r.job.seed):
        job = result.job
        grouped[(job.dataset, job.target, job.option)].append(result)
    return tuple(
        Scored(d, t, o, tuple(r.folds for r in found), tuple(r.test for r in found))
        for (d, t, o), found in grouped.items()
    )


def scores_to_json(scores: Sequence[Scored]) -> str:
    """The scores as plain JSON: per pair and option, the fold figures per seed and fold and
    the test figures per seed."""
    return json.dumps([asdict(s) for s in scores], indent=2) + "\n"


def scores_from_json(text: str) -> tuple[Scored, ...]:
    """The scores scores_to_json wrote."""
    return tuple(
        Scored(
            row["dataset"],
            row["target"],
            row["option"],
            tuple(tuple(seed) for seed in row["folds"]),
            tuple(row["test"]),
        )
        for row in json.loads(text)
    )


def pair_prefix(dataset: str, target: str) -> str:
    """The macro stem of a pair, as precNsMass."""
    return f"prec{DATASET_TEX[dataset].capitalize()}{target.capitalize()}"


def _figures(value: float) -> str:
    return f"{value:.2f}"


def _text(word: str) -> str:
    return f"\\ensuremath{{\\mathrm{{{word}}}}}"


# The option's word in a macro name: letters only, since digits end a control-sequence name.
OPTION_WORD = {"tuned": "Tuned", "float64": "Double", "lbfgs": "Lbfgs", "residual": "Residual"}


def precision_numbers(scores: Sequence[Scored]) -> dict[str, str]:
    """The macros of the figure's numbers: per pair and option the mean fold and test figures
    (n/a for an option that was not scored), the tuned spread, the best option's gain and the
    kept option, and the seeds."""
    numbers: dict[str, str] = {}
    for (dataset, target), found in by_pair(scores).items():
        stem = pair_prefix(dataset, target)
        scored = {s.option: s for s in found}
        for option in OPTIONS:
            name = f"{stem}{OPTION_WORD[option]}"
            if option in scored:
                numbers[f"{name}Folds"] = _figures(mean_folds(scored[option]))
                numbers[f"{name}Test"] = _figures(float(np.mean(scored[option].test)))
            else:
                numbers[f"{name}Folds"] = numbers[f"{name}Test"] = _text("n/a")
        tuned = _tuned_of(found)
        best = best_gain(found)
        numbers[f"{stem}Spread"] = _figures(spread(tuned))
        numbers[f"{stem}Gain"] = _text("n/a") if best is None else _figures(gain(best, tuned))
        numbers[f"{stem}Kept"] = _text(kept_option(found))
    first = next(iter(scores), None)
    numbers["precSeeds"] = str(0 if first is None else len(first.folds))
    return numbers


def precision_caption() -> str:
    """The caption of the precision figure."""
    return (
        "The precision options on the tuned winner of each pair: the mean significant figures "
        "on the frozen folds (dot) with the spread of the tuned option over folds and seeds "
        "(bar), the figures on the test curves (open mark), the data ceiling of the pair "
        "(dashed) and the kept option (star), the one whose gain exceeds the spread. An option "
        "the family does not take is n/a."
    )


def precision_tex() -> str:
    """The figure environment of the precision figure."""
    return (
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{FIGURE}}}\n"
        f"\\caption{{{precision_caption()}}}\n"
        "\\label{fig:precision}\n\\end{figure}\n"
    )


def precision_figure(
    scores: Sequence[Scored], ceilings: Mapping[PairKey, float], style: PlotStyle
) -> Figure:
    """A panel per pair, a row per option: a dot at the mean fold figures with a bar of the
    tuned spread, an open mark at the mean test figures, n/a for an option not scored, the
    pair's ceiling as a dashed line and a star beside the kept option."""
    grouped = by_pair(scores)
    figure, grid = plt.subplots(
        len(grouped),
        1,
        figsize=figure_size(style, len(grouped), 0.3),
        layout="constrained",
        squeeze=False,
        sharex=True,
    )
    colors = dict(zip(OPTIONS, sns.color_palette(style.palette, len(OPTIONS)), strict=True))
    for axes, ((dataset, target), found) in zip(grid[:, 0], grouped.items(), strict=True):
        _panel(axes, dataset, target, found, ceilings[(dataset, target)], colors, style)
    grid[-1, 0].set_xlabel("significant figures of the 95th percentile")
    handles = [
        Line2D([], [], marker="o", color="black", linestyle=""),
        Line2D([], [], marker="o", markerfacecolor="none", color="black", linestyle=""),
        Line2D([], [], color="black", linestyle="--"),
        Line2D([], [], marker="*", color="black", linestyle=""),
    ]
    labels = ["folds, tuned spread", "test", "ceiling", "kept"]
    figure.legend(handles, labels, loc="outside lower center", ncols=4, fontsize="x-small")
    return figure


def _panel(
    axes: Axes,
    dataset: str,
    target: Target,
    found: Sequence[Scored],
    ceiling: float,
    colors: Mapping[Option, Any],
    style: PlotStyle,
) -> None:
    """One pair's panel of precision_figure."""
    tuned = _tuned_of(found)
    kept = kept_option(found)
    scored = {s.option: s for s in found}
    bar = spread(tuned)
    for row, option in enumerate(OPTIONS):
        if option not in scored:
            axes.text(0.01, row, "n/a", transform=axes.get_yaxis_transform(), va="center")
            continue
        axes.errorbar(
            mean_folds(scored[option]),
            row,
            xerr=bar,
            marker="o",
            markersize=mark_size(style, 4),
            color=colors[option],
            elinewidth=line_width(style, 1.2),
        )
        axes.plot(
            float(np.mean(scored[option].test)),
            row,
            marker="o",
            markersize=mark_size(style, 5),
            markerfacecolor="none",
            color=colors[option],
        )
        if option == kept:
            axes.plot(
                0.97,
                row,
                marker="*",
                markersize=mark_size(style, 8),
                color="black",
                transform=axes.get_yaxis_transform(),
            )
    axes.axvline(ceiling, color="black", linestyle="--", linewidth=line_width(style, 0.8))
    axes.set_yticks(range(len(OPTIONS)), OPTIONS)
    axes.set_ylim(len(OPTIONS) - 0.5, -0.5)
    axes.set_title(f"{DATASET_TEX[dataset]} {SYMBOLS[target_symbol(target)]}", fontsize="small")


def job_line(result: Result, done: int, total: int, elapsed: float) -> str:
    """One finished job as a log line: its pair, option and seed, its mean fold and test
    figures, the elapsed time and the time left."""
    job = result.job
    return (
        f"{done}/{total} {DATASET_TEX[job.dataset]} {job.target} {job.option} seed {job.seed}: "
        f"folds {np.mean(result.folds):.2f}, test {result.test:.2f}, "
        f"{progress_tail(done, total, elapsed)}"
    )


def jobs(
    saved: Mapping[PairKey, Saved], wanted: Sequence[PairKey], seeds: Sequence[int]
) -> tuple[Job, ...]:
    """One job per wanted pair, applicable option and seed, in that order."""
    return tuple(
        Job(dataset, target, option, seed)
        for dataset, target in wanted
        for option, takes in applicable(saved[(dataset, target)].winner.candidate).items()
        if takes
        for seed in seeds
    )


# --- adapters and the run ---------------------------------------------------------------------


def read_saved(path: Path) -> tuple[Saved, ...]:
    """The saved surrogates of make pipeline from its summary.json; a missing file raises
    FileNotFoundError naming the path."""
    if not path.is_file():
        raise FileNotFoundError(f"no saved surrogates at {path}: run make pipeline first")
    return tuple(
        Saved(
            Winner.of(row["dataset"], row["target"], row["family"], row["unit"], row["params"]),
            float(row["test"]),
            float(row["ceiling"]),
        )
        for row in json.loads(path.read_text())
    )


def scores_path(state: Path, batch: str) -> Path:
    """Where a batch keeps its scores."""
    return state / SECTION / f"{batch}-precision.json"


def run_job(work: Work) -> Result:
    """Run one job through the harness, record its ledger entry and return its figures."""
    job, context = work.job, work.context
    winner = work.saved.winner
    batch = context.batch
    fit = context.fitter(
        job.option,
        winner.candidate,
        work.pair.space,
        context.knobs,
        context.training,
        winner.params,
        context.residual_neighbours,
    )
    settings: dict[str, Setting] = {
        "unit": winner.candidate.unit,
        "option": job.option,
        **asdict(context.knobs),
        "residual_neighbours": context.residual_neighbours,
        **winner.params,
    }
    meta = RunMetadata(
        id=new_id(),
        batch=batch.id,
        dataset=job.dataset,
        target=job.target,
        candidate=winner.candidate.family,
        settings=settings,
        commit=batch.commit,
        dirty=batch.dirty,
        started=batch.started,
        epochs=None,
    )
    entries: list[str] = []
    (run,) = harness(
        work.pair.design,
        job.target,
        walled(fit, context.seconds),
        [job.seed],
        context.training.valid_fraction,
        record=lambda ran: entries.append(write_entry(batch.ledger, ran, meta).meta.id),
    )
    test = significant_figures(run.test.zones["test"].p95)
    return Result(job, fold_figures(run), test, entries[0])


def _run_batch(
    args: argparse.Namespace,
    saved: Mapping[PairKey, Saved],
    by_key: Mapping[PairKey, Pair],
    config: PaperConfig,
    fitter: OptionFitter,
    commit: Callable[[], str] | None,
    dirty: Callable[[], bool] | None,
    now: Callable[[], datetime] | None,
) -> str:
    """Run every job as a new batch on the workers, a line logged and the scores saved as each
    ends; return the batch id."""
    batch = make_batch(
        args.state,
        journal=lambda found: scores_path(args.state, found).with_suffix(".journal"),
        commit=commit or git_commit,
        dirty=dirty or git_dirty,
        now=now or utc_now,
        new_id=new_id,
    )
    batch.ledger.mkdir(parents=True, exist_ok=True)
    target = scores_path(args.state, batch.id)
    target.parent.mkdir(parents=True, exist_ok=True)
    settings = config.network
    context = Context(
        batch,
        Knobs(args.k, args.neighbours, args.knots, per_curve=0),
        make_training(settings),
        args.residual_neighbours,
        args.minutes * 60.0,
        fitter,
    )
    seeds = [settings.seed + offset for offset in range(args.seeds)]
    found = jobs(saved, args.pair, seeds)
    works = [
        partial(
            run_job,
            Work(job, saved[(job.dataset, job.target)], by_key[(job.dataset, job.target)], context),
        )
        for job in found
    ]
    logger.info("scoring %d jobs on %d workers, batch %s", len(works), args.workers, batch.id)
    started = time.monotonic()
    finished: dict[int, Result] = {}

    def report(index: int, result: Result) -> None:
        # Saved after every job, in job order, so a parent that dies leaves the finished ones.
        finished[index] = result
        logger.info(job_line(result, len(finished), len(works), time.monotonic() - started))
        write_checked(target, scores_to_json(scores_of([finished[i] for i in sorted(finished)])))

    results = run_jobs(
        works,
        args.workers,
        log_level=settings.log_level,
        quiet=(surrogate.__name__,),
        on_done=report,
    )
    write_checked(target, scores_to_json(scores_of(results)))
    return batch.id


def main(
    argv: Sequence[str],
    config: PaperConfig | None = None,
    fitter: OptionFitter = option_fitter,
    commit: Callable[[], str] | None = None,
    dirty: Callable[[], bool] | None = None,
    now: Callable[[], datetime] | None = None,
) -> None:
    """Score every option of every wanted pair on the workers, saving the scores under the state
    folder, and write the figure wrapper, the PNG and the macros -- or, given --batch, fit
    nothing and take that batch's scores."""
    args = _arguments(argv)
    for path in (args.ns, args.bh, args.split):
        assert path.is_file(), f"input not found: {path}"
    config = config or load_config()
    saved: dict[PairKey, Saved] = {
        (s.winner.dataset, s.winner.target): s for s in read_saved(args.state / SUMMARY)
    }
    tables = {
        NEUTRON_STARS.dataset: pd.read_parquet(args.ns),
        BLACK_HOLES.dataset: pd.read_parquet(args.bh),
    }
    by_key: dict[PairKey, Pair] = {
        (p.dataset, p.target): p for p in pairs(tables, pd.read_parquet(args.split), SPECS, TARGETS)
    }
    with at_level(config.network.log_level):
        if args.batch is None:
            batch = _run_batch(args, saved, by_key, config, fitter, commit, dirty, now)
        else:
            batch = args.batch
            logger.info("rebuilding the precision figure from batch %s; nothing is fitted", batch)
        scores = scores_from_json(scores_path(args.state, batch).read_text())
        out = args.assets / SECTION
        (out / "figures").mkdir(parents=True, exist_ok=True)
        apply_style(config.plot, NOTATION_TEX.read_text())
        ceilings: dict[PairKey, float] = {key: s.ceiling for key, s in saved.items()}
        figure = precision_figure(scores, ceilings, config.plot)
        figure.savefig(out / "figures" / f"{FIGURE}.png", dpi=config.plot.dpi)
        plt.close(figure)
        write_checked(out / f"{FIGURE}.tex", precision_tex())
        write_checked(out / f"{MACROS}.tex", render_macros(precision_numbers(scores)))
        logger.info("done: precision figure and macros of batch %s -> %s", batch, out)


def _pair(text: str) -> PairKey:
    """A --pair value, dataset:target."""
    dataset, _, target = text.partition(":")
    known = [(spec.dataset, t) for spec in SPECS for t in TARGETS]
    if (dataset, target) not in known:
        raise argparse.ArgumentTypeError(f"{text!r} is none of {[f'{d}:{t}' for d, t in known]}")
    return dataset, target  # type: ignore[return-value]


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    """The inputs, the asset and state folders and the knobs; 99_precision.mk passes every knob,
    so it is the one place their values are written."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ns", "bh", "split", "assets", "state"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--k", type=int, required=True, help="the k-NN's k a winner lies over")
    parser.add_argument("--neighbours", type=int, required=True, help="the local RBF's, same")
    parser.add_argument("--knots", type=int, required=True, help="the curve-wise spline's, same")
    parser.add_argument("--residual-neighbours", type=int, required=True, help="the residual's")
    parser.add_argument("--seeds", type=int, required=True, help="seeds per option and pair")
    parser.add_argument("--minutes", type=float, required=True, help="the wall of one fit")
    parser.add_argument("--workers", type=int, required=True, help="jobs at once")
    parser.add_argument(
        "--pair",
        type=_pair,
        action="append",
        help="dataset:target, repeatable; default every pair (the one-pair smoke run)",
    )
    parser.add_argument("--batch", help="fit nothing; rebuild from this saved batch")
    args = parser.parse_args(argv)
    args.pair = args.pair or [(spec.dataset, t) for spec in SPECS for t in TARGETS]
    return args


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s")
    main(sys.argv[1:])

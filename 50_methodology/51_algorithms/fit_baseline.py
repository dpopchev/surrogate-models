"""The baseline surrogate behind Section 4.1 (W-034): a plain MLP fitted to the NS mass.

Inputs: the prepared NS table and the frozen curve split (local/state/). The net trains on every
non-test curve through shared/design.py and shared/surrogate.py, with early stopping on held-out
training curves, and is scored once on the test curves. Outputs, in the section's asset folder:
the \\baseNsMass... macros (test MARE, RMSE, fit seconds, epochs) and the parity figure; in
<state dir>/51_algorithms/<run>/, the run record and the diagnostics for the developer (W-035).

Run as `uv run python <this file> <ns.parquet> <split.parquet> <asset dir> <state dir>`
(mk/paper.mk does).
"""

import logging
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from shared.config import PaperConfig, load_config
from shared.design import NEUTRON_STARS, Design, DesignSpec, design
from shared.diagnostics import curve_overlay, error_cdf, loss_curve
from shared.eda import curve_ids, render_macros, sci_tex
from shared.plots import PlotStyle, anchor_color, apply_style
from shared.runs import RunRecord, from_json, run_name, run_stem, to_json
from shared.surrogate import LIVE_EVERY, Training, duration_text, make_estimator, mare, rmse

logger = logging.getLogger(__name__)

SECTION = "51_algorithms"
# The diagnostics are for the screen, not print: a lower resolution than the paper's figures.
DIAGNOSTICS_DPI = 150

# --- vocabulary and types ---------------------------------------------------------------------


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


# --- pure functions ---------------------------------------------------------------------------


def fit_and_score(
    design: Design, training: Training, clock: Callable[[], float], live_plot: Path | None = None
) -> Fit:
    """Fit on the non-test rows, score on the test rows; clock times the fit; with live_plot,
    the loss curve is redrawn there while it trains."""
    train, test = ~design.test, design.test
    estimator = make_estimator(training, n_inputs=design.X.shape[1], live_plot=live_plot)
    started = clock()
    estimator.fit(design.X[train], design.y[train], net__groups=design.groups[train])
    seconds = clock() - started
    y_true, y_pred = design.y[test], estimator.predict(design.X[test])
    history = tuple(
        {key: value for key, value in row.items() if key != "batches"}
        for row in estimator.named_steps["net"].history
    )
    best_epoch = int(min(history, key=lambda row: row["valid_loss"])["epoch"])
    return Fit(
        y_true=y_true,
        y_pred=y_pred,
        mare=mare(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        seconds=seconds,
        epochs=len(history),
        best_epoch=best_epoch,
        history=history,
        x_test=design.X[test],
        groups_test=design.groups[test],
    )


def numbers(fit: Fit) -> dict[str, str]:
    """The \\baseNsMass... macros of one fit."""
    return {
        "baseNsMassMare": sci_tex(fit.mare),
        "baseNsMassRmse": sci_tex(fit.rmse),
        "baseNsMassFitSeconds": f"{fit.seconds:.0f}",
        "baseNsMassEpochs": str(fit.epochs),
    }


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
    """Fit the NS mass baseline and write its numbers and parity figure into the section's
    folder of the asset dir, after emptying that folder of an earlier run; leave the run's
    record and diagnostics in a folder named after the run under the state dir."""
    assert len(argv) == 4, (
        f"usage: fit_baseline.py <ns.parquet> <split.parquet> <asset dir> <state dir>, got {argv}"
    )
    source, split_file, out = Path(argv[0]), Path(argv[1]), Path(argv[2]) / SECTION
    state = Path(argv[3]) / SECTION
    for path in (source, split_file):
        assert path.is_file(), f"input not found: {path}"
    config = config or load_config()
    commit, dirty, now = commit or _git_commit, dirty or _git_dirty, now or _utc_now
    settings = config.methodology.algorithms
    started = now()
    run = state / run_stem(NEUTRON_STARS.dataset, "mass", started)
    run.mkdir(parents=True, exist_ok=True)
    # latest.log is what make follow tails: repointed at each run's log as the run starts.
    latest = state / "latest.log"
    latest.unlink(missing_ok=True)
    latest.symlink_to(Path(run.name) / "train.log")
    with _logging_into(run / "train.log", settings.log_level):
        _fit_and_write(source, split_file, out, run, config, started, commit(), dirty())


def _fit_and_write(
    source: Path,
    split_file: Path,
    out: Path,
    run: Path,
    config: PaperConfig,
    started: datetime,
    commit: str,
    uncommitted: bool,
) -> None:
    """The run itself, its log lines going to the console and to the run's train.log."""
    if uncommitted:
        logger.info("the code holds uncommitted changes: run.json marks this run dirty")
    apply_style(config.plot)
    threads = config.methodology.algorithms.threads
    torch.set_num_threads(threads)
    training = Training(
        **config.methodology.algorithms.model_dump(exclude={"log_level", "threads"})
    )
    table = pd.read_parquet(source)
    data = design(
        table, pd.read_parquet(split_file), NEUTRON_STARS, "mass", config.data_analysis.charge_floor
    )
    train = ~data.test
    rows = (int(train.sum()), int(data.test.sum()))
    curves = (len(np.unique(data.groups[train])), len(np.unique(data.groups[data.test])))
    logger.info("== baseline run: %s / mass ==", NEUTRON_STARS.dataset)
    for line in start_banner(run, out, training, rows, curves, threads):
        logger.info(line)
    fit = fit_and_score(data, training, time.perf_counter, live_plot=run / "loss_curve.png")
    logger.info(progress_line(1, 1, fit.seconds))
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.iterdir():
        stale.unlink()
    figure = parity(fit, config.plot)
    figure.savefig(out / f"{PARITY}.png", dpi=config.plot.dpi)
    plt.close(figure)
    (out / f"{PARITY}.tex").write_text(figure_tex())
    (out / f"{SECTION}_num.tex").write_text(render_macros(numbers(fit)))
    record = RunRecord(
        dataset=NEUTRON_STARS.dataset,
        target="mass",
        training=training,
        commit=commit,
        dirty=uncommitted,
        started=started,
        seconds=fit.seconds,
        epochs=fit.epochs,
        best_epoch=fit.best_epoch,
        mare=fit.mare,
        rmse=fit.rmse,
    )
    assert run.name == run_name(record), f"run folder {run.name} is not {run_name(record)}"
    write_diagnostics(fit, record, run, curve_names(table, NEUTRON_STARS))
    assert from_json((run / "run.json").read_text()) == record, (
        f"{run}/run.json does not round-trip"
    )
    logger.info("== baseline run finished ==")
    for line in end_banner(fit, training, sorted(out.iterdir()) + sorted(run.iterdir())):
        logger.info(line)
    logger.info(
        "done: baseline NS mass numbers and parity figure -> %s, diagnostics -> %s", out, run
    )


def write_diagnostics(fit: Fit, record: RunRecord, run: Path, names: dict[int, str]) -> None:
    """Write the run record, the loss curve, the error CDF and the curve overlay (curves
    labelled by names) into run."""
    run.mkdir(parents=True, exist_ok=True)
    (run / "run.json").write_text(to_json(record))
    overlay = curve_overlay(
        fit.x_test[:, 0], fit.y_true, fit.y_pred, fit.groups_test, k=3, names=names
    )
    overlay.axes[0].set_xlabel("$\\log_{10}\\rho_c$")
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


def _utc_now() -> datetime:
    """The current time in UTC."""
    return datetime.now(UTC)


if __name__ == "__main__":
    # The console handler only; main sets the level from the config for the run's duration.
    logging.basicConfig(format="%(message)s")
    main(sys.argv[1:])

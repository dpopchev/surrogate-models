"""The baseline surrogate behind Section 5.1 (W-034): a plain MLP fitted to the NS mass.

Inputs: the prepared NS table and the frozen curve split (local/state/). The net trains on every
non-test curve through shared/design.py and shared/surrogate.py, with early stopping on held-out
training curves, and is scored once on the test curves. Outputs, in the section's asset folder:
the \\baseNsMass... macros (test MARE, RMSE, fit seconds, epochs) and the parity figure.

Run as `uv run python <this file> <ns.parquet> <split.parquet> <asset dir>` (mk/paper.mk does).
"""

import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from shared.config import PaperConfig, load_config
from shared.design import NEUTRON_STARS, Design, design
from shared.eda import number_tex, render_macros
from shared.plots import PlotStyle, anchor_color, apply_style
from shared.surrogate import Training, make_estimator, mare, rmse

logger = logging.getLogger(__name__)

SECTION = "51_algorithms"

# --- vocabulary and types ---------------------------------------------------------------------


@dataclass(frozen=True)
class Fit:
    """One fitted baseline scored on the test curves."""

    y_true: np.ndarray
    y_pred: np.ndarray
    mare: float
    rmse: float
    seconds: float
    epochs: int


# --- pure functions ---------------------------------------------------------------------------


def fit_and_score(design: Design, training: Training, clock: Callable[[], float]) -> Fit:
    """Fit on the non-test rows, score on the test rows; clock times the fit."""
    train, test = ~design.test, design.test
    estimator = make_estimator(training, n_inputs=design.X.shape[1])
    started = clock()
    estimator.fit(design.X[train], design.y[train], net__groups=design.groups[train])
    seconds = clock() - started
    y_true, y_pred = design.y[test], estimator.predict(design.X[test])
    epochs = len(estimator.named_steps["net"].history)
    return Fit(y_true, y_pred, mare(y_true, y_pred), rmse(y_true, y_pred), seconds, epochs)


def numbers(fit: Fit) -> dict[str, str]:
    """The \\baseNsMass... macros of one fit."""
    return {
        "baseNsMassMare": number_tex(fit.mare),
        "baseNsMassRmse": number_tex(fit.rmse),
        "baseNsMassFitSeconds": f"{fit.seconds:.0f}",
        "baseNsMassEpochs": str(fit.epochs),
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


def main(argv: list[str], config: PaperConfig | None = None) -> None:
    """Fit the NS mass baseline and write its numbers and parity figure into the section's
    folder of the asset dir, after emptying that folder of an earlier run."""
    assert len(argv) == 3, (
        f"usage: fit_baseline.py <ns.parquet> <split.parquet> <asset dir>, got {argv}"
    )
    source, split_file, out = Path(argv[0]), Path(argv[1]), Path(argv[2]) / SECTION
    for path in (source, split_file):
        assert path.is_file(), f"input not found: {path}"
    config = config or load_config()
    apply_style(config.plot)
    settings = config.methodology.algorithms
    training = Training(**settings.model_dump(exclude={"log_level"}))
    data = design(
        pd.read_parquet(source),
        pd.read_parquet(split_file),
        NEUTRON_STARS,
        "mass",
        config.data_analysis.charge_floor,
    )
    train = ~data.test
    logger.info(
        "neutron_stars / mass: %d training rows on %d curves, %d test rows on %d curves",
        train.sum(),
        len(np.unique(data.groups[train])),
        data.test.sum(),
        len(np.unique(data.groups[data.test])),
    )
    fit = fit_and_score(data, training, time.perf_counter)
    logger.info(
        "test: MARE %.4g, RMSE %.4g, %d epochs, fit %.0f s",
        fit.mare,
        fit.rmse,
        fit.epochs,
        fit.seconds,
    )
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.iterdir():
        stale.unlink()
    figure = parity(fit, config.plot)
    figure.savefig(out / f"{PARITY}.png", dpi=config.plot.dpi)
    plt.close(figure)
    (out / f"{PARITY}.tex").write_text(figure_tex())
    (out / f"{SECTION}_num.tex").write_text(render_macros(numbers(fit)))
    logger.info("done: baseline NS mass numbers and parity figure -> %s", out)


if __name__ == "__main__":
    settings = load_config()
    logging.basicConfig(level=settings.methodology.algorithms.log_level, format="%(message)s")
    main(sys.argv[1:], settings)

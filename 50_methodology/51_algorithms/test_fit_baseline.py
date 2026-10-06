"""Facts about the baseline fit behind Section 5.1, on tiny synthetic curves."""

from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from fit_baseline import (
    SECTION,
    Fit,
    figure_tex,
    fit_and_score,
    main,
    numbers,
    parity,
)

from shared.config import PaperConfig
from shared.design import Design
from shared.plots import PlotStyle
from shared.surrogate import Training

# Six curves keyed by p along x in [0, 1]: y = 1 + 0.1 p + x^2; curve p = 2 is the test curve.
P = np.repeat(np.arange(6.0), 10)
X_ALONG = np.tile(np.linspace(0.0, 1.0, 10), 6)
TOY = Design(
    X=np.column_stack([X_ALONG, P]).astype(np.float32),
    y=(1.0 + 0.1 * P + X_ALONG**2).astype(np.float32),
    groups=P.astype(int),
    test=P == 2,
    ablation=np.zeros(len(P), dtype=bool),
)
SHORT = Training(
    width=8,
    depth=1,
    activation="relu",
    loss="mse",
    lr=1e-2,
    max_epochs=3,
    batch_size=16,
    patience=3,
    valid_fraction=0.25,
    seed=0,
)


def clock(*readings: float) -> Callable[[], float]:
    """A clock that returns the given readings in turn."""
    ticks = iter(readings)
    return lambda: next(ticks)


@pytest.fixture(scope="module")
def fit() -> Fit:
    """One short fit of the toy curves, timed by a clock reading 10.0 then 12.5."""
    return fit_and_score(TOY, SHORT, clock(10.0, 12.5))


SCORED = Fit(
    y_true=np.array([1.0, 2.0]),
    y_pred=np.array([1.1, 1.8]),
    mare=0.0123,
    rmse=0.0456,
    seconds=246.0,
    epochs=143,
)


class TestNumbers:
    def test_the_test_mare(self) -> None:
        assert numbers(SCORED)["baseNsMassMare"] == "0.0123"

    def test_names_the_rmse_seconds_and_epochs_too(self) -> None:
        assert sorted(numbers(SCORED)) == [
            "baseNsMassEpochs",
            "baseNsMassFitSeconds",
            "baseNsMassMare",
            "baseNsMassRmse",
        ]


def test_the_parity_figure_draws_the_identity_line() -> None:
    assert len(parity(SCORED, PlotStyle(usetex=False)).axes[0].lines) == 1


def test_the_figure_wrapper_includes_the_parity_image() -> None:
    assert "{51_algorithms_fig_parity}" in figure_tex()


def ns_like(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny NS-shaped table of five (beta, lambda) curves and its split, one curve in test."""
    keys = [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0), (3.0, 1.0)]
    rows = [
        {
            "beta": b,
            "lambda": lam,
            "rho_c": 10.0 ** (i + 1),
            "M": 1.0 + 0.1 * i + 0.05 * b,
            "D": 0.1,
        }
        for b, lam in keys
        for i in range(4)
    ]
    labels = ["fold0", "fold1", "test", "fold0", "fold1"]
    split = [
        {"dataset": "neutron_stars", "beta": b, "lambda": lam, "label": label, "ablation": False}
        for (b, lam), label in zip(keys, labels, strict=True)
    ]
    table, split_file = tmp_path / "ns.parquet", tmp_path / "split.parquet"
    pd.DataFrame(rows).to_parquet(table)
    pd.DataFrame(split).to_parquet(split_file)
    return table, split_file


def test_main_writes_the_numbers_and_the_parity_figure_into_its_section(tmp_path: Path) -> None:
    table, split_file = ns_like(tmp_path)
    assets = tmp_path / "assets"
    (assets / SECTION).mkdir(parents=True)
    (assets / SECTION / "stale.tex").write_text("from an earlier run")
    network = {"width": 8, "depth": 1, "max_epochs": 2, "batch_size": 8}
    config = PaperConfig.model_validate(
        {"plot": {"usetex": False}, "methodology": {"algorithms": network}}
    )
    main([str(table), str(split_file), str(assets)], config)
    assert sorted(p.name for p in (assets / SECTION).iterdir()) == [
        "51_algorithms_fig_parity.png",
        "51_algorithms_fig_parity.tex",
        "51_algorithms_num.tex",
    ]


class TestFitAndScore:
    def test_is_scored_on_the_test_rows_only(self, fit: Fit) -> None:
        assert fit.y_true.tolist() == TOY.y[TOY.test].tolist()

    def test_times_the_fit_with_the_clock(self, fit: Fit) -> None:
        assert fit.seconds == 2.5

    def test_counts_the_epochs_run(self, fit: Fit) -> None:
        assert fit.epochs == SHORT.max_epochs

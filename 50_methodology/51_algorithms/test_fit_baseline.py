"""Facts about the baseline fit behind Section 5.1, on tiny synthetic curves."""

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from fit_baseline import (
    SECTION,
    Fit,
    curve_names,
    end_banner,
    figure_tex,
    fit_and_score,
    main,
    numbers,
    parity,
    progress_line,
    start_banner,
)

from shared.config import PaperConfig
from shared.design import NEUTRON_STARS, Design
from shared.plots import PlotStyle
from shared.runs import from_json
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
    best_epoch=123,
    history=(),
    x_test=np.array([[1.0], [2.0]]),
    groups_test=np.array([0, 0]),
)


class TestNumbers:
    def test_the_test_mare_is_scientific(self) -> None:
        assert numbers(SCORED)["baseNsMassMare"] == "1.23\\times 10^{-2}"

    def test_the_test_rmse_is_scientific(self) -> None:
        assert numbers(SCORED)["baseNsMassRmse"] == "4.56\\times 10^{-2}"

    def test_names_the_rmse_seconds_and_epochs_too(self) -> None:
        assert sorted(numbers(SCORED)) == [
            "baseNsMassEpochs",
            "baseNsMassFitSeconds",
            "baseNsMassMare",
            "baseNsMassRmse",
        ]


BOUNDED = replace(SHORT, max_epochs=500, patience=20)


class TestStartBanner:
    @pytest.fixture
    def lines(self) -> list[str]:
        return start_banner(Path("state/run-1"), BOUNDED, rows=(100, 20), curves=(10, 2))

    def test_names_the_run_folder(self, lines: list[str]) -> None:
        assert "folder: state/run-1" in lines

    def test_states_the_rows_and_curves_it_trains_and_scores_on(self, lines: list[str]) -> None:
        assert "data: 100 training rows on 10 curves, 20 test rows on 2 curves" in lines

    def test_says_how_to_follow_the_run(self, lines: list[str]) -> None:
        assert "follow live: make follow" in lines

    def test_states_what_bounds_the_fit(self, lines: list[str]) -> None:
        assert (
            "bounds: at most 500 epochs; early stop after 20 epochs without a better "
            "validation loss; 25% of the training curves validate"
        ) in lines


FILES = [Path("assets/51_algorithms/51_algorithms_num.tex"), Path("state/run-1/run.json")]


class TestEndBanner:
    def test_says_an_early_stop_and_after_which_epoch(self) -> None:
        assert (
            "stopped: early -- no better validation loss for 20 epochs after epoch 123"
            in end_banner(SCORED, BOUNDED, FILES)
        )

    def test_gives_the_restored_epoch_the_scores_and_the_time(self) -> None:
        assert (
            "best epoch 123 restored; test MARE 1.23e-02, RMSE 4.56e-02; fit 4m 6s"
            in end_banner(SCORED, BOUNDED, FILES)
        )

    def test_lists_every_file_written(self) -> None:
        lines = end_banner(SCORED, BOUNDED, FILES)
        assert [f"  {path}" for path in FILES] == lines[lines.index("wrote:") + 1 :][:2]

    def test_ends_with_how_to_see_the_runs(self) -> None:
        assert end_banner(SCORED, BOUNDED, FILES)[-1] == (
            "list runs: make runs; show this run: make run"
        )

    def test_says_when_max_epochs_ended_the_fit(self) -> None:
        ran_out = replace(SCORED, epochs=500)
        assert "stopped: max_epochs 500 reached" in end_banner(ran_out, BOUNDED, FILES)


class TestProgressLine:
    def test_estimates_the_time_left_from_the_mean_fit(self) -> None:
        assert progress_line(3, 24, 252.0) == "fit 3/24 done, elapsed 4m 12s, about 29m 24s left"

    def test_says_nothing_is_left_after_the_last_fit(self) -> None:
        assert progress_line(1, 1, 12.0) == "fit 1/1 done, elapsed 12s, about 0s left"


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


def test_each_curve_is_named_by_its_key(tmp_path: Path) -> None:
    table, _ = ns_like(tmp_path)
    names = curve_names(pd.read_parquet(table), NEUTRON_STARS)
    assert names[1] == "$\\beta$ = 1, $\\lambda$ = 2"


@pytest.fixture(scope="module")
def ran(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One main run on the NS-like table, a stale asset left from an earlier run, the commit
    and the clock injected; returns its working folder (assets/ and state/ inside)."""
    folder = tmp_path_factory.mktemp("run")
    table, split_file = ns_like(folder)
    assets, state = folder / "assets", folder / "state"
    (assets / SECTION).mkdir(parents=True)
    (assets / SECTION / "stale.tex").write_text("from an earlier run")
    network = {"width": 8, "depth": 1, "max_epochs": 2, "batch_size": 8}
    config = PaperConfig.model_validate(
        {"plot": {"usetex": False}, "methodology": {"algorithms": network}}
    )
    main(
        [str(table), str(split_file), str(assets), str(state)],
        config,
        commit=lambda: "abc1234",
        dirty=lambda: True,
        now=lambda: datetime(2026, 10, 6, 10, 56, 0, tzinfo=UTC),
    )
    return folder


class TestMain:
    def test_writes_the_numbers_and_the_parity_figure_into_its_section(self, ran: Path) -> None:
        assert sorted(p.name for p in (ran / "assets" / SECTION).iterdir()) == [
            "51_algorithms_fig_parity.png",
            "51_algorithms_fig_parity.tex",
            "51_algorithms_num.tex",
        ]

    def test_records_the_injected_commit(self, ran: Path) -> None:
        run = ran / "state" / SECTION / "neutron_stars-mass-20261006T105600Z"
        assert from_json((run / "run.json").read_text()).commit == "abc1234"

    def test_records_uncommitted_code(self, ran: Path) -> None:
        run = ran / "state" / SECTION / "neutron_stars-mass-20261006T105600Z"
        assert from_json((run / "run.json").read_text()).dirty is True

    def test_logs_the_epoch_table_into_the_run_s_train_log(self, ran: Path) -> None:
        run = ran / "state" / SECTION / "neutron_stars-mass-20261006T105600Z"
        assert "valid_mare" in (run / "train.log").read_text()

    def test_opens_its_log_with_the_start_banner(self, ran: Path) -> None:
        run = ran / "state" / SECTION / "neutron_stars-mass-20261006T105600Z"
        assert "follow live: make follow" in (run / "train.log").read_text()

    def test_closes_its_log_with_the_end_banner(self, ran: Path) -> None:
        run = ran / "state" / SECTION / "neutron_stars-mass-20261006T105600Z"
        assert "list runs: make runs" in (run / "train.log").read_text()

    def test_logs_the_progress_after_the_fit(self, ran: Path) -> None:
        run = ran / "state" / SECTION / "neutron_stars-mass-20261006T105600Z"
        assert "fit 1/1 done" in (run / "train.log").read_text()

    def test_points_latest_log_at_the_run_s_train_log(self, ran: Path) -> None:
        run = ran / "state" / SECTION / "neutron_stars-mass-20261006T105600Z"
        latest = ran / "state" / SECTION / "latest.log"
        assert latest.resolve() == (run / "train.log").resolve()

    def test_leaves_the_diagnostics_in_a_folder_named_after_the_run(self, ran: Path) -> None:
        run = ran / "state" / SECTION / "neutron_stars-mass-20261006T105600Z"
        assert sorted(p.name for p in run.iterdir()) == [
            "curves.png",
            "error_cdf.png",
            "loss_curve.png",
            "run.json",
            "train.log",
        ]


class TestFitAndScore:
    def test_is_scored_on_the_test_rows_only(self, fit: Fit) -> None:
        assert fit.y_true.tolist() == TOY.y[TOY.test].tolist()

    def test_times_the_fit_with_the_clock(self, fit: Fit) -> None:
        assert fit.seconds == 2.5

    def test_counts_the_epochs_run(self, fit: Fit) -> None:
        assert fit.epochs == SHORT.max_epochs

    def test_names_the_epoch_of_the_lowest_valid_loss(self, fit: Fit) -> None:
        assert fit.best_epoch == min(fit.history, key=lambda row: row["valid_loss"])["epoch"]

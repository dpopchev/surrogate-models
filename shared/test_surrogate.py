"""Facts about the shared surrogate estimator, on tiny synthetic curves."""

import logging
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from skorch.dataset import Dataset
from skorch.utils import Ansi
from torch import nn

from shared.surrogate import (
    MLP,
    Activation,
    EpochTable,
    Loss,
    Training,
    approx_minutes,
    curve_valid_split,
    make_estimator,
    make_mean_reference,
    make_nearest_reference,
    mare,
    mare_in_d,
    network_fitter,
    rebuild_charge,
    rmse,
    sklearn_fitter,
    stop_window,
)


def test_mare_is_the_mean_absolute_relative_error() -> None:
    assert mare(np.array([1.0, 2.0]), np.array([1.1, 1.8])) == pytest.approx(0.1)


def test_mare_in_d_turns_a_constant_offset_in_y_into_its_relative_error_in_d() -> None:
    y_true = np.array([-1.0, -5.0])
    assert mare_in_d(y_true, y_true + np.log10(1.1)) == pytest.approx(0.1)


def test_rebuild_charge_inverts_the_charge_target_down_to_tiny_charges() -> None:
    charge, mass = np.array([1e-7, 0.2]), np.array([2.0, 4.0])
    rebuilt = rebuild_charge(np.log10(charge / mass), mass)
    assert rebuilt.tolist() == pytest.approx([1e-7, 0.2], rel=1e-12)


@pytest.mark.parametrize(
    ("name", "module"), [("relu", nn.ReLU), ("gelu", nn.GELU), ("tanh", nn.Tanh)]
)
def test_each_activation_builds_its_module(name: Activation, module: type[nn.Module]) -> None:
    assert isinstance(MLP(n_inputs=2, width=4, depth=1, activation=name).layers[1], module)


@pytest.mark.parametrize(("name", "criterion"), [("mse", nn.MSELoss), ("huber", nn.HuberLoss)])
def test_each_loss_selects_its_criterion(name: Loss, criterion: type[nn.Module]) -> None:
    estimator = make_estimator(replace(TOY, loss=name), n_inputs=2)
    assert estimator.named_steps["net"].criterion is criterion


def test_the_validation_split_shares_no_curve_with_training() -> None:
    groups = np.repeat(np.arange(6), 3)
    rows = np.zeros((len(groups), 1), dtype=np.float32)
    train, valid = curve_valid_split(0.34, seed=0)(Dataset(rows, rows), rows, groups=groups)
    assert set(groups[train.indices]) & set(groups[valid.indices]) == set()


def test_rmse_is_the_root_mean_squared_error() -> None:
    assert rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(12.5**0.5)


# Six curves keyed by p along x in [0, 1]: y = 1 + 0.1 p + x^2, smooth and positive.
P = np.repeat(np.arange(6.0), 10)
X_ALONG = np.tile(np.linspace(0.0, 1.0, 10), 6)
INPUTS = np.column_stack([X_ALONG, P]).astype(np.float32)
TARGET = (1.0 + 0.1 * P + X_ALONG**2).astype(np.float32)
GROUPS = P.astype(int)
HELD_OUT = GROUPS == 2

TOY = Training(
    width=32,
    depth=2,
    activation="relu",
    loss="mse",
    lr=1e-2,
    max_epochs=100,
    batch_size=16,
    patience=100,
    valid_fraction=0.25,
    seed=0,
)


def fitted(training: Training) -> Pipeline:
    """The estimator fitted on every curve but the held-out one."""
    keep = ~HELD_OUT
    estimator = make_estimator(training, n_inputs=2)
    return estimator.fit(INPUTS[keep], TARGET[keep], net__groups=GROUPS[keep])


class TestNetworkFitter:
    """The harness hands the network its fit rows and its validation rows (W-065)."""

    def test_validates_on_exactly_the_rows_it_is_handed(self) -> None:
        fit, valid = GROUPS < 4, (GROUPS == 4) | (GROUPS == 5)
        captured: list[Pipeline] = []
        fitter = network_fitter(replace(TOY, max_epochs=5), on_fit=captured.append)
        predict = fitter(INPUTS[fit], TARGET[fit], INPUTS[valid], TARGET[valid], 0)
        net = captured[0].named_steps["net"]
        best = int(np.argmin(net.history[:, "valid_loss"]))
        scored = mare(TARGET[valid], predict(INPUTS[valid]))
        assert scored == pytest.approx(net.history[best, "valid_mare"], rel=1e-4)


def test_a_scikit_learn_fitter_fits_the_fit_rows_alone() -> None:
    fit, valid = GROUPS < 4, GROUPS >= 4
    predict = sklearn_fitter(make_mean_reference)(
        INPUTS[fit], TARGET[fit], INPUTS[valid], TARGET[valid], 0
    )
    assert predict(INPUTS[:1]).tolist() == pytest.approx([float(TARGET[fit].mean())])


def test_valid_mare_is_scored_on_the_original_scale() -> None:
    keep = ~HELD_OUT
    estimator = make_estimator(replace(TOY, max_epochs=3), n_inputs=2)
    estimator.fit(INPUTS[keep], TARGET[keep] + 1000.0, net__groups=GROUPS[keep])
    assert estimator.named_steps["net"].history[-1, "valid_mare"] < 0.01


def test_the_learning_rate_anneals_and_is_recorded() -> None:
    net = fitted(replace(TOY, max_epochs=3)).named_steps["net"]
    assert net.history[-1, "event_lr"] < TOY.lr


def test_training_stops_once_the_validation_loss_stalls() -> None:
    stalled = replace(TOY, lr=0.0, patience=2, max_epochs=50)
    assert len(fitted(stalled).named_steps["net"].history) < stalled.max_epochs


def test_an_early_stop_keeps_the_best_epoch_s_weights() -> None:
    keep = ~HELD_OUT
    estimator = fitted(replace(TOY, lr=5e-2, patience=2))
    net = estimator.named_steps["net"]
    inputs = estimator.named_steps["scale"].transform(INPUTS[keep])
    _, valid = net.get_split_datasets(inputs, TARGET[keep], groups=GROUPS[keep])
    best = int(np.argmin(net.history[:, "valid_loss"]))
    rows = np.asarray(valid.indices)
    scored = mare(TARGET[keep][rows], net.predict(inputs[rows]))
    assert scored == pytest.approx(net.history[best, "valid_mare"], rel=1e-4)


def test_the_epoch_table_reaches_the_logger(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="shared.surrogate"):
        fitted(replace(TOY, max_epochs=2))
    assert "valid_mare" in caplog.text


class TestReferences:
    def test_the_mean_reference_predicts_the_training_mean(self) -> None:
        mean = make_mean_reference().fit(INPUTS[~HELD_OUT], TARGET[~HELD_OUT])
        assert mean.predict(INPUTS[HELD_OUT][:1])[0] == pytest.approx(TARGET[~HELD_OUT].mean())

    def test_the_nearest_reference_reproduces_a_training_row(self) -> None:
        nearest = make_nearest_reference().fit(INPUTS[~HELD_OUT], TARGET[~HELD_OUT])
        assert nearest.predict(INPUTS[:1])[0] == pytest.approx(TARGET[0])

    def test_the_nearest_reference_measures_distance_on_standardized_inputs(self) -> None:
        inputs = np.array([[0.0, 0.0], [1.0, 100.0]])
        nearest = make_nearest_reference().fit(inputs, np.array([10.0, 20.0]))
        assert nearest.predict(np.array([[0.9, 40.0]]))[0] == 20.0


class TestStopWindow:
    def test_spans_patience_to_max_epochs_after_a_new_best(self) -> None:
        assert stop_window(epoch=17, best_epoch=17, patience=20, max_epochs=500) == (20, 483)

    def test_never_lets_the_earliest_stop_pass_max_epochs(self) -> None:
        assert stop_window(epoch=495, best_epoch=490, patience=20, max_epochs=500) == (5, 5)


class TestApproxMinutes:
    def test_rounds_to_the_nearest_minute(self) -> None:
        assert approx_minutes(epochs=20, epoch_seconds=8.0) == "~3m"

    def test_states_hours_beyond_an_hour(self) -> None:
        assert approx_minutes(epochs=483, epoch_seconds=8.0) == "~1h 4m"

    def test_reads_less_than_a_minute_under_half_a_minute(self) -> None:
        assert approx_minutes(epochs=3, epoch_seconds=8.0) == "~<1m"


def epoch_table(caplog: pytest.LogCaptureFixture) -> str:
    """The epoch table a two-epoch toy fit logs."""
    with caplog.at_level(logging.INFO, logger="shared.surrogate"):
        fitted(replace(TOY, max_epochs=2))
    return caplog.text


class TestEpochTable:
    def test_names_the_epoch_time_elapsed_s(self, caplog: pytest.LogCaptureFixture) -> None:
        assert "elapsed_s" in epoch_table(caplog)

    def test_shows_the_epoch_out_of_max_epochs(self, caplog: pytest.LogCaptureFixture) -> None:
        assert re.search(r"\b2/2\b", epoch_table(caplog)) is not None

    def test_counts_the_epochs_since_the_best_against_patience(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        assert re.search(r"\b0/100\b", epoch_table(caplog)) is not None

    def test_shows_the_time_to_the_stop_if_no_gain_comes(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        assert re.search(r"\bstop_if_no_gain\b", epoch_table(caplog)) is not None

    def test_shows_the_time_to_the_stop_at_max_epochs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        assert re.search(r"\bstop_at_max\b", epoch_table(caplog)) is not None

    def test_records_the_epoch_time_rounded_to_a_tenth(self) -> None:
        net = fitted(replace(TOY, max_epochs=1)).named_steps["net"]
        assert net.history[-1, "elapsed_s"] == round(net.history[-1, "dur"], 1)

    def test_pads_no_value_with_zeros(self, caplog: pytest.LogCaptureFixture) -> None:
        assert re.search(r"\b\d+\.\d*00\b", epoch_table(caplog)) is None

    def test_no_longer_prints_dur(self, caplog: pytest.LogCaptureFixture) -> None:
        assert re.search(r"\bdur\b", epoch_table(caplog)) is None


def test_the_loss_curve_is_drawn_live_while_training(tmp_path: Path) -> None:
    live = tmp_path / "loss_curve.png"
    keep = ~HELD_OUT
    estimator = make_estimator(replace(TOY, max_epochs=5), n_inputs=2, live_plot=live)
    estimator.fit(INPUTS[keep], TARGET[keep], net__groups=GROUPS[keep])
    assert live.stat().st_size > 0


def test_a_non_finite_training_loss_raises() -> None:
    poisoned = TARGET.copy()
    poisoned[0] = np.nan
    estimator = make_estimator(replace(TOY, max_epochs=2), n_inputs=2)
    with pytest.raises(FloatingPointError):
        estimator.fit(INPUTS, poisoned, net__groups=GROUPS)


def test_the_same_seed_gives_the_same_predictions() -> None:
    short = replace(TOY, max_epochs=3)
    first, second = (fitted(short).predict(INPUTS[HELD_OUT]) for _ in range(2))
    assert np.array_equal(first, second)


def test_a_held_out_curve_is_predicted_better_than_by_the_mean() -> None:
    predicted = fitted(TOY).predict(INPUTS[HELD_OUT])
    baseline = np.full(HELD_OUT.sum(), TARGET[~HELD_OUT].mean())
    assert mare(TARGET[HELD_OUT], predicted) < mare(TARGET[HELD_OUT], baseline)


class TestEpochTableCells:
    def test_epoch_table_writes_a_loss_in_e_notation_to_3_figures(self) -> None:
        assert EpochTable().format_row({"train_loss": 0.0007835}, "train_loss", "") == "7.84e-04"

    def test_epoch_table_writes_the_lr_to_4_figures_so_its_decay_shows(self) -> None:
        assert EpochTable().format_row({"lr": 0.0009999}, "lr", "") == "9.999e-04"

    def test_epoch_table_keeps_a_tenth_on_whole_seconds(self) -> None:
        assert EpochTable().format_row({"elapsed_s": 2.0}, "elapsed_s", "") == "2.0"

    def test_epoch_table_keeps_the_colour_of_a_best_cell(self) -> None:
        row = {"valid_loss": 0.0123, "valid_loss_best": True}
        assert EpochTable().format_row(row, "valid_loss", "<c>") == f"<c>1.23e-02{Ansi.ENDC.value}"

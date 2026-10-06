"""Facts about the shared surrogate estimator, on tiny synthetic curves."""

import logging
import re
from dataclasses import replace

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from skorch.dataset import Dataset
from torch import nn

from shared.surrogate import (
    MLP,
    Activation,
    Loss,
    Training,
    curve_valid_split,
    make_estimator,
    mare,
    rmse,
    stop_window,
    time_left,
)


def test_mare_is_the_mean_absolute_relative_error() -> None:
    assert mare(np.array([1.0, 2.0]), np.array([1.1, 1.8])) == pytest.approx(0.1)


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


class TestStopWindow:
    def test_spans_patience_to_max_epochs_after_a_new_best(self) -> None:
        assert stop_window(epoch=17, best_epoch=17, patience=20, max_epochs=500) == (20, 483)

    def test_never_lets_the_earliest_stop_pass_max_epochs(self) -> None:
        assert stop_window(epoch=495, best_epoch=490, patience=20, max_epochs=500) == (5, 5)


def test_the_time_left_is_the_window_at_the_mean_epoch_time() -> None:
    assert time_left((20, 483), mean_epoch_seconds=8.0) == "~3m-1h 4m"


def test_the_time_left_under_half_a_minute_reads_less_than_a_minute() -> None:
    assert time_left((0, 3), mean_epoch_seconds=8.0) == "~<1m-<1m"


def epoch_table(caplog: pytest.LogCaptureFixture) -> str:
    """The epoch table a two-epoch toy fit logs."""
    with caplog.at_level(logging.INFO, logger="shared.surrogate"):
        fitted(replace(TOY, max_epochs=2))
    return caplog.text


class TestEpochTable:
    def test_names_the_epoch_time_elapse_s(self, caplog: pytest.LogCaptureFixture) -> None:
        assert "elapse_s" in epoch_table(caplog)

    def test_shows_the_epoch_out_of_max_epochs(self, caplog: pytest.LogCaptureFixture) -> None:
        assert re.search(r"\b2/2\b", epoch_table(caplog)) is not None

    def test_counts_the_epochs_since_the_best_against_patience(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        assert re.search(r"\b0/100\b", epoch_table(caplog)) is not None

    def test_shows_the_time_left_until_the_earliest_and_latest_stop(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        assert re.search(r"\btime_left\b", epoch_table(caplog)) is not None

    def test_shows_the_epoch_time_to_a_tenth_of_a_second(self) -> None:
        net = fitted(replace(TOY, max_epochs=1)).named_steps["net"]
        assert re.fullmatch(r"\d+\.\d", str(net.history[-1, "elapse_s"])) is not None

    def test_no_longer_prints_dur(self, caplog: pytest.LogCaptureFixture) -> None:
        assert re.search(r"\bdur\b", epoch_table(caplog)) is None


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

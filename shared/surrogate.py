"""The surrogate estimator every model family shares, its metrics and its training feedback (W-034).

make_estimator builds Pipeline(StandardScaler, ScaledNetRegressor): a skorch MLP that
standardizes its target itself, so its predictions and its per-epoch valid_mare are on the
target's original scale (Section 3.3: M in M_sun, the dimensionless charge target). Early
stopping validates on whole held-out curves -- the fit passes the rows' curve ids as
`net__groups` -- never on skorch's default random rows (W-017).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, assert_never, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback, EarlyStopping, EpochScoring, LRScheduler
from skorch.dataset import ValidSplit
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from shared.diagnostics import loss_curve

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

# The live loss curve is redrawn every LIVE_EVERY epochs, at a screen resolution.
LIVE_EVERY = 5
LIVE_DPI = 100

Activation = Literal["relu", "gelu", "tanh"]
Loss = Literal["mse", "huber"]


@dataclass(frozen=True)
class Training:
    """How one network is built and trained."""

    width: int
    depth: int
    activation: Activation
    loss: Loss
    lr: float
    max_epochs: int
    batch_size: int
    patience: int
    valid_fraction: float
    seed: int


# --- metrics ----------------------------------------------------------------------------------


def mare(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The mean absolute relative error."""
    return float(np.mean(np.abs((y_pred - y_true) / y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The root mean squared error."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# --- progress ---------------------------------------------------------------------------------


def stop_window(epoch: int, best_epoch: int, patience: int, max_epochs: int) -> tuple[int, int]:
    """Epochs left until the earliest stop (patience runs out without a new best) and the
    latest (max_epochs)."""
    latest = max_epochs - epoch
    return min(patience - (epoch - best_epoch), latest), latest


def duration_text(seconds: float) -> str:
    """Seconds as "1h 4m 24s", "2m 40s" or "12s"."""
    hours, rest = divmod(round(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def _minutes_text(seconds: float) -> str:
    """Seconds to the nearest minute: "1h 4m", "3m", or "<1m" under half a minute."""
    hours, minutes = divmod(round(seconds / 60), 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m" if minutes else "<1m"


def approx_minutes(epochs: int, epoch_seconds: float) -> str:
    """The approximate time a number of epochs takes, to the nearest minute: "~3m"."""
    return f"~{_minutes_text(epochs * epoch_seconds)}"


# --- the network ------------------------------------------------------------------------------


class MLP(nn.Module):
    """A plain multilayer perceptron with one output."""

    def __init__(self, n_inputs: int, width: int, depth: int, activation: Activation) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for size in [n_inputs] + [width] * (depth - 1):
            layers += [nn.Linear(size, width), _activation(activation)]
        self.layers = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, x: torch.Tensor, **_fit_params: Any) -> torch.Tensor:
        """The output for a batch; skorch also hands the fit params (the curve groups) here."""
        return self.layers(x)


def _activation(name: Activation) -> nn.Module:
    """The torch module of an activation name."""
    match name:
        case "relu":
            return nn.ReLU()
        case "gelu":
            return nn.GELU()
        case "tanh":
            return nn.Tanh()
        case _:
            assert_never(name)


def _criterion(name: Loss) -> type[nn.Module]:
    """The torch loss of a loss name; Huber keeps torch's delta 1.0 on the standardized target."""
    match name:
        case "mse":
            return nn.MSELoss
        case "huber":
            return nn.HuberLoss
        case _:
            assert_never(name)


class ScaledNetRegressor(NeuralNetRegressor):
    """A skorch regressor that trains on its standardized target and predicts on its scale.

    `seed` seeds torch before each fit, so the initial weights and the batch order repeat.
    """

    def __init__(self, *args: Any, seed: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> ScaledNetRegressor:  # noqa: N803
        """Standardize the target, seed torch, then fit the net on the target."""
        torch.manual_seed(self.seed)
        column = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.target_scaler_ = StandardScaler().fit(column)
        scaled = np.asarray(self.target_scaler_.transform(column), dtype=np.float32)
        return super().fit(X, scaled, **fit_params)

    def predict(self, X: Any) -> np.ndarray:  # noqa: N803
        """Predictions on the target's original scale."""
        return self.target_scaler_.inverse_transform(super().predict(X)).ravel()


class FiniteLoss(Callback):
    """Stop training with FloatingPointError when an epoch's train or valid loss is not finite."""

    def on_epoch_end(
        self,
        net: NeuralNetRegressor,
        dataset_train: Any = None,
        dataset_valid: Any = None,
        **kwargs: Any,
    ) -> None:
        """Check the losses the epoch just recorded."""
        epoch = cast(dict[str, Any], net.history[-1])
        for key in ("train_loss", "valid_loss"):
            if key in epoch and not np.isfinite(epoch[key]):
                raise FloatingPointError(f"epoch {epoch['epoch']}: {key} is {epoch[key]}")


class ElapsedSeconds(Callback):
    """Record the seconds an epoch took, to a tenth, as elapsed_s, the epoch table's time
    column."""

    def on_epoch_end(
        self,
        net: NeuralNetRegressor,
        dataset_train: Any = None,
        dataset_valid: Any = None,
        **kwargs: Any,
    ) -> None:
        """Copy skorch's dur (recorded by its EpochTimer, which runs first) into elapsed_s."""
        net.history.record("elapsed_s", round(cast(float, net.history[-1, "dur"]), 1))


class Progress(Callback):
    """Record where a fit stands: at_epoch (k/MAX); patience, EarlyStopping's own count of
    epochs without a 0.01% better validation loss out of those allowed ("3/20"); and the
    approximate time to the stop if no gain comes (stop_if_no_gain) and to max_epochs
    (stop_at_max), at the median of the last ROLLING epoch times.

    Runs after EarlyStopping, whose counter it reads. PrintLog drops every key ending in
    _best, hence patience, not since_best.
    """

    ROLLING = 5

    def __init__(self, patience: int) -> None:
        self.patience = patience

    def on_epoch_end(
        self,
        net: NeuralNetRegressor,
        dataset_train: Any = None,
        dataset_valid: Any = None,
        **kwargs: Any,
    ) -> None:
        """Record this epoch's progress columns."""
        rows = cast(list[dict[str, Any]], net.history)
        epoch = int(rows[-1]["epoch"])
        misses = int(dict(net.callbacks_)["early_stopping"].misses_)
        net.history.record("at_epoch", f"{epoch}/{net.max_epochs}")
        net.history.record("patience", f"{misses}/{self.patience}")
        if_no_gain, at_max = stop_window(epoch, epoch - misses, self.patience, net.max_epochs)
        epoch_seconds = float(np.median([row["dur"] for row in rows[-self.ROLLING :]]))
        net.history.record("stop_if_no_gain", approx_minutes(if_no_gain, epoch_seconds))
        net.history.record("stop_at_max", approx_minutes(at_max, epoch_seconds))


class LiveLossCurve(Callback):
    """Rewrite the loss curve at path every `every` epochs while the fit runs.

    The image is written beside path and then moved over it, so a viewer never reads half a
    file.
    """

    def __init__(self, path: Path, every: int) -> None:
        self.path = path
        self.every = every

    def on_epoch_end(
        self,
        net: NeuralNetRegressor,
        dataset_train: Any = None,
        dataset_valid: Any = None,
        **kwargs: Any,
    ) -> None:
        """Redraw the curve on every `every`-th epoch."""
        rows = cast(list[dict[str, Any]], net.history)
        if int(rows[-1]["epoch"]) % self.every:
            return
        figure = loss_curve([{k: v for k, v in row.items() if k != "batches"} for row in rows])
        partial = self.path.with_name(f".{self.path.name}")
        figure.savefig(partial, format="png", dpi=LIVE_DPI)
        plt.close(figure)
        partial.replace(self.path)


def _valid_mare(net: ScaledNetRegressor, X: Any, y: Any) -> float:  # noqa: N803
    """MARE on the validation curves, the standardized target turned back to its scale."""
    column = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return mare(net.target_scaler_.inverse_transform(column).ravel(), net.predict(X))


def curve_valid_split(fraction: float, seed: int) -> ValidSplit:
    """A validation split of whole curves, from the groups passed to fit."""
    # skorch types `cv` from its default (5); it accepts any sklearn splitter.
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    return ValidSplit(cast(Any, splitter))


def make_mean_reference() -> DummyRegressor:
    """The mean predictor: every row gets the mean target of the training rows."""
    return DummyRegressor(strategy="mean")


def make_nearest_reference() -> Pipeline:
    """The nearest-curve predictor: the target of the closest training row in inputs
    standardized on the training rows. Test curves are never in training, so the closest row
    always lies on another curve."""
    return Pipeline([("scale", StandardScaler()), ("nearest", KNeighborsRegressor(n_neighbors=1))])


def make_estimator(training: Training, n_inputs: int, live_plot: Path | None = None) -> Pipeline:
    """Pipeline(StandardScaler, ScaledNetRegressor) with the training callbacks; with live_plot,
    the loss curve is also redrawn there every LIVE_EVERY epochs while it trains."""
    live = [("live_loss_curve", LiveLossCurve(live_plot, LIVE_EVERY))] if live_plot else []
    net = ScaledNetRegressor(
        module=MLP,
        module__n_inputs=n_inputs,
        module__width=training.width,
        module__depth=training.depth,
        module__activation=training.activation,
        criterion=_criterion(training.loss),
        optimizer=torch.optim.AdamW,
        lr=training.lr,
        max_epochs=training.max_epochs,
        batch_size=training.batch_size,
        train_split=curve_valid_split(training.valid_fraction, training.seed),
        callbacks=[
            (
                "valid_mare",
                EpochScoring(_valid_mare, lower_is_better=True, on_train=False, name="valid_mare"),
            ),
            # skorch types `policy` from its default (a name); it accepts a scheduler class.
            ("lr", LRScheduler(cast(Any, CosineAnnealingLR), T_max=training.max_epochs)),
            ("early_stopping", EarlyStopping(patience=training.patience, load_best=True)),
            ("finite_loss", FiniteLoss()),
            ("elapsed_s", ElapsedSeconds()),
            ("progress", Progress(training.patience)),
            *live,
        ],
        # at_epoch (k/MAX) replaces the bare epoch and sorts first; elapsed_s replaces dur.
        callbacks__print_log__keys_ignored=["dur", "epoch"],
        # 4 significant figures, not 4 decimals: 15.1 not 15.1000, and the lr's small steps show.
        callbacks__print_log__floatfmt=".4g",
        callbacks__print_log__sink=logger.info,
        seed=training.seed,
    )
    return Pipeline([("scale", StandardScaler()), ("net", net)])

"""The surrogate estimator every model family shares, its metrics and its training feedback (W-034).

make_estimator builds Pipeline(StandardScaler, ScaledNetRegressor): a skorch MLP that
standardizes its target itself, so its predictions and its per-epoch valid_mare are on the
target's original scale (Section 3.3: M in M_sun, the dimensionless charge target). Early
stopping validates on whole held-out curves -- the fit passes the rows' curve ids as
`net__groups` -- never on skorch's default random rows (W-017).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, assert_never, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupShuffleSplit, PredefinedSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback, EarlyStopping, EpochScoring, LRScheduler, PrintLog
from skorch.dataset import ValidSplit
from skorch.utils import Ansi
from tabulate import tabulate
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


def mare_in_d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The MARE of the charge D rebuilt with the true M from charge targets Y = log10(D/M).

    The mass cancels: D_pred / D_true = 10^(Y_pred - Y_true), so no M is needed.
    """
    return float(np.mean(np.abs(10.0 ** (y_pred - y_true) - 1.0)))


def rebuild_charge(y_charge: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """The charge D from the charge target Y = log10(D/M) and a mass M."""
    return mass * 10.0**y_charge


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
    """A plain multilayer perceptron, one output unit per target column."""

    def __init__(
        self, n_inputs: int, width: int, depth: int, activation: Activation, n_outputs: int = 1
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for size in [n_inputs] + [width] * (depth - 1):
            layers += [nn.Linear(size, width), _activation(activation)]
        self.layers = nn.Sequential(*layers, nn.Linear(width, n_outputs))

    def forward(self, x: torch.Tensor, **_fit_params: Any) -> torch.Tensor:
        """The output for a batch; skorch also hands the fit params (the curve groups) here."""
        return self.layers(x)


class ResMLP(nn.Module):
    """A multilayer perceptron whose hidden layers are residual blocks, h + act(W h + b): each
    block adds to an identity skip, so a deep network starts near a shallow one (H3)."""

    def __init__(
        self, n_inputs: int, width: int, depth: int, activation: Activation, n_outputs: int = 1
    ) -> None:
        super().__init__()
        self.first = nn.Linear(n_inputs, width)
        self.blocks = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 1))
        self.act = _activation(activation)
        self.head = nn.Linear(width, n_outputs)

    def forward(self, x: torch.Tensor, **_fit_params: Any) -> torch.Tensor:
        """The output for a batch; skorch also hands the fit params (the curve groups) here."""
        h = self.act(self.first(x))
        for block in self.blocks:
            h = h + self.act(block(h))
        return self.head(h)


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
        """Standardize each target column, seed torch, then fit the net on the target."""
        torch.manual_seed(self.seed)
        target = np.asarray(y, dtype=np.float32)
        columns = target.reshape(len(target), -1)
        self.target_scaler_ = StandardScaler().fit(columns)
        scaled = np.asarray(self.target_scaler_.transform(columns), dtype=np.float32)
        return super().fit(X, scaled, **fit_params)

    def predict(self, X: Any) -> np.ndarray:  # noqa: N803
        """Predictions on the target's original scale: one column per target column, a flat
        array for a one-column target."""
        found = self.target_scaler_.inverse_transform(super().predict(X))
        return found.ravel() if found.shape[1] == 1 else found


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


class EpochTable(PrintLog):
    """skorch's epoch table with a format per column (W-078): losses and errors in e-notation
    to 3 significant figures, the lr to 4 so the cosine decay shows, the seconds to a tenth;
    every other column as skorch formats it."""

    def format_row(self, row: dict[str, Any], key: str, color: str) -> str:
        """The cell of key in row, wrapped in color when it is the column's best so far."""
        spec = EPOCH_FORMATS.get(key)
        if spec is None:
            return cast(str, super().format_row(row, key, color))
        cell = f"{row[key]:{spec}}"
        return f"{color}{cell}{Ansi.ENDC.value}" if row.get(f"{key}_best") else cell

    def table(self, row: dict[str, Any]) -> str:
        """The one-row table skorch prints, each cell as format_row wrote it: tabulate would
        re-parse a number-like cell and format it again (7.84e-04 as 0.000784)."""
        headers, cells = zip(*self._yield_keys_formatted(row), strict=True)
        return tabulate(
            [cells],
            headers=headers,
            tablefmt=self.tablefmt,
            stralign=self.stralign,
            disable_numparse=True,
        )


# The epoch table's cell format per history key (the lr is recorded as event_lr and shown as
# lr); a column not named keeps skorch's own format.
EPOCH_FORMATS = {"train_loss": ".2e", "valid_loss": ".2e", "valid_mare": ".2e", "event_lr": ".3e"}


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
    target = np.asarray(y, dtype=np.float32)
    truth = net.target_scaler_.inverse_transform(target.reshape(len(target), -1))
    return mare(truth.ravel() if truth.shape[1] == 1 else truth, net.predict(X))


def curve_valid_split(fraction: float, seed: int) -> ValidSplit:
    """A validation split of whole curves, from the groups passed to fit."""
    # skorch types `cv` from its default (5); it accepts any sklearn splitter.
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    return ValidSplit(cast(Any, splitter))


def network_fitter(
    training: Training,
    live_plot: Path | None = None,
    on_fit: Callable[[Pipeline], None] = lambda _: None,
    before_fit: Callable[[Pipeline], None] = lambda _: None,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], Callable[..., np.ndarray]]:
    """A harness fitter of the network: it early-stops on exactly the validation rows it is
    handed, seeded by the harness's seed; before_fit receives each built pipeline before it
    trains (to add a callback), on_fit each fitted one."""

    def fit(
        x_fit: np.ndarray, y_fit: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, seed: int
    ) -> Callable[..., np.ndarray]:
        fold = np.concatenate([np.full(len(x_fit), -1), np.zeros(len(x_valid), dtype=int)])
        estimator = make_estimator(
            replace(training, seed=seed), x_fit.shape[1], live_plot, valid_fold=fold
        )
        x = np.vstack([x_fit, x_valid]).astype(np.float32)
        before_fit(estimator)
        estimator.fit(x, np.concatenate([y_fit, y_valid]).astype(np.float32))
        on_fit(estimator)
        return lambda rows: estimator.predict(np.asarray(rows, dtype=np.float32))

    return fit


def sklearn_fitter(
    make: Callable[[], Any],
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], Callable[..., np.ndarray]]:
    """A harness fitter of a scikit-learn estimator, fitted on the fit rows alone."""

    def fit(
        x_fit: np.ndarray, y_fit: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, seed: int
    ) -> Callable[..., np.ndarray]:
        return make().fit(x_fit, y_fit).predict

    return fit


def make_mean_reference() -> DummyRegressor:
    """The mean predictor: every row gets the mean target of the training rows."""
    return DummyRegressor(strategy="mean")


def make_nearest_reference() -> Pipeline:
    """The nearest-curve predictor: the target of the closest training row in inputs
    standardized on the training rows. Test curves are never in training, so the closest row
    always lies on another curve."""
    return Pipeline([("scale", StandardScaler()), ("nearest", KNeighborsRegressor(n_neighbors=1))])


def make_estimator(
    training: Training,
    n_inputs: int,
    live_plot: Path | None = None,
    valid_fold: np.ndarray | None = None,
    n_outputs: int = 1,
    skips: bool = False,
) -> Pipeline:
    """Pipeline(StandardScaler, ScaledNetRegressor) with the training callbacks, n_outputs
    target columns (the curve-wise spline coefficients, W-067), the residual network when
    skips is set (W-067); with live_plot,
    the loss curve is also redrawn there every LIVE_EVERY epochs while it trains. With
    valid_fold (-1 for a fit row, 0 for a validation row) the net validates on exactly those
    rows; without it, on a share of whole curves from the groups passed to fit."""
    split = (
        ValidSplit(cast(Any, PredefinedSplit(valid_fold)))
        if valid_fold is not None
        else curve_valid_split(training.valid_fraction, training.seed)
    )
    live = [("live_loss_curve", LiveLossCurve(live_plot, LIVE_EVERY))] if live_plot else []
    net = ScaledNetRegressor(
        module=ResMLP if skips else MLP,
        module__n_inputs=n_inputs,
        module__width=training.width,
        module__depth=training.depth,
        module__activation=training.activation,
        module__n_outputs=n_outputs,
        criterion=_criterion(training.loss),
        optimizer=torch.optim.AdamW,
        lr=training.lr,
        max_epochs=training.max_epochs,
        batch_size=training.batch_size,
        train_split=split,
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
        # at_epoch (k/MAX) replaces the bare epoch and sorts first; elapsed_s replaces dur. The
        # losses, errors and lr are formatted per column (EpochTable, W-078); any other float keeps
        # 4 significant figures, not 4 decimals: 15.1 not 15.1000.
        callbacks__print_log=EpochTable(
            keys_ignored=["dur", "epoch"], floatfmt=".4g", sink=logger.info
        ),
        seed=training.seed,
    )
    return Pipeline([("scale", StandardScaler()), ("net", net)])

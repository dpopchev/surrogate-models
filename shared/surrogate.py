"""The surrogate estimator every model family shares, its metrics and its training feedback (W-034).

make_estimator builds Pipeline(StandardScaler, ScaledNetRegressor): a skorch MLP that
standardizes its target itself, so its predictions and its per-epoch valid_mare are on the
target's original scale (Section 4.3: M in M_sun, the dimensionless charge target). Early
stopping validates on whole held-out curves -- the fit passes the rows' curve ids as
`net__groups` -- never on skorch's default random rows (W-017).
"""

import logging
from dataclasses import dataclass
from typing import Any, Literal, assert_never, cast

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback, EarlyStopping, EpochScoring, LRScheduler
from skorch.dataset import ValidSplit
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

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


def _valid_mare(net: ScaledNetRegressor, X: Any, y: Any) -> float:  # noqa: N803
    """MARE on the validation curves, the standardized target turned back to its scale."""
    column = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return mare(net.target_scaler_.inverse_transform(column).ravel(), net.predict(X))


def curve_valid_split(fraction: float, seed: int) -> ValidSplit:
    """A validation split of whole curves, from the groups passed to fit."""
    # skorch types `cv` from its default (5); it accepts any sklearn splitter.
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    return ValidSplit(cast(Any, splitter))


def make_estimator(training: Training, n_inputs: int) -> Pipeline:
    """Pipeline(StandardScaler, ScaledNetRegressor) with the training callbacks."""
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
            ("early_stopping", EarlyStopping(patience=training.patience)),
            ("finite_loss", FiniteLoss()),
        ],
        callbacks__print_log__sink=logger.info,
        seed=training.seed,
    )
    return Pipeline([("scale", StandardScaler()), ("net", net)])

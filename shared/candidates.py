"""The candidates a screen or a tuning scores: a model family in one unit of prediction, and
its harness fitter (W-067, moved here from the family screen for W-068's tuning).
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Literal, assert_never

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Kernel,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.pipeline import Pipeline

from shared.curvewise import curvewise_fitter
from shared.eda import CurveSpace
from shared.families import make_gpr, make_knn, make_rbf, make_xgboost
from shared.harness import Fitter
from shared.search import Unit
from shared.surrogate import ResMLP, Training, make_estimator, network_fitter, sklearn_fitter

# --- vocabulary and types ---------------------------------------------------------------------

Family = Literal["k-NN", "local RBF", "GPR", "XGBoost", "MLP", "ResNet"]
FAMILIES: tuple[Family, ...] = ("k-NN", "local RBF", "GPR", "XGBoost", "MLP", "ResNet")
UNITS: tuple[Unit, ...] = ("pointwise", "curve-wise")

# The GPR's kernel shapes a trial chooses from (shared/search.py SPACES), each scaled and
# joined by white noise as in shared/families.py make_gpr.
GPR_KERNELS: Mapping[str, Callable[[], Kernel]] = {
    "rbf": RBF,
    "matern52": partial(Matern, nu=2.5),
    "rational_quadratic": RationalQuadratic,
}


@dataclass(frozen=True)
class Candidate:
    """One family in one unit of prediction."""

    family: Family
    unit: Unit


@dataclass(frozen=True)
class Knobs:
    """The families' settings: the k-NN's k, the local RBF's neighbours, the curve-wise
    spline's knots (ends included), and the fewest rows a thinned curve keeps."""

    k: int
    neighbours: int
    knots: int
    per_curve: int


# --- pure functions ---------------------------------------------------------------------------


def tuned_knobs(knobs: Knobs, params: Mapping[str, Any]) -> Knobs:
    """The knobs with a trial's k, neighbours and knots laid over them."""
    return replace(knobs, **_picked(params, ("k", "neighbours", "knots")))


def tuned_training(training: Training, params: Mapping[str, Any]) -> Training:
    """The training with a trial's width, depth, activation, loss and lr laid over it."""
    return replace(training, **_picked(params, ("width", "depth", "activation", "loss", "lr")))


def _picked(params: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    """The params among the names."""
    return {name: params[name] for name in names if name in params}


def tuned_fitter(
    candidate: Candidate,
    space: CurveSpace,
    knobs: Knobs,
    training: Training,
    params: Mapping[str, Any],
) -> Fitter:
    """The candidate's harness fitter with one trial's params laid over the knobs, the
    training and the estimator."""
    laid = (tuned_knobs(knobs, params), tuned_training(training, params))
    return fitter_of(candidate, space, *laid, params=params)


def fitter_of(
    candidate: Candidate,
    space: CurveSpace,
    knobs: Knobs,
    training: Training,
    params: Mapping[str, Any] | None = None,
) -> Fitter:
    """The harness fitter of the candidate on a design of the curve space: the family fitted
    row by row, or as the base that predicts each curve's spline from its curve keys; params
    are a trial's settings of the estimator itself (estimator_of)."""
    family = candidate.family
    match candidate.unit:
        case "pointwise":
            if family == "MLP":
                return network_fitter(training)
            if family == "ResNet":
                return network_fitter(training, before_fit=_with_skips)
            return sklearn_fitter(estimator_of(family, knobs, params))
        case "curve-wise":
            keys, along = curve_columns(space)
            if family in ("MLP", "ResNet"):
                outputs = knobs.knots + 4  # the spline's knots + 2 coefficients, start, end
                base = partial(CurveNet, training, len(keys), outputs, family == "ResNet")
                return curvewise_fitter(base, keys, along, knobs.knots)
            base = estimator_of(family, knobs, params)
            return curvewise_fitter(base, keys, along, knobs.knots)
        case _:
            assert_never(candidate.unit)


def estimator_of(
    family: Family, knobs: Knobs, params: Mapping[str, Any] | None = None
) -> Callable[[], Any]:
    """The scikit-learn estimator of a family that is not a network, with a trial's params of
    its own (the k-NN's weights, the GPR's kernel, XGBoost's trees) set on it."""
    own = params or {}
    match family:
        case "k-NN":
            weights = _picked(own, ("weights",))
            return lambda: make_knn(knobs.k).set_params(
                **{f"knn__{n}": v for n, v in weights.items()}
            )
        case "local RBF":
            return partial(make_rbf, knobs.neighbours)
        case "GPR":
            if "kernel" not in own:
                return make_gpr
            shape = GPR_KERNELS[own["kernel"]]
            return lambda: make_gpr().set_params(
                gpr__kernel=ConstantKernel() * shape() + WhiteKernel()
            )
        case "XGBoost":
            trees = _picked(own, ("max_depth", "n_estimators", "learning_rate"))
            return lambda: make_xgboost().set_params(**trees)
        case "MLP" | "ResNet":
            raise ValueError(f"{family} is a network, fitted by network_fitter")
        case _:
            assert_never(family)


def _with_skips(pipeline: Pipeline) -> None:
    """Turn a built network into the residual network of the same size (identity skips)."""
    pipeline.named_steps["net"].set_params(module=ResMLP)


def curve_columns(space: CurveSpace) -> tuple[tuple[int, ...], int]:
    """The design columns of the curve keys and of the one input along a curve."""
    names = [name for name, _ in space.inputs]
    along = next(i for i, name in enumerate(names) if name not in space.curve)
    return tuple(names.index(name) for name in space.curve), along


class CurveNet(RegressorMixin, BaseEstimator):
    """A network as the curve-wise base: one row per curve, so each row is its own group and
    the network validates on whole curves."""

    def __init__(self, training: Training, n_inputs: int, n_outputs: int, skips: bool) -> None:
        self.training = training
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.skips = skips

    def fit(self, x: np.ndarray, y: np.ndarray) -> CurveNet:
        """Fit the network on the curves' keys and outputs."""
        self.pipeline_ = make_estimator(
            self.training, self.n_inputs, n_outputs=self.n_outputs, skips=self.skips
        )
        rows = np.asarray(x, dtype=np.float32)
        self.pipeline_.fit(rows, np.asarray(y, dtype=np.float32), net__groups=np.arange(len(x)))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Every output of each curve."""
        return self.pipeline_.predict(np.asarray(x, dtype=np.float32))

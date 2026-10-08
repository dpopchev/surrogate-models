"""Facts about the candidates and their harness fitters, on tiny synthetic data."""

from typing import Any

import numpy as np
import pytest
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from shared.candidates import (
    FAMILIES,
    UNITS,
    Candidate,
    Knobs,
    estimator_of,
    fitter_of,
    tuned_fitter,
    tuned_knobs,
    tuned_training,
)
from shared.design import Design
from shared.eda import make_curve_space
from shared.harness import harness
from shared.search import Space, space_of
from shared.surrogate import Training

TRAINING = Training(
    width=8,
    depth=2,
    activation="relu",
    loss="mse",
    lr=1e-2,
    max_epochs=2,
    batch_size=16,
    patience=2,
    valid_fraction=0.34,
    seed=0,
)
KNOBS = Knobs(k=2, neighbours=4, knots=4, per_curve=2)
SPACE = make_curve_space({"x": "raw", "p": "raw"}, ("p",))

# Twelve curves of ten rows, y = p + x: curves 10 and 11 are test, enough training curves left
# for a curve-wise network to validate on whole curves within a fold's fit.
P12 = np.repeat(np.arange(12.0), 10)
X12 = np.tile(np.linspace(0.0, 1.0, 10), 12)
WIDE = Design(
    X=np.column_stack([X12, P12]),
    y=P12 + X12,
    groups=P12.astype(int),
    test=P12 >= 10,
    fold=np.where(P12 >= 10, -1, P12.astype(int) % 2),
    ablation=np.zeros(120, dtype=bool),
)

CANDIDATES = [Candidate(family, unit) for family in FAMILIES for unit in UNITS]


@pytest.mark.parametrize("candidate", CANDIDATES, ids=str)
def test_every_candidate_predicts_the_toy_s_test_curves(candidate: Candidate) -> None:
    fitter = fitter_of(candidate, SPACE, KNOBS, TRAINING)
    (run,) = harness(WIDE, "mass", fitter, seeds=[0], valid_fraction=0.34)
    assert np.isfinite(run.predictions).all()


def lowest(space: Space) -> dict[str, Any]:
    """The first choice or the low end of every distribution of the space."""
    return {name: _lowest(dist) for name, dist in space.items()}


def _lowest(dist: BaseDistribution) -> Any:
    """The first choice of a categorical distribution, the low end of a numeric one."""
    match dist:
        case CategoricalDistribution():
            return dist.choices[0]
        case IntDistribution() | FloatDistribution():
            return dist.low
        case _:
            raise TypeError(f"no lowest value of {dist!r}")


@pytest.mark.parametrize("candidate", CANDIDATES, ids=str)
def test_every_candidate_s_searched_params_predict_the_toy_s_test_curves(
    candidate: Candidate,
) -> None:
    params = lowest(space_of(candidate.family, candidate.unit))
    fitter = tuned_fitter(candidate, SPACE, KNOBS, TRAINING, params)
    (run,) = harness(WIDE, "mass", fitter, seeds=[0], valid_fraction=0.34)
    assert np.isfinite(run.predictions).all()


def test_a_trial_s_knots_replace_the_knobs_knots() -> None:
    assert tuned_knobs(KNOBS, {"knots": 9}).knots == 9


def test_a_trial_s_width_replaces_the_training_width() -> None:
    assert tuned_training(TRAINING, {"width": 32, "knots": 9}).width == 32


def test_a_k_nn_trial_with_uniform_weights_builds_a_uniform_k_nn() -> None:
    estimator = estimator_of("k-NN", KNOBS, {"k": 3, "weights": "uniform"})()
    assert estimator.get_params()["knn__weights"] == "uniform"


def test_an_xgboost_trial_sets_its_trees_depth() -> None:
    estimator = estimator_of("XGBoost", KNOBS, {"max_depth": 5, "n_estimators": 10})()
    assert estimator.get_params()["max_depth"] == 5


def test_a_gpr_trial_s_matern52_kernel_is_a_matern_of_nu_2_5() -> None:
    estimator = estimator_of("GPR", KNOBS, {"kernel": "matern52"})()
    assert estimator.get_params()["gpr__kernel__k1__k2__nu"] == 2.5

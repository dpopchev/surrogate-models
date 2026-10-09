"""Facts about the equal-budget search, on a tiny synthetic design and a toy family."""

import numpy as np
import optuna
import torch
from optuna.distributions import FloatDistribution
from threadpoolctl import threadpool_info

from shared.design import Design
from shared.search import SPACES, search, space_of

# Eight curves of five rows, y = 1 + 0.1 p + x: curves 6 and 7 are test; the six training
# curves fall into two frozen folds.
P = np.repeat(np.arange(8.0), 5)
X_ALONG = np.tile(np.linspace(0.0, 1.0, 5), 8)
TOY = Design(
    X=np.column_stack([X_ALONG, P]),
    y=1.0 + 0.1 * P + X_ALONG,
    groups=P.astype(int),
    test=P >= 6,
    fold=np.where(P >= 6, -1, P.astype(int) % 2),
    ablation=P == 7,
)

# One knob: the toy predicts the truth times (1 + scale).
SPACE = {"scale": FloatDistribution(0.0, 0.5)}


def scaled(params):
    """A toy family: a fitter predicting the true y times 1 + params["scale"]."""

    def fit(x_fit, y_fit, x_valid, y_valid, seed):
        return lambda x: (1.0 + params["scale"]) * (1.0 + 0.1 * x[:, 1] + x[:, 0])

    return fit


def test_the_search_runs_exactly_its_budget_of_trials() -> None:
    found = search(scaled, SPACE, 3, TOY, "mass", seed=0, valid_fraction=0.34)
    assert found.trials == 3


def test_every_library_runs_on_one_thread_during_a_trial() -> None:
    threads: list[int] = []

    def counting(params):
        def fit(x_fit, y_fit, x_valid, y_valid, seed):
            threads.append(torch.get_num_threads())
            threads.extend(pool["num_threads"] for pool in threadpool_info())
            return lambda x: np.ones(len(x))

        return fit

    search(counting, SPACE, 1, TOY, "mass", seed=0, valid_fraction=0.34)
    assert set(threads) == {1}


def test_every_family_of_the_screen_has_a_space() -> None:
    screened = {"k-NN", "local RBF", "GPR", "XGBoost", "MLP", "ResNet"}
    assert set(SPACES) == screened


def test_a_curve_wise_candidate_also_tunes_the_spline_knots() -> None:
    assert "knots" in space_of("k-NN", "curve-wise")


def test_a_pointwise_candidate_has_no_spline_knots() -> None:
    assert "knots" not in space_of("k-NN", "pointwise")


def test_one_seed_gives_one_search() -> None:
    first = search(scaled, SPACE, 3, TOY, "mass", seed=7, valid_fraction=0.34)
    second = search(scaled, SPACE, 3, TOY, "mass", seed=7, valid_fraction=0.34)
    assert first == second


def test_a_trial_never_fits_nor_predicts_a_test_curve() -> None:
    seen: list[np.ndarray] = []

    def recording(params):
        def fit(x_fit, y_fit, x_valid, y_valid, seed):
            seen.extend([x_fit[:, 1], x_valid[:, 1]])

            def predict(x):
                seen.append(x[:, 1])
                return np.ones(len(x))

            return predict

        return fit

    search(recording, SPACE, 3, TOY, "mass", seed=0, valid_fraction=0.34)
    assert max(float(p.max()) for p in seen) < 6.0


def test_the_best_trial_is_the_one_nearest_the_truth_on_the_folds() -> None:
    study = optuna.create_study(direction="maximize")
    found = search(scaled, SPACE, 8, TOY, "mass", seed=0, valid_fraction=0.34, study=study)
    scales = [trial.params["scale"] for trial in study.trials]
    assert found.params["scale"] == min(scales)


def raising_second(raised: list[dict]):
    """A toy family whose second trial's fitter raises FloatingPointError, as FiniteLoss does on
    a non-finite loss; the raising trial's params are appended to `raised`."""
    calls: list[int] = []

    def family(params):
        calls.append(1)
        if len(calls) == 2:
            raised.append(params)

        def fit(x_fit, y_fit, x_valid, y_valid, seed):
            if len(calls) == 2:
                raise FloatingPointError("loss is nan")
            return lambda x: (1.0 + params["scale"]) * (1.0 + 0.1 * x[:, 1] + x[:, 0])

        return fit

    return family, calls


def test_a_raising_trial_does_not_end_the_search() -> None:
    family, calls = raising_second([])
    search(family, SPACE, 4, TOY, "mass", seed=0, valid_fraction=0.34)
    assert len(calls) == 4


def test_a_raising_trial_is_told_fail_in_the_study() -> None:
    family, _ = raising_second([])
    study = optuna.create_study(direction="maximize")
    search(family, SPACE, 4, TOY, "mass", seed=0, valid_fraction=0.34, study=study)
    states = [trial.state for trial in study.trials]
    assert states.count(optuna.trial.TrialState.FAIL) == 1


def test_the_best_params_come_from_the_trials_that_did_not_raise() -> None:
    raised: list[dict] = []
    family, _ = raising_second(raised)
    found = search(family, SPACE, 4, TOY, "mass", seed=0, valid_fraction=0.34)
    assert found.params != raised[0]


def test_a_search_counts_its_failed_trials() -> None:
    family, _ = raising_second([])
    found = search(family, SPACE, 4, TOY, "mass", seed=0, valid_fraction=0.34)
    assert found.failed == 1

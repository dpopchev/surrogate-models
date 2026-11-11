"""Facts about the final surrogate: read a winner, fit it, save it and load it back."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paperlib.candidates import Candidate, Knobs
from paperlib.design import Design, Pair
from paperlib.domain import make_black_hole_domain
from paperlib.eda import make_curve_space
from paperlib.final import (
    FinalSurrogate,
    InvalidWinnerError,
    Refused,
    SavedSurrogateError,
    Winner,
    fit_winner,
    load_domain,
    load_surrogate,
    predict,
    read_winners,
    save_domain,
    save_surrogate,
)
from paperlib.surrogate import Training

pytestmark = pytest.mark.model

SPACE = make_curve_space({"x": "raw", "p": "raw"}, ("p",))
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

# Three curves of ten rows, y = 1 + p + x: curve 2 is the test curve.
P3 = np.repeat(np.arange(3.0), 10)
X3 = np.tile(np.linspace(0.0, 1.0, 10), 3)
TOY = Design(
    X=np.column_stack([X3, P3]),
    y=1.0 + P3 + X3,
    groups=P3.astype(int),
    test=P3 >= 2,
    fold=np.where(P3 >= 2, -1, P3.astype(int)),
    ablation=np.zeros(30, dtype=bool),
)
PAIR = Pair("toy", "mass", TOY, SPACE)
WINNER = Winner("toy", "mass", Candidate("k-NN", "pointwise"), {"weights": "distance"})


def fitted() -> FinalSurrogate:
    return fit_winner(PAIR, WINNER, KNOBS, TRAINING, seed=0, valid_fraction=0.5, ceiling=4.0)


def test_saved_surrogate_predicts_as_fitted(tmp_path: Path) -> None:
    surrogate = fitted()
    save_surrogate(tmp_path / "toy.pkl", surrogate)
    loaded = load_surrogate(tmp_path / "toy.pkl")
    np.testing.assert_array_equal(loaded.predict(TOY.X), surrogate.predict(TOY.X))


def test_a_fitted_surrogate_names_its_pair() -> None:
    surrogate = fitted()
    assert (surrogate.dataset, surrogate.target, surrogate.family, surrogate.unit) == (
        "toy",
        "mass",
        "k-NN",
        "pointwise",
    )


def test_a_fitted_surrogate_keeps_the_ceiling_it_was_given() -> None:
    assert fitted().ceiling == 4.0


def test_a_fitted_surrogate_scores_its_test_rows_in_significant_figures() -> None:
    assert fitted().test > 0.0


def test_winners_are_read_from_the_tuning_json(tmp_path: Path) -> None:
    path = tmp_path / "winners.json"
    path.write_text(
        json.dumps(
            [
                {
                    "dataset": "toy",
                    "target": "charge",
                    "family": "GPR",
                    "unit": "pointwise",
                    "params": {"kernel": "rbf"},
                    "folds": 1.0,
                    "test": 2.0,
                }
            ]
        )
    )
    assert read_winners(path) == (
        Winner("toy", "charge", Candidate("GPR", "pointwise"), {"kernel": "rbf"}),
    )


def test_a_winner_of_an_unknown_family_is_refused() -> None:
    with pytest.raises(InvalidWinnerError, match="family"):
        Winner.of("toy", "mass", "SVM", "pointwise", {})


def test_a_failed_reload_restores_the_earlier_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "toy.pkl"
    save_surrogate(path, fitted())
    before = path.read_bytes()

    def broken(_: Path) -> FinalSurrogate:
        raise OSError("unreadable")

    monkeypatch.setattr("paperlib.final.load_surrogate", broken)
    with pytest.raises(SavedSurrogateError, match=r"toy\.pkl"):
        save_surrogate(path, fitted())
    assert path.read_bytes() == before


BH_TABLE = pd.DataFrame({"beta": [0.0, 0.0, 1.0, 1.0], "r_h": [2.0, 5.0, 3.0, 6.0]})


def test_guarded_predict_refuses_outside_the_domain() -> None:
    domain = make_black_hole_domain(BH_TABLE)

    def predictor(x: np.ndarray) -> np.ndarray:
        if (x[:, 0] < 2.0).any():
            raise RuntimeError("the surrogate was evaluated on a refused row")
        return x[:, 0] * 2.0

    toy = FinalSurrogate("bh", "mass", "k-NN", "pointwise", {}, 1.0, 1.0, predictor)
    # Columns r_h, beta: the first row is below the existence edge 2.0, the second inside.
    result = predict(toy, domain, np.array([[1.0, 0.0], [4.0, 0.0]]))
    assert result == (Refused("existence edge", {"beta": 0.0, "r_h": 2.0}), 8.0)


def test_a_domain_is_saved_and_loaded_back(tmp_path: Path) -> None:
    domain = make_black_hole_domain(BH_TABLE)
    save_domain(tmp_path / "bh-domain.json", domain)
    assert load_domain(tmp_path / "bh-domain.json") == domain

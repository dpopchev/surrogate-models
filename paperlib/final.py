"""The final surrogate of one pair: the tuning's winner fitted on every non-test row, scored on
the test rows beside the audit's ceiling, saved and loaded by the sections that read it (W-152).

The harness predictor is a closure, so the saved form is the whole FinalSurrogate through
cloudpickle (standard pickle bytes, read back with pickle).
"""

import json
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cloudpickle
import numpy as np

from paperlib.candidates import FAMILIES, UNITS, Candidate, Family, Knobs, Unit, tuned_fitter
from paperlib.design import TARGETS, Pair, Target
from paperlib.harness import Predictor, fit_on, one_thread
from paperlib.scorecard import relative_errors, significant_figures
from paperlib.surrogate import Training

# --- vocabulary and types ---------------------------------------------------------------------


class InvalidWinnerError(ValueError):
    """A winner names a family, unit or target the pipeline does not know."""


class SavedSurrogateError(OSError):
    """A saved surrogate does not read back as the one written."""


@dataclass(frozen=True)
class Winner:
    """The tuned candidate of one pair: its family and unit and the trial's params."""

    dataset: str
    target: Target
    candidate: Candidate
    params: Mapping[str, Any]

    @classmethod
    def of(
        cls, dataset: str, target: str, family: str, unit: str, params: Mapping[str, Any]
    ) -> Winner:
        """The winner of plain values.

        Raises InvalidWinnerError when the family, unit or target is unknown.
        """
        for name, value, known in (
            ("family", family, FAMILIES),
            ("unit", unit, UNITS),
            ("target", target, TARGETS),
        ):
            if value not in known:
                raise InvalidWinnerError(f"unknown {name} {value!r}, expected one of {known}")
        candidate = Candidate(cast(Family, family), cast(Unit, unit))
        return cls(dataset, cast(Target, target), candidate, dict(params))


@dataclass(frozen=True)
class FinalSurrogate:
    """One fitted pair: its winner's identity, the p95 test and the ceiling figures in
    significant figures, and its predictor."""

    dataset: str
    target: Target
    family: Family
    unit: Unit
    params: Mapping[str, Any]
    test: float
    ceiling: float
    predictor: Predictor

    def predict(self, x: np.ndarray) -> np.ndarray:
        """The target of each row of inputs."""
        return self.predictor(x)


# --- pure functions ---------------------------------------------------------------------------


def fit_winner(
    pair: Pair,
    winner: Winner,
    knobs: Knobs,
    training: Training,
    seed: int,
    valid_fraction: float,
    ceiling: float,
) -> FinalSurrogate:
    """The winner fitted on every row that is not a test row, scored by the p95 relative error
    of the test rows (the charge in D rebuilt with the true M) as significant figures."""
    fitter = tuned_fitter(winner.candidate, pair.space, knobs, training, winner.params)
    design = pair.design
    with one_thread():
        predictor = fit_on(design, ~design.test, fitter, seed, valid_fraction)
        predicted = predictor(design.X[design.test])
    errors = relative_errors(design.y[design.test], predicted, pair.target)
    p95 = float(np.quantile(errors[np.isfinite(errors)], 0.95))
    return FinalSurrogate(
        winner.dataset,
        winner.target,
        winner.candidate.family,
        winner.candidate.unit,
        winner.params,
        significant_figures(p95),
        ceiling,
        predictor,
    )


# --- adapters ---------------------------------------------------------------------------------


def read_winners(path: Path) -> tuple[Winner, ...]:
    """The winners the tuning stored as JSON; a missing file raises naming the path."""
    if not path.is_file():
        raise FileNotFoundError(f"no winners at {path}: run the tuning first")
    return tuple(
        Winner.of(w["dataset"], w["target"], w["family"], w["unit"], w["params"])
        for w in json.loads(path.read_text())
    )


def save_surrogate(path: Path, surrogate: FinalSurrogate) -> None:
    """Write the surrogate, read it back and compare; on a failure restore what the file held
    (or remove it) and raise SavedSurrogateError."""
    before = path.read_bytes() if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cloudpickle.dumps(surrogate))
    try:
        back = load_surrogate(path)
        same = (back.dataset, back.target, back.family, back.unit, back.test, back.ceiling) == (
            surrogate.dataset,
            surrogate.target,
            surrogate.family,
            surrogate.unit,
            surrogate.test,
            surrogate.ceiling,
        )
        if not same:
            raise SavedSurrogateError("the reloaded surrogate differs from the saved one")
    except Exception as error:
        if before is None:
            path.unlink()
        else:
            path.write_bytes(before)
        raise SavedSurrogateError(f"{path} does not read back: {error}") from error


def load_surrogate(path: Path) -> FinalSurrogate:
    """The surrogate saved at path; a missing file raises naming the path."""
    if not path.is_file():
        raise FileNotFoundError(f"no surrogate at {path}: run make pipeline first")
    loaded = pickle.loads(path.read_bytes())
    if not isinstance(loaded, FinalSurrogate):
        raise SavedSurrogateError(f"{path} holds a {type(loaded).__name__}, not a surrogate")
    return loaded

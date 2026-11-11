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
from paperlib.domain import (
    BlackHoleDomain,
    Domain,
    Inside,
    NeutronStarDomain,
    Reason,
    check,
)
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


@dataclass(frozen=True)
class HeldOut:
    """The test rows of a pair scored one by one: inputs, curve ids, relative errors and, when
    given, the true charge D of each row."""

    inputs: np.ndarray
    curves: np.ndarray
    errors: np.ndarray
    charge: np.ndarray | None


@dataclass(frozen=True)
class Refused:
    """A query outside the physical domain: why, and the nearest valid input."""

    reason: Reason
    nearest: Mapping[str, float]


# The columns of an input row by domain, in the order of the dataset's curve space; a neutron-star
# rho_c is the physical value, the surrogate reads its log10.
BLACK_HOLE_COLUMNS = ("r_h", "beta")
NEUTRON_STAR_COLUMNS = ("rho_c", "beta", "lambda")

# --- pure functions ---------------------------------------------------------------------------


def predict(
    surrogate: FinalSurrogate, domain: Domain, x: np.ndarray
) -> tuple[float | Refused, ...]:
    """Per input row the prediction when the row lies inside the domain, else Refused; the
    surrogate is never evaluated on a refused row."""
    columns = BLACK_HOLE_COLUMNS if isinstance(domain, BlackHoleDomain) else NEUTRON_STAR_COLUMNS
    verdicts = [check(domain, dict(zip(columns, map(float, row), strict=True))) for row in x]
    inside = np.array([isinstance(v, Inside) for v in verdicts], dtype=bool)
    values = np.empty(len(x))
    if inside.any():
        rows = np.array(x[inside], dtype=np.float64)
        if isinstance(domain, NeutronStarDomain):
            rows[:, 0] = np.log10(rows[:, 0])
        values[inside] = surrogate.predict(rows)
    return tuple(
        float(values[i]) if isinstance(v, Inside) else Refused(v.reason, v.nearest)
        for i, v in enumerate(verdicts)
    )


def held_out_errors(surrogate: FinalSurrogate, pair: Pair, charge: np.ndarray | None) -> HeldOut:
    """The relative error of each test row of the pair's design (the charge in D rebuilt with the
    true M); `charge` is the true D of every design row, kept for the test rows."""
    design = pair.design
    test = design.test
    errors = relative_errors(design.y[test], surrogate.predict(design.X[test]), pair.target)
    return HeldOut(
        design.X[test], design.groups[test], errors, None if charge is None else charge[test]
    )


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


def domain_json(domain: Domain) -> str:
    """The domain as plain JSON, tagged by its kind."""
    if isinstance(domain, BlackHoleDomain):
        body: dict[str, Any] = {
            "kind": "black_hole",
            "betas": domain.betas,
            "r_min": domain.r_min,
            "r_max": domain.r_max,
        }
    else:
        body = {
            "kind": "neutron_star",
            "keys": domain.keys,
            "rho_min": domain.rho_min,
            "rho_max": domain.rho_max,
        }
    return json.dumps(body, indent=2) + "\n"


def save_domain(path: Path, domain: Domain) -> None:
    """Write the domain as JSON and read it back; a difference removes the file (or restores what
    it held) and raises SavedSurrogateError."""
    before = path.read_text() if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(domain_json(domain))
    if load_domain(path) != domain:
        if before is None:
            path.unlink()
        else:
            path.write_text(before)
        raise SavedSurrogateError(f"{path} does not read back as the domain written")


def load_domain(path: Path) -> Domain:
    """The domain saved at path; a missing file raises naming the path."""
    if not path.is_file():
        raise FileNotFoundError(f"no domain at {path}: run make pipeline first")
    body = json.loads(path.read_text())
    if body["kind"] == "black_hole":
        return BlackHoleDomain(tuple(body["betas"]), tuple(body["r_min"]), tuple(body["r_max"]))
    return NeutronStarDomain(
        tuple((float(b), float(lam)) for b, lam in body["keys"]),
        tuple(body["rho_min"]),
        tuple(body["rho_max"]),
    )


def load_surrogate(path: Path) -> FinalSurrogate:
    """The surrogate saved at path; a missing file raises naming the path."""
    if not path.is_file():
        raise FileNotFoundError(f"no surrogate at {path}: run make pipeline first")
    loaded = pickle.loads(path.read_bytes())
    if not isinstance(loaded, FinalSurrogate):
        raise SavedSurrogateError(f"{path} holds a {type(loaded).__name__}, not a surrogate")
    return loaded

"""The design-matrix contract every surrogate reads its data through (W-034).

A prepared table and the frozen curve split (50_methodology/51_algorithms/split_datasets.py)
become one Design per dataset and target: the inputs of the dataset's curve space, each raw or
in log10, the target (M in M_sun, or the charge target log10(D/M) of Section 3.3, W-063), one
integer per curve for grouped splitting, and each row's test and ablation flags.
"""

from dataclasses import dataclass
from typing import Literal, assert_never

import numpy as np
import pandas as pd

from shared.eda import CurveSpace, curve_ids, make_curve_space

# --- vocabulary and types ---------------------------------------------------------------------

Target = Literal["mass", "charge"]


@dataclass(frozen=True)
class DesignSpec:
    """One dataset's name in the split file and its curve space."""

    dataset: str
    space: CurveSpace


NEUTRON_STARS = DesignSpec(
    "neutron_stars",
    make_curve_space({"rho_c": "log10", "beta": "raw", "lambda": "raw"}, ("beta", "lambda")),
)
BLACK_HOLES = DesignSpec(
    "black_holes", make_curve_space({"r_h": "raw", "beta": "raw"}, ("beta",))
)


class UnlabelledCurveError(ValueError):
    """A row's curve has no label for its dataset in the split."""


@dataclass(frozen=True)
class Design:
    """The rows of one dataset as a model sees them, aligned row by row."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    test: np.ndarray
    fold: np.ndarray
    ablation: np.ndarray


# --- pure functions ---------------------------------------------------------------------------


def _inputs(table: pd.DataFrame, space: CurveSpace) -> np.ndarray:
    """The inputs of the space in its order, each raw or in log10."""
    columns = [
        np.log10(table[column]) if scale == "log10" else table[column]
        for column, scale in space.inputs
    ]
    return np.column_stack(columns).astype(np.float32)


def _target(table: pd.DataFrame, target: Target) -> np.ndarray:
    """M in M_sun, or the charge target log10(D/M); every prepared D is positive."""
    mass = table["M"].to_numpy(np.float64)
    match target:
        case "mass":
            values = mass
        case "charge":
            values = np.log10(table["D"].to_numpy(np.float64) / mass)
        case _:
            assert_never(target)
    return values.astype(np.float32)


def _labels(table: pd.DataFrame, split: pd.DataFrame, spec: DesignSpec) -> pd.DataFrame:
    """Each row's split label and ablation flag, from its curve's row for this dataset."""
    curve = list(spec.space.curve)
    own = split.loc[split["dataset"] == spec.dataset, [*curve, "label", "ablation"]]
    labelled = table[curve].merge(own, on=curve, how="left", validate="many_to_one")
    if (missing := labelled["label"].isna()).any():
        unlabelled = labelled.loc[missing, curve].drop_duplicates()
        raise UnlabelledCurveError(
            f"{spec.dataset}: {len(unlabelled)} curve(s) without a split label, first "
            f"{unlabelled.iloc[0].to_dict()}"
        )
    return labelled


def design(
    table: pd.DataFrame, split: pd.DataFrame, spec: DesignSpec, target: Target
) -> Design:
    """The design of one dataset and target under the frozen split.

    Raises UnlabelledCurveError when a row's curve has no label for the dataset.
    """
    labelled = _labels(table, split, spec)
    return Design(
        X=_inputs(table, spec.space),
        y=_target(table, target),
        groups=curve_ids(table, spec.space),
        test=(labelled["label"] == "test").to_numpy(),
        fold=_folds(labelled["label"]),
        ablation=labelled["ablation"].to_numpy(bool),
    )


def _folds(labels: pd.Series) -> np.ndarray:
    """Each row's frozen fold, k for "fold<k>", -1 for a test row."""
    return np.array(
        [-1 if label == "test" else int(str(label).removeprefix("fold")) for label in labels]
    )

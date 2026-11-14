"""Facts about the zone masks and scores, on two hand-made curves of five rows."""

import numpy as np
import pytest

from paperlib.design import Design
from paperlib.harness import FitRows, Predictor
from paperlib.zones import (
    curve_position,
    edge_held_out,
    physical_edge,
    zone_figures,
    zone_masks,
)

pytestmark = pytest.mark.model

# Six curves of two rows, y = 1 + curve: curves 0 and 1 are test, curve 2 is the edge, curves 3
# to 5 train; the first input is the row's own index.
CURVES = np.repeat(np.arange(6), 2)
HELD = Design(
    X=np.column_stack([np.arange(12.0), CURVES.astype(float)]),
    y=1.0 + CURVES,
    groups=CURVES,
    test=CURVES <= 1,
    fold=np.full(12, -1),
    ablation=CURVES == 2,
)


class Clock:
    """A clock that ticks one second at each reading."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


GROUPS = np.array([0] * 5 + [1] * 5)
ALONG = np.array([5.0, 1.0, 2.0, 4.0, 3.0, 10.0, 20.0, 30.0, 40.0, 50.0])
ABLATION = np.array([False] * 4 + [True] + [False] * 4 + [True])
ERRORS = np.array([0.1, 0.01, 0.1, 0.1, 0.001, 0.1, 0.1, 0.1, 0.1, 0.01])


def test_the_position_ranks_each_row_along_its_own_curve() -> None:
    assert curve_position(GROUPS, ALONG).tolist() == [
        1.0, 0.0, 0.25, 0.75, 0.5, 0.0, 0.25, 0.5, 0.75, 1.0,
    ]  # fmt: skip


def test_a_one_row_curve_sits_at_the_start() -> None:
    assert curve_position(np.array([7]), np.array([3.0])).tolist() == [0.0]


def test_the_physical_edge_at_the_start_is_the_first_rows_of_each_curve() -> None:
    position = curve_position(GROUPS, ALONG)
    assert np.flatnonzero(physical_edge(position, "start", 0.25)).tolist() == [1, 2, 5, 6]


def test_the_physical_edge_at_the_end_is_the_last_rows_of_each_curve() -> None:
    position = curve_position(GROUPS, ALONG)
    assert np.flatnonzero(physical_edge(position, "end", 0.25)).tolist() == [0, 3, 8, 9]


def test_the_masks_split_the_rows_into_interior_edge_and_physical() -> None:
    masks = zone_masks(ABLATION, curve_position(GROUPS, ALONG), "start", 0.0)
    assert {name: np.flatnonzero(mask).tolist() for name, mask in masks.items()} == {
        "interior": [0, 1, 2, 3, 5, 6, 7, 8],
        "edge": [4, 9],
        "physical": [1, 5],
    }


def test_zone_figures_report_p95_and_worst_row_per_zone() -> None:
    masks = zone_masks(ABLATION, curve_position(GROUPS, ALONG), "start", 0.0)
    masks["empty"] = np.zeros(10, dtype=bool)
    found = zone_figures(ERRORS, masks)
    assert {
        name: None if f is None else (round(f.p95, 4), round(f.worst, 4))
        for name, f in found.items()
    } == {
        "interior": (1.0, 1.0),
        "edge": (2.02, 2.0),
        "physical": (1.02, 1.0),
        "empty": None,
    }


def test_edge_held_out_fits_without_the_edge_curves() -> None:
    seen: list[FitRows] = []

    def fitter(rows: FitRows, seed: int) -> Predictor:
        seen.append(rows)
        return lambda x: np.full(len(x), rows.y_fit.mean())

    found = edge_held_out(HELD, "mass", fitter, 0, 0.34, Clock())
    (rows,) = seen
    shown = {int(i) for i in np.concatenate([rows.x_fit[:, 0], rows.x_valid[:, 0]])}
    assert (shown, sorted(found.zones), found.fit_seconds, found.predict_seconds) == (
        set(range(6, 12)),
        ["edge", "interior"],
        1.0,
        1.0,
    )

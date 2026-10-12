"""Facts about the knobs macro file: the baseline network's settings and the screens' make knobs."""

import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from paper_knobs import knob_numbers, space_numbers

from shared.config import AlgorithmsSection

pytestmark = pytest.mark.model


def test_knob_macros_name_every_setting() -> None:
    found = knob_numbers(
        AlgorithmsSection(),
        {"Repr": "--k 8 --seeds 3", "Families": "--budgets 1000 10000 --memory-gb 4"},
    )
    assert set(found) == {
        "knobNetWidth",
        "knobNetDepth",
        "knobNetActivation",
        "knobNetLoss",
        "knobNetLr",
        "knobNetMaxEpochs",
        "knobNetBatchSize",
        "knobNetPatience",
        "knobNetValidFraction",
        "knobNetSeed",
        "knobNetThreads",
        "knobNetWorkers",
        "knobReprK",
        "knobReprSeeds",
        "knobFamiliesBudgetsFirst",
        "knobFamiliesBudgetsSecond",
        "knobFamiliesMemoryGb",
    }


def test_space_macros_name_each_range() -> None:
    space = {
        "neighbours": IntDistribution(50, 400, log=True),
        "lr": FloatDistribution(1e-4, 1e-2, log=True),
        "kernel": CategoricalDistribution(("rbf", "matern52")),
    }
    assert space_numbers({"Rbf": space}, IntDistribution(6, 24)) == {
        "knobSpaceRbfNeighboursLow": "50",
        "knobSpaceRbfNeighboursHigh": "400",
        "knobSpaceRbfLrLow": "10^{-4}",
        "knobSpaceRbfLrHigh": "10^{-2}",
        "knobSpaceKnotsLow": "6",
        "knobSpaceKnotsHigh": "24",
    }

"""Facts about the knobs macro file: the baseline network's settings and the screens' make knobs."""

import pytest
from paper_knobs import knob_numbers

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

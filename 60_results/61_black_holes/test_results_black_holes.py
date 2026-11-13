"""Facts about Section 5.1's figure and numbers, on a tiny synthetic black-hole map and toy
surrogates."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from results_black_holes import (
    SECTION_TOML,
    BlackHoleSection,
    MapInput,
    TargetInput,
    VariantName,
    draw,
    figure_tex,
    main,
    numbers,
    run,
)

from paperlib.config import PaperConfig, load_section
from paperlib.design import BLACK_HOLES, Target, pairs
from paperlib.domain import make_black_hole_domain
from paperlib.final import FinalSurrogate, save_domain, save_surrogate
from paperlib.harness import Predictor
from paperlib.plots import NOTATION_TEX, PlotStyle, apply_style


@pytest.fixture(autouse=True)
def _notation():
    """The style with the paper's notation, as main applies it; restored after each test."""
    with plt.rc_context():
        apply_style(PlotStyle(usetex=False), NOTATION_TEX.read_text())
        yield


def total(x: np.ndarray) -> np.ndarray:
    """The toy mass prediction: the sum of the inputs of each row (module level, so it pickles)."""
    return x.sum(axis=1)


def flat(x: np.ndarray) -> np.ndarray:
    """The toy charge prediction: log10(D/M) = 0 everywhere."""
    return np.zeros(len(x))


def toy_surrogate(
    target: Target, predictor: Predictor, test: float, ceiling: float
) -> FinalSurrogate:
    """A black-hole surrogate with the given p95 test and ceiling figures."""
    return FinalSurrogate(
        dataset="black_holes",
        target=target,
        family="local RBF",
        unit="curve-wise",
        params={},
        test=test,
        ceiling=ceiling,
        predictor=predictor,
    )


def table() -> pd.DataFrame:
    """Four beta curves of four rows each, every curve starting a step later in r_h."""
    rows = [
        {"beta": b, "r_h": 1.0 + 0.5 * (b - 1) + 0.5 * i, "M": 1.0 + 0.5 * b, "D": 2.0 + b}
        for b in (1.0, 2.0, 3.0, 4.0)
        for i in range(4)
    ]
    return pd.DataFrame(rows)


def split() -> pd.DataFrame:
    """Curves 1 and 2 train, curves 3 and 4 are the test curves."""
    return pd.DataFrame(
        [
            {"dataset": "black_holes", "beta": b, "lambda": np.nan, "label": label, "ablation": 0}
            for b, label in ((1.0, "fold0"), (2.0, "fold1"), (3.0, "test"), (4.0, "test"))
        ]
    )


def toy_map() -> MapInput:
    """The toy map: mass 1.0 figures against a ceiling of 2.0, charge 1.5 against 4.0."""
    data = table()
    found = pairs({"black_holes": data}, split(), (BLACK_HOLES,), ("mass", "charge"))
    surrogates = (
        toy_surrogate("mass", total, 1.0, 2.0),
        toy_surrogate("charge", flat, 1.5, 4.0),
    )
    return MapInput(
        make_black_hole_domain(data),
        tuple(
            TargetInput(pair, surrogate, data["D"].to_numpy(float), surrogate.ceiling)
            for pair, surrogate in zip(found, surrogates, strict=True)
        ),
    )


@pytest.mark.model
def test_the_numbers_hold_the_gap_of_the_mass_target() -> None:
    assert numbers(run(toy_map()))["resultsBhMassGap"] == "1.0"


@pytest.mark.model
def test_the_numbers_hold_the_family_and_unit_of_the_winner() -> None:
    macros = numbers(run(toy_map()))
    assert (macros["resultsBhChargeFamily"], macros["resultsBhChargeUnit"]) == (
        "local RBF",
        "curve-wise",
    )


@pytest.mark.model
def test_the_evidence_scores_only_the_test_rows() -> None:
    assert [len(t.held.errors) for t in run(toy_map()).targets] == [8, 8]


@pytest.mark.functional
def test_curves_draws_one_line_per_held_out_curve_in_each_panel() -> None:
    figure = draw(run(toy_map()), PlotStyle(usetex=False), "curves")
    assert [len(ax.lines) - 1 for ax in figure.axes[:2]] == [2, 2]


@pytest.mark.functional
def test_curves_marks_the_data_ceiling_of_each_target() -> None:
    figure = draw(run(toy_map()), PlotStyle(usetex=False), "curves")
    heights = [float(np.ravel(ax.lines[-1].get_ydata())[0]) for ax in figure.axes[:2]]
    assert heights == pytest.approx([1e-2, 10**-4.0])


@pytest.mark.functional
def test_plane_draws_a_scatter_of_the_test_rows_in_each_panel() -> None:
    figure = draw(run(toy_map()), PlotStyle(usetex=False), "plane")
    shapes = [np.shape(ax.collections[0].get_offsets())[0] for ax in figure.axes[:2]]
    assert shapes == [8, 8]


@pytest.mark.view
@pytest.mark.parametrize("variant", ["curves", "plane"])
def test_figure_tex_includes_the_picked_variant(variant: VariantName) -> None:
    assert f"{{61_black_holes_fig_errors_{variant}}}" in figure_tex(variant)


@pytest.fixture
def written(tmp_path: Path) -> tuple[Path, Path]:
    """The state and asset folders after main ran on the toy table, domain and surrogates."""
    state, assets = tmp_path / "state", tmp_path / "assets"
    data = table()
    data.to_parquet(tmp_path / "bh.parquet")
    split().to_parquet(tmp_path / "split.parquet")
    pipeline = state / "41_algorithm"
    save_domain(pipeline / "black_holes-domain.json", make_black_hole_domain(data))
    save_surrogate(pipeline / "black_holes-mass.pkl", toy_surrogate("mass", total, 1.0, 2.0))
    save_surrogate(pipeline / "black_holes-charge.pkl", toy_surrogate("charge", flat, 1.5, 4.0))
    reachable = state / "92_black_holes"
    reachable.mkdir(parents=True)
    (reachable / "reachable_figures.json").write_text(json.dumps({"mass": 2.0, "charge": 4.0}))
    main(
        [str(tmp_path / "bh.parquet"), str(tmp_path / "split.parquet"), str(state), str(assets)],
        config=PaperConfig.model_validate({"plot": {"usetex": False}}),
        section=BlackHoleSection(),
    )
    return state, assets


@pytest.mark.functional
def test_main_writes_the_figure_wrapper(written: tuple[Path, Path]) -> None:
    assert (written[1] / "61_black_holes" / "61_black_holes_fig_errors.tex").is_file()


@pytest.mark.functional
def test_main_stores_the_numbers(written: tuple[Path, Path]) -> None:
    stored = json.loads((written[0] / "61_black_holes" / "numbers.json").read_text())
    assert stored["resultsBhMassGap"] == "1.0"


@pytest.mark.model
def test_shipped_section_toml_selects_the_papers_picks() -> None:
    expected = BlackHoleSection.model_validate({"variants": {"errors": "curves"}})
    assert load_section(BlackHoleSection, SECTION_TOML) == expected

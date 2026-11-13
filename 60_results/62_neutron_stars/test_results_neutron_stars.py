"""Facts about Section 5.2's figure and numbers, on a tiny synthetic neutron-star map and toy
surrogates."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from results_neutron_stars import (
    SECTION_TOML,
    MapInput,
    NeutronStarSection,
    TargetInput,
    VariantName,
    draw,
    figure_tex,
    main,
    numbers,
    run,
)

from paperlib.config import PaperConfig, load_section
from paperlib.design import NEUTRON_STARS, Target, pairs
from paperlib.final import FinalSurrogate, save_surrogate
from paperlib.harness import Predictor
from paperlib.plots import NOTATION_TEX, PlotStyle, apply_style

CURVES = ((1.0, 1.0, "fold0"), (1.0, 2.0, "fold1"), (2.0, 1.0, "test"), (2.0, 2.0, "test"))


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
    """A neutron-star surrogate with the given p95 test and ceiling figures."""
    return FinalSurrogate(
        dataset="neutron_stars",
        target=target,
        family="local RBF",
        unit="curve-wise",
        params={},
        test=test,
        ceiling=ceiling,
        predictor=predictor,
    )


def table() -> pd.DataFrame:
    """Four (beta, lambda) curves of four rows each, the central density rising along each."""
    rows = [
        {
            "beta": b,
            "lambda": lam,
            "rho_c": 10.0 ** (1.0 + 0.1 * i),
            "M": 1.0 + 0.1 * i + b,
            "D": 2.0 + b + lam,
        }
        for b, lam, _ in CURVES
        for i in range(4)
    ]
    return pd.DataFrame(rows)


def split() -> pd.DataFrame:
    """Curves 1 and 2 train, curves 3 and 4 are the test curves."""
    return pd.DataFrame(
        [
            {"dataset": "neutron_stars", "beta": b, "lambda": lam, "label": label, "ablation": 0}
            for b, lam, label in CURVES
        ]
    )


def toy_map() -> MapInput:
    """The toy map: mass 1.0 figures against a ceiling of 2.0, charge 1.5 against 4.0."""
    data = table()
    found = pairs({"neutron_stars": data}, split(), (NEUTRON_STARS,), ("mass", "charge"))
    surrogates = (
        toy_surrogate("mass", total, 1.0, 2.0),
        toy_surrogate("charge", flat, 1.5, 4.0),
    )
    return MapInput(
        tuple(
            TargetInput(pair, surrogate, data["D"].to_numpy(float), surrogate.ceiling)
            for pair, surrogate in zip(found, surrogates, strict=True)
        )
    )


@pytest.mark.model
def test_the_numbers_hold_the_gap_of_the_mass_target() -> None:
    assert numbers(run(toy_map()))["resultsNsMassGap"] == "1.0"


@pytest.mark.model
def test_the_numbers_hold_the_family_and_unit_of_the_winner() -> None:
    macros = numbers(run(toy_map()))
    assert (macros["resultsNsChargeFamily"], macros["resultsNsChargeUnit"]) == (
        "local RBF",
        "curve-wise",
    )


@pytest.mark.model
def test_the_evidence_scores_only_the_test_rows() -> None:
    assert [len(t.held.errors) for t in run(toy_map()).targets] == [8, 8]


@pytest.mark.functional
@pytest.mark.parametrize("variant", ["by_lambda", "by_beta"])
def test_draws_one_line_per_held_out_curve_in_each_panel(variant: VariantName) -> None:
    figure = draw(run(toy_map()), PlotStyle(usetex=False), variant)
    assert [len(ax.lines) - 1 for ax in figure.axes[:2]] == [2, 2]


@pytest.mark.functional
def test_draws_the_data_ceiling_of_each_target() -> None:
    figure = draw(run(toy_map()), PlotStyle(usetex=False), "by_lambda")
    heights = [float(np.ravel(ax.lines[-1].get_ydata())[0]) for ax in figure.axes[:2]]
    assert heights == pytest.approx([1e-2, 10**-4.0])


@pytest.mark.functional
def test_draws_the_maximum_mass_end_of_each_held_out_curve() -> None:
    figure = draw(run(toy_map()), PlotStyle(usetex=False), "by_lambda")
    ends = [len(ax.collections) for ax in figure.axes[:2]]
    assert ends == [2, 2]


@pytest.mark.view
@pytest.mark.parametrize("variant", ["by_lambda", "by_beta"])
def test_figure_tex_includes_the_picked_variant(variant: VariantName) -> None:
    assert f"{{62_neutron_stars_fig_errors_{variant}}}" in figure_tex(variant)


@pytest.fixture
def written(tmp_path: Path) -> tuple[Path, Path]:
    """The state and asset folders after main ran on the toy table and surrogates."""
    state, assets = tmp_path / "state", tmp_path / "assets"
    table().to_parquet(tmp_path / "ns.parquet")
    split().to_parquet(tmp_path / "split.parquet")
    pipeline = state / "41_algorithm"
    save_surrogate(pipeline / "neutron_stars-mass.pkl", toy_surrogate("mass", total, 1.0, 2.0))
    save_surrogate(pipeline / "neutron_stars-charge.pkl", toy_surrogate("charge", flat, 1.5, 4.0))
    reachable = state / "91_neutron_stars"
    reachable.mkdir(parents=True)
    (reachable / "reachable_figures.json").write_text(json.dumps({"mass": 2.0, "charge": 4.0}))
    main(
        [str(tmp_path / "ns.parquet"), str(tmp_path / "split.parquet"), str(state), str(assets)],
        config=PaperConfig.model_validate({"plot": {"usetex": False}}),
        section=NeutronStarSection(),
    )
    return state, assets


@pytest.mark.functional
def test_main_writes_the_figure_wrapper(written: tuple[Path, Path]) -> None:
    assert (written[1] / "62_neutron_stars" / "62_neutron_stars_fig_errors.tex").is_file()


@pytest.mark.functional
def test_main_stores_the_numbers(written: tuple[Path, Path]) -> None:
    stored = json.loads((written[0] / "62_neutron_stars" / "numbers.json").read_text())
    assert stored["resultsNsMassGap"] == "1.0"


@pytest.mark.functional
def test_ns_results_figures_folder_holds_each_variant_png(
    written: tuple[Path, Path], tmp_path: Path
) -> None:
    out = tmp_path / "figures"
    config = PaperConfig.model_validate({"plot": {"usetex": False}})
    main(["--figures", str(out), "--state", str(written[0])], config=config)
    stem = "62_neutron_stars_fig_errors_"
    expected = {f"{stem}by_lambda.png", f"{stem}by_beta.png"}
    assert {p.name for p in (out / "62_neutron_stars" / "figures").iterdir()} == expected


@pytest.mark.model
def test_shipped_section_toml_selects_the_papers_picks() -> None:
    expected = NeutronStarSection.model_validate({"variants": {"errors": "by_lambda"}})
    assert load_section(NeutronStarSection, SECTION_TOML) == expected

"""Facts about the domain figure of Section 4, on tiny synthetic maps and a toy surrogate."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from check_domain import (
    SECTION_TOML,
    DomainSection,
    DomainVariant,
    MapInput,
    draw,
    figure_tex,
    main,
    numbers,
    run,
)

from paperlib.config import PaperConfig, load_section
from paperlib.domain import make_black_hole_domain, make_neutron_star_domain
from paperlib.final import FinalSurrogate, save_domain, save_surrogate
from paperlib.plots import NOTATION_TEX, PlotStyle, apply_style


@pytest.fixture(autouse=True)
def _notation():
    """The style with the paper's notation, as main applies it; restored after each test."""
    with plt.rc_context():
        apply_style(PlotStyle(usetex=False), NOTATION_TEX.read_text())
        yield


def total(x: np.ndarray) -> np.ndarray:
    """The toy prediction: the sum of the inputs of each row (module level, so it pickles)."""
    return x.sum(axis=1)


def toy_surrogate(dataset: str) -> FinalSurrogate:
    """A surrogate whose prediction is the sum of its inputs."""
    return FinalSurrogate(
        dataset=dataset,
        target="mass",
        family="k-NN",
        unit="pointwise",
        params={},
        test=1.0,
        ceiling=2.0,
        predictor=total,
    )


def black_holes() -> pd.DataFrame:
    """Three beta curves on an r_h grid of step 0.5 up to 3.0, each starting later in r_h."""
    rows = [
        {"beta": b, "r_h": 1.0 + 0.5 * (b - 1) + 0.5 * i, "M": 0.5 * b}
        for b in (1.0, 2.0, 3.0)
        for i in range(5)
        if 1.0 + 0.5 * (b - 1) + 0.5 * i <= 3.0
    ]
    return pd.DataFrame(rows)


def neutron_stars() -> pd.DataFrame:
    """Four curves of five rho_c rows on the corners of a beta, lambda square."""
    rows = [
        {"beta": b, "lambda": lam, "rho_c": 1e14 * 2.0**i, "M": 0.5 + 0.1 * i}
        for b in (0.5, 1.5)
        for lam in (1.0, 2.0)
        for i in range(5)
    ]
    return pd.DataFrame(rows)


def bh_map() -> MapInput:
    table = black_holes()
    return MapInput(
        "black_holes",
        table,
        make_black_hole_domain(table),
        toy_surrogate("black_holes"),
        np.arange(len(table)) % 5 == 4,
    )


def ns_map() -> MapInput:
    table = neutron_stars()
    return MapInput(
        "neutron_stars",
        table,
        make_neutron_star_domain(table),
        toy_surrogate("neutron_stars"),
        np.arange(len(table)) % 5 == 4,
    )


@pytest.mark.functional
def test_a_probe_outside_the_toy_domain_is_drawn_as_refused() -> None:
    found = run((bh_map(),))
    figure = draw(found, PlotStyle(usetex=False), "planes")
    refused = [c for c in figure.axes[0].collections if c.get_label() == "refused"]
    outside = [p for p in found[0].probes if not p.answered]
    assert (len(outside) > 0, np.shape(refused[0].get_offsets())[0]) == (True, len(outside))


@pytest.mark.functional
def test_each_refused_probe_has_an_arrow_to_its_nearest_valid_input() -> None:
    found = run((bh_map(),))
    figure = draw(found, PlotStyle(usetex=False), "planes")
    assert len(figure.axes[0].texts) == sum(not p.answered for p in found[0].probes)


@pytest.mark.functional
def test_planes_no_arrows_draws_no_arrow() -> None:
    figure = draw(run((bh_map(), ns_map())), PlotStyle(usetex=False), "planes_no_arrows")
    assert sum(len(ax.texts) for ax in figure.axes) == 0


@pytest.mark.functional
@pytest.mark.parametrize("variant", ["planes", "planes_no_arrows"])
def test_the_legend_is_one_figure_legend_below_every_panel(variant: DomainVariant) -> None:
    figure = draw(run((bh_map(), ns_map())), PlotStyle(usetex=False), variant)
    figure.canvas.draw()
    top = (
        figure.legends[0].get_window_extent().transformed(figure.transFigure.inverted()).y1
        if figure.legends
        else 1.0
    )
    assert (
        sum(ax.get_legend() is not None for ax in figure.axes),
        len(figure.legends),
        top < min(ax.get_position().y0 for ax in figure.axes),
    ) == (0, 1, True)


@pytest.mark.model
def test_every_test_row_of_a_map_is_inside_its_domain() -> None:
    macros = numbers(run((bh_map(), ns_map())))
    assert (macros["validityBhTestInside"], macros["validityNsTestInside"]) == ("100", "100")


@pytest.mark.model
def test_the_neutron_star_curve_probes_cross_the_stable_branch() -> None:
    found = run((ns_map(),))
    assert {p.answered for p in found[0].curve_probes} == {True, False}


@pytest.mark.view
def test_figure_tex_includes_the_picked_variant() -> None:
    assert "{50_validity_fig_domain_planes}" in figure_tex("planes")


@pytest.fixture
def written(tmp_path: Path) -> tuple[Path, Path]:
    """The state and asset folders after main ran on toy tables, domains and surrogates."""
    state, assets = tmp_path / "state", tmp_path / "assets"
    bh, ns = black_holes(), neutron_stars()
    bh.to_parquet(tmp_path / "bh.parquet")
    ns.to_parquet(tmp_path / "ns.parquet")
    split = pd.DataFrame(
        [
            {"dataset": "black_holes", "beta": b, "lambda": np.nan, "label": "test", "ablation": 0}
            for b in (1.0, 2.0, 3.0)
        ]
        + [
            {"dataset": "neutron_stars", "beta": b, "lambda": lam, "label": "test", "ablation": 0}
            for b in (0.5, 1.5)
            for lam in (1.0, 2.0)
        ]
    )
    split.to_parquet(tmp_path / "split.parquet")
    pipeline = state / "41_algorithm"
    for name, table, make in (
        ("black_holes", bh, make_black_hole_domain),
        ("neutron_stars", ns, make_neutron_star_domain),
    ):
        save_domain(pipeline / f"{name}-domain.json", make(table))
        save_surrogate(pipeline / f"{name}-mass.pkl", toy_surrogate(name))
    config = PaperConfig.model_validate({"plot": {"usetex": False}})
    main(
        [
            str(tmp_path / "ns.parquet"),
            str(tmp_path / "bh.parquet"),
            str(tmp_path / "split.parquet"),
            str(state),
            str(assets),
        ],
        config=config,
        section=DomainSection(),
    )
    return state, assets


@pytest.mark.functional
def test_main_writes_the_figure_wrapper(written: tuple[Path, Path]) -> None:
    assert (written[1] / "50_validity" / "50_validity_fig_domain.tex").is_file()


@pytest.mark.functional
def test_main_stores_the_numbers(written: tuple[Path, Path]) -> None:
    stored = json.loads((written[0] / "50_validity" / "numbers.json").read_text())
    assert stored["validityBhTestInside"] == "100"


@pytest.mark.functional
def test_domain_figures_folder_holds_each_variant_png(
    written: tuple[Path, Path], tmp_path: Path
) -> None:
    out = tmp_path / "figures"
    config = PaperConfig.model_validate({"plot": {"usetex": False}})
    main(["--figures", str(out), "--state", str(written[0])], config=config)
    expected = {"50_validity_fig_domain_planes.png", "50_validity_fig_domain_planes_no_arrows.png"}
    assert {p.name for p in (out / "50_validity" / "figures").iterdir()} == expected


@pytest.mark.model
def test_shipped_section_toml_selects_the_papers_picks() -> None:
    expected = DomainSection.model_validate({"variants": {"domain": "planes"}})
    assert load_section(DomainSection, SECTION_TOML) == expected, (
        f"{SECTION_TOML} drifted from the paper's picks"
    )

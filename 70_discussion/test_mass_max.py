"""Facts about the predicted maximum mass of a held-out curve, on tiny synthetic curves."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mass_max import (
    SECTION_TOML,
    DiscussionSection,
    MapInput,
    main,
    numbers,
    predicted_mass_max,
    run,
)

from paperlib.config import PaperConfig, load_section
from paperlib.final import FinalSurrogate, save_surrogate

# Three training curves (fold labels) and one test curve at the middle of the edge between the
# first two. Each curve ends at its stable edge: 100, 1000 and 100 for the training curves.
CURVES = (
    (0.0, 0.0, "fold0", [10.0, 100.0], [1.0, 2.0]),
    (2.0, 0.0, "fold1", [10.0, 1000.0], [1.0, 3.0]),
    (0.0, 2.0, "fold0", [10.0, 100.0], [1.0, 2.0]),
    (1.0, 0.0, "test", [10.0, 500.0], [1.0, 2.5]),
)


def toy_table() -> pd.DataFrame:
    """One row per (curve, rho_c), the columns the prepared neutron-star table has."""
    rows = [
        {"beta": beta, "lambda": lam, "rho_c": rho, "M": mass}
        for beta, lam, _, rhos, masses in CURVES
        for rho, mass in zip(rhos, masses, strict=True)
    ]
    return pd.DataFrame(rows)


def toy_split() -> pd.DataFrame:
    """The split file's rows: one per curve and dataset, with the curve's label."""
    return pd.DataFrame(
        [
            {"dataset": "neutron_stars", "beta": beta, "lambda": lam, "label": label}
            for beta, lam, label, _, _ in CURVES
        ]
    )


def rho_back(x: np.ndarray) -> np.ndarray:
    """The toy mass predictor: its rho_c input, undone from the log10 the design gives it."""
    return 10.0 ** x[:, 0]


@pytest.mark.model
def test_predicted_mass_max_is_the_mass_at_the_interpolated_edge():
    result = predicted_mass_max(toy_table(), toy_split(), rho_back)
    assert result.curves[0].predicted == pytest.approx(550.0, rel=1e-4)


def tenth_over(x: np.ndarray) -> np.ndarray:
    """The toy mass predictor that overshoots the test curve's maximum mass (2.5) by a tenth."""
    return np.full(len(x), 2.75)


def toy_numbers() -> dict[str, str]:
    """The macros of Section 6 on the toy curves."""
    return numbers(run(MapInput(toy_table(), toy_split(), tenth_over, 2.5)))


@pytest.mark.model
def test_run_numbers_hold_the_p95_relative_error():
    assert toy_numbers()["discussionMassPNinetyFive"] == "0.100"


@pytest.mark.model
def test_run_numbers_hold_the_p95_in_significant_figures():
    assert toy_numbers()["discussionMassPNinetyFiveDigits"] == "1.0"


@pytest.mark.model
def test_run_numbers_hold_the_ceiling_beside_it():
    assert toy_numbers()["discussionMassCeiling"] == "2.5"


@pytest.mark.model
def test_run_numbers_hold_the_curves_scored():
    assert toy_numbers()["discussionMassCurves"] == "1"


@pytest.mark.functional
def test_mass_max_figures_folder_holds_each_variant_png(tmp_path: Path) -> None:
    state = tmp_path / "state"
    toy_table().to_parquet(tmp_path / "ns.parquet")
    toy_split().to_parquet(tmp_path / "split.parquet")
    surrogate = FinalSurrogate(
        dataset="neutron_stars",
        target="mass",
        family="local RBF",
        unit="curve-wise",
        params={},
        test=1.0,
        ceiling=2.5,
        predictor=tenth_over,
    )
    save_surrogate(state / "41_algorithm" / "neutron_stars-mass.pkl", surrogate)
    (state / "91_neutron_stars").mkdir(parents=True)
    (state / "91_neutron_stars" / "reachable_figures.json").write_text(json.dumps({"mass": 2.5}))
    config = PaperConfig.model_validate({"plot": {"usetex": False}})
    paths = [str(tmp_path / "ns.parquet"), str(tmp_path / "split.parquet"), str(state)]
    main([*paths, str(tmp_path / "assets")], config=config, section=DiscussionSection())
    out = tmp_path / "figures"
    main(["--figures", str(out), "--state", str(state)], config=config)
    stem = "70_discussion_fig_mass_max_"
    expected = {f"{stem}parity.png", f"{stem}map.png"}
    assert {p.name for p in (out / "70_discussion" / "figures").iterdir()} == expected


@pytest.mark.model
def test_shipped_section_toml_selects_the_papers_picks() -> None:
    expected = DiscussionSection.model_validate({"variants": {"mass_max": "parity"}})
    assert load_section(DiscussionSection, SECTION_TOML) == expected

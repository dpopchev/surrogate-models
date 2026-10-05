"""Facts about the shared plot style and its per-dataset colors."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import seaborn as sns
from matplotlib.colors import to_hex
from pydantic import ValidationError

from shared.plots import BlackHoleColors, PlotStyle, anchor_color, apply_style, colormap

STYLE = PlotStyle(usetex=False, font_size=9.0, text_width_pt=144.54)


@pytest.fixture(autouse=True)
def _restore_rcparams():
    with plt.rc_context():
        yield


def test_style_sets_usetex() -> None:
    apply_style(STYLE)
    assert plt.rcParams["text.usetex"] is False


def test_style_sets_a_serif_font() -> None:
    apply_style(STYLE)
    assert plt.rcParams["font.family"] == ["serif"]


def test_style_sets_the_font_size() -> None:
    apply_style(STYLE)
    assert plt.rcParams["font.size"] == 9.0


def test_figure_width_is_the_text_width_in_inches() -> None:
    apply_style(STYLE)
    assert plt.rcParams["figure.figsize"][0] == pytest.approx(2.0)


def test_figure_height_follows_the_aspect() -> None:
    apply_style(STYLE)
    assert plt.rcParams["figure.figsize"][1] == pytest.approx(2.0 * 0.62)


def test_color_cycle_is_the_seaborn_palette() -> None:
    apply_style(STYLE)
    cycle = [to_hex(c) for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]]
    assert cycle == sns.color_palette("colorblind").as_hex()


def test_colormap_resolves_a_seaborn_name() -> None:
    assert colormap("crest").name == "crest"


def test_colormap_resolves_a_matplotlib_name() -> None:
    assert colormap("Oranges").name == "Oranges"


def test_ns_anchor_is_the_first_palette_color() -> None:
    assert anchor_color(STYLE, "neutron_stars") == sns.color_palette("colorblind")[0]


def test_bh_anchor_is_the_second_palette_color() -> None:
    assert anchor_color(STYLE, "black_holes") == sns.color_palette("colorblind")[1]


def test_figures_are_written_at_600_dpi_by_default() -> None:
    assert PlotStyle().dpi == 600


def test_a_non_positive_dpi_is_rejected() -> None:
    with pytest.raises(ValidationError, match="dpi"):
        PlotStyle.model_validate({"dpi": 0})


def test_unknown_palette_is_rejected() -> None:
    with pytest.raises(ValidationError, match="palette"):
        PlotStyle.model_validate({"palette": "rainbow"})


def test_bh_colors_refuse_a_cool_colormap() -> None:
    with pytest.raises(ValidationError, match="beta_cmap"):
        BlackHoleColors.model_validate({"beta_cmap": "crest"})

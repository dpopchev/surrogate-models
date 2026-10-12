"""Facts about the shared plot style and its per-dataset colors."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from pydantic import ValidationError

from shared.plots import (
    SYMBOLS,
    BlackHoleColors,
    Marks,
    PlotStyle,
    anchor_color,
    apply_style,
    colormap,
    contact_sheet,
    figure_size,
    make_symbols,
    write_variant_sheet,
)

# A 2 in wide, 3 in tall text block.
STYLE = PlotStyle(usetex=False, font_size=9.0, text_width_pt=144.54, text_height_pt=216.81)


@pytest.fixture(autouse=True)
def _restore_rcparams():
    with plt.rc_context():
        yield


@pytest.mark.view
def test_contact_sheet_places_one_labelled_panel_per_image(tmp_path: Path) -> None:
    images = {label: tmp_path / f"{label}.png" for label in ("inset", "departure")}
    for path in images.values():
        plt.imsave(path, np.zeros((4, 6)))
    sheet = contact_sheet(images)
    assert [axes.get_title() for axes in sheet.axes] == ["inset", "departure"]


@pytest.mark.view
def test_variant_sheet_writes_one_png_per_variant_and_the_sheet(tmp_path: Path) -> None:
    def drawer() -> Figure:
        return plt.figure(figsize=(1, 1))

    write_variant_sheet(tmp_path, "toy", {"one": drawer, "two": drawer}, STYLE)
    written = sorted(path.name for path in tmp_path.glob("*.png"))
    assert written == ["toy_one.png", "toy_two.png", "toy_variants.png"]


@pytest.mark.model
def test_make_symbols_refuses_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown symbols"):
        make_symbols(("M", "no_such_column"))


@pytest.mark.view
def test_a_symbol_lookup_without_a_preamble_raises_lookup_error() -> None:
    plt.rcParams["text.usetex"] = False
    plt.rcParams["text.latex.preamble"] = ""
    with pytest.raises(LookupError, match="apply_style"):
        SYMBOLS["D"]


@pytest.mark.view
def test_style_hands_the_notation_to_latex() -> None:
    apply_style(STYLE, notation="\\newcommand{\\Dch}{D}")
    assert plt.rcParams["text.latex.preamble"] == "\\newcommand{\\Dch}{D}"


@pytest.mark.view
def test_a_symbol_is_the_paper_s_macro_under_usetex() -> None:
    apply_style(PlotStyle(usetex=True), notation="\\newcommand{\\Dch}{D}")
    assert SYMBOLS["D"] == "$\\Dch$"


@pytest.mark.view
def test_symbols_name_the_maximum_mass() -> None:
    apply_style(PlotStyle(usetex=True), notation="\\newcommand{\\Mmax}{M_{\\max}}")
    assert SYMBOLS["M_max"] == "$\\Mmax$"


@pytest.mark.view
def test_a_symbol_expands_to_its_notation_when_latex_is_off() -> None:
    # Mathtext knows no macros: the notation's \newcommand bodies replace them.
    apply_style(STYLE, notation="\\newcommand{\\Dch}{D}\n\\newcommand{\\rhoc}{\\rho_c}")
    assert (SYMBOLS["D"], SYMBOLS["rho_c"]) == ("$D$", "$\\rho_c$")


@pytest.mark.view
def test_style_sets_usetex() -> None:
    apply_style(STYLE)
    assert plt.rcParams["text.usetex"] is False


@pytest.mark.view
def test_style_sets_a_serif_font() -> None:
    apply_style(STYLE)
    assert plt.rcParams["font.family"] == ["serif"]


@pytest.mark.view
def test_style_sets_the_font_size() -> None:
    apply_style(STYLE)
    assert plt.rcParams["font.size"] == 9.0


@pytest.mark.view
def test_figure_width_is_the_text_width_in_inches() -> None:
    apply_style(STYLE)
    assert plt.rcParams["figure.figsize"][0] == pytest.approx(2.0)


@pytest.mark.view
def test_figure_height_follows_the_aspect() -> None:
    apply_style(STYLE)
    assert plt.rcParams["figure.figsize"][1] == pytest.approx(2.0 * 0.62)


@pytest.mark.view
def test_a_multi_row_figure_is_capped_at_a_share_of_the_text_height() -> None:
    # Four rows at 0.42 of the width each would be 3.36 in; the cap is 0.85 of 3 in.
    assert figure_size(STYLE, rows=4, per_row=0.42)[1] == pytest.approx(0.85 * 3.0)


@pytest.mark.view
def test_color_cycle_is_the_seaborn_palette() -> None:
    apply_style(STYLE)
    cycle = [to_hex(c) for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]]
    assert cycle == sns.color_palette("colorblind").as_hex()


@pytest.mark.view
def test_colormap_resolves_a_seaborn_name() -> None:
    assert colormap("crest").name == "crest"


@pytest.mark.view
def test_colormap_resolves_a_matplotlib_name() -> None:
    assert colormap("Oranges").name == "Oranges"


@pytest.mark.view
def test_ns_anchor_is_the_first_palette_color() -> None:
    assert anchor_color(STYLE, "neutron_stars") == sns.color_palette("colorblind")[0]


@pytest.mark.view
def test_bh_anchor_is_the_second_palette_color() -> None:
    assert anchor_color(STYLE, "black_holes") == sns.color_palette("colorblind")[1]


@pytest.mark.model
def test_figures_are_written_at_600_dpi_by_default() -> None:
    assert PlotStyle().dpi == 600


@pytest.mark.model
def test_plot_style_reads_the_grid_values_and_diagnostics_dpi() -> None:
    keys = {"grid_values": 250, "univariate_grid_values": 60, "diagnostics_dpi": 150}
    style = PlotStyle.model_validate({**keys, "live_every": 5, "live_dpi": 100})
    read = (style.grid_values, style.univariate_grid_values, style.diagnostics_dpi)
    assert (*read, style.live_every, style.live_dpi) == (250, 60, 150, 5, 100)


@pytest.mark.model
def test_a_preset_scales_the_marks() -> None:
    style = PlotStyle(presets={"large": Marks(mark_scale=2.0)}, preset="large")
    assert (style.marks.mark_scale, PlotStyle().marks) == (2.0, Marks())


@pytest.mark.model
def test_an_unknown_preset_is_refused() -> None:
    with pytest.raises(ValidationError, match="preset"):
        PlotStyle(preset="none")


@pytest.mark.model
def test_a_non_positive_dpi_is_rejected() -> None:
    with pytest.raises(ValidationError, match="dpi"):
        PlotStyle.model_validate({"dpi": 0})


@pytest.mark.model
def test_unknown_palette_is_rejected() -> None:
    with pytest.raises(ValidationError, match="palette"):
        PlotStyle.model_validate({"palette": "rainbow"})


@pytest.mark.model
def test_bh_colors_refuse_a_cool_colormap() -> None:
    with pytest.raises(ValidationError, match="beta_cmap"):
        BlackHoleColors.model_validate({"beta_cmap": "crest"})

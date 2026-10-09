"""One paper-wide matplotlib and seaborn style, with a color family per dataset.

NS figures use cool colormaps and BH figures warm ones (E-003 Decision), so a colorbar alone
tells which dataset is shown. The style is a PlotStyle, chosen in paper.toml's [plot] section.
"""

from typing import Literal, assert_never

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Colormap
from pydantic import BaseModel, ConfigDict, PositiveInt

# --- vocabulary and types ---------------------------------------------------------------------

Dataset = Literal["neutron_stars", "black_holes"]

# The two families are disjoint: no NS colormap is a BH colormap.
CoolCmap = Literal["crest", "mako", "Blues", "GnBu", "PuBu"]
WarmCmap = Literal["flare", "rocket", "Oranges", "YlOrRd", "OrRd"]

# 11pt a4paper article text width and height, measured with \showthe\textwidth (W-014
# Finding) and \showthe\textheight (W-097).
ARTICLE_TEXT_WIDTH_PT = 360.0
ARTICLE_TEXT_HEIGHT_PT = 595.80026
POINTS_PER_INCH = 72.27


class NeutronStarColors(BaseModel):
    """Colormaps for the NS free parameters, and the palette index of the NS anchor color."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lambda_cmap: CoolCmap = "crest"
    beta_cmap: CoolCmap = "mako"
    anchor: int = 0


class BlackHoleColors(BaseModel):
    """Colormap for the BH free parameter, and the palette index of the BH anchor color."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    beta_cmap: WarmCmap = "flare"
    anchor: int = 1


class PlotStyle(BaseModel):
    """The look every figure shares: seaborn context and palette, LaTeX text, article width."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    context: Literal["paper", "notebook", "talk"] = "paper"
    palette: Literal["colorblind", "deep", "muted"] = "colorblind"
    font_size: float = 11.0
    usetex: bool = True
    text_width_pt: float = ARTICLE_TEXT_WIDTH_PT
    text_height_pt: float = ARTICLE_TEXT_HEIGHT_PT
    aspect: float = 0.62
    # Figures are written as PNG at this resolution (W-029).
    dpi: PositiveInt = 600
    neutron_stars: NeutronStarColors = NeutronStarColors()
    black_holes: BlackHoleColors = BlackHoleColors()


# --- behaviour --------------------------------------------------------------------------------


def apply_style(style: PlotStyle, notation: str = "") -> None:
    """Set the seaborn theme, then the matplotlib rcParams the style fixes; `notation` is the
    paper's macro file (00_metadata/notation.tex), handed to LaTeX as the preamble so a label
    may use the macros the captions use (W-098)."""
    width = style.text_width_pt / POINTS_PER_INCH
    sns.set_theme(context=style.context, style="ticks", palette=style.palette)
    plt.rcParams.update(
        {
            "text.usetex": style.usetex,
            "text.latex.preamble": notation,
            "font.family": "serif",
            "font.size": style.font_size,
            "figure.figsize": (width, width * style.aspect),
        }
    )


def figure_size(
    style: PlotStyle, rows: int, per_row: float, max_share: float = 0.85
) -> tuple[float, float]:
    """The size in inches of a figure of `rows` rows, each `per_row` of the text width tall,
    capped at `max_share` of the text height so it fits a page."""
    width = style.text_width_pt / POINTS_PER_INCH
    height = style.text_height_pt / POINTS_PER_INCH
    return width, min(per_row * width * rows, max_share * height)


def colormap(name: CoolCmap | WarmCmap) -> Colormap:
    """Return the named seaborn or matplotlib colormap."""
    return sns.color_palette(name, as_cmap=True)


def anchor_color(style: PlotStyle, dataset: Dataset) -> tuple[float, float, float]:
    """Return the dataset's single color for marks that carry no parameter (histograms)."""
    match dataset:
        case "neutron_stars":
            index = style.neutron_stars.anchor
        case "black_holes":
            index = style.black_holes.anchor
        case _:
            assert_never(dataset)
    return sns.color_palette(style.palette)[index]

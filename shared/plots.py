"""One paper-wide matplotlib and seaborn style, with a color family per dataset.

NS figures use cool colormaps and BH figures warm ones (E-003 Decision), so a colorbar alone
tells which dataset is shown. The style is a PlotStyle, chosen in paper.toml's [plot] section.
"""

import re
from collections.abc import Iterator, Mapping
from pathlib import Path
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


# Every symbol a figure or a table names, by column name, in the paper's notation
# (00_metadata/notation.tex). Tables are LaTeX and take the macros as they are (MACROS); a
# figure reads SYMBOLS, which hands the macro to usetex and, when LaTeX is off (mathtext, the
# tests), the macro's body from the notation apply_style received.
MACROS: Mapping[str, str] = {
    "rho_c": "$\\rhoc$",
    "M": "$M$",
    "D": "$\\Dch$",
    "D_over_M": "$\\Dch/M$",
    "log10_D": "$\\log_{10}\\Dch$",
    "log10_D_over_M": "$\\log_{10}(\\Dch/M)$",
    "log10_rho_c": "$\\log_{10}\\rhoc$",
    "beta": "$\\beta$",
    "lambda": "$\\lambda$",
    "r_h": "$\\rh$",
    "Msun": "$\\Msun$",
    "Y": "$Y = \\log_{10}(\\Dch/M)$",
}


# The paper's notation macros: input by the preamble, handed to LaTeX by apply_style, read by
# SYMBOLS when LaTeX is off.
NOTATION_TEX = Path(__file__).resolve().parents[1] / "00_metadata" / "notation.tex"


class Symbols(Mapping[str, str]):
    """MACROS, or the named subset of them in that order, as a figure label reads them at the
    time of the lookup: the macro under usetex, else its body from the notation apply_style
    received, or from the notation file before any style was applied (mathtext knows no
    macros)."""

    def __init__(self, names: tuple[str, ...] = tuple(MACROS)) -> None:
        assert all(name in MACROS for name in names), f"unknown symbols: {names}"
        self._names = names

    def __getitem__(self, key: str) -> str:
        if key not in self._names:
            raise KeyError(key)
        macro = MACROS[key]
        if plt.rcParams["text.usetex"]:
            return macro
        notation = str(plt.rcParams["text.latex.preamble"]) or NOTATION_TEX.read_text()
        return expand(macro, notation)

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def of(self, *names: str) -> Symbols:
        """The symbols of these columns, in this order."""
        return Symbols(names)


SYMBOLS = Symbols()

_NEWCOMMAND = re.compile(r"\\newcommand\{\\(\w+)\}\{(.*)\}")


def expand(text: str, notation: str) -> str:
    """The text with every argument-free \\newcommand of the notation replaced by its body."""
    for name, body in _NEWCOMMAND.findall(notation):
        text = re.sub(rf"\\{name}(?![A-Za-z])", lambda _, body=body: body, text)
    return text


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

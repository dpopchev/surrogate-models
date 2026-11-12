"""The error figure of Section 5.2 and its numbers (W-155).

The saved neutron-star surrogates of the mass and the charge score every held-out row
(paperlib/final.py held_out_errors). The figure draws, per target, the relative error of those
rows against log10 rho_c beside the data ceiling of the target; the numbers are the p95 test and
ceiling figures, their gap and the family and unit of the winner. Nothing is fitted.
"""

import argparse
import json
import logging
import pickle
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict

from paperlib.config import NOTATION_TEX, PaperConfig, load_config, load_section
from paperlib.design import NEUTRON_STARS, TARGETS, Pair, Target, pairs, target_symbol
from paperlib.eda import text_width_size
from paperlib.final import FinalSurrogate, HeldOut, held_out_errors, load_surrogate
from paperlib.plots import SYMBOLS, PlotStyle, apply_style, colormap, line_width, mark_size

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

# by_lambda: the lines colored by lambda (the pick); by_beta: colored by beta.
VariantName = Literal["by_lambda", "by_beta"]


class ErrorVariants(BaseModel):
    """Section 5.2: which variant of the error figure the paper places."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    errors: VariantName = "by_lambda"


class NeutronStarSection(BaseModel):
    """Section 5.2's own settings: the picked variant of its figure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    variants: ErrorVariants = ErrorVariants()


SECTION = "62_neutron_stars"
SECTION_TOML = Path(__file__).with_name("62_neutron_stars.toml")
PIPELINE = "41_algorithm"
REACHABLE = Path("91_neutron_stars") / "reachable_figures.json"

# An error of zero has no place on a log axis: it is drawn at this floor.
FLOOR = 1e-16
TAG = {"mass": "Mass", "charge": "Charge"}
# The columns of a held-out row's inputs: log10 rho_c, beta, lambda.
RHO, BETA, LAMBDA = 0, 1, 2


@dataclass(frozen=True)
class TargetInput:
    """One target as the figure reads it: its pair, its saved surrogate, the true D of every
    design row and the data ceiling in significant figures."""

    pair: Pair
    surrogate: FinalSurrogate
    charge: np.ndarray
    ceiling: float


@dataclass(frozen=True)
class MapInput:
    """The neutron-star map: its targets, the mass first."""

    targets: tuple[TargetInput, ...]


@dataclass(frozen=True)
class TargetEvidence:
    """What one target's panel plots and its numbers cite: the held-out rows scored, the p95 test
    and ceiling figures and the winner's family and unit."""

    target: Target
    held: HeldOut
    test: float
    ceiling: float
    family: str
    unit: str


@dataclass(frozen=True)
class Evidence:
    """What the figure plots and the numbers count."""

    targets: tuple[TargetEvidence, ...]


# --- pure functions ---------------------------------------------------------------------------


def run(found: MapInput) -> Evidence:
    """Every target's held-out rows scored by its saved surrogate."""
    return Evidence(
        tuple(
            TargetEvidence(
                one.pair.target,
                held_out_errors(one.surrogate, one.pair, one.charge),
                one.surrogate.test,
                one.ceiling,
                one.surrogate.family,
                one.surrogate.unit,
            )
            for one in found.targets
        )
    )


def numbers(found: Evidence) -> dict[str, str]:
    """Every number Section 5.2 cites, keyed by its \\resultsNs macro name: per target the p95
    test and ceiling figures, their gap and the family and unit of the winner."""
    macros: dict[str, str] = {}
    for one in found.targets:
        tag = f"resultsNs{TAG[one.target]}"
        macros |= {
            f"{tag}Test": f"{one.test:.1f}",
            f"{tag}Ceiling": f"{one.ceiling:.1f}",
            f"{tag}Gap": f"{one.ceiling - one.test:.1f}",
            f"{tag}Family": one.family,
            f"{tag}Unit": one.unit,
        }
    return macros


# --- figures ----------------------------------------------------------------------------------

CAPTIONS: Mapping[VariantName, str] = {
    "by_lambda": (
        "Relative error of the saved neutron-star surrogates on the held-out curves against "
        "$\\log_{10}\\rhoc$, for the mass (left) and the dilaton charge (right): one line per "
        "held-out $(\\beta, \\lambda)$ curve colored by $\\lambda$, the dotted line the data "
        "ceiling of the target, the dot the maximum-mass end of the curve."
    ),
    "by_beta": (
        "Relative error of the saved neutron-star surrogates on the held-out curves against "
        "$\\log_{10}\\rhoc$, for the mass (left) and the dilaton charge (right): one line per "
        "held-out $(\\beta, \\lambda)$ curve colored by $\\beta$, the dotted line the data "
        "ceiling of the target, the dot the maximum-mass end of the curve."
    ),
}


def asset() -> str:
    """The stem one figure's image and its .tex wrapper share."""
    return f"{SECTION}_fig_errors"


def picked_stem(variant: VariantName) -> str:
    """The stem of the picked variant's PNG under build/sections/<section>/figures/."""
    return f"{asset()}_{variant}"


def figure_tex(variant: VariantName) -> str:
    """The figure environment of the error figure in its picked variant."""
    return (
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{picked_stem(variant)}}}\n"
        f"\\caption{{{CAPTIONS[variant]}}}\n\\label{{fig:results-ns-errors}}\n\\end{{figure}}\n"
    )


def _colored_by(variant: VariantName) -> tuple[int, str]:
    """The input column that colors the lines of a variant and its symbol."""
    match variant:
        case "by_lambda":
            return LAMBDA, "lambda"
        case "by_beta":
            return BETA, "beta"
        case _:
            assert_never(variant)


def _cmap(style: PlotStyle, name: str) -> Colormap:
    """The cool colormap of the colored parameter."""
    return colormap(
        style.neutron_stars.lambda_cmap if name == "lambda" else style.neutron_stars.beta_cmap
    )


def _panel(
    ax: Axes, one: TargetEvidence, style: PlotStyle, variant: VariantName, norm: Normalize
) -> None:
    """One line per held-out curve against log10 rho_c, the maximum-mass end of each marked, and
    the data ceiling of the target."""
    column, name = _colored_by(variant)
    cmap = _cmap(style, name)
    rho = one.held.inputs[:, RHO]
    for curve in np.unique(one.held.curves):
        rows = np.flatnonzero(one.held.curves == curve)
        rows = rows[np.argsort(rho[rows])]
        color = cmap(norm(float(one.held.inputs[rows[0], column])))
        ax.plot(
            rho[rows],
            np.maximum(one.held.errors[rows], FLOOR),
            color=color,
            lw=line_width(style, 1.0),
        )
        ax.scatter(
            [float(rho[rows[-1]])],
            [float(max(one.held.errors[rows[-1]], FLOOR))],
            s=mark_size(style, 10.0),
            color=color,
            zorder=3,
        )
    ax.set_yscale("log")
    ax.set_xlabel(SYMBOLS["log10_rho_c"])
    ax.set_ylabel(f"relative error of {SYMBOLS[target_symbol(one.target)]}")
    ax.axhline(10.0**-one.ceiling, color="0.25", ls=":", lw=line_width(style, 1.0))


def draw(found: Evidence, style: PlotStyle, variant: VariantName) -> Figure:
    """One panel per target, the mass first, and one colorbar of the colored parameter."""
    fig, axes = plt.subplots(
        1, len(found.targets), figsize=text_width_size(0.45), squeeze=False, layout="constrained"
    )
    column, name = _colored_by(variant)
    values = np.concatenate([t.held.inputs[:, column] for t in found.targets])
    norm = Normalize(float(values.min()), float(values.max()))
    for ax, one in zip(axes[0], found.targets, strict=True):
        _panel(ax, one, style, variant, norm)
        ax.set_title(f"neutron stars, {SYMBOLS[target_symbol(one.target)]}")
    mappable = ScalarMappable(norm=norm, cmap=_cmap(style, name))
    fig.colorbar(mappable, ax=list(axes[0]), label=SYMBOLS[name])
    return fig


# --- shell ------------------------------------------------------------------------------------


def _arguments(argv: list[str]) -> argparse.Namespace:
    """The prepared neutron-star table, the split, the state folder and the asset folder."""
    parser = argparse.ArgumentParser(prog="results_neutron_stars.py")
    for name in ("ns", "split", "state", "assets"):
        parser.add_argument(name, type=Path)
    return parser.parse_args(argv)


def _figure_arguments(argv: list[str]) -> argparse.Namespace:
    """The figures mode: the out dir and the state dir the main run stored its data in."""
    parser = argparse.ArgumentParser(prog="results_neutron_stars.py")
    parser.add_argument("--figures", type=Path, required=True, help="draw every variant here")
    parser.add_argument("--state", type=Path, required=True, help="the state dir of the main run")
    return parser.parse_args(argv)


def _figure_data_path(state: Path) -> Path:
    """Where the main run stores what the figure plots, machine-local interim."""
    return state / SECTION / "figures.pkl"


def _inputs(args: argparse.Namespace) -> MapInput:
    """Read the neutron-star map: its table and split, its saved surrogates, the data ceilings
    recorded by the tuning."""
    table = pd.read_parquet(args.ns)
    saved = args.state / PIPELINE
    ceilings = json.loads((args.state / REACHABLE).read_text())
    found = pairs(
        {NEUTRON_STARS.dataset: table}, pd.read_parquet(args.split), (NEUTRON_STARS,), TARGETS
    )
    return MapInput(
        tuple(
            TargetInput(
                pair,
                load_surrogate(saved / f"{pair.dataset}-{pair.target}.pkl"),
                table["D"].to_numpy(np.float64),
                float(ceilings[pair.target]),
            )
            for pair in found
        )
    )


def _store(state: Path, macros: Mapping[str, str]) -> None:
    """Store the numbers under the section's state folder, for paperlib/tools/render_assets.py."""
    (state / SECTION).mkdir(parents=True, exist_ok=True)
    (state / SECTION / "numbers.json").write_text(json.dumps(dict(macros)))


def main(
    argv: Sequence[str],
    config: PaperConfig | None = None,
    section: NeutronStarSection | None = None,
) -> None:
    """Write the figure wrapper and store the numbers and the figure data -- or, given
    --figures, draw every variant as its own PNG from the data the main run stored."""
    argv = list(argv)
    config = config or load_config()
    apply_style(config.plot, NOTATION_TEX.read_text())
    if "--figures" in argv:
        figure_args = _figure_arguments(argv)
        found = pickle.loads(_figure_data_path(figure_args.state).read_bytes())
        out = figure_args.figures / SECTION / "figures"
        out.mkdir(parents=True, exist_ok=True)
        for variant in get_args(VariantName):
            fig = draw(found, config.plot, variant)
            fig.savefig(out / f"{picked_stem(variant)}.png", dpi=config.plot.dpi)
            plt.close(fig)
        logger.info("done: %d error figure variants -> %s", len(get_args(VariantName)), out)
        return
    args = _arguments(argv)
    section = section or load_section(NeutronStarSection, SECTION_TOML)
    found = run(_inputs(args))
    out = args.assets / SECTION
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{asset()}.tex").write_text(figure_tex(section.variants.errors))
    _store(args.state, numbers(found))
    _figure_data_path(args.state).write_bytes(pickle.dumps(found))
    logger.info("done: error figure wrapper -> %s, numbers -> %s", out, args.state / SECTION)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])

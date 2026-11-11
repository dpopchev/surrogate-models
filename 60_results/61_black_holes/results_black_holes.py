"""The error figure of Section 5.1 and its numbers (W-154).

The saved black-hole surrogates of the mass and the charge score every held-out row
(paperlib/final.py held_out_errors). The figure draws, per target, the relative error of those
rows beside the data ceiling of the target; the numbers are the p95 test and ceiling figures, their
gap and the family and unit of the winner. Nothing is fitted.
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
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict

from paperlib.config import NOTATION_TEX, PaperConfig, load_config, load_section
from paperlib.design import BLACK_HOLES, TARGETS, Pair, Target, pairs, target_symbol
from paperlib.domain import BlackHoleDomain
from paperlib.eda import text_width_size
from paperlib.final import (
    FinalSurrogate,
    HeldOut,
    held_out_errors,
    load_domain,
    load_surrogate,
)
from paperlib.plots import SYMBOLS, PlotStyle, apply_style, colormap, line_width, mark_size

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

# curves: the error of every held-out curve against r_h, one line each (the pick);
# plane: the error as color over the (beta, r_h) plane.
VariantName = Literal["curves", "plane"]


class ErrorVariants(BaseModel):
    """Section 5.1: which variant of the error figure the paper places."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    errors: VariantName = "curves"


class BlackHoleSection(BaseModel):
    """Section 5.1's own settings: the picked variant of its figure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    variants: ErrorVariants = ErrorVariants()


SECTION = "61_black_holes"
SECTION_TOML = Path(__file__).with_name("61_black_holes.toml")
PIPELINE = "41_algorithm"
REACHABLE = Path("92_black_holes") / "reachable_figures.json"

# An error of zero has no place on a log axis: it is drawn at this floor.
FLOOR = 1e-16
TAG = {"mass": "Mass", "charge": "Charge"}


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
    """The black-hole map: its domain and its targets, the mass first."""

    domain: BlackHoleDomain
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

    domain: BlackHoleDomain
    targets: tuple[TargetEvidence, ...]


# --- pure functions ---------------------------------------------------------------------------


def run(found: MapInput) -> Evidence:
    """Every target's held-out rows scored by its saved surrogate."""
    return Evidence(
        found.domain,
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
        ),
    )


def numbers(found: Evidence) -> dict[str, str]:
    """Every number Section 5.1 cites, keyed by its \\resultsBh macro name: per target the p95
    test and ceiling figures, their gap and the family and unit of the winner."""
    macros: dict[str, str] = {}
    for one in found.targets:
        tag = f"resultsBh{TAG[one.target]}"
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
    "curves": (
        "Relative error of the saved black-hole surrogates on the held-out curves against $\\rh$, "
        "for the mass (left) and the dilaton charge (right): one line per held-out $\\beta$, the "
        "dotted line the data ceiling of the target, the triangle on the axis the existence edge "
        "$r_{h,\\min}(\\beta)$ of the curve."
    ),
    "plane": (
        "Relative error of the saved black-hole surrogates on the held-out curves over the "
        "$(\\beta, \\rh)$ plane, for the mass (left) and the dilaton charge (right): the color is "
        "$\\log_{10}$ of the error of each held-out row, the line the existence edge "
        "$r_{h,\\min}(\\beta)$."
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
        f"\\caption{{{CAPTIONS[variant]}}}\n\\label{{fig:results-bh-errors}}\n\\end{{figure}}\n"
    )


def _curves(
    ax: Axes, found: Evidence, one: TargetEvidence, style: PlotStyle, norm: Normalize
) -> None:
    """One line per held-out curve against r_h, the existence edge of each on the axis, and the
    data ceiling of the target."""
    cmap = colormap(style.black_holes.beta_cmap)
    r_h, beta = one.held.inputs[:, 0], one.held.inputs[:, 1]
    for curve in np.unique(one.held.curves):
        rows = np.flatnonzero(one.held.curves == curve)
        rows = rows[np.argsort(r_h[rows])]
        color = cmap(norm(float(beta[rows[0]])))
        ax.plot(
            r_h[rows],
            np.maximum(one.held.errors[rows], FLOOR),
            color=color,
            lw=line_width(style, 1.0),
        )
        edge = float(np.interp(beta[rows[0]], found.domain.betas, found.domain.r_min))
        ax.scatter(
            [edge],
            [0.0],
            marker="^",
            s=mark_size(style, 14.0),
            color=color,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            zorder=3,
        )
    ax.set_yscale("log")
    ax.set_xlabel(SYMBOLS["r_h"])
    ax.set_ylabel(f"relative error of {SYMBOLS[target_symbol(one.target)]}")
    ax.axhline(10.0**-one.ceiling, color="0.25", ls=":", lw=line_width(style, 1.0))


def _plane(
    ax: Axes, found: Evidence, one: TargetEvidence, style: PlotStyle, norm: Normalize
) -> None:
    """The held-out rows over (beta, r_h), the log10 of their error as color."""
    cmap = colormap(style.black_holes.beta_cmap)
    ax.scatter(
        one.held.inputs[:, 1],
        one.held.inputs[:, 0],
        c=np.log10(np.maximum(one.held.errors, FLOOR)),
        cmap=cmap,
        norm=norm,
        s=mark_size(style, 10.0),
        zorder=2,
    )
    ax.plot(found.domain.betas, found.domain.r_min, color="0.25", lw=line_width(style, 1.0))
    ax.set_xlabel(SYMBOLS["beta"])
    ax.set_ylabel(SYMBOLS["r_h"])


def draw(found: Evidence, style: PlotStyle, variant: VariantName) -> Figure:
    """One panel per target, the mass first, and one colorbar: beta under curves, the log10 of
    the error under plane."""
    fig, axes = plt.subplots(
        1, len(found.targets), figsize=text_width_size(0.45), squeeze=False, layout="constrained"
    )
    match variant:
        case "curves":
            betas = np.concatenate([t.held.inputs[:, 1] for t in found.targets])
            norm = Normalize(float(betas.min()), float(betas.max()))
            label = SYMBOLS["beta"]
            panel = _curves
        case "plane":
            logs = np.log10(
                np.maximum(np.concatenate([t.held.errors for t in found.targets]), FLOOR)
            )
            norm = Normalize(float(logs.min()), float(logs.max()))
            label = "$\\log_{10}$ relative error"
            panel = _plane
        case _:
            assert_never(variant)
    for ax, one in zip(axes[0], found.targets, strict=True):
        panel(ax, found, one, style, norm)
        ax.set_title(f"black holes, {SYMBOLS[target_symbol(one.target)]}")
    mappable = ScalarMappable(norm=norm, cmap=colormap(style.black_holes.beta_cmap))
    fig.colorbar(mappable, ax=list(axes[0]), label=label)
    return fig


# --- shell ------------------------------------------------------------------------------------


def _arguments(argv: list[str]) -> argparse.Namespace:
    """The prepared black-hole table, the split, the state folder and the asset folder."""
    parser = argparse.ArgumentParser(prog="results_black_holes.py")
    for name in ("bh", "split", "state", "assets"):
        parser.add_argument(name, type=Path)
    return parser.parse_args(argv)


def _figure_arguments(argv: list[str]) -> argparse.Namespace:
    """The figures mode: the out dir and the state dir the main run stored its data in."""
    parser = argparse.ArgumentParser(prog="results_black_holes.py")
    parser.add_argument("--figures", type=Path, required=True, help="draw every variant here")
    parser.add_argument("--state", type=Path, required=True, help="the state dir of the main run")
    return parser.parse_args(argv)


def _figure_data_path(state: Path) -> Path:
    """Where the main run stores what the figure plots, machine-local interim."""
    return state / SECTION / "figures.pkl"


def _inputs(args: argparse.Namespace) -> MapInput:
    """Read the black-hole map: its table and split, its saved domain and surrogates, the data
    ceilings recorded by Section 5's tuning."""
    table = pd.read_parquet(args.bh)
    saved = args.state / PIPELINE
    ceilings = json.loads((args.state / REACHABLE).read_text())
    found = pairs(
        {BLACK_HOLES.dataset: table}, pd.read_parquet(args.split), (BLACK_HOLES,), TARGETS
    )
    domain = load_domain(saved / f"{BLACK_HOLES.dataset}-domain.json")
    assert isinstance(domain, BlackHoleDomain)
    return MapInput(
        domain,
        tuple(
            TargetInput(
                pair,
                load_surrogate(saved / f"{pair.dataset}-{pair.target}.pkl"),
                table["D"].to_numpy(np.float64),
                float(ceilings[pair.target]),
            )
            for pair in found
        ),
    )


def _store(state: Path, macros: Mapping[str, str]) -> None:
    """Store the numbers under the section's state folder, for paperlib/tools/render_assets.py."""
    (state / SECTION).mkdir(parents=True, exist_ok=True)
    (state / SECTION / "numbers.json").write_text(json.dumps(dict(macros)))


def main(
    argv: Sequence[str],
    config: PaperConfig | None = None,
    section: BlackHoleSection | None = None,
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
    section = section or load_section(BlackHoleSection, SECTION_TOML)
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

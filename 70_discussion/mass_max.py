"""The maximum neutron-star mass the surrogate predicts for each held-out curve (W-156).

The stable edge in rho_c of a held-out (beta, lambda) is interpolated from the training curves
alone (paperlib/domain.py); the mass surrogate is evaluated there and set against the last row of
the held-out curve, the maximum mass of the data. Keys outside the training keys are skipped and
counted. Nothing is fitted.
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
from paperlib.design import NEUTRON_STARS
from paperlib.domain import make_neutron_star_domain, stable_edge
from paperlib.eda import number_tex, text_width_size
from paperlib.final import load_surrogate
from paperlib.harness import Predictor
from paperlib.plots import SYMBOLS, PlotStyle, apply_style, colormap, line_width, mark_size
from paperlib.tools.render_assets import store_section

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

# parity: the predicted against the data maximum mass colored by lambda (the pick); map: the
# relative error as color over the (beta, lambda) plane.
VariantName = Literal["parity", "map"]


class MassMaxVariants(BaseModel):
    """Section 6: which variant of the maximum-mass figure the paper places."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mass_max: VariantName = "parity"


class DiscussionSection(BaseModel):
    """Section 6's own settings: the picked variant of its figure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    variants: MassMaxVariants = MassMaxVariants()


SECTION = "70_discussion"
SECTION_TOML = Path(__file__).with_name("70_discussion.toml")
PIPELINE = "41_algorithm"
REACHABLE = Path("91_neutron_stars") / "reachable_figures.json"

# An error of zero has no place on a log axis: it is drawn at this floor.
FLOOR = 1e-16


@dataclass(frozen=True)
class CurveMassMax:
    """One held-out curve: its keys, the predicted and the data maximum mass (M_sun) and the
    relative error of the prediction."""

    beta: float
    lam: float
    predicted: float
    data: float
    error: float


@dataclass(frozen=True)
class MassMaxes:
    """The held-out curves with a prediction, and how many were skipped for lying outside the
    training keys."""

    curves: tuple[CurveMassMax, ...]
    skipped: int


@dataclass(frozen=True)
class MapInput:
    """The neutron-star map as the figure reads it: its prepared table, the split, the saved mass
    predictor and the data ceiling of the mass in significant figures."""

    table: pd.DataFrame
    split: pd.DataFrame
    predict: Predictor
    ceiling: float


@dataclass(frozen=True)
class Evidence:
    """What the figure plots and the numbers cite: the maxima and the ceiling."""

    maxes: MassMaxes
    ceiling: float


# --- pure functions ---------------------------------------------------------------------------


def predicted_mass_max(table: pd.DataFrame, split: pd.DataFrame, predict: Predictor) -> MassMaxes:
    """The maximum mass of each held-out curve, read from the surrogate at the interpolated edge.

    `table` is the prepared neutron-star table (beta, lambda, rho_c, M; each curve cut at its
    maximum mass), `split` the frozen split file; `predict` takes the design inputs in the order
    of paperlib.design.NEUTRON_STARS (log10 rho_c, beta, lambda).
    """
    space = NEUTRON_STARS.space
    curve = list(space.curve)
    own = split.loc[split["dataset"] == NEUTRON_STARS.dataset, [*curve, "label"]]
    labelled = table.merge(own, on=curve, how="left", validate="many_to_one")
    training = labelled[labelled["label"].notna() & (labelled["label"] != "test")]
    domain = make_neutron_star_domain(training)
    found: list[CurveMassMax] = []
    skipped = 0
    for (beta, lam), rows in labelled[labelled["label"] == "test"].groupby(curve, sort=True):
        edge = stable_edge(domain, float(beta), float(lam))[1]  # type: ignore[arg-type]
        if np.isnan(edge):
            skipped += 1
            continue
        x = np.array([[np.log10(edge), beta, lam]], dtype=np.float32)
        predicted = float(predict(x)[0])
        data = float(rows["M"].iloc[-1])
        found.append(
            CurveMassMax(float(beta), float(lam), predicted, data, abs(predicted - data) / data)  # type: ignore[arg-type]
        )
    return MassMaxes(curves=tuple(found), skipped=skipped)


def run(found: MapInput) -> Evidence:
    """The maximum mass of each held-out curve, with the data ceiling beside it."""
    return Evidence(predicted_mass_max(found.table, found.split, found.predict), found.ceiling)


def numbers(found: Evidence) -> dict[str, str]:
    """Every number Section 6 cites, keyed by its \\discussionMass macro name: the curves scored
    and skipped, the median and p95 relative error, the p95 in significant figures and the
    ceiling."""
    errors = np.array([c.error for c in found.maxes.curves])
    assert errors.size, "no held-out curve has a predicted maximum mass"
    p95 = float(np.quantile(errors, 0.95))
    return {
        "discussionMassCurves": str(errors.size),
        "discussionMassSkipped": str(found.maxes.skipped),
        "discussionMassMedian": number_tex(float(np.median(errors))),
        "discussionMassPNinetyFive": number_tex(p95),
        "discussionMassPNinetyFiveDigits": f"{-np.log10(max(p95, FLOOR)):.1f}",
        "discussionMassCeiling": f"{found.ceiling:.1f}",
    }


# --- figures ----------------------------------------------------------------------------------

CAPTIONS: Mapping[VariantName, str] = {
    "parity": (
        "The maximum mass of each held-out $(\\beta, \\lambda)$ curve: the saved neutron-star "
        "surrogate at the stable edge interpolated from the training curves, against the data "
        "(top, the diagonal the agreement), colored by $\\lambda$, and its relative error "
        "(bottom), the dotted line the data ceiling of the mass."
    ),
    "map": (
        "The relative error of the predicted maximum mass over the $(\\beta, \\lambda)$ plane "
        "of the held-out curves (top), and against the data maximum mass (bottom), the dotted "
        "line the data ceiling of the mass."
    ),
}


def asset() -> str:
    """The stem one figure's image and its .tex wrapper share."""
    return f"{SECTION}_fig_mass_max"


def picked_stem(variant: VariantName) -> str:
    """The stem of the picked variant's PNG under build/sections/<section>/figures/."""
    return f"{asset()}_{variant}"


def figure_tex(variant: VariantName) -> str:
    """The figure environment of the maximum-mass figure in its picked variant."""
    return (
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{picked_stem(variant)}}}\n"
        f"\\caption{{{CAPTIONS[variant]}}}\n\\label{{fig:discussion-mass-max}}\n\\end{{figure}}\n"
    )


def _cmap(style: PlotStyle, variant: VariantName) -> Colormap:
    """The cool colormap of the colored quantity: lambda for the parity, beta for the map."""
    cool = style.neutron_stars
    return colormap(cool.lambda_cmap if variant == "parity" else cool.beta_cmap)


def _top(ax: Axes, found: Evidence, style: PlotStyle, variant: VariantName) -> None:
    """The upper panel of the variant, with its colorbar."""
    curves = found.maxes.curves
    data = np.array([c.data for c in curves])
    predicted = np.array([c.predicted for c in curves])
    cmap = _cmap(style, variant)
    match variant:
        case "parity":
            color = np.array([c.lam for c in curves])
            both = np.concatenate([data, predicted])
            reach = (float(both.min()), float(both.max()))
            ax.plot(reach, reach, color="0.25", lw=line_width(style, 0.8), ls="--", zorder=1)
            ax.set_xlabel(f"data {SYMBOLS['M_max']}")
            ax.set_ylabel(f"predicted {SYMBOLS['M_max']}")
        case "map":
            color = np.log10(np.maximum([c.error for c in curves], FLOOR))
            ax.set_xlabel(SYMBOLS["beta"])
            ax.set_ylabel(SYMBOLS["lambda"])
        case _:
            assert_never(variant)
    x = data if variant == "parity" else np.array([c.beta for c in curves])
    y = predicted if variant == "parity" else np.array([c.lam for c in curves])
    norm = Normalize(float(color.min()), float(color.max()))
    ax.scatter(x, y, c=color, cmap=cmap, norm=norm, s=mark_size(style, 14.0), zorder=3)
    label = SYMBOLS["lambda"] if variant == "parity" else "$\\log_{10}$ relative error"
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(mappable, ax=ax, label=label)


def _bottom(ax: Axes, found: Evidence, style: PlotStyle, variant: VariantName) -> None:
    """The relative error against the data maximum mass on a log axis, with the ceiling."""
    curves = found.maxes.curves
    cmap = _cmap(style, variant)
    lam = np.array([c.lam for c in curves])
    norm = Normalize(float(lam.min()), float(lam.max()))
    ax.scatter(
        [c.data for c in curves],
        np.maximum([c.error for c in curves], FLOOR),
        c=lam,
        cmap=cmap,
        norm=norm,
        s=mark_size(style, 14.0),
        zorder=3,
    )
    ax.set_yscale("log")
    ax.axhline(10.0**-found.ceiling, color="0.25", ls=":", lw=line_width(style, 1.0))
    ax.set_xlabel(f"data {SYMBOLS['M_max']}")
    ax.set_ylabel(f"relative error of {SYMBOLS['M_max']}")


def draw(found: Evidence, style: PlotStyle, variant: VariantName) -> Figure:
    """The upper panel of the variant, the relative error below it with the mass ceiling."""
    fig, axes = plt.subplots(
        2, 1, figsize=text_width_size(0.9), squeeze=False, layout="constrained"
    )
    _top(axes[0][0], found, style, variant)
    _bottom(axes[1][0], found, style, variant)
    return fig


# --- shell ------------------------------------------------------------------------------------


def _arguments(argv: list[str]) -> argparse.Namespace:
    """The prepared neutron-star table, the split, the state folder and the asset folder."""
    parser = argparse.ArgumentParser(prog="mass_max.py")
    for name in ("ns", "split", "state", "assets"):
        parser.add_argument(name, type=Path)
    return parser.parse_args(argv)


def _figure_arguments(argv: list[str]) -> argparse.Namespace:
    """The figures mode: the out dir and the state dir the main run stored its data in."""
    parser = argparse.ArgumentParser(prog="mass_max.py")
    parser.add_argument("--figures", type=Path, required=True, help="draw every variant here")
    parser.add_argument("--state", type=Path, required=True, help="the state dir of the main run")
    return parser.parse_args(argv)


def _figure_data_path(state: Path) -> Path:
    """Where the main run stores what the figure plots, machine-local interim."""
    return state / SECTION / "figures.pkl"


def _inputs(args: argparse.Namespace) -> MapInput:
    """Read the neutron-star table and split, the saved mass surrogate and its recorded ceiling."""
    surrogate = load_surrogate(args.state / PIPELINE / f"{NEUTRON_STARS.dataset}-mass.pkl")
    ceilings = json.loads((args.state / REACHABLE).read_text())
    return MapInput(
        pd.read_parquet(args.ns),
        pd.read_parquet(args.split),
        surrogate.predict,
        float(ceilings["mass"]),
    )


def main(
    argv: Sequence[str],
    config: PaperConfig | None = None,
    section: DiscussionSection | None = None,
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
        logger.info("done: %d maximum-mass figure variants -> %s", len(get_args(VariantName)), out)
        return
    args = _arguments(argv)
    section = section or load_section(DiscussionSection, SECTION_TOML)
    found = run(_inputs(args))
    out = args.assets / SECTION
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{asset()}.tex").write_text(figure_tex(section.variants.mass_max))
    store_section(args.state, SECTION, {}, numbers(found))
    _figure_data_path(args.state).write_bytes(pickle.dumps(found))
    logger.info("done: maximum-mass figure wrapper -> %s, numbers -> %s", out, args.state / SECTION)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])

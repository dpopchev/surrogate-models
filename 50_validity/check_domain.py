"""The domain figure of Section 4 and its numbers (W-153).

Per map -- black holes in the (beta, r_h) plane, neutron stars in the (beta, lambda) plane and along
one curve of rho_c -- a regular grid of probe queries goes through predict of paperlib/final.py: an
answered probe is a filled mark, a refused one an open mark with an arrow to its nearest valid
input. The numbers are the probes answered and refused per map and the share of the test rows that
lie inside the domain, which must be all of them.
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
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict

from paperlib.config import NOTATION_TEX, PaperConfig, load_config, load_section
from paperlib.design import BLACK_HOLES, NEUTRON_STARS, design
from paperlib.domain import BlackHoleDomain, Domain, Inside, NeutronStarDomain, check
from paperlib.eda import number_tex, text_width_size
from paperlib.final import (
    BLACK_HOLE_COLUMNS,
    NEUTRON_STAR_COLUMNS,
    FinalSurrogate,
    Refused,
    load_domain,
    load_surrogate,
    predict,
)
from paperlib.plots import SYMBOLS, PlotStyle, anchor_color, apply_style, line_width, mark_size

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

# planes: the probes with an arrow from each refused one to its nearest valid input;
# planes_no_arrows: the same without the arrows.
DomainVariant = Literal["planes", "planes_no_arrows"]


class DomainVariants(BaseModel):
    """Section 4: which variant of the domain figure the paper places."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: DomainVariant = "planes"


class DomainSection(BaseModel):
    """Section 4's own settings: the picked variant of its figure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    variants: DomainVariants = DomainVariants()


SECTION = "50_validity"
SECTION_TOML = Path(__file__).with_name("50_validity.toml")
PIPELINE = "41_algorithm"

# The probe grid reaches this share of a coupling's span beyond the sampled range, so some
# probes lie outside the domain.
PAD = 0.15
GRID = 9
CURVE_PROBES = 18


@dataclass(frozen=True)
class MapInput:
    """One map as the figure reads it: its prepared table (r_h, beta, M for the black holes;
    rho_c, beta, lambda, M for the neutron stars), the domain read from it, the saved mass
    surrogate and the mask of its test rows."""

    dataset: str
    table: pd.DataFrame
    domain: Domain
    surrogate: FinalSurrogate
    test: np.ndarray


@dataclass(frozen=True)
class Probe:
    """One query: its input row by column, whether the surrogate answered it, the prediction
    when it did and the nearest valid input when it did not."""

    x: Mapping[str, float]
    answered: bool
    value: float | None
    nearest: Mapping[str, float] | None


@dataclass(frozen=True)
class CurveView:
    """One neutron-star curve as drawn: log10 rho_c and M of its rows, and its stable range."""

    key: tuple[float, float]
    log_rho: np.ndarray
    mass: np.ndarray
    low: float
    high: float


@dataclass(frozen=True)
class MapEvidence:
    """What one map's panels plot and its numbers count."""

    dataset: str
    domain: Domain
    probes: tuple[Probe, ...]
    curve_probes: tuple[Probe, ...]
    curve: CurveView | None
    test_rows: int
    test_inside: int


# --- pure functions ---------------------------------------------------------------------------


def _axis(low: float, high: float, floor: float | None = None) -> np.ndarray:
    """A regular grid over [low, high] padded by PAD of its span on each side, not below floor."""
    pad = PAD * (high - low)
    start = low - pad if floor is None else max(low - pad, floor)
    return np.linspace(start, high + pad, GRID)


def _probes(
    surrogate: FinalSurrogate, domain: Domain, columns: tuple[str, ...], rows: np.ndarray
) -> tuple[Probe, ...]:
    """The probe of every row, answered or refused by predict."""
    answers = predict(surrogate, domain, rows)
    return tuple(
        Probe(dict(zip(columns, map(float, row), strict=True)), True, answer, None)
        if not isinstance(answer, Refused)
        else Probe(
            dict(zip(columns, map(float, row), strict=True)), False, None, dict(answer.nearest)
        )
        for row, answer in zip(rows, answers, strict=True)
    )


def _black_hole_probes(found: MapInput) -> tuple[Probe, ...]:
    """A grid over beta and r_h around the sampled ranges."""
    domain = found.domain
    assert isinstance(domain, BlackHoleDomain)
    betas = _axis(domain.betas[0], domain.betas[-1])
    radii = _axis(min(domain.r_min), max(domain.r_max), floor=0.0)
    rows = np.array([(r, b) for b in betas for r in radii])
    return _probes(found.surrogate, domain, BLACK_HOLE_COLUMNS, rows)


def _neutron_star_probes(found: MapInput) -> tuple[Probe, ...]:
    """A grid over beta and lambda at the median rho_c of the table."""
    domain = found.domain
    assert isinstance(domain, NeutronStarDomain)
    keys = np.array(domain.keys)
    rho = float(10 ** np.median(np.log10(found.table["rho_c"])))
    betas = _axis(keys[:, 0].min(), keys[:, 0].max())
    lambdas = _axis(keys[:, 1].min(), keys[:, 1].max())
    rows = np.array([(rho, b, lam) for b in betas for lam in lambdas])
    return _probes(found.surrogate, domain, NEUTRON_STAR_COLUMNS, rows)


def _curve_view(found: MapInput) -> CurveView:
    """The sampled curve nearest the centre of the keys: its rows and its stable range."""
    domain = found.domain
    assert isinstance(domain, NeutronStarDomain)
    keys = np.array(domain.keys)
    i = int(np.argmin(np.hypot(*(keys - keys.mean(axis=0)).T)))
    beta, lam = domain.keys[i]
    rows = found.table[(found.table["beta"] == beta) & (found.table["lambda"] == lam)]
    rows = rows.sort_values("rho_c")
    return CurveView(
        (beta, lam),
        np.log10(rows["rho_c"].to_numpy(float)),
        rows["M"].to_numpy(float),
        float(np.log10(domain.rho_min[i])),
        float(np.log10(domain.rho_max[i])),
    )


def _curve_probes(found: MapInput, curve: CurveView) -> tuple[Probe, ...]:
    """Probes along the curve's rho_c, from below its range to beyond its maximum mass."""
    beta, lam = curve.key
    span = curve.high - curve.low
    log_rho = np.linspace(curve.low - PAD * span, curve.high + PAD * span, CURVE_PROBES)
    rows = np.array([(10**r, beta, lam) for r in log_rho])
    return _probes(found.surrogate, found.domain, NEUTRON_STAR_COLUMNS, rows)


def _curve_extremes(found: MapInput, curve: Sequence[str], along: str) -> tuple[int, int]:
    """The test rows of the map and how many lie inside its domain. A verdict depends on a row's
    curve and its coordinate along it, bounded by the curve's smallest and largest test row, so
    those two rows decide every row of the curve between them."""
    test = found.table[found.test]
    inside = 0
    for _, rows in test.groupby(list(curve), sort=False):
        ends = rows.loc[[rows[along].idxmin(), rows[along].idxmax()]]
        verdicts = [
            check(found.domain, {k: float(r[k]) for k in (*curve, along)})
            for _, r in ends.iterrows()
        ]
        if all(isinstance(v, Inside) for v in verdicts):
            inside += len(rows)
    return len(test), inside


def run(maps: Sequence[MapInput]) -> tuple[MapEvidence, ...]:
    """The probes and the test-row counts of each map."""
    found: list[MapEvidence] = []
    for one in maps:
        match one.domain:
            case BlackHoleDomain():
                total, inside = _curve_extremes(one, ("beta",), "r_h")
                found.append(
                    MapEvidence(
                        one.dataset, one.domain, _black_hole_probes(one), (), None, total, inside
                    )
                )
            case NeutronStarDomain():
                total, inside = _curve_extremes(one, ("beta", "lambda"), "rho_c")
                curve = _curve_view(one)
                found.append(
                    MapEvidence(
                        one.dataset,
                        one.domain,
                        _neutron_star_probes(one),
                        _curve_probes(one, curve),
                        curve,
                        total,
                        inside,
                    )
                )
            case _:
                assert_never(one.domain)
    return tuple(found)


def numbers(found: Sequence[MapEvidence]) -> dict[str, str]:
    """Every number Section 4 cites, keyed by its \\validity macro name."""
    macros: dict[str, str] = {}
    for evidence in found:
        tag = "Bh" if isinstance(evidence.domain, BlackHoleDomain) else "Ns"
        probes = evidence.probes + evidence.curve_probes
        answered = sum(p.answered for p in probes)
        share = 100.0 * evidence.test_inside / max(evidence.test_rows, 1)
        macros |= {
            f"validity{tag}Answered": str(answered),
            f"validity{tag}Refused": str(len(probes) - answered),
            f"validity{tag}TestRows": str(evidence.test_rows),
            f"validity{tag}TestInside": number_tex(share),
        }
    return macros


# --- figures ----------------------------------------------------------------------------------

CAPTION = (
    "The physical domain of each map. Left: black holes in the $(\\beta, r_h)$ plane, the "
    "existence edge $r_{h,\\min}(\\beta)$ of the sampled curves. Centre: neutron stars in the "
    "$(\\beta, \\lambda)$ plane, the sampled keys. Right: the stable range of $\\rhoc$ along one "
    "curve, up to its maximum mass. A regular grid of probe queries goes through the surrogate: "
    "an answered probe is a filled mark, a refused one an open mark%s."
)
ARROWS = " with an arrow to its nearest valid input"


def asset() -> str:
    """The stem one figure's image and its .tex wrapper share."""
    return f"{SECTION}_fig_domain"


def picked_stem(variant: DomainVariant) -> str:
    """The stem of the picked variant's PNG under build/sections/<section>/figures/."""
    return f"{asset()}_{variant}"


def figure_tex(variant: DomainVariant) -> str:
    """The figure environment of the domain figure in its picked variant."""
    caption = CAPTION % (ARROWS if variant == "planes" else "")
    return (
        "\\begin{figure}[!htb]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{picked_stem(variant)}}}\n"
        f"\\caption{{{caption}}}\n\\label{{fig:validity-domain}}\n\\end{{figure}}\n"
    )


def _marks(
    ax: Axes,
    probes: Sequence[Probe],
    place: tuple[str, str],
    style: PlotStyle,
    color: tuple[float, float, float],
    arrows: bool,
) -> None:
    """The answered probes as filled marks, the refused as open marks, each with its arrow to the
    nearest valid input when arrows are drawn."""
    size = mark_size(style, 14.0)
    for answered in (True, False):
        chosen = [p for p in probes if p.answered == answered]
        points = np.array([[p.x[place[0]], p.x[place[1]]] for p in chosen]).reshape(-1, 2)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=size,
            facecolors=color if answered else "none",
            edgecolors=color if answered else "0.25",
            linewidths=line_width(style, 0.8),
            label="answered" if answered else "refused",
            zorder=3,
        )
    if arrows:
        for p in probes:
            if p.nearest is not None:
                ax.annotate(
                    "",
                    xy=(p.nearest[place[0]], p.nearest[place[1]]),
                    xytext=(p.x[place[0]], p.x[place[1]]),
                    arrowprops={"arrowstyle": "->", "color": "0.4", "lw": line_width(style, 0.6)},
                )


def _black_hole_panel(ax: Axes, found: MapEvidence, style: PlotStyle, arrows: bool) -> None:
    domain = found.domain
    assert isinstance(domain, BlackHoleDomain)
    color = anchor_color(style, "black_holes")
    ax.fill_between(domain.betas, domain.r_min, domain.r_max, color=color, alpha=0.18, lw=0)
    ax.plot(domain.betas, domain.r_min, color=color, lw=line_width(style, 1.2))
    ax.plot(domain.betas, domain.r_max, color=color, lw=line_width(style, 0.6), ls=":")
    ax.scatter(domain.betas, domain.r_min, s=mark_size(style, 6.0), color=color, zorder=2)
    _marks(ax, found.probes, ("beta", "r_h"), style, color, arrows)
    ax.set_xlabel(SYMBOLS["beta"])
    ax.set_ylabel(SYMBOLS["r_h"])
    ax.set_title("black holes")


def _neutron_star_panel(ax: Axes, found: MapEvidence, style: PlotStyle, arrows: bool) -> None:
    domain = found.domain
    assert isinstance(domain, NeutronStarDomain)
    color = anchor_color(style, "neutron_stars")
    keys = np.array(domain.keys)
    ax.scatter(keys[:, 0], keys[:, 1], s=mark_size(style, 3.0), color=color, alpha=0.5, zorder=2)
    _marks(ax, found.probes, ("beta", "lambda"), style, color, arrows)
    ax.set_xlabel(SYMBOLS["beta"])
    ax.set_ylabel(SYMBOLS["lambda"])
    ax.set_title("neutron stars")


def _curve_panel(ax: Axes, found: MapEvidence, style: PlotStyle, arrows: bool) -> None:
    curve = found.curve
    assert curve is not None
    color = anchor_color(style, "neutron_stars")
    ax.axvspan(curve.low, curve.high, color=color, alpha=0.18, lw=0)
    ax.plot(curve.log_rho, curve.mass, color=color, lw=line_width(style, 1.2))
    base = float(curve.mass.min() - 0.15 * np.ptp(curve.mass))
    drawn: list[Probe] = []
    for p in found.curve_probes:
        here = float(np.log10(p.x["rho_c"]))
        if p.answered:
            drawn.append(Probe({"x": here, "y": float(p.value or 0.0)}, True, p.value, None))
        else:
            assert p.nearest is not None
            edge = float(np.log10(p.nearest["rho_c"]))
            target = {"x": edge, "y": float(np.interp(edge, curve.log_rho, curve.mass))}
            drawn.append(Probe({"x": here, "y": base}, False, None, target))
    _marks(ax, drawn, ("x", "y"), style, color, arrows)
    ax.set_xlabel(SYMBOLS["log10_rho_c"])
    ax.set_ylabel(SYMBOLS["M"])
    ax.set_title("one stable branch")


def draw(found: Sequence[MapEvidence], style: PlotStyle, variant: DomainVariant) -> Figure:
    """One panel per map, the neutron stars with a second panel for the stable range of rho_c."""
    arrows = variant == "planes"
    panels = [
        (panel, evidence)
        for evidence in found
        for panel in (
            (_black_hole_panel,)
            if isinstance(evidence.domain, BlackHoleDomain)
            else (_neutron_star_panel, _curve_panel)
        )
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=text_width_size(0.4), squeeze=False)
    for ax, (panel, evidence) in zip(axes[0], panels, strict=True):
        panel(ax, evidence, style, arrows)
    axes[0][0].legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig


# --- shell ------------------------------------------------------------------------------------


def _arguments(argv: list[str]) -> argparse.Namespace:
    """The prepared tables, the split, the state folder and the asset folder."""
    parser = argparse.ArgumentParser(prog="check_domain.py")
    for name in ("ns", "bh", "split", "state", "assets"):
        parser.add_argument(name, type=Path)
    return parser.parse_args(argv)


def _figure_arguments(argv: list[str]) -> argparse.Namespace:
    """The figures mode: the out dir and the state dir the main run stored its data in."""
    parser = argparse.ArgumentParser(prog="check_domain.py")
    parser.add_argument("--figures", type=Path, required=True, help="draw every variant here")
    parser.add_argument("--state", type=Path, required=True, help="the state dir of the main run")
    return parser.parse_args(argv)


def _figure_data_path(state: Path) -> Path:
    """Where the main run stores what the figure plots, machine-local interim."""
    return state / SECTION / "figures.pkl"


def _inputs(args: argparse.Namespace) -> tuple[MapInput, ...]:
    """Read each map: its table, its saved domain and mass surrogate, its test rows."""
    split = pd.read_parquet(args.split)
    inputs: list[MapInput] = []
    for spec, source in ((BLACK_HOLES, args.bh), (NEUTRON_STARS, args.ns)):
        table = pd.read_parquet(source)
        saved = args.state / PIPELINE
        inputs.append(
            MapInput(
                spec.dataset,
                table,
                load_domain(saved / f"{spec.dataset}-domain.json"),
                load_surrogate(saved / f"{spec.dataset}-mass.pkl"),
                design(table, split, spec, "mass").test,
            )
        )
    return tuple(inputs)


def _store(state: Path, macros: Mapping[str, str]) -> None:
    """Store the numbers under the section's state folder, for paperlib/tools/render_assets.py."""
    (state / SECTION).mkdir(parents=True, exist_ok=True)
    (state / SECTION / "numbers.json").write_text(json.dumps(dict(macros)))


def main(
    argv: list[str],
    config: PaperConfig | None = None,
    section: DomainSection | None = None,
) -> None:
    """Write the figure wrapper and store the numbers and the figure data -- or, given
    --figures, draw every variant as its own PNG from the data the main run stored."""
    config = config or load_config()
    apply_style(config.plot, NOTATION_TEX.read_text())
    if "--figures" in argv:
        figure_args = _figure_arguments(argv)
        found = pickle.loads(_figure_data_path(figure_args.state).read_bytes())
        out = figure_args.figures / SECTION / "figures"
        out.mkdir(parents=True, exist_ok=True)
        for variant in get_args(DomainVariant):
            fig = draw(found, config.plot, variant)
            fig.savefig(out / f"{picked_stem(variant)}.png", dpi=config.plot.dpi)
            plt.close(fig)
        logger.info("done: %d domain figure variants -> %s", len(get_args(DomainVariant)), out)
        return
    args = _arguments(argv)
    section = section or load_section(DomainSection, SECTION_TOML)
    found = run(_inputs(args))
    out = args.assets / SECTION
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{asset()}.tex").write_text(figure_tex(section.variants.domain))
    _store(args.state, numbers(found))
    _figure_data_path(args.state).write_bytes(pickle.dumps(found))
    logger.info("done: domain figure wrapper -> %s, numbers -> %s", out, args.state / SECTION)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])

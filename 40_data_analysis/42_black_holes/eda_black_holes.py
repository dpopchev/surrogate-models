"""Exploratory analysis of the prepared BH table behind Section 4.2 (W-018).

The BH data are 21 dense beta curves over the horizon radius r_h. The computations ask what the
NS analysis asked, and find different answers: how far M is from r_h/2 (the mass correction),
how much beta moves M and D at fixed r_h, where each curve starts (the existence edge), which
charge target the data support, and which split the curves support. Numbers become \\bhEda...
macros; the dataset-agnostic machinery lives in shared/eda.py.
"""

import logging
from dataclasses import dataclass
from typing import Literal, assert_never, get_args

import numpy as np
import pandas as pd

from shared.eda import (
    Adjacency,
    ColumnSummary,
    camel,
    curve_adjacency,
    curve_ids,
    group_splitter,
    holdout_mae,
    kfold_mae,
    make_curve_space,
    number_tex,
    orders_of_magnitude,
    row_splitter,
    summarize,
)

logger = logging.getLogger(__name__)

# --- vocabulary and types ---------------------------------------------------------------------

CURVE = ("beta",)
BH_SPACE = make_curve_space({"beta": "raw", "r_h": "raw"}, CURVE)
CHARGE_TARGETS = ("D", "log10_D", "D_over_M", "log10_D_over_M")


@dataclass(frozen=True)
class ChargeTarget:
    """How one candidate charge target correlates with M, beta and r_h."""

    target: str
    pearson_m: float
    spearman_m: float
    pearson_beta: float
    pearson_rh: float


@dataclass(frozen=True)
class BetaEffect:
    """How far beta moves a target at fixed r_h: the median spread across beta at one r_h value
    over the target's total spread (0 when beta does not matter)."""

    target: str
    spread_ratio: float


@dataclass(frozen=True)
class MassCorrection:
    """The range of M - r_h/2 and how it follows r_h and beta."""

    minimum: float
    maximum: float
    pearson_rh: float
    pearson_beta: float


@dataclass(frozen=True)
class ExistenceEdge:
    """Where each beta curve starts: (beta, smallest r_h), ordered by beta."""

    starts: tuple[tuple[float, float], ...]


BhSplitStrategy = Literal["random_rows", "curves", "outer_curves"]


@dataclass(frozen=True)
class SplitScore:
    """1-nearest-neighbour error of a target when the split holds out what the strategy names.

    random_rows: rows at random; curves: whole beta curves (GroupKFold); outer_curves: the
    smallest and largest beta curves, predicted from the others (extrapolation in beta).
    """

    strategy: BhSplitStrategy
    target: str
    mae: float


# --- pure functions ---------------------------------------------------------------------------


def with_mass_correction(table: pd.DataFrame) -> pd.DataFrame:
    """Add M_correction = M - r_h/2 to a table with M and r_h."""
    return table.assign(M_correction=table["M"] - table["r_h"] / 2)


def charge_targets(table: pd.DataFrame) -> tuple[ChargeTarget, ...]:
    """Correlate each candidate charge target with M, beta and r_h, in CHARGE_TARGETS order."""
    return tuple(
        ChargeTarget(
            target,
            float(table[target].corr(table["M"])),
            float(table[target].corr(table["M"], method="spearman")),
            float(table[target].corr(table["beta"])),
            float(table[target].corr(table["r_h"])),
        )
        for target in CHARGE_TARGETS
    )


def beta_effect(table: pd.DataFrame, target: str) -> BetaEffect:
    """Measure how far beta moves target at fixed r_h.

    The curves share one r_h grid, so each r_h value holds one row per curve present there;
    r_h values with a single curve carry no spread and are skipped.
    """
    across_beta = table.groupby("r_h")[target].std().dropna()
    return BetaEffect(target, float(across_beta.median() / table[target].std()))


def mass_correction(table: pd.DataFrame) -> MassCorrection:
    """Summarize M - r_h/2 from a table with M_correction."""
    correction = table["M_correction"]
    return MassCorrection(
        float(correction.min()),
        float(correction.max()),
        float(correction.corr(table["r_h"])),
        float(correction.corr(table["beta"])),
    )


def existence_edge(table: pd.DataFrame) -> ExistenceEdge:
    """Find where each beta curve starts in r_h."""
    starts = table.groupby("beta")["r_h"].min()
    pairs = zip(starts.index.to_numpy(dtype=float), starts.to_numpy(dtype=float), strict=True)
    return ExistenceEdge(tuple((float(b), float(r)) for b, r in pairs))


def _score(
    table: pd.DataFrame, target: str, strategy: BhSplitStrategy, folds: int, seed: int
) -> float:
    """The 1-NN error of one strategy."""
    match strategy:
        case "random_rows":
            return kfold_mae(table, BH_SPACE, target, row_splitter(folds, seed), None)
        case "curves":
            groups = curve_ids(table, BH_SPACE)
            return kfold_mae(table, BH_SPACE, target, group_splitter(folds, seed), groups)
        case "outer_curves":
            outer = table["beta"].isin([table["beta"].min(), table["beta"].max()])
            return holdout_mae(table, BH_SPACE, target, outer.to_numpy())
        case _:
            assert_never(strategy)


def split_strategies(
    table: pd.DataFrame, target: str, folds: int, seed: int
) -> tuple[SplitScore, ...]:
    """1-NN error of target under each split strategy, in BhSplitStrategy order."""
    return tuple(
        SplitScore(strategy, target, _score(table, target, strategy, folds, seed))
        for strategy in get_args(BhSplitStrategy)
    )


SUMMARY_COLUMNS = ("r_h", "M", "D", "D_over_M")
BETA_EFFECT_TARGETS = ("M", "M_correction", "log10_D")
# Macro tags of the split targets: M, the mass correction, log10 D.
SPLIT_TAGS = {"M": "M", "M_correction": "MC", "log10_D": "LD"}


@dataclass(frozen=True)
class Evidence:
    """Every computed result Section 4.2 cites, computed once for the macros and the tables."""

    rows: int
    curves: int
    rows_per_curve: tuple[int, int]
    rh_step: float
    pearson_m_rh: float
    folds: int
    summaries: tuple[ColumnSummary, ...]
    targets: tuple[ChargeTarget, ...]
    beta_effects: tuple[BetaEffect, ...]
    correction: MassCorrection
    edge: ExistenceEdge
    adjacency: Adjacency
    splits: tuple[SplitScore, ...]


def evidence(table: pd.DataFrame, folds: int, seed: int) -> Evidence:
    """Compute every Section 4.2 result from a table with the charge targets and correction."""
    sizes = table.groupby(list(CURVE)).size()
    grid = np.unique(table["r_h"].to_numpy())
    return Evidence(
        len(table),
        len(sizes),
        (int(sizes.min()), int(sizes.max())),
        float(np.diff(grid).min()),
        float(table["M"].corr(table["r_h"])),
        folds,
        summarize(table, SUMMARY_COLUMNS),
        charge_targets(table),
        tuple(beta_effect(table, target) for target in BETA_EFFECT_TARGETS),
        mass_correction(table),
        existence_edge(table),
        curve_adjacency(table, BH_SPACE),
        tuple(s for t in SPLIT_TAGS for s in split_strategies(table, t, folds, seed)),
    )


def numbers(found: Evidence) -> dict[str, str]:
    """Every number Section 4.2 cites, keyed by its \\bhEda macro name."""
    summary = {(s.column, s.scale): s for s in found.summaries}
    target = {t.target: t for t in found.targets}
    effect = {e.target: e.spread_ratio for e in found.beta_effects}
    (beta_low, rh_low), (beta_high, rh_high) = found.edge.starts[0], found.edge.starts[-1]
    rh, d, dm = summary["r_h", "raw"], summary["D", "raw"], summary["D_over_M", "raw"]
    return {
        "bhEdaRows": f"{found.rows}",
        "bhEdaCurves": f"{found.curves}",
        "bhEdaRowsPerCurveMin": f"{found.rows_per_curve[0]}",
        "bhEdaRowsPerCurveMax": f"{found.rows_per_curve[1]}",
        "bhEdaRhStep": f"{found.rh_step:.3f}",
        "bhEdaRhMin": f"{rh.minimum:.3f}",
        "bhEdaRhMax": f"{rh.maximum:.3f}",
        "bhEdaRhOrders": f"{orders_of_magnitude(rh):.2f}",
        "bhEdaRhRawSkew": f"{rh.skew:.2f}",
        "bhEdaPearsonMRh": f"{found.pearson_m_rh:.4f}",
        "bhEdaDMin": number_tex(d.minimum),
        "bhEdaDMax": number_tex(d.maximum),
        "bhEdaDOrders": f"{orders_of_magnitude(d):.2f}",
        "bhEdaDMOrders": f"{orders_of_magnitude(dm):.2f}",
        "bhEdaSpearmanMD": f"{target['D'].spearman_m:.3f}",
        "bhEdaSpearmanMLogD": f"{target['log10_D'].spearman_m:.3f}",
        "bhEdaSpearmanMLogDM": f"{target['log10_D_over_M'].spearman_m:.3f}",
        "bhEdaPearsonRhLogD": f"{target['log10_D'].pearson_rh:.3f}",
        "bhEdaPearsonRhLogDM": f"{target['log10_D_over_M'].pearson_rh:.3f}",
        "bhEdaPearsonBetaLogD": f"{target['log10_D'].pearson_beta:.3f}",
        "bhEdaPearsonBetaLogDM": f"{target['log10_D_over_M'].pearson_beta:.3f}",
        "bhEdaCorrectionMin": f"{found.correction.minimum:.3f}",
        "bhEdaCorrectionMax": f"{found.correction.maximum:.3f}",
        "bhEdaCorrectionPearsonRh": f"{found.correction.pearson_rh:.2f}",
        "bhEdaCorrectionPearsonBeta": f"{found.correction.pearson_beta:.2f}",
        "bhEdaBetaEffectPercentM": f"{100 * effect['M']:.2f}",
        "bhEdaBetaEffectPercentMC": f"{100 * effect['M_correction']:.1f}",
        "bhEdaBetaEffectPercentLD": f"{100 * effect['log10_D']:.1f}",
        "bhEdaEdgeBetaLow": f"{beta_low:.2f}",
        "bhEdaEdgeBetaHigh": f"{beta_high:.2f}",
        "bhEdaEdgeRhMinLow": f"{rh_low:.3f}",
        "bhEdaEdgeRhMinHigh": f"{rh_high:.3f}",
        "bhEdaSameCurvePercent": f"{100 * found.adjacency.same_curve_share:.1f}",
        "bhEdaWithinMedian": f"{found.adjacency.within_median:.4f}",
        "bhEdaAcrossMedian": f"{found.adjacency.across_median:.4f}",
        "bhEdaFolds": f"{found.folds}",
    } | {
        f"bhEdaSplit{camel(s.strategy)}{SPLIT_TAGS[s.target]}": _split_error(s)
        for s in found.splits
    }


def _split_error(score: SplitScore) -> str:
    """One split error for math mode; BH errors span orders of magnitude, so the format adapts."""
    return number_tex(score.mae)

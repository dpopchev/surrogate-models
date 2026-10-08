"""Facts about the family screen behind Section 5.2, on tiny synthetic data."""

import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from screen_families import (
    SECTION,
    Candidate,
    FitTimeoutError,
    Job,
    Panel,
    Scored,
    Series,
    Trajectory,
    Wall,
    candidates,
    error_figure,
    family_table,
    job_line,
    main,
    next_jobs,
    outcomes_from_json,
    outcomes_to_json,
    over_memory,
    parity_figure,
    parity_picks,
    run_job,
    scaling_figure,
    survivors,
    thin,
    timed,
    trajectories,
)

from shared.config import PaperConfig
from shared.design import Design, Target
from shared.plots import PlotStyle

# Five curves of ten rows, y = p + x: curves 0-3 train (two frozen folds), curve 4 is test.
P = np.repeat(np.arange(5.0), 10)
X_ALONG = np.tile(np.linspace(0.0, 1.0, 10), 5)
TOY = Design(
    X=np.column_stack([X_ALONG, P]),
    y=P + X_ALONG,
    groups=P.astype(int),
    test=P == 4,
    fold=np.where(P == 4, -1, P.astype(int) % 2),
    ablation=np.zeros(50, dtype=bool),
)


def test_every_family_is_a_candidate_in_both_units() -> None:
    assert len(set(candidates())) == 12


def test_thinning_keeps_every_test_row() -> None:
    assert int(thin(TOY, rows=20, per_curve=2, seed=0).test.sum()) == 10


def test_thinning_keeps_every_training_curve_when_each_can_keep_its_share() -> None:
    thinned = thin(TOY, rows=20, per_curve=2, seed=0)
    kept = thinned.groups[~thinned.test]
    assert np.bincount(kept).tolist() == [5, 5, 5, 5]


def test_thinning_below_the_share_of_every_curve_keeps_fewer_curves() -> None:
    thinned = thin(TOY, rows=6, per_curve=2, seed=0)
    kept = thinned.groups[~thinned.test]
    assert sorted(np.bincount(kept, minlength=4).tolist()) == [0, 2, 2, 2]


KNN = Candidate("k-NN", "pointwise")
RBF = Candidate("local RBF", "pointwise")
GPR = Candidate("GPR", "pointwise")
MLP = Candidate("MLP", "curve-wise")


def scored(candidate: Candidate, folds: float) -> Scored:
    """A toy outcome of the candidate on one pair in the first round."""
    job = Job("toy", "mass", candidate, rows=1000, round=0)
    return Scored(job, folds, test=folds, ripple=(), fit_seconds=1.0, entry="e")


def walled(candidate: Candidate) -> Wall:
    """A toy wall of the candidate on one pair in the first round."""
    return Wall(Job("toy", "mass", candidate, rows=1000, round=0), "past 30 min")


def test_the_best_candidates_on_the_folds_survive() -> None:
    round_ = [scored(KNN, 1.0), scored(RBF, 3.0), scored(MLP, 2.0)]
    assert survivors(round_, keep=2) == (RBF, MLP)


def test_a_wall_never_survives() -> None:
    assert survivors([scored(KNN, 1.0), walled(GPR)], keep=2) == (KNN,)


def test_a_pointwise_gpr_whose_kernel_matrix_passes_the_budget_is_over_memory() -> None:
    # 100 rows: a kernel matrix of 100 x 100 doubles, 80 000 bytes.
    assert over_memory(GPR, rows=100, memory_bytes=50_000)


def test_a_pointwise_gpr_within_the_budget_is_not_over_memory() -> None:
    assert not over_memory(GPR, rows=100, memory_bytes=100_000)


def test_a_curve_wise_gpr_is_not_over_memory_by_its_rows() -> None:
    # Curve-wise, the GPR is fitted on one row per curve, not on the rows.
    assert not over_memory(Candidate("GPR", "curve-wise"), rows=100, memory_bytes=50_000)


def test_a_job_line_tells_the_scores_how_far_and_the_time_left() -> None:
    line = job_line(scored(KNN, 2.5), done=2, total=6, elapsed=60.0)
    assert line == (
        "job 2/6 of round 1 done: toy mass, k-NN pointwise, 1000 rows: folds 2.50, test 2.50 "
        "figures; elapsed 1m 0s, about 2m 0s left"
    )


def test_a_wall_s_job_line_names_its_wall() -> None:
    line = job_line(walled(GPR), done=1, total=1, elapsed=5.0)
    assert line == (
        "job 1/1 of round 1 done: toy mass, GPR pointwise, 1000 rows: wall, past 30 min; "
        "elapsed 5s, about 0s left"
    )


def slow(x_fit, y_fit, x_valid, y_valid, seed):
    """A toy fitter that takes a second."""
    time.sleep(1.0)
    return lambda x: np.zeros(len(x))


def test_a_fit_past_its_time_budget_is_stopped() -> None:
    with pytest.raises(FitTimeoutError):
        timed(slow, seconds=0.05)(TOY.X, TOY.y, TOY.X, TOY.y, 0)


# Twelve curves of ten rows, y = p + x: curves 10 and 11 are test.
P12 = np.repeat(np.arange(12.0), 10)
X12 = np.tile(np.linspace(0.0, 1.0, 10), 12)
WIDE = Design(
    X=np.column_stack([X12, P12]),
    y=P12 + X12,
    groups=P12.astype(int),
    test=P12 >= 10,
    fold=np.where(P12 >= 10, -1, P12.astype(int) % 2),
    ablation=np.zeros(120, dtype=bool),
)


def off_by_a_tenth(x_fit, y_fit, x_valid, y_valid, seed):
    """A toy fitter that predicts the truth (p + x) times 1.1: a relative error of 0.1."""
    return lambda x: 1.1 * (x[:, 1] + x[:, 0])


def job_of(candidate: Candidate, rows: int = 120) -> Job:
    """A toy job of the candidate on the wide toy design."""
    return Job("toy", "mass", candidate, rows=rows, round=0)


def run_toy(job: Job, fitter, seconds: float = 60.0, memory_bytes: float = 1e9):
    """The job's outcome on the wide toy design, its ledger entry named "entry"."""
    return run_job(
        job,
        WIDE,
        fitter,
        seed=0,
        valid_fraction=0.34,
        per_curve=2,
        seconds=seconds,
        memory_bytes=memory_bytes,
        record=lambda run: "entry",
    )


def test_a_scored_job_keeps_the_mean_of_its_folds_figures() -> None:
    match run_toy(job_of(KNN), off_by_a_tenth):
        case Scored(folds=folds):
            assert folds == pytest.approx(1.0)
        case outcome:
            pytest.fail(f"not scored: {outcome}")


def test_a_gpr_over_the_memory_budget_is_a_wall_unrun() -> None:
    outcome = run_toy(job_of(GPR), slow, memory_bytes=1_000)
    assert outcome == Wall(job_of(GPR), "kernel matrix over 1e-06 GB")


def test_a_fit_past_the_time_budget_makes_the_job_a_wall() -> None:
    outcome = run_toy(job_of(KNN), slow, seconds=0.05)
    assert outcome == Wall(job_of(KNN), "past 0.000833333 min")


def hungry(x_fit, y_fit, x_valid, y_valid, seed):
    """A toy fitter that runs out of memory."""
    raise MemoryError


def test_a_fit_out_of_memory_makes_the_job_a_wall() -> None:
    assert run_toy(job_of(KNN), hungry) == Wall(job_of(KNN), "out of memory")


def test_the_outcomes_round_trip_through_json() -> None:
    outcomes = (
        Scored(job_of(MLP), 2.0, 1.5, (0.25, 0.5), 3.0, "e"),
        walled(GPR),
    )
    assert outcomes_from_json(outcomes_to_json(outcomes)) == outcomes


# Ten curves per dataset, eight rows each: the NS keys lie on a parabola, so no three are
# collinear and the local RBF's linear part has full rank on any three curves.
CURVES = range(1, 11)
TOY_LABELS = ["test", "test", *["fold0"] * 4, *["fold1"] * 4]


def toy_inputs(folder: Path) -> tuple[Path, Path, Path]:
    """Tiny NS- and BH-shaped tables and one split for both: two curves of each in test, four in
    each of two folds."""
    ns_rows = [
        {"beta": float(i), "lambda": float(i * i), "rho_c": 10.0 ** (1 + j / 7)}
        | {"M": 1.0 + 0.1 * j + 0.01 * i, "D": 0.1 + 0.01 * j}
        for i in CURVES
        for j in range(8)
    ]
    bh_rows = [
        {"r_h": 4.0 + j, "beta": float(i), "M": 2.0 + 0.5 * j + 0.01 * i, "D": 0.3 - 0.02 * j}
        for i in CURVES
        for j in range(8)
    ]
    split = [
        {"dataset": "neutron_stars", "beta": float(i), "lambda": float(i * i)}
        | {"label": label, "ablation": False}
        for i, label in zip(CURVES, TOY_LABELS, strict=True)
    ] + [
        {"dataset": "black_holes", "beta": float(i), "label": label, "ablation": False}
        for i, label in zip(CURVES, TOY_LABELS, strict=True)
    ]
    ns, bh, split_file = folder / "ns.parquet", folder / "bh.parquet", folder / "split.parquet"
    pd.DataFrame(ns_rows).to_parquet(ns)
    pd.DataFrame(bh_rows).to_parquet(bh)
    pd.DataFrame(split).to_parquet(split_file)
    return ns, bh, split_file


@pytest.fixture(scope="module")
def ran(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One main run on the toy tables, one worker, the commit and clock injected; returns its
    working folder (assets/ and state/ inside)."""
    folder = tmp_path_factory.mktemp("screen")
    ns, bh, split_file = toy_inputs(folder)
    network = {"width": 8, "depth": 1, "max_epochs": 2, "batch_size": 8}
    config = PaperConfig.model_validate(
        {"plot": {"usetex": False}, "methodology": {"algorithms": network}}
    )
    paths = [str(ns), str(bh), str(split_file), str(folder / "assets"), str(folder / "state")]
    main(
        [*paths, *TOY_KNOBS],
        config,
        commit=lambda: "abc1234",
        dirty=lambda: False,
        now=lambda: datetime(2026, 10, 8, 12, 0, 0, tzinfo=UTC),
    )
    return folder


TOY_KNOBS = [
    *("--k", "2", "--neighbours", "1000", "--knots", "2", "--per-curve", "2"),
    *("--budgets", "16", "32", "64", "--keeps", "3", "2"),
    *("--minutes", "30", "--memory-gb", "4", "--workers", "1"),
]


def test_main_saves_the_outcomes_of_every_round(ran: Path) -> None:
    (saved,) = (ran / "state" / SECTION).glob("*.json")
    # Per pair: 12 candidates, then 3 of each unit, then 2 of each unit.
    assert len(outcomes_from_json(saved.read_text())) == 4 * (12 + 6 + 4)


def test_main_draws_the_parity_figure(ran: Path) -> None:
    assert (ran / "assets" / SECTION / f"{SECTION}_fig_parity.png").stat().st_size > 0


def test_main_draws_the_error_figure(ran: Path) -> None:
    assert (ran / "assets" / SECTION / f"{SECTION}_fig_errors.png").stat().st_size > 0


def test_main_draws_the_scaling_figure(ran: Path) -> None:
    assert (ran / "assets" / SECTION / f"{SECTION}_fig_scaling.png").stat().st_size > 0


def test_main_writes_the_family_table(ran: Path) -> None:
    table = ran / "assets" / SECTION / f"{SECTION}_tab_families.tex"
    assert len(table_rows(table.read_text())) == 12


def on_ns_mass(candidate: Candidate, rows: int, folds: float) -> Scored:
    """A toy outcome of the candidate on the NS mass at that many rows."""
    job = Job("neutron_stars", "mass", candidate, rows=rows, round=0)
    return Scored(job, folds, test=folds, ripple=(), fit_seconds=1.0, entry="e")


def table_rows(table: str) -> list[str]:
    """The body rows of a booktabs table."""
    body = table.split("\\midrule\n")[1].split("\\bottomrule")[0]
    return [row for row in body.split(" \\\\\n") if row]


def test_the_family_table_has_a_row_per_candidate() -> None:
    table = family_table([on_ns_mass(KNN, 1000, 2.0)])
    assert len(table_rows(table)) == 12


def test_a_cell_shows_the_folds_figures_at_the_most_rows_with_those_rows() -> None:
    table = family_table(
        [on_ns_mass(KNN, 1000, 2.0), on_ns_mass(KNN, 10_000, 2.5), on_ns_mass(RBF, 10_000, 3.0)]
    )
    assert table_rows(table)[0].split(" & ")[2] == "$2.50_{10^{4}}$"


def test_a_wall_s_cell_names_the_rows_it_stood_at() -> None:
    wall = Wall(Job("neutron_stars", "mass", GPR, rows=100_000, round=2), "past 30 min")
    table = family_table([on_ns_mass(KNN, 100_000, 2.0), wall])
    assert table_rows(table)[4].split(" & ")[2] == "wall$_{10^{5}}$"


def test_the_pair_s_best_at_the_most_rows_is_bold() -> None:
    table = family_table(
        [on_ns_mass(KNN, 1000, 2.0), on_ns_mass(KNN, 10_000, 2.5), on_ns_mass(RBF, 10_000, 3.0)]
    )
    assert table_rows(table)[2].split(" & ")[2] == "$\\mathbf{3.00}_{10^{4}}$"


def test_the_parity_plot_shows_each_unit_s_best_at_the_pair_s_most_rows() -> None:
    picks = parity_picks(
        [
            on_ns_mass(KNN, 1000, 3.5),
            on_ns_mass(KNN, 10_000, 2.0),
            on_ns_mass(RBF, 10_000, 3.0),
            on_ns_mass(MLP, 10_000, 1.0),
        ]
    )
    assert [pick.job.candidate for pick in picks] == [RBF, MLP]


def toy_panel(dataset: str) -> Panel:
    """A panel of two toy series, both off by 1%, on four rows."""
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    errors = np.full(4, 0.01)
    return Panel(
        dataset,
        "toy",
        "$M$",
        (Series("k-NN", truth, 1.01 * truth, errors), Series("MLP", truth, 0.99 * truth, errors)),
    )


TOY_PANELS = [toy_panel(dataset) for dataset in ("neutron_stars",) * 2 + ("black_holes",) * 2]


def test_the_parity_figure_has_a_panel_per_pair() -> None:
    figure = parity_figure(TOY_PANELS, PlotStyle(usetex=False))
    assert len([axes for axes in figure.axes if axes.collections]) == 4


def test_the_error_figure_has_a_panel_per_pair_and_candidate() -> None:
    figure = error_figure(TOY_PANELS, PlotStyle(usetex=False))
    assert len([axes for axes in figure.axes if axes.collections]) == 4 * 2


def test_an_error_panel_draws_its_candidate_s_median_and_p95() -> None:
    figure = error_figure(TOY_PANELS, PlotStyle(usetex=False))
    assert len(figure.axes[0].lines) == 2


def test_a_panel_of_a_dataset_without_a_colormap_is_refused() -> None:
    with pytest.raises(ValueError, match="no colormap for dataset toy"):
        parity_figure([toy_panel("toy")], PlotStyle(usetex=False))


def test_a_walled_candidate_is_marked_at_its_rows() -> None:
    wall = Wall(Job("neutron_stars", "mass", GPR, rows=10_000, round=1), "past 30 min")
    (path,) = trajectories([on_ns_mass(GPR, 1000, 2.0), wall])
    assert path.walls == (10_000,)


def toy_path(dataset: str, target: Target, walls: tuple[int, ...] = ()) -> Trajectory:
    """A toy candidate scored on 1e3 and 1e4 rows."""
    return Trajectory(dataset, target, KNN, (1000, 10_000), (2.0, 2.5), (1.0, 9.0), walls)


TOY_PATHS = [
    toy_path("neutron_stars", "mass"),
    toy_path("neutron_stars", "charge"),
    toy_path("black_holes", "mass"),
    toy_path("black_holes", "charge"),
]


def test_the_scaling_figure_has_a_figures_and_a_seconds_panel_per_pair() -> None:
    figure = scaling_figure(TOY_PATHS, PlotStyle(usetex=False))
    assert len([axes for axes in figure.axes if axes.lines]) == 4 * 2


def test_a_wall_is_a_cross_at_its_rows_on_the_seconds_panel() -> None:
    paths = [toy_path("neutron_stars", "mass", walls=(100_000,))]
    figure = scaling_figure(paths, PlotStyle(usetex=False), wall_seconds=1800.0)
    crosses = [line for line in figure.axes[1].lines if line.get_marker() == "x"]
    assert [np.asarray(c.get_xydata())[0].tolist() for c in crosses] == [[100_000.0, 1800.0]]


def test_the_scaling_legend_names_every_family_and_both_units() -> None:
    figure = scaling_figure(TOY_PATHS, PlotStyle(usetex=False))
    (legend,) = figure.legends
    assert [text.get_text() for text in legend.get_texts()] == [
        "k-NN",
        "local RBF",
        "GPR",
        "XGBoost",
        "MLP",
        "ResNet",
        "pointwise",
        "curve-wise",
        "wall",
    ]


def test_the_next_round_keeps_the_best_of_each_unit() -> None:
    round_ = [scored(KNN, 1.0), scored(RBF, 3.0), scored(MLP, 0.5)]
    assert next_jobs(round_, budgets=(1000, 10_000), keeps=(1,)) == (
        Job("toy", "mass", RBF, rows=10_000, round=1),
        Job("toy", "mass", MLP, rows=10_000, round=1),
    )


def test_the_next_round_runs_each_pair_s_survivors_at_the_next_budget() -> None:
    other = Job("other", "charge", KNN, rows=1000, round=0)
    round_ = [scored(KNN, 1.0), scored(RBF, 3.0), Scored(other, 2.0, 2.0, (), 1.0, "e")]
    assert next_jobs(round_, budgets=(1000, 10_000), keeps=(1,)) == (
        Job("toy", "mass", RBF, rows=10_000, round=1),
        Job("other", "charge", KNN, rows=10_000, round=1),
    )

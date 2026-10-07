"""Facts about the baseline fit behind Section 4.1, on tiny synthetic curves."""

import re
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import optuna
import pandas as pd
import pytest
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

matplotlib.use("Agg")

from fit_baseline import (
    SECTION,
    Fit,
    PairScores,
    Scored,
    baseline_table,
    charge_mares,
    charge_numbers,
    curve_names,
    end_banner,
    figure_tex,
    main,
    metric_for,
    numbers,
    parity,
    progress_line,
    score_pair,
    start_banner,
)

from shared.ceilings import Spread
from shared.config import PaperConfig
from shared.design import NEUTRON_STARS, Design
from shared.harness import Timing
from shared.plots import PlotStyle
from shared.runs import from_json
from shared.surrogate import Training, mare, mare_in_d

# Six curves keyed by p along x in [0, 1]: y = 1 + 0.1 p + x^2; curve p = 2 is the test curve.
P = np.repeat(np.arange(6.0), 10)
X_ALONG = np.tile(np.linspace(0.0, 1.0, 10), 6)
TOY = Design(
    X=np.column_stack([X_ALONG, P]).astype(np.float32),
    y=(1.0 + 0.1 * P + X_ALONG**2).astype(np.float32),
    groups=P.astype(int),
    test=P == 2,
    fold=np.where(P == 2, -1, P.astype(int) % 2),  # curves 0, 4 in fold 0; 1, 3, 5 in fold 1
    ablation=np.zeros(len(P), dtype=bool),
)
SHORT = Training(
    width=8,
    depth=1,
    activation="relu",
    loss="mse",
    lr=1e-2,
    max_epochs=3,
    batch_size=16,
    patience=3,
    valid_fraction=0.25,
    seed=0,
)


def clock(*readings: float) -> Callable[[], float]:
    """A clock that returns the given readings in turn."""
    ticks = iter(readings)
    return lambda: next(ticks)


# The harness reads the clock four times per predictor: before and after the final fit, after a
# one-row and after a batch prediction; the MLP is scored first.
READINGS = (10.0, 12.5, 12.6, 12.7, *([0.0] * 8))


@pytest.fixture(scope="module")
def fit() -> Fit:
    """The MLP's final fit of the toy curves, its fit timed at 2.5 s by the clock."""
    return score_pair(TOY, "toy", "mass", SHORT, clock=clock(*READINGS))[1]


SCORED = Fit(
    y_true=np.array([1.0, 2.0]),
    y_pred=np.array([1.1, 1.8]),
    mare=0.0123,
    rmse=0.0456,
    seconds=246.0,
    epochs=143,
    best_epoch=123,
    history=(),
    x_test=np.array([[1.0], [2.0]]),
    groups_test=np.array([0, 0]),
)


def spread_of_mean(mean: float, p95: float = 1e-2) -> Spread:
    """A spread whose mean and 95th percentile the table reads."""
    return Spread(mean, p95, p95, mean)


NS_MASS = PairScores(
    dataset="neutron_stars",
    target="mass",
    scored=(
        Scored(
            "MLP",
            spread_of_mean(0.00662),
            (spread_of_mean(0.007), spread_of_mean(0.009)),
            Timing(213.4, 2e-4, 5e-4),
        ),
        Scored("mean", spread_of_mean(0.2), (spread_of_mean(0.2),) * 2, Timing(0.0, 1e-6, 1e-6)),
        Scored(
            "nearest curve",
            spread_of_mean(0.05),
            (spread_of_mean(0.04), spread_of_mean(0.06)),
            Timing(0.1, 1e-4, 2e-4),
        ),
    ),
)


@pytest.fixture(scope="module")
def pair() -> tuple[PairScores, Fit]:
    """The toy curves scored as one pair through the harness: the MLP, then the references."""
    return score_pair(TOY, "toy", "mass", SHORT)


class TestScorePair:
    def test_scores_the_mlp_and_both_references(self, pair: tuple[PairScores, Fit]) -> None:
        assert [s.predictor for s in pair[0].scored] == ["MLP", "mean", "nearest curve"]

    def test_reports_after_every_mlp_fit(self) -> None:
        fits: list[int] = []
        short = replace(SHORT, max_epochs=1)
        score_pair(TOY, "toy", "mass", short, after_fit=lambda: fits.append(1))
        assert len(fits) == 3  # two folds and the final fit

    def test_times_the_references_too(self, pair: tuple[PairScores, Fit]) -> None:
        assert isinstance(pair[0].scored[2].timing, Timing)

    def test_scores_the_mlp_s_final_fit_with_the_target_s_metric(self) -> None:
        _, fit = score_pair(TOY, "toy", "charge", replace(SHORT, max_epochs=1))
        assert fit.mare == pytest.approx(mare_in_d(fit.y_true, fit.y_pred))


class TestMetricFor:
    def test_scores_the_charge_in_d(self) -> None:
        assert metric_for("charge") is mare_in_d

    def test_scores_the_mass_by_its_mare(self) -> None:
        assert metric_for("mass") is mare


class TestBaselineTable:
    def test_gives_the_mlp_its_mare_fold_spread_figures_and_seconds(self) -> None:
        assert (
            "NS & $M$ & MLP & $6.62\\times 10^{-3}$ & "
            "$8.00\\times 10^{-3} \\pm 1.00\\times 10^{-3}$ & 2.0 & 213 & $2.00\\times 10^{-4}$"
        ) in baseline_table([NS_MASS])

    def test_labels_the_charge_rows_by_d(self) -> None:
        ns_charge = replace(NS_MASS, target="charge")
        assert "NS & $\\Dch$ & MLP &" in baseline_table([ns_charge])

    def test_says_the_charge_rows_are_scored_in_d(self) -> None:
        assert "rebuilt with the true" in baseline_table([NS_MASS])


class TestNumbers:
    def test_the_test_mare_is_scientific(self) -> None:
        assert numbers(SCORED)["baseNsMassMare"] == "1.23\\times 10^{-2}"

    def test_the_test_rmse_is_scientific(self) -> None:
        assert numbers(SCORED)["baseNsMassRmse"] == "4.56\\times 10^{-2}"

    def test_names_the_rmse_seconds_and_epochs_too(self) -> None:
        assert sorted(numbers(SCORED)) == [
            "baseNsMassEpochs",
            "baseNsMassFitSeconds",
            "baseNsMassMare",
            "baseNsMassRmse",
        ]


class TestChargeMares:
    """D = M 10^Y on two test rows: the charge model is exact, the mass model misses row 0."""

    @pytest.fixture
    def mares(self) -> tuple[float, float]:
        mass = replace(SCORED, y_true=np.array([2.0, 4.0]), y_pred=np.array([2.2, 4.0]))
        charge = replace(SCORED, y_true=np.array([-1.0, -1.0]), y_pred=np.array([-1.0, -1.0]))
        return charge_mares(mass, charge)

    def test_with_the_true_mass_only_the_charge_model_errs(self, mares) -> None:
        assert mares[0] == pytest.approx(0.0)

    def test_with_the_predicted_mass_its_error_carries_into_d(self, mares) -> None:
        assert mares[1] == pytest.approx(0.05)


class TestChargeNumbers:
    def test_names_the_true_mass_mare_per_dataset(self) -> None:
        found = charge_numbers({"black_holes": (0.0123, 0.0456)})
        assert found["baseBhChargeMareTrueM"] == "1.23\\times 10^{-2}"

    def test_names_the_predicted_mass_mare_per_dataset(self) -> None:
        found = charge_numbers({"neutron_stars": (0.0123, 0.0456)})
        assert found["baseNsChargeMarePredM"] == "4.56\\times 10^{-2}"


BOUNDED = replace(SHORT, max_epochs=500, patience=20)


class TestStartBanner:
    @pytest.fixture
    def lines(self) -> list[str]:
        return start_banner(
            Path("state/run-1"),
            Path("assets/51_algorithms"),
            BOUNDED,
            rows=(100, 20),
            curves=(10, 2),
            threads=2,
        )

    def test_states_the_torch_threads_per_fit(self, lines: list[str]) -> None:
        assert "threads: 2 torch threads per fit" in lines

    def test_opens_with_the_stop_criterion(self, lines: list[str]) -> None:
        assert lines[:4] == [
            "stop criterion:",
            "  - early stop after 20 epochs in a row without a validation loss "
            "0.01% below the best",
            "  - the best epoch is then restored",
            "  - at most 500 epochs",
        ]

    def test_names_the_run_folder(self, lines: list[str]) -> None:
        assert "folder: state/run-1" in lines

    def test_states_the_rows_and_curves_it_trains_and_scores_on(self, lines: list[str]) -> None:
        assert "data: 100 training rows on 10 curves, 20 test rows on 2 curves" in lines

    def test_states_the_validation_share(self, lines: list[str]) -> None:
        assert "validation: 25% of the training curves" in lines

    def test_lists_the_live_loss_curve(self, lines: list[str]) -> None:
        assert "  - state/run-1/loss_curve.png (redrawn every 5 epochs)" in lines

    def test_lists_the_paper_assets_written_at_the_end(self, lines: list[str]) -> None:
        assert "  - assets/51_algorithms/ (Section 4.1 numbers, parity figure)" in lines

    def test_says_how_to_follow_the_run(self, lines: list[str]) -> None:
        assert "follow live: make follow" in lines


FILES = [Path("assets/51_algorithms/51_algorithms_num.tex"), Path("state/run-1/run.json")]


class TestEndBanner:
    def test_says_an_early_stop_and_after_which_epoch(self) -> None:
        assert (
            "stopped: early -- no better validation loss for 20 epochs after epoch 123"
            in end_banner(SCORED, BOUNDED, FILES)
        )

    def test_gives_the_restored_epoch_the_scores_and_the_time(self) -> None:
        assert (
            "best epoch 123 restored; test MARE 1.23e-02, RMSE 4.56e-02; fit 4m 6s"
            in end_banner(SCORED, BOUNDED, FILES)
        )

    def test_lists_every_file_written(self) -> None:
        lines = end_banner(SCORED, BOUNDED, FILES)
        assert [f"  {path}" for path in FILES] == lines[lines.index("wrote:") + 1 :][:2]

    def test_ends_with_how_to_see_the_runs(self) -> None:
        assert end_banner(SCORED, BOUNDED, FILES)[-1] == (
            "list runs: make runs; show this run: make run"
        )

    def test_says_when_max_epochs_ended_the_fit(self) -> None:
        ran_out = replace(SCORED, epochs=500)
        assert "stopped: max_epochs 500 reached" in end_banner(ran_out, BOUNDED, FILES)


class TestProgressLine:
    def test_estimates_the_time_left_from_the_mean_fit(self) -> None:
        assert progress_line(3, 24, 252.0) == "fit 3/24 done, elapsed 4m 12s, about 29m 24s left"

    def test_says_nothing_is_left_after_the_last_fit(self) -> None:
        assert progress_line(1, 1, 12.0) == "fit 1/1 done, elapsed 12s, about 0s left"


def test_the_parity_figure_draws_the_identity_line() -> None:
    assert len(parity(SCORED, PlotStyle(usetex=False)).axes[0].lines) == 1


def test_the_figure_wrapper_includes_the_parity_image() -> None:
    assert "{51_algorithms_fig_parity}" in figure_tex()


NS_KEYS = [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0), (3.0, 1.0)]
BH_KEYS = [1.0, 2.0, 3.0, 4.0, 5.0]
LABELS = ["fold0", "fold1", "test", "fold0", "fold1"]


def toy_inputs(folder: Path) -> tuple[Path, Path, Path]:
    """Tiny NS- and BH-shaped tables and one split for both: five (beta, lambda) NS curves of
    four rows, five beta BH curves of five rows; one curve of each in test, two in each of two
    folds (a fold fit then trains on two curves, one left for its validation)."""
    ns_rows = [
        {"beta": b, "lambda": lam, "rho_c": 10.0 ** (i + 1), "M": 1.0 + 0.1 * i + 0.05 * b}
        | {"D": 0.1}
        for b, lam in NS_KEYS
        for i in range(4)
    ]
    bh_rows = [
        {"r_h": 4.0 + i, "beta": b, "M": 2.0 + 0.5 * i + 0.01 * b, "D": 0.3 - 0.02 * i}
        for b in BH_KEYS
        for i in range(5)
    ]
    split = [
        {"dataset": "neutron_stars", "beta": b, "lambda": lam, "label": label, "ablation": False}
        for (b, lam), label in zip(NS_KEYS, LABELS, strict=True)
    ] + [
        {"dataset": "black_holes", "beta": b, "label": label, "ablation": False}
        for b, label in zip(BH_KEYS, LABELS, strict=True)
    ]
    ns, bh, split_file = folder / "ns.parquet", folder / "bh.parquet", folder / "split.parquet"
    pd.DataFrame(ns_rows).to_parquet(ns)
    pd.DataFrame(bh_rows).to_parquet(bh)
    pd.DataFrame(split).to_parquet(split_file)
    return ns, bh, split_file


def test_each_curve_is_named_by_its_key(tmp_path: Path) -> None:
    ns, _, _ = toy_inputs(tmp_path)
    names = curve_names(pd.read_parquet(ns), NEUTRON_STARS)
    assert names[1] == "$\\beta$ = 1, $\\lambda$ = 2"


def first_group(pattern: str, path: Path) -> str:
    """The first capture group of pattern in the file at path; a missing match fails loudly."""
    match = re.search(pattern, path.read_text())
    if match is None:
        raise AssertionError(f"{pattern} not found in {path.name}")
    return match.group(1)


@pytest.fixture(scope="module")
def ran(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One main run on the toy NS and BH tables, a stale asset left from an earlier run, the
    commit and the clock injected; returns its working folder (assets/ and state/ inside)."""
    folder = tmp_path_factory.mktemp("run")
    ns, bh, split_file = toy_inputs(folder)
    assets, state = folder / "assets", folder / "state"
    (assets / SECTION).mkdir(parents=True)
    (assets / SECTION / "stale.tex").write_text("from an earlier run")
    # One worker: the shared fit count interleaves across pairs on several (T-179 covers those).
    network = {"width": 8, "depth": 1, "max_epochs": 2, "batch_size": 8, "workers": 1}
    config = PaperConfig.model_validate(
        {"plot": {"usetex": False}, "methodology": {"algorithms": network}}
    )
    main(
        [str(ns), str(bh), str(split_file), str(assets), str(state)],
        config,
        commit=lambda: "abc1234",
        dirty=lambda: True,
        now=lambda: datetime(2026, 10, 6, 10, 56, 0, tzinfo=UTC),
    )
    return folder


class TestMain:
    def test_writes_the_numbers_and_the_parity_figure_into_its_section(self, ran: Path) -> None:
        assert sorted(p.name for p in (ran / "assets" / SECTION).iterdir()) == [
            "51_algorithms_fig_parity.png",
            "51_algorithms_fig_parity.tex",
            "51_algorithms_num.tex",
            "51_algorithms_tab_baseline.tex",
        ]

    def test_writes_the_charge_mare_with_true_and_predicted_mass_per_dataset(
        self, ran: Path
    ) -> None:
        numbers_tex = (ran / "assets" / SECTION / "51_algorithms_num.tex").read_text()
        assert len(re.findall(r"\\base(?:Ns|Bh)ChargeMare(?:True|Pred)M\}", numbers_tex)) == 4

    def test_scores_the_charge_rows_in_d_like_the_true_mass_macro(self, ran: Path) -> None:
        folder = ran / "assets" / SECTION
        numbers_tex, table_tex = (
            folder / "51_algorithms_num.tex",
            folder / "51_algorithms_tab_baseline.tex",
        )
        macro = first_group(r"\\baseBhChargeMareTrueM\}\{(.*)\}\n", numbers_tex)
        row = first_group(r"BH & \$\\Dch\$ & MLP & \$([^$]*)\$", table_tex)
        assert row == macro

    def test_writes_one_mlp_row_per_dataset_and_target(self, ran: Path) -> None:
        table = (ran / "assets" / SECTION / "51_algorithms_tab_baseline.tex").read_text()
        assert table.count("& MLP &") == 4

    def test_leaves_a_run_folder_per_dataset_and_target(self, ran: Path) -> None:
        runs = sorted(p.name for p in (ran / "state" / SECTION).iterdir() if p.is_dir())
        assert [name.rsplit("-", 1)[0] for name in runs] == [
            "black_holes-charge",
            "black_holes-mass",
            "neutron_stars-charge",
            "neutron_stars-mass",
        ]

    def test_leaves_a_ledger_entry_per_predictor_pair_and_seed(self, ran: Path) -> None:
        assert len(list((ran / "state" / "ledger").glob("*.json"))) == 3 * 4 * 1

    def test_leaves_a_completed_trial_naming_each_ledger_entry(self, ran: Path) -> None:
        storage = JournalStorage(JournalFileBackend(str(ran / "state" / "optuna" / "journal.log")))
        (summary,) = optuna.get_all_study_summaries(storage)
        study = optuna.load_study(study_name=summary.study_name, storage=storage)
        told = {t.user_attrs["ledger_id"] for t in study.get_trials(states=[TrialState.COMPLETE])}
        assert told == {path.stem for path in (ran / "state" / "ledger").glob("*.json")}

    def test_records_the_injected_commit(self, ran: Path) -> None:
        run = ran / "state" / SECTION / "neutron_stars-mass-20261006T105600Z"
        assert from_json((run / "run.json").read_text()).commit == "abc1234"

    def test_records_uncommitted_code(self, ran: Path) -> None:
        run = ran / "state" / SECTION / "neutron_stars-mass-20261006T105600Z"
        assert from_json((run / "run.json").read_text()).dirty is True

    def test_logs_the_epoch_table_into_the_run_s_train_log(self, ran: Path) -> None:
        run = ran / "state" / SECTION / "neutron_stars-mass-20261006T105600Z"
        assert "valid_mare" in (run / "train.log").read_text()

    def test_trains_on_the_configured_thread_count(self, ran: Path) -> None:
        assert torch.get_num_threads() == 1

    def test_opens_its_log_with_the_start_banner(self, ran: Path) -> None:
        run = ran / "state" / SECTION / "neutron_stars-mass-20261006T105600Z"
        assert "follow live: make follow" in (run / "train.log").read_text()

    def test_closes_its_log_with_the_end_banner(self, ran: Path) -> None:
        run = ran / "state" / SECTION / "neutron_stars-mass-20261006T105600Z"
        assert "list runs: make runs" in (run / "train.log").read_text()

    def test_counts_the_progress_over_every_fit_of_every_pair(self, ran: Path) -> None:
        run = ran / "state" / SECTION / "neutron_stars-mass-20261006T105600Z"
        assert "fit 3/12 done" in (run / "train.log").read_text()  # 4 pairs x (2 folds + 1)

    def test_points_latest_log_at_the_last_pair_s_train_log(self, ran: Path) -> None:
        run = ran / "state" / SECTION / "black_holes-charge-20261006T105600Z"
        latest = ran / "state" / SECTION / "latest.log"
        assert latest.resolve() == (run / "train.log").resolve()

    def test_leaves_the_diagnostics_in_a_folder_named_after_the_run(self, ran: Path) -> None:
        run = ran / "state" / SECTION / "neutron_stars-mass-20261006T105600Z"
        assert sorted(p.name for p in run.iterdir()) == [
            "curves.png",
            "error_cdf.png",
            "loss_curve.png",
            "run.json",
            "train.log",
        ]


class TestFinalFit:
    """The MLP's final fit, built from the harness run and the fitted network."""

    def test_is_scored_on_the_test_rows_only(self, fit: Fit) -> None:
        assert fit.y_true.tolist() == TOY.y[TOY.test].tolist()

    def test_times_the_fit_with_the_clock(self, fit: Fit) -> None:
        assert fit.seconds == pytest.approx(2.5)

    def test_counts_the_epochs_run(self, fit: Fit) -> None:
        assert fit.epochs == SHORT.max_epochs

    def test_names_the_epoch_of_the_lowest_valid_loss(self, fit: Fit) -> None:
        assert fit.best_epoch == min(fit.history, key=lambda row: row["valid_loss"])["epoch"]

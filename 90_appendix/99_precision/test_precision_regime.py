"""Facts about the precision run of the tuning and precision appendix (W-093), on the toy tables
and a toy summary of k-NN pointwise winners: nothing here fits a real surrogate."""

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from precision_options import Option
from precision_regime import SECTION, Scored, kept_option, main, precision_numbers, read_saved

from paperlib.batches import read_ledger
from paperlib.config import PaperConfig

ToyInputs = Callable[[Path], tuple[Path, Path, Path]]

PAIRS = [
    (dataset, target)
    for dataset in ("neutron_stars", "black_holes")
    for target in ("mass", "charge")
]
APPLICABLE = ("tuned", "residual")
SEEDS = 2

TOY_CONFIG = PaperConfig.model_validate(
    {
        "plot": {"usetex": False},
        "network": {"width": 8, "depth": 1, "max_epochs": 2, "batch_size": 8},
    }
)
TOY_KNOBS = [
    *("--k", "2", "--neighbours", "4", "--knots", "4", "--residual-neighbours", "5"),
    *("--seeds", str(SEEDS), "--minutes", "30", "--workers", "1"),
]


def write_summary(state: Path) -> Path:
    """The saved surrogates of make pipeline: a k-NN pointwise winner per pair."""
    rows = [
        {
            "dataset": dataset,
            "target": target,
            "family": "k-NN",
            "unit": "pointwise",
            "params": {"weights": "distance"},
            "test": 3.0,
            "ceiling": 4.1,
        }
        for dataset, target in PAIRS
    ]
    path = state / "41_algorithm" / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rows))
    return path


def paths(folder: Path, assets: str = "assets") -> list[str]:
    """The toy inputs (toy_inputs writes them) and the asset and state folders, as arguments."""
    names = ("ns.parquet", "bh.parquet", "split.parquet")
    return [*(str(folder / name) for name in names), str(folder / assets), str(folder / "state")]


@pytest.fixture(scope="module")
def ran(tmp_path_factory: pytest.TempPathFactory, toy_inputs: ToyInputs) -> Path:
    """One full run on the toy tables, one worker, the commit and clock injected; returns its
    working folder (assets/ and state/ inside)."""
    folder = tmp_path_factory.mktemp("precision")
    toy_inputs(folder)
    write_summary(folder / "state")
    main(
        [*paths(folder), *TOY_KNOBS],
        TOY_CONFIG,
        commit=lambda: "abc1234",
        dirty=lambda: False,
        now=lambda: datetime(2026, 11, 11, 12, 0, 0, tzinfo=UTC),
    )
    return folder


def refuse(*args: Any, **kwargs: Any) -> None:
    """A fitter that must never be built."""
    raise AssertionError("a saved batch fits nothing")


def batch_of(folder: Path) -> str:
    """The id of the one saved batch of the run."""
    (saved,) = (folder / "state" / SECTION).glob("*-precision.json")
    return saved.name.removesuffix("-precision.json")


@pytest.mark.functional
def test_the_run_scores_one_entry_per_pair_option_and_seed(ran: Path) -> None:
    entries = read_ledger(ran / "state" / "ledger")
    assert len(entries) == len(PAIRS) * len(APPLICABLE) * SEEDS


@pytest.mark.functional
def test_the_run_saves_the_scores_per_pair_and_option(ran: Path) -> None:
    saved = ran / "state" / SECTION / f"{batch_of(ran)}-precision.json"
    rows = json.loads(saved.read_text())
    assert {(r["dataset"], r["target"], r["option"]) for r in rows} == {
        (d, t, o) for d, t in PAIRS for o in APPLICABLE
    }


@pytest.mark.functional
def test_the_run_writes_the_figure_png(ran: Path) -> None:
    png = ran / "assets" / SECTION / "figures" / f"{SECTION}_fig_precision.png"
    assert png.is_file()


@pytest.mark.functional
def test_the_wrapper_includes_the_png_stem(ran: Path) -> None:
    wrapper = (ran / "assets" / SECTION / f"{SECTION}_fig_precision.tex").read_text()
    assert f"{{{SECTION}_fig_precision}}" in wrapper


@pytest.mark.functional
def test_the_wrapper_labels_the_figure(ran: Path) -> None:
    wrapper = (ran / "assets" / SECTION / f"{SECTION}_fig_precision.tex").read_text()
    assert "\\label{fig:precision}" in wrapper


@pytest.mark.functional
def test_the_macros_name_each_pair_and_option(ran: Path) -> None:
    macros = (ran / "assets" / SECTION / f"{SECTION}_precision_num.tex").read_text()
    pairs = ("NsMass", "NsCharge", "BhMass", "BhCharge")
    options = ("Tuned", "Double", "Lbfgs", "Residual")
    names = [
        f"\\newcommand{{\\prec{pair}{option}{kind}}}"
        for pair in pairs
        for option in options
        for kind in ("Folds", "Test")
    ]
    names += [
        f"\\newcommand{{\\prec{pair}{kind}}}"
        for pair in pairs
        for kind in ("Spread", "Gain", "Kept")
    ]
    assert all(name in macros for name in [*names, "\\newcommand{\\precSeeds}"])


@pytest.mark.functional
def test_a_saved_batch_rebuilds_the_wrapper_with_no_fit(ran: Path) -> None:
    main([*paths(ran, "rebuilt"), *TOY_KNOBS, "--batch", batch_of(ran)], TOY_CONFIG, fitter=refuse)
    wrapper = ran / "rebuilt" / SECTION / f"{SECTION}_fig_precision.tex"
    assert wrapper.read_text() == (ran / "assets" / SECTION / wrapper.name).read_text()


@pytest.mark.functional
def test_a_saved_batch_rebuilds_the_macros_with_no_fit(ran: Path) -> None:
    main([*paths(ran, "again"), *TOY_KNOBS, "--batch", batch_of(ran)], TOY_CONFIG, fitter=refuse)
    macros = ran / "again" / SECTION / f"{SECTION}_precision_num.tex"
    assert macros.read_text() == (ran / "assets" / SECTION / macros.name).read_text()


def scored(option: Option, *folds: tuple[float, ...]) -> Scored:
    """Hand-made figures of one option on one pair: a tuple of fold figures per seed."""
    return Scored("neutron_stars", "mass", option, folds, tuple(1.0 for _ in folds))


TUNED = scored("tuned", (3.0, 3.2), (3.0, 3.2))  # mean 3.1, spread 0.1


@pytest.mark.model
def test_an_option_whose_gain_exceeds_the_spread_is_kept() -> None:
    better = scored("residual", (3.5, 3.7), (3.5, 3.7))  # gain 0.5
    assert kept_option([TUNED, better]) == "residual"


@pytest.mark.model
def test_tuned_is_kept_when_no_gain_exceeds_the_spread() -> None:
    barely = scored("residual", (3.1, 3.2), (3.1, 3.2))  # gain 0.05, under the spread
    assert kept_option([TUNED, barely]) == "tuned"


@pytest.mark.model
def test_every_macro_name_is_letters_only() -> None:
    numbers = precision_numbers([TUNED, scored("residual", (3.5, 3.7), (3.5, 3.7))])
    assert all(re.fullmatch(r"prec[A-Za-z]+", name) for name in numbers)


@pytest.mark.model
def test_every_na_body_is_wrapped_in_ensuremath() -> None:
    numbers = precision_numbers([TUNED, scored("residual", (3.5, 3.7), (3.5, 3.7))])
    na = [body for body in numbers.values() if "n/a" in body]
    assert na == ["\\ensuremath{\\mathrm{n/a}}"] * len(na) != []


@pytest.mark.edge
def test_a_missing_summary_raises_naming_its_path(tmp_path: Path) -> None:
    missing = tmp_path / "41_algorithm" / "summary.json"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        read_saved(missing)

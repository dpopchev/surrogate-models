"""Facts about the pipeline as one algorithm (W-152), on tiny synthetic tables, a recording save."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from build_surrogates import Sources, run

from paperlib.candidates import Knobs
from paperlib.config import NetworkSettings
from paperlib.domain import Domain
from paperlib.final import FinalSurrogate
from paperlib.surrogate import make_training

ToyInputs = Callable[[Path], tuple[Path, Path, Path]]

KNOBS = Knobs(k=3, neighbours=20, knots=4, per_curve=0)
PAIRS = [
    (dataset, target)
    for dataset in ("neutron_stars", "black_holes")
    for target in ("mass", "charge")
]


class Recorded:
    """What one toy run did: the surrogates it saved by path, the text it wrote, its lines."""

    def __init__(self) -> None:
        self.saved: dict[Path, FinalSurrogate] = {}
        self.written: dict[Path, str] = {}
        self.domains: dict[Path, Domain] = {}
        self.lines: list[str] = []


def write_state(state: Path, *, with_winners: bool = True) -> None:
    """The recorded inputs of the pipeline: both ceilings and a k-NN winner per pair."""
    for folder in ("91_neutron_stars", "92_black_holes"):
        (state / folder).mkdir(parents=True)
        (state / folder / "reachable_figures.json").write_text('{"mass": 4.1, "charge": 2.4}')
    if with_winners:
        (state / "99_precision").mkdir(parents=True)
        winners = [
            {
                "dataset": d,
                "target": t,
                "family": "k-NN",
                "unit": "pointwise",
                "params": {"weights": "distance"},
            }
            for d, t in PAIRS
        ]
        (state / "99_precision" / "winners.json").write_text(json.dumps(winners))


def run_toy(folder: Path, toy_inputs: ToyInputs) -> Recorded:
    """One run on the toy tables with the effects recorded."""
    ns, bh, split = toy_inputs(folder)
    write_state(folder / "state")
    done = Recorded()
    run(
        Sources(ns, bh, split, folder / "state"),
        KNOBS,
        make_training(NetworkSettings()),
        seed=1,
        save=done.saved.__setitem__,
        save_domain=done.domains.__setitem__,
        write=done.written.__setitem__,
        say=done.lines.append,
    )
    return done


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory: pytest.TempPathFactory, toy_inputs: ToyInputs) -> Recorded:
    return run_toy(tmp_path_factory.mktemp("pipeline"), toy_inputs)


@pytest.mark.functional
def test_run_saves_one_surrogate_per_pair(toy_run: Recorded) -> None:
    assert sorted(path.name for path in toy_run.saved) == sorted(f"{d}-{t}.pkl" for d, t in PAIRS)


@pytest.mark.functional
def test_run_saves_the_domain_of_each_map(toy_run: Recorded) -> None:
    assert sorted(path.name for path in toy_run.domains) == [
        "black_holes-domain.json",
        "neutron_stars-domain.json",
    ]


@pytest.mark.functional
def test_run_prints_each_pair_beside_its_ceiling(toy_run: Recorded) -> None:
    assert toy_run.lines[0].startswith("neutron_stars mass: k-NN pointwise, test p95 ")


@pytest.mark.functional
def test_run_summarises_the_pairs_as_plain_data(toy_run: Recorded) -> None:
    (summary,) = (json.loads(text) for path, text in toy_run.written.items())
    assert [(row["dataset"], row["target"], row["ceiling"]) for row in summary] == [
        (d, t, 4.1 if t == "mass" else 2.4) for d, t in PAIRS
    ]


@pytest.mark.edge
def test_a_missing_input_fails_naming_its_path(tmp_path: Path, toy_inputs: ToyInputs) -> None:
    ns, bh, split = toy_inputs(tmp_path)
    write_state(tmp_path / "state", with_winners=False)
    with pytest.raises(FileNotFoundError, match=r"winners\.json"):
        run(
            Sources(ns, bh, split, tmp_path / "state"),
            KNOBS,
            make_training(NetworkSettings()),
            seed=1,
            save=lambda path, surrogate: None,
            save_domain=lambda path, domain: None,
            write=lambda path, text: None,
            say=lambda line: None,
        )

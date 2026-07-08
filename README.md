# surrogate-models

Machine-learning surrogate models over scientific datasets, packaged as
`surrogate_models`.

## Overview

A surrogate model replaces an expensive numerical solve with a fast learned
approximation. This repository hosts that work -- data preparation, model
training, and evaluation -- developed under a strict functional-core /
onion-architecture discipline with test-driven development.

Status: the toolchain (uv, Make, JupyterLab) and project conventions are in place,
and two bounded contexts are live. The `datasets` context ingests raw
neutron-star `.dat` output into typed, schema-certified frames and exposes a
one-call public loader, `load_neutron_stars`. The `mlmodels` context manages
training as a thin vertical slice: you GIVE it a model -- an injected PyTorch
Lightning module -- and it runs training behind the imperative shell under a
certified configuration (epochs, learning rate, batch size, optimizer) and writes
a checkpoint, streaming a live progress bar with per-step loss for interactive
runs. It is exercised end to end today with a stub regressor (the real
neutron-stars surrogate is the next slice). The `python -m surrogate_models`
command-line entry point is still a placeholder; the library API below works
today.

## Public API

Load the neutron-stars dataset as a pandas `DataFrame` with one call:

```python
from surrogate_models import load_neutron_stars

df = load_neutron_stars()
```

`load_neutron_stars()` is get-or-build with read-through caching:

1. It looks for a stored `neutron-stars` dataset under `datasets.path`
   (the parquet store) and returns it if present.
2. On a miss, it digests the raw source at `datasets.neutron_stars_source`,
   persists the certified dataset to the store, then returns the freshly read
   frame.

The returned frame carries every column from the raw batches plus a `pc_init`
column -- each batch's initial central pressure, taken from its comment header.
On failure (for example a missing source file) the call raises.

### Configuration

Settings resolve highest-priority first: OS environment (and `.env`), then
`surrogate_models.toml`, then the built-in defaults. Variables are prefixed
`SURROGATE_MODELS__` and nest into sections with the same `__` delimiter.

| Variable                                       | Default                                | Purpose                                              |
|------------------------------------------------|----------------------------------------|------------------------------------------------------|
| `SURROGATE_MODELS__DATASETS__PATH`             | `var/data/surrogate_models/datasets`   | Where certified datasets are persisted (parquet)     |
| `SURROGATE_MODELS__DATASETS__NEUTRON_STARS_SOURCE` | `data/neutron-stars/neutron-stars.dat` | Raw concatenated neutron-stars `.dat` the loader digests |
| `SURROGATE_MODELS__MLMODELS__CHECKPOINT_DIR`   | `var/data/surrogate_models/checkpoints` | Where a training run writes its Lightning `.ckpt`    |

See `.env.example` and `surrogate_models.toml.example` for a copy-ready template.

## Prerequisites

| Tool   | Version    | Notes                                              |
|--------|------------|----------------------------------------------------|
| uv     | >= 0.11    | Python packaging and environment manager           |
| Python | 3.14.5     | Pinned in `.python-version`; provisioned by uv     |
| make   | any GNU    | Drives every workflow (run `make help`)            |

You only need uv installed. It provisions the pinned Python interpreter and the
virtual environment for you.

## Getting Started

```sh
git clone <repo-url> surrogate-models
cd surrogate-models

make venv       # create .venv with Python 3.14.5 (from .python-version)
make sync       # sync dependencies from uv.lock into .venv
make install    # install the project into .venv (editable)
```

List every available task with its description:

```sh
make help
```

## JupyterLab

Exploration happens in notebooks under `notebooks/`. The `lab` dependency group
provides JupyterLab with vim bindings, LSP, and `nbstripout` (which keeps
notebook diffs clean by stripping outputs on commit).

```sh
make lab-install     # add and sync the lab toolchain into .venv
make lab-hooks       # install the nbstripout git filter + notebooks/.gitattributes
make lab             # start JupyterLab rooted at notebooks/ (http://127.0.0.1:8888)
```

Override the host or port:

```sh
make lab LAB_HOST=0.0.0.0 LAB_PORT=9000
```

## Make Targets

| Target              | Purpose                                                |
|---------------------|--------------------------------------------------------|
| `make help`         | List all targets with descriptions                     |
| `make venv`         | Build `.venv` from `.python-version`                    |
| `make sync`         | Sync dependencies from `uv.lock`                        |
| `make install`      | Install the project into `.venv` (editable)            |
| `make run`          | Start the `surrogate_models` entry point (placeholder) |
| `make test`         | Run the test suite with coverage                       |
| `make test-quick`   | Run the test suite quietly (per TDD cycle)             |
| `make typecheck`    | Static type-check `src/` with mypy                     |
| `make lint`         | Ruff lint + format check + import contracts            |
| `make lint-imports` | Check onion-ring import contracts with import-linter   |
| `make format`       | Auto-format and fix lint with ruff                     |
| `make build`        | Build sdist + wheel into `dist/` (tests must pass)     |
| `make lab-install`  | Install the JupyterLab toolchain (group: `lab`)        |
| `make lab-hooks`    | Install the nbstripout git filter                      |
| `make lab`          | Run JupyterLab rooted at `notebooks/`                  |

## Project Layout

```
.
|-- src/surrogate_models/
|   |-- __init__.py           # public API (re-exports load_neutron_stars)
|   |-- __main__.py           # placeholder CLI entry point
|   |-- config.py             # app-wide settings shell (pydantic-settings)
|   |-- datasets/             # datasets bounded context
|   |   |-- domain.py         # functional core: Dataset, DatasetID, smart constructors
|   |   |-- application.py    # CQRS handlers (make / get dataset)
|   |   |-- infrastructure.py # imperative shell: parquet I/O, .dat ingest
|   |   `-- __main__.py       # context root + load_neutron_stars facade
|   |-- mlmodels/             # mlmodels (training) bounded context
|   |   |-- domain.py         # functional core: TrainingRun states, RunID, TrainingConfig
|   |   |-- application.py    # CQRS handler (train run) over injected build_model / train
|   |   |-- infrastructure.py # imperative shell: Lightning Trainer adapter + stub model
|   |   `-- __main__.py       # context root + train_stub_run facade
|   `-- railway_adts/         # Result / Option / @safe railway primitives
|-- tests/                    # mirror of src/, test-first
|-- Makefile                  # task runner (environment, quality, build, JupyterLab)
|-- pyproject.toml            # project metadata, dependency groups, import contracts
|-- .python-version           # pinned interpreter (3.14.5)
`-- notebooks/                # JupyterLab exploration (outputs stripped on commit)
```

## Development

The codebase follows an onion architecture with bounded contexts under
`src/surrogate_models/`, built test-first. Each context splits into a pure
`domain` core, an `application` layer of CQRS handlers wired over injected
callables, and an `infrastructure` shell that owns all I/O; the composition root
(`__main__.py`) wires the rings and never reaches a domain directly. Import
direction is enforced by import-linter contracts (`make lint-imports`).

The quality gates run as: `make format`, `make lint`, `make typecheck`, then
`make test`. The build gate (`make build`) requires a green test suite before
producing artifacts.

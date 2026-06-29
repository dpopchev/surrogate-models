# regression-surrogate-pde-solver

Machine-learning regression surrogates that approximate solutions to partial
differential equations (PDEs), packaged as `regression_surrogate_pde_solver`.

## Overview

A surrogate model replaces an expensive numerical PDE solve with a fast learned
approximation. This repository hosts that work: data preparation, model training,
and evaluation for regression-based PDE surrogates, developed under a strict
functional-core / onion-architecture discipline with test-driven development.

Status: early scaffold. The toolchain (uv, Make, JupyterLab) and project
conventions are in place. The `regression_surrogate_pde_solver` package, its `src/` tree, and the test
suite are not implemented yet; `main.py` is a placeholder. Commands below are
split into what runs today and what activates once the package lands.

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
git clone <repo-url> regression-surrogate-pde-solver
cd regression-surrogate-pde-solver

make venv     # create .venv with Python 3.14.5 (from .python-version)
make sync     # sync dependencies from uv.lock into .venv
```

List every available task with its description:

```sh
make help
```

Run the current placeholder entry point as a sanity check:

```sh
uv run python main.py     # prints: Hello from regression-surrogate-pde-solver!
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

Available now:

| Target            | Purpose                                                  |
|-------------------|----------------------------------------------------------|
| `make help`       | List all targets with descriptions                       |
| `make venv`       | Build `.venv` from `.python-version`                      |
| `make sync`       | Sync dependencies from `uv.lock`                          |
| `make lab-install`| Install the JupyterLab toolchain (group: `lab`)          |
| `make lab-hooks`  | Install the nbstripout git filter                        |
| `make lab`        | Run JupyterLab rooted at `notebooks/`                     |

Active once the `regression_surrogate_pde_solver` package and tests exist:

| Target              | Purpose                                                |
|---------------------|--------------------------------------------------------|
| `make install`      | Install the project into `.venv` (editable)            |
| `make run`          | Start the `regression_surrogate_pde_solver` entry point and show its help    |
| `make test`         | Run the test suite with coverage                        |
| `make test-quick`   | Run the test suite quietly (per TDD cycle)             |
| `make typecheck`    | Static type-check `src/` with mypy                      |
| `make lint`         | Ruff lint + format check + import contracts            |
| `make lint-imports` | Check onion-ring import contracts with import-linter    |
| `make format`       | Auto-format and fix lint with ruff                      |
| `make build`        | Build sdist + wheel into `dist/` (tests must pass)     |

## Project Layout

```
.
|-- main.py            # placeholder entry point
|-- Makefile           # task runner (environment, quality, build, JupyterLab)
|-- pyproject.toml     # project metadata and dependency groups
|-- .python-version    # pinned interpreter (3.14.5)
|-- uv.lock            # resolved dependency lockfile
`-- notebooks/         # JupyterLab exploration (outputs stripped on commit)
```

## Development

The codebase follows an onion architecture with bounded contexts under
`src/regression_surrogate_pde_solver/`, built test-first. Once the package lands, the quality gates run
as: `make format`, `make lint`, `make typecheck`, then `make test`. The build
gate (`make build`) requires a green test suite before producing artifacts.

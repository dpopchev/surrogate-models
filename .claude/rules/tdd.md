---
description: Python TDD mechanics -- how RED-GREEN-REFACTOR binds to this stack (test runner, stub form, dependency injection, real-data source). The universal cycle lives in the global build-with-tdd skill; this rule is its python binding.
paths:
  - "tests/**/*.py"
  - "src/**/*.py"
---

# TDD mechanics (python)

Binds the global `build-with-tdd` cycle to this stack. Read that skill for the
RED-GREEN-REFACTOR loop and vertical-slice framing; apply these mechanics inside it.

- [do] Run tests through Make targets: `make test-quick` per cycle, `make test`
  (full) after REFACTOR. NEVER bare `pytest` (Makefile gate). No pytest-arg
  passthrough exists -- add a target before running a focused test.
- [do] Place slice stubs as `Err(NotImplemented)` (from `shared_kernel.result`) that
  fix the typed signature; replace a stub only when a RED test forces it.
- [do] Construct test inputs as DOMAIN objects directly. NEVER parse a source string
  to build a test input.
- [do] Inject I/O dependencies as typed lambdas matching the port `Callable` aliases.
  NEVER `mock.patch` or `monkeypatch`.
- [do] Tests ALWAYS use MINIMAL, small, hand-authored fixtures -- the fewest tiny
  rows/columns (2-3 rows; generic names like `x1`/`y`) that prove the point -- for
  EVERY tier, e2e included. If a test needs many rows or the real schema to pass, it is
  over-specified; shrink it to the minimal example.
- [don't] NEVER put a real/production dataset -- or its SCHEMA (real column names),
  its SCALE (true row counts), or its CONTENTS -- into `tests/`. The real datasets are
  private and NOT in the repo; a test must not depend on them, read `data/**`, or
  hard-code their column names or sizes (that leaks private data). Real data is loaded
  only by app/infrastructure adapters at RUNTIME, never by a test.
- [don't] Return bare values from command handlers -- they return `Result[T, E]`
  (see the architecture rule).

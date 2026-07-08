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
- [do] Source concrete values from `data/{concept}/prepared/` (per the data-layout
  rule). If absent, HALT and ask -- never read `orig/`, never invent domain data.
- [don't] Return bare values from command handlers -- they return `Result[T, E]`
  (see the architecture rule).

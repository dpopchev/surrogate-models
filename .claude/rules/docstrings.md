---
paths:
  - "**/*.py"
  - "*.py"
---

# Docstrings -- Hard Gates

Straightforward, plain-language documentation INSIDE Python code. Every module and
every public object says what it is and why it exists, in the fewest clear words.
Part of the python foundation profile: applies to every `.py` file in a project that
instantiates the profile. (Prose documentation under `docs/` is governed separately by
the `docs` gate.)

## Module docstrings

- ALWAYS: open every module with a docstring. Its first line is one plain sentence
  naming what the module holds and what it is for -- e.g. `"""Order aggregate: states,
  transitions, and smart constructors."""`.
- ALWAYS: add at most one short paragraph after that line when the module needs it --
  its role in the system, not a listing of its contents.
- NEVER: a module with no docstring, or a docstring that just repeats the filename.

## Function, method, and class docstrings

- ALWAYS: start with a one-line imperative summary on its own line, ending with a
  period -- `"""Confirm a draft order."""`.
- ALWAYS: say WHAT it does and WHY it matters -- the intent a reader cannot recover
  from the signature. Document only the non-obvious: errors returned/raised, side
  effects, invariants, units, edge cases.
- NEVER: restate the signature in prose ("Takes an order_id string and returns a
  string"). Types live in the annotations; the docstring adds meaning, not noise.
- NEVER: ceremony or boilerplate -- no empty `Args:`/`Returns:` scaffolding that only
  echoes the annotations, no author/date/changelog tags, no restating the obvious.
- ALWAYS: plain language over jargon. Straightforward beats clever.

## Style and upkeep

- ALWAYS: triple-double-quoted `"""..."""`; the summary line fits on one physical
  line; wrap any following prose at the project's line width.
- ALWAYS: a trivial, self-evident private helper may carry a one-line docstring or
  none -- clarity is the goal, not coverage for its own sake.
- ALWAYS: keep docstrings current. A docstring that no longer matches the code is a
  bug -- fix it in the same change.

---
paths:
  - "Makefile"
---

# Makefile -- Hard Gates

Hard constraints for the build system.

- ALWAYS use `uv run` for every Python target -- never bare `python` or `pytest`.
- NEVER run bare `pytest` or `uv run pytest` outside Make targets.
- NEVER hardcode the Python version -- derive from `.python-version`.
- ALWAYS mark every target `.PHONY` and give it a `## <description>` help comment.
- ALWAYS prefix recipe commands with `@`.
- NEVER add a target without a help comment -- `make help` must list it.

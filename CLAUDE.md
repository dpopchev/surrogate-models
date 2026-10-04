# CLAUDE.md -- surrogate-models

The work record is `.board/` (decisions on E-001 and E-002). This file holds only
what `~/.claude/rules/python.md` asks a repository to record here.

## Gate deviations

- rules/python.md, Directories: `local/state/` holds runtime state -- prepared tables, splits, trained weights, metrics -- written by make targets and kept by `make clean`; `build/` holds only what make builds and orchestrates into the paper. Reason: E-002 Decisions (developer). Pending the `~/.claude` rule change requested from claude-f5 on 2026-10-04.

---
paths:
  - ".state-file.md"
---

# State File -- Hard Gates

The full protocol (the 7 update triggers, the tree-format template, the
acknowledgement gates) lives in CLAUDE.md -> "State File Protocol". The
hypothesis-tree lifecycle is the operating mode of the whole system -- the main loop
and every agent inherit it from CLAUDE.md -- so whichever agent holds the active
task drives it; there is no separate owner. The path-scoped gates:

- ALWAYS mutate `.state-file.md` with `Edit`/`Write` only.
- NEVER update the state file just because a tool ran -- only on one of the 7
  structural triggers.
- ALWAYS halt for user acknowledgement before altering the Day-One
  Hypothesis, adding a user-provided constraint, or crossing a phase boundary.
- NEVER delete refuted branches -- move them to a "Pruned:" section with the
  refutation reason and evidence citation.

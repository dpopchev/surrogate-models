---
paths:
  - "data/**"
---

# Data Layout -- Hard Gates

The layout and invariants for `data/` in a data-science project: raw inputs are
preserved untouched, and everything downstream reads from a machine-ready prepared
form. The data overlay sits ON TOP of the bounded-context core -- prepared data is
what the domain is modeled from and what tests source fixtures from. The
characterization/prioritization and sampling PROCEDURES live in the
`characterize-and-prioritize` and `sample-trace-data` skills; this rule owns the
STRUCTURE and its invariants. `data/` holds MACHINE artifacts only -- the human-facing
understanding of this data (summaries, explanations, data dictionaries, illustrative
subsets) lives in `docs/wiki`, which is data-aware and summarizes + links this tree
(the `docs` rule).

## Layout

```text
data/
+-- {concept}/              <- one directory per source / concept / dataset
|   +-- orig/               <- raw inputs AND outputs, exactly as received
|   +-- prepared/           <- massaged, machine-ready in/out
|       +-- examples/       <- machine-readable in/out pairs: CSV with headers, JSON, YAML
+-- MANIFEST.md             <- provenance index: every concept, source, timestamp, transformation
```

## Gates

- ALWAYS: raw data lands in `data/{concept}/orig/` and is IMMUTABLE after
  placement -- the source of truth. NEVER modify a file under `orig/`.
- ALWAYS: everything downstream reads from `data/{concept}/prepared/`, never from
  `orig/` directly. Preparation is orig -> prepared, one direction.
- ALWAYS: preparation is DETERMINISTIC -- the same `orig/` input yields the same
  `prepared/` output. Strip non-deterministic noise (timestamps, PIDs, absolute
  paths, random tokens); replace with deterministic placeholders where structure
  must be preserved.
- ALWAYS: prepared files are self-documenting and machine-ready -- field names
  carry the semantics; CSV has headers; formats are JSON / CSV / YAML. Names read
  in a machine-ready, consistent pattern.
- ALWAYS: a sample set is the SMALLEST subset covering all observed logical
  pathways -- at minimum one happy-path, one boundary, one fault case.
- ALWAYS: every `{concept}` in `data/` has a `data/MANIFEST.md` entry recording its
  source, timestamp, and what each transformation changed and why (reproducible).
- NEVER: write fixtures into `tests/` from here -- tests are built FROM
  `prepared/` (fixtures constructed as domain objects, never invented). Prepared
  data is also where the domain model is discovered (entities, states, vocabulary).

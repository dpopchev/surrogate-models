---
paths:
  - "docs/**"
---

# Docs -- Hard Gates

`docs/` is the project's KNOWLEDGE BASE: a browsable, summary-first, cross-linked wiki
that COMPLEMENTS the code and is AWARE of `data/`. It never mirrors code and never holds
heavy data -- it SUMMARIZES and LINKS them. Three layers: `raw/` (source document dumps),
`wiki/` (distilled understanding), `outputs/` (deliverables). At this scale indexes +
summaries beat RAG. The `docs-init` / `docs-update` / `docs-health-check` skills scaffold,
grow, and lint it; this rule owns the STRUCTURE and its invariants. Every claim traces to
a source artifact -- a `src/` signature, a `tests/` case, a `data/` example -- never
invent claims.

## Layout

```text
docs/
+-- README.md            <- front door: what this base is + the map (the hub)
+-- CHANGELOG.md         <- running log: ingestions, new articles, promotions, health checks
+-- raw/                 <- source DOCUMENT dumps (papers, specs, exports, notes)
|   +-- MANIFEST.md      <- the raw registry ("ingested" list): one row per source
+-- wiki/                <- distilled, indexed, backlinked knowledge (summary-first)
|   +-- index.md         <- map of concepts + articles
|   +-- QUESTIONS.md     <- open questions + cross-article tensions worth resolving
|   +-- concepts/        <- one idea per file, backlinked; singular source:
|   +-- articles/        <- syntheses spanning several concepts; a sources: list + optional status
+-- outputs/             <- answers, reports, charts, machine-ready feature specs
    +-- index.md         <- log; reusable outputs promoted back into wiki/
```

`docs/raw/` holds source DOCUMENTS you read to understand (distinct from `data/{concept}/orig/`,
which holds numeric datasets the pipeline computes on -- see the `data-layout` rule).

## Format gates (absorbed from the former documentation rule)

- ALWAYS GitHub Flavored Markdown, ASCII-only, trailing newline, ATX headers, fenced code
  blocks with a language id. NEVER binary formats (.pdf/.docx/.pptx).
- ALWAYS front-load the conclusion (Synopsis / Summary / Data); tables for 3+ item
  comparisons; Mermaid fenced blocks for diagrams.
- ALWAYS lowercase-hyphenated descriptive filenames; slugs kebab-case, unique in folder.
- NEVER document speculative or unimplemented features -- only what exists in code / data.

## Knowledge gates

- ALWAYS `wiki/` is DISTILLED, not dumped: a wiki note is the paragraph you would tell a
  colleague, not the source pasted in. Raw material lands in `raw/` (with a MANIFEST row);
  the value is the compiled `wiki/` summary and the index that points at it.
- ALWAYS every wiki note CITES what it distills in frontmatter + inline links: a CONCEPT
  carries a singular `source:` (a `src/` signature `file.py:function`, a `tests/` case, a
  `data/` example, or a `raw/` document); an ARTICLE carries a `sources:` LIST whose entries
  each resolve to a `raw/` file, plus an optional `status: speculative` while provisional.
  Complement the code; NEVER mirror it (link to the source of truth, do not copy it).
- ALWAYS index in the SAME change: adding/moving a note updates the enclosing `index.md`
  and the backlinks on BOTH ends. An unindexed note does not exist.
- ALWAYS record change + tension: every ingestion / new article / promotion earns a
  `docs/CHANGELOG.md` line, and every open question or cross-article tension lives in
  `wiki/QUESTIONS.md`. These are the base's memory -- `docs-health-check` reads them to
  measure the delta and to avoid re-flagging known gaps.
- ALWAYS `outputs/` that hold reusable knowledge are PROMOTED: distill a `wiki/` summary
  and backlink the output (the self-improving loop).

## Data-awareness gates (`docs/` is aware of `data/`)

- ALWAYS heavy data stays in `data/` -- NEVER copy a dataset, feature matrix, or large CSV
  into `docs/`. The wiki SUMMARIZES and LINKS them.
- ALWAYS for each `data/{concept}` the wiki carries three things: a SUMMARY (key stats --
  counts, ranges, distributions), a REFERENCE (data dictionary / column schema /
  vocabulary), and a small illustrative high-variance SUBSET (inline table or small linked
  example). The machine fixture set is separate and stays in
  `data/{concept}/prepared/examples/` (the `data-layout` rule + `sample-trace-data` skill).
- ALWAYS generating or refreshing a summary / reference / subset from `data/` is PROPOSED
  then HALTS for confirmation -- never auto-materialize one without the user's OK.
  Human-in-the-loop is the gate.

## Traceability

- ALWAYS cite real source paths and real `data/` examples; update docs in the SAME commit
  as the code / data they describe.

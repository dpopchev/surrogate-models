---
name: sample-trace-data
description: Ingest raw traces or logs, profile distributions, clean noise, and export minimal high-variance fixture sets. Use when asked to "sample data", "clean traces", "generate fixtures from logs", or "profile test data".
---

# Sample Trace Data

Prepares fixtures under the data-layout gate (`.claude/rules/data-layout.md`).

## Goal & Triggers

Ingest raw runtime data (logs, terminal dumps, I/O pairs), profile the logical
distribution, strip non-deterministic noise, and export the smallest possible
fixture set covering all observed pathways. Output lands in
`data/{context}/prepared/` ready for downstream domain modeling and TDD fixtures.

Triggers: "sample data", "clean traces", "generate fixtures from logs",
"profile test data", "extract test cases from output".

## Inputs / Outputs

| Input | Required | Description |
|-------|:--------:|-------------|
| Raw data source | Yes | File path, terminal output, or pasted log content |
| Target consumer | No | domain modeling or TDD fixtures (default: infer from content) |
| Variance goal | No | Minimum category count (default: 3 -- happy, boundary, fault) |

| Output | Description |
|--------|-------------|
| Prepared sample(s) | JSON/YAML in `data/{context}/prepared/examples/` |
| Manifest | Category distribution summary; update `data/MANIFEST.md` |

## Step-by-Step Workflow

### Step 1: Ingest and Identify Format

Read the raw data. Classify format: JSON, CSV, multi-line log, terminal dump, or
unstructured text. If format is ambiguous, emit sample lines and ask.

### Step 2: Profile Distribution

Categorize each data point into logical pathways:
- Happy path -- standard successful execution
- Boundary -- min/max values, empty inputs, limit cases
- Fault -- exceptions, errors, malformed inputs
- State transition -- intermediate steps (if applicable)

Report category counts before proceeding.

### Step 3: Clean Non-Deterministic Noise

Strip unless explicitly needed for debugging:
- Timestamps, dates, elapsed time
- Process IDs, thread IDs
- Absolute local filesystem paths
- Random UUIDs or session tokens

Replace with deterministic placeholders where structure must be preserved.

### Step 4: Sample for Maximum Variance

Select the smallest subset covering 100% of observed categories:
- At least 1 sample per category
- Prefer extremes (min/max) over midpoints
- Include at least 1 fault case even if rare in source
- Total sample size: aim for 3-10 items (never exceed 20 without justification)

### Step 5: Format for Consumer

- For domain modeling: token sequences, syntax pairs, edge-case inputs as JSON.
- For TDD fixtures: structured input/output pairs with `id`, `description`,
  `input`, `expected_output`.
- File format: JSON (single fixtures), YAML (multi-document streams).

### Step 6: Write and Report

Write the prepared sample to `data/{context}/prepared/examples/` (NEVER `tests/`
-- `tests/` is built from here under `build-with-tdd`). Emit a manifest and update
`data/MANIFEST.md`:
- Source path, total records, categories found, samples selected
- Target consumer and output path

## Exit Criteria

- [ ] All logical categories from source data represented in the sample.
- [ ] Non-deterministic noise stripped (no timestamps, PIDs, local paths).
- [ ] Sample size <= 20 items.
- [ ] Output format matches the target consumer schema.
- [ ] Prepared file written under `data/{context}/prepared/`; `data/MANIFEST.md` updated.

## Anti-Rationalization

| Failure Mode | Correct Behavior |
|---|---|
| Include all data points ("more is better") | Minimal variance-maximizing subset. 3-10 items covers most cases. |
| Skip profiling, jump to formatting | Profile first. Unknown distribution = unknown coverage. |
| Leave timestamps in "just in case" | Strip all non-deterministic data. Determinism is non-negotiable. |
| Write fixtures into `tests/` | The data-layout rule governs `data/`; write to `data/{context}/prepared/`. `build-with-tdd` builds `tests/` from it. |
| Sample only happy paths | Every set needs at least 1 boundary + 1 fault case. |

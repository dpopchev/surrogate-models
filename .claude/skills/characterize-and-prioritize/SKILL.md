---
name: characterize-and-prioritize
description: Characterize a problem's inputs and outputs, name the features linking them, then prioritize the work by MoSCoW or POC/MVP to move forward fast. Use when asked to "characterize the data", "what are the inputs and outputs", "prioritize the work", "moscow this", "scope an MVP", or "where do I start?".
---

# Characterize and Prioritize

The framing move before building: turn a vague problem into a characterized
INPUT -> FEATURES -> OUTPUT map, then cut it into a prioritized backlog so the
smallest valuable slice ships first. Analysis and planning only -- produces a
decision menu, writes no `src`.

## Steps

If inputs, outputs, or the goal are unstated, ASK before characterizing -- never
assume the target variable.

1. CHARACTERIZE I/O -- state the INPUTS (source, shape, dtype, ranges, invariants)
   and the OUTPUTS (target, shape, units, acceptance). For data, profile before
   asserting: row/column counts, uniqueness of key candidates, value ranges,
   vocabulary. Every claim traces to observed data (per the data-layout rule).
2. NAME FEATURES -- describe the patterns linking input to output: candidate
   features, transforms, and the hypothesized input -> output relationship. Flag each
   as observed or assumed.
3. PRIORITIZE -- score each deliverable/feature by ONE frame (pick per the ask):
   - MoSCoW: Must / Should / Could / Won't (now).
   - POC -> MVP -> increments: the thinnest end-to-end path that proves value.
   Rank by leverage / effort; name the smallest slice that moves the goal.
4. HAND OFF -- present the ranked menu + the recommended first slice, then HALT. The
   user picks; implementation is `build-with-tdd`, not this skill.

## Reference

- Data profiling mechanics (dimensions, uniqueness, variants) + the immutable
  `orig/` vs `prepared/` layout: the `data-layout` rule.
- Building the chosen slice: the global `build-with-tdd` skill.

## Exit criteria

- [ ] Inputs and outputs stated with shape + invariants; data claims traced to
      observation, not assumed.
- [ ] Input -> output features named, observed-vs-assumed flagged.
- [ ] Work prioritized by ONE frame (MoSCoW or POC/MVP); smallest valuable slice named.
- [ ] Ranked menu presented; no `src` written; user selects.

## Anti-rationalization

| Failure mode | Correct behavior |
|---|---|
| Assume the target variable | ASK. The output defines the whole frame. |
| "The format is obvious, skip profiling" | Measure. Obvious assumptions hide edge cases. |
| Boil-the-ocean backlog | Prioritize by ONE frame; name the smallest slice that ships. |
| State an assumed feature as fact | Flag observed vs assumed. Traceability over guess. |
| Start implementing option 1 | HALT. Present the menu. `build-with-tdd` builds it. |

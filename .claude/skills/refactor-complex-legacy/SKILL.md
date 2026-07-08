---
name: refactor-complex-legacy
description: Modernize complex legacy functions using the 4-phase wrap-extract-replace-remove strangler fig. Use when asked to "rewrite legacy", "strangle", "modernize function", "wrap and replace", or "too complex to extract".
---

# Refactor Complex Legacy

Applies the size + railway gates in `.claude/rules/architecture.md`.

## Goal & Triggers

Decomposes complex legacy functions (>~200 lines, many callers, mixed concerns)
into typed railway pipelines. Each phase is a separate `build-with-tdd` task with
tests green throughout.

Triggers: "rewrite legacy", "strangle", "modernize function", "wrap and replace",
"too complex to extract".

## Inputs / Outputs

| Input | Required | Description |
|-------|----------|-------------|
| Legacy function path | Yes | File and function name |
| Caller count | Auto | `grep -rn "<function>" src/` |
| Concern inventory | Yes | List of mixed responsibilities |

| Output | Description |
|--------|-------------|
| Phase 1 | `_v2` typed wrapper -- zero behavior change |
| Phase 2 | Extraction stubs -- one per concern, `Err(NotImplemented)` |
| Phase 3 | Pure `.and_then()` railway -- no legacy call |
| Phase 4 | Legacy deletion TODO -- grep confirms zero callers |

## Step-by-Step Workflow

### Step 1: Inventory

1. Count lines with `wc -l <target_file>`.
2. Count callers with `grep -rn "<function_name>" src/ | wc -l`.
3. List concerns: validation, transformation, persistence, notification, etc.
4. If <~200 lines and <3 concerns -> too small for strangler fig; decompose it
   directly with `build-with-tdd` under the size gate in `.claude/rules/architecture.md`.

### Step 2: Phase 1 -- WRAP

Create a typed wrapper that delegates to the legacy function. Zero behavior
change. All callers redirect to the wrapper.

```python
# OLD: called by 12 callers
def process_order(raw_data: dict, db, mailer) -> dict:
    ...  # 200 lines of mixed concerns

# NEW: typed wrapper -- delegates to legacy
def process_order_v2(
    cmd: ProcessOrderCommand,
    *,
    find: FindOrderFn,
    save: SaveOrderFn,
    notify: NotifyFn,
) -> Result[OrderResult, ProcessError]:
    # TODO: DEVELOPER TASK Phase 1 -- redirect all callers to _v2
    legacy_result = process_order(
        {"order_id": cmd.order_id, ...},
        db=_get_db(), mailer=_get_mailer(),
    )
    return Ok(OrderResult.from_legacy(legacy_result))
```

### Step 3: Phase 2 -- EXTRACT (repeat per concern)

Peel one concern at a time. Place a stub. `build-with-tdd` implements it. Tests
stay green after each extraction.

```python
def _validate_order(cmd: ProcessOrderCommand) -> Result[ValidatedOrder, ValidationError]:
    # TODO: DEVELOPER TASK Phase 2a -- extract validation
    return Err(NotImplemented)

# Wrapper uses extracted step + legacy remainder:
def process_order_v2(cmd, *, find, save, notify):
    return (
        _validate_order(cmd)
        .and_then(lambda v: _legacy_remainder(v, find, save, notify))
    )
```

(Repeat for enrichment, persistence, notification, etc.)

### Step 4: Phase 3 -- REPLACE

All concerns extracted. The wrapper IS the new implementation. No legacy call
remains.

```python
def process_order_v2(cmd, *, find, save, notify):
    return (
        _validate_order(cmd)
        .and_then(_enrich)
        .and_then(lambda e: _persist(e, save=save))
        .and_then(lambda s: _notify(s, notify=notify))
    )
```

### Step 5: Phase 4 -- REMOVE

Search for legacy function references. Expect only `_v2` references remain. Then
create a `build-with-tdd` task to delete the legacy function.

### Step 6: Verify

Ensure all tests pass (`make test`) and there is no dead code or legacy callers
remaining (`grep -rn "<function_name>" src/`).

## Exit Criteria

- [ ] Phase 1: wrapper exists, all callers redirected, tests green.
- [ ] Phase 2: each concern has a typed stub, tests green after each.
- [ ] Phase 3: wrapper has zero legacy calls, pure `.and_then()` railway.
- [ ] Phase 4: legacy function deleted, grep confirms zero references.
- [ ] Each phase is a separate `build-with-tdd` task.
- [ ] No behavior change at any phase boundary.

## Anti-Rationalization

| Failure Mode | Correct Behavior |
|---|---|
| Rewrite in place | NEVER. Wrap first. The wrapper is the safety net. |
| Skip Phase 1 | The typed wrapper is mandatory. Callers redirect before extraction. |
| Batch extractions | One concern per Phase 2 step. Tests green after each. |
| Leave legacy after Phase 3 | Dead code is entropy. Phase 4 (REMOVE) is mandatory. |
| Apply to <200-line function | Strangler fig is overkill. Decompose directly with `build-with-tdd` under the size gate in `architecture.md`. |
| Phase 2 without tests | Each extraction must be independently testable. No exceptions. |

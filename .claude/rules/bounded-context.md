---
paths:
  - "src/**/*.py"
---

# Bounded Context -- Layout & Growth Hard Gates

The canonical shape of a bounded context, and the module -> package growth path.
This is the single source of truth for WHERE code goes. The behavioral gates
(purity, railway, CQRS read/write rules, DIP, aggregates) live in
`.claude/rules/architecture.md`; this file owns STRUCTURE only. The skills apply
these gates -- they do not restate them.

Core stance: modules first, uniformly. EVERY ring -- domain, application,
infrastructure -- starts as one flat `.py` module and earns a package only when it
crosses the size gate. The application's CQRS command/query/projection split is a
correctness boundary preserved as ORDERED SECTIONS inside `application.py`
(Commands -> Queries -> Projections); it becomes separate files only at the split.

## Canonical Skeleton (a new context starts here)

```text
src/{package}/{context}/
+-- domain.py            <- functional core: identities, value objects, aggregates,
|                           state transitions, smart constructors, workflow signatures
+-- application.py       <- one flat module; CQRS as ORDERED SECTIONS:
|                             (1) Commands:    handle_<verb>_<noun>(*, deps, cmd) -> Result
|                             (2) Queries:     handle_get_<noun>(*, read, query) -> Result[DTO]
|                             (3) Projections: to_<noun>_<format>(domain | row) -> DTO
|                           OPTIONAL published port Callables (e.g. ReserveStockFn) live here
|                           too. Splits to application/ (commands.py/queries.py/projections.py)
|                           at the size gate.
+-- infrastructure.py    <- imperative shell: @safe I/O, persistence, mappers, emit, AND
|                           the CONSUMING-side ACL -- a factory that wraps a FOREIGN
|                           context's published port (`ports.py`) and translates the
|                           foreign DTO -> this context's own type, foreign error -> a
|                           local error. Splits to `infrastructure/acl.py`, then
|                           `infrastructure/acl/<read_model>.py`, at the size gate.
+-- __main__.py          <- context root: get_settings() + wires this context's OWN
|                           APPLICATION + INFRASTRUCTURE rings ONLY -- never domain
|                           (reached transitively through them; naming it here is a leak).
|                           Foreign ports arrive injected via compose(*, <foreign_port>);
|                           cross-context wiring lives in the top-level project main.
```

- ALWAYS: a new context starts with flat `domain.py`, `application.py`, and
  `infrastructure.py` -- all three modules-first.
- ALWAYS: `application.py` keeps the CQRS split as ORDERED SECTIONS (Commands ->
  Queries -> Projections) -- a correctness boundary present from the first handler,
  enforced by section order, not by separate files.
- NEVER: let the CQRS section boundary erode (a query handler in the Commands
  section, a command driving a read) -- the split stays logical from the first
  handler even though it is one file until the split.
- NEVER: pre-split a ring into a package before it crosses the size gate (no
  empty `domain/`, no premature `application/`, no `commands/` directory holding one file).
- ALWAYS: the moment a context CONSUMES another, its consuming-side ACL exists from
  day one in `infrastructure.py` -- a factory `make_<foreign>_<verb>(foreign_port)
  -> <LocalFn>` that maps the foreign DTO -> a local type and the foreign error -> a
  local error. The handler depends on the LOCAL alias; the real foreign callable is
  wired at the TOP-LEVEL project main (the sole cross-context composition root) and
  reaches this context injected via `compose(*, <foreign_port>)` -- this context's
  `__main__.py` never imports the producer.
- NEVER: a consumer imports a foreign `domain/` or `application.{commands,queries,
  projections}` to borrow a type -- that is `ACL_001`, mechanically forbidden by the
  `.importlinter` cross-context contracts (see `architecture.md` Verify). The ONLY
  legal foreign import is the producer's `application/ports.py`.

## Ring -> Module Placement

- `domain.py`            -- pure core. No I/O. Aggregates, states-as-types, smart
                            constructors (see `architecture.md` Aggregates).
- `application/commands.py`   -- write-side handlers. Drive the aggregate lifecycle
                                 (`load_<noun>` -> transition -> `save_<noun>`). Return `Result`.
- `application/queries.py`    -- read-side handlers. Bypass the aggregate: bind a
                                 `find_<noun>_<view>` read port returning a primitive
                                 read-row / DTO, never the aggregate. Return a DTO.
- `application/projections.py`-- pure `to_*` transforms: domain or read-row -> DTO.
- `application/ports.py`  -- OPTIONAL. The `Callable` type aliases this context
                             PUBLISHES for other contexts to depend on (e.g.
                             `ReserveStockFn`). Internal handler deps stay co-located
                             in `commands.py`/`queries.py`; `ports.py` is only the
                             outward-facing contract. Add it when another context
                             consumes this one; omit it otherwise.
- `infrastructure.py`    -- `@safe` I/O, persistence, mappers, ACL adapters, emitters.
- `__main__.py`          -- wires this context's OWN `application` + `infrastructure`
                            rings ONLY (never `domain` directly -- a domain import here
                            is a leak); exposes `compose(*, <foreign_port>)` for the
                            top-level main to inject foreign ports. Cross-context wiring
                            is the top-level main's alone.

### Ports & repositories -- dependencies are callables, never classes

- ALWAYS: a dependency is a `Callable` type alias (DIP via callables, per
  `architecture.md`). Store access splits by CQRS side -- never a class, never a
  `Repository`/`*Repo` type:
  - WRITE: the aggregate PAIR `load_<noun>` / `save_<noun>`. `load_<noun>(id) ->
    Result[<Aggregate>, E]` HYDRATES the whole aggregate (re-certified from the
    store -- the trust boundary) for a command's load -> transition -> save.
  - READ: `find_<noun>_<view>(id) -> Result[<ReadRow | DTO>, E]` returns PRIMITIVES
    (a read-row / DTO), bypassing the aggregate and its certification.
  - Mnemonic: LOAD an aggregate to act on it; FIND a view to show it. A query that
    binds an aggregate-returning port and then projects it is the read-model bypass
    smell -> `CQRS_004`.
- ALWAYS: a port another context consumes is published in `application/ports.py`;
  the consumer depends on that alias, and the real callable is wired at the
  top-level project main through the consumer's ACL adapter (see `architecture.md`
  ACL).
- NEVER: a consuming context imports the producing context's `domain/` to get a
  type -- it depends on the published port alias only (cross-context = `ACL_001`).
- An aggregate has exactly ONE home context, and only THAT context's
  `infrastructure` may turn it into rows and back (persist/load it). No other
  context loads or saves a foreign aggregate.

### Handlers vs transitions -- what each module holds

A "handler" is the application-layer `handle_<verb>_<noun>` / `handle_get_<noun>`
function. The application layer holds all handlers; the domain holds none.

- ALWAYS: a handler lives in `commands.py` (write) or `queries.py` (read),
  co-located with its one request message. `commands.py` holds, per command: the
  message (`ConfirmOrderCommand`), the injected callable type aliases, the error
  union, and `handle_confirm_order`. Same shape in `queries.py` for the read side.
- ALWAYS: a handler signature is FULLY keyword-only, deps BEFORE the message --
  `handle_<verb>_<noun>(*, <deps>, cmd)` (query: `(*, <read>, query)`). The leading
  `*` makes every argument keyword-only: call sites are self-documenting and
  transposition-proof, deps read first in the DI-currying order, and no caller can
  accidentally `partial(handler, cmd)` and curry the DATA instead of the deps. The
  produced partial is therefore invoked `handler(cmd=...)` -- keyword, by design.
- `projections.py` holds pure `to_*` transforms only -- the read functions a query
  handler maps over. A projection is not a handler.
- `domain.py` holds the pure TRANSITION the command handler drives
  (`confirm_order(order) -> Result`), never the handler. Domain has no injected
  deps, no I/O, no orchestration.
- NEVER: a `handlers.py`, `services.py`, or `usecases.py` module -- the handler is
  the command/query module's primary content, not a layer of its own.

## Core vs Delivery Contexts

The skeleton above is a CORE context (it owns a domain). A DELIVERY context
(`cli`, `web`, `api`) is a thin edge that drives a core through its published
application API.

- A delivery context is its OWN bounded context with its OWN input model -- the
  delivery vocabulary parsed into typed values (argv / flags / exit codes for
  `cli`; route / form / status for `web`). That parse is the delivery's own grammar.
- ALWAYS: a delivery context depends ONLY on the published application API
  (command/query handlers + `ports.py`). Reaching past the application API into a
  core's `domain/` is `ACL_001`.
- ALWAYS: structural `match/case` dispatch (input -> handler, then `Result` ->
  output) lives at the delivery edge -- never in the core's domain.
- ALWAYS: the core is unchanged to gain a delivery. A second delivery (web after
  cli) is just another upstream consumer of the same handlers -- zero core edits.
- NEVER: business rules, aggregates, or persistence in a delivery context -- those
  belong to the core it calls. A delivery owns only its input model and the
  mapping parsed-input -> core handler -> rendered output.

## Module -> Package Growth (the ~500-line trigger)

When a flat module crosses ~500 lines (or a function ~50), split it into a package
named after the same ring/module, one file per concept, with `__init__.py`
re-exporting the public API so external imports never break.

| Flat module | Splits into (package) | One file per | File name |
|---|---|---|---|
| `domain.py` | `domain/` | domain concept | `<concept>.py` (`order.py`, `cart.py`, `stock.py`) |
| `application.py` | `application/` | CQRS section -> file | `commands.py`, `queries.py`, `projections.py` (the 3 ordered sections become the 3 files) |
| `application/commands.py` | `application/commands/` | command | `<verb>_<noun>.py` (`confirm_order.py`) |
| `application/queries.py` | `application/queries/` | query | `get_<noun>.py` (`get_cart_items.py`) |
| `application/projections.py` | `application/projections/` | domain concept | `<concept>.py` (`order.py`) |
| `infrastructure.py` | `infrastructure/` | concern + adapters | `persistence.py`, `emitter.py`, `mappers/<concept>_mapper.py`, `acl.py` |
| `infrastructure/acl.py` | `infrastructure/acl/` | consumed read-model | `<read_model>.py` (`credit_status.py`, `stock_level.py`) |

- ALWAYS: split by domain CONCEPT, never by technical pattern.
- NEVER: `aggregates.py`, `value_objects.py`, `entities.py`, `handlers.py`,
  `dtos.py` -- name files after the concept (`order.py`), the command
  (`confirm_order.py`), or the query (`get_cart_items.py`).
- ALWAYS: `__init__.py` re-exports the public API after a split -- the package's
  import surface equals the flat module's it replaced.
- NEVER: split `commands.py` into `queries/` files or vice versa -- a package
  keeps the CQRS side of the module it grew from.

## Migrate While Refactoring (no big-bang)

This layout is reached by continuous improvement, not a rewrite.

- ALWAYS: a module stays flat until you are ALREADY editing its context for
  another reason AND it has crossed the size gate. Split it then.
- NEVER: open a migration pass that splits modules nobody is touching.
- NEVER: batch -- one concept extracted per developer task (place stubs across the
  slice, then fill each under `build-with-tdd`).
- ALWAYS: behavior is identical before and after a split. A split is a move +
  re-export, never a rewrite.
- When a legacy context predates this layout (e.g. flat `application.py`, or
  `commands/` directories with one file each), record the gap and converge on the
  next edit that touches it -- do not rewrite it in place.

## Verify

- `make typecheck` and `make lint` are the targets that exist today.
- Structural checks: import-linter enforces ring/ACL imports (`make lint-imports`);
  the rest by inspection --
  ```bash
  # premature package: a ring directory holding a single module
  find src -type d \( -name commands -o -name queries -o -name domain \) \
    -exec sh -c 'test $(ls -1 "$1"/*.py 2>/dev/null | grep -vc __init__) -le 1 && echo "premature: $1"' _ {} \;
  # missing CQRS split: read + write in one application module
  grep -rln "def handle_get_" src --include=commands.py     # query in commands.py
  grep -rlnE "def handle_(confirm|add|create|update|delete|reserve)_" src --include=queries.py
  # oversize: a flat module past the gate with no package
  find src -name "*.py" -exec wc -l {} + | sort -rn | head -20
  ```

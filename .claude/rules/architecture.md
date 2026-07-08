---
paths:
  - "src/**/*.py"
---

# Architecture -- Hard Gates

Functional Core / Imperative Shell, Onion rings, railway-oriented error handling.
These are the falsifiable BEHAVIORAL gates only. WHERE code goes -- the canonical
context skeleton, modules-first start, and the module -> package growth path -- is
single-sourced in `.claude/rules/bounded-context.md` (also auto-loaded on
`src/**/*.py`). Domain-modeling gates (aggregates, states-as-types, smart
constructors) are in the Aggregates section below. These rules are the authority;
the `build-with-tdd` and `refactor-complex-legacy` skills apply them, and
import-linter (`make lint-imports`) mechanically enforces the ring/ACL imports.

## Functional Core / Imperative Shell

- ALWAYS: `domain/` is the functional core -- pure, total, deterministic, no I/O.
- ALWAYS: `infrastructure/` is the imperative shell -- all I/O and side effects;
  it IMPLEMENTS the application's port `Callable` aliases (e.g. `SaveFn`, `LoadFn`,
  `FindFn`) structurally via DIP, WITHOUT importing `application/`.
- ALWAYS: `application/` orchestrates the core via injected callables; no direct I/O.

## Ring Import Direction

`application/` and `infrastructure/` are INDEPENDENT siblings over a shared
`domain/` bottom -- neither imports the other. The composition root (`__main__`)
injects the infrastructure adapter into the application handler (DIP). The only
internal import arrows are `application -> domain` and `infrastructure -> domain`
(the latter for TYPES ONLY).

- NEVER: `domain/` imports from `application/` or `infrastructure/` -- it is the
  dependency-free bottom.
- ALWAYS: `application/` imports from `domain/` -- workflows, port aliases, and
  handlers are typed against domain types. This is an allowed inward arrow.
- NEVER: `application/` imports from `infrastructure/`.
- ALWAYS: `infrastructure/` imports from `domain/` for TYPES ONLY -- the domain
  types its port implementations and mappers (row -> domain) are annotated against.
  Never for domain behavior beyond constructing/reading those types.
- NEVER: `infrastructure/` imports from `application/` -- it implements the
  application's port `Callable` aliases STRUCTURALLY (DIP) and is bound to them only
  at the composition root, never by importing `application/`. (Cross-context: a
  CONSUMING context's `infrastructure/` ACL wrapping a FOREIGN context's
  `application/ports.py` is the one documented exception -- see `bounded-context.md`.)

## Forbidden in Domain (purity)

- NEVER: import `logging`, `os`, `pathlib`, `io` in `domain/`.
- NEVER: `raise` in `domain/` -- return `Result[T, E]` instead.
- NEVER: `datetime.now()`, `random.*`, `uuid4()` in `domain/` -- inject as callables.
- NEVER: mutable globals in `domain/`.

## Railway (functional error handling)

- ALWAYS: compose fallible steps with `.and_then()` / `.fmap()` / `.map_err()`.
- NEVER: branch on a `Result` (no `isinstance`, no mid-pipeline `unwrap`) -- railway only.
- ALWAYS: `@safe` / `@safe_async` wraps I/O in `infrastructure/` ONLY -- never
  `domain/` or `application/`; never nested.

### Errors (structured failures)

- ALWAYS: a failure crossing UP a ring is carried as a structured, serializable
  cause -- never the lower ring's exception type -- so a higher ring raises its
  OWN error without leaking lower types (the railway analogue of `raise X from
  Y`). Distinguish a BOUNDARY failure (carries that cause) from a pure VALIDATION
  failure (carries the offending value).
- ALWAYS: a failure is rendered at the edge by the error VALUE itself, never by
  branching on the variant -- the delivery edge imports no `domain/` type.

## Forbidden Everywhere

- NEVER: `Protocol` or `ABC` -- use `Callable` type aliases (DIP via callables).
- NEVER: `isinstance()` -- use structural `match/case`.
- NEVER: local imports inside functions -- all imports at file top.
- NEVER: `print()` for operational output -- use the logger.
- NEVER: f-strings in log calls -- use `%s` lazy formatting.

## CQRS

- NEVER: mixed read/write in the same handler.
- NEVER: return a domain aggregate from a query -- use a DTO.
- NEVER: a query's read port returns the aggregate. The read side binds a
  `find_<noun>_<view>` port yielding a primitive read-row / DTO and bypasses
  certification; loading the aggregate (a `load_<noun>` port) only to project it is
  the read-model bypass smell -> `CQRS_004`. The write side hydrates via
  `load_<noun>`; the read side never does.
- ALWAYS: a command (write) loads ONE aggregate, applies ONE transition, and saves
  the WHOLE aggregate -- the root AND its owned entities -- in a single atomic call.
- NEVER: a partial save (root without its new child, or a child without its root) --
  after every command all aggregate invariants hold; no partial states persist.
- Module placement of commands / queries / projections is in
  `.claude/rules/bounded-context.md` (write side, read side, pure `to_*`).

## Boundaries (edge DTOs)

- ALWAYS: bare `str` / `int` / `float` survive ONLY at the outer edge -- a command
  or query DTO, or a row/dict crossing an I/O boundary.
- ALWAYS: the handler re-wraps edge primitives into domain value objects (via the
  `make_*` smart constructors) BEFORE the domain is touched -- inside the domain a
  bare primitive is a bug.
- ALWAYS: a mapper translates a foreign row/dict/frame into a domain type at the
  `infrastructure/` boundary -- foreign shapes never cross inward into `domain/`.

## Aggregates (domain modeling)

The aggregate is the domain's consistency boundary -- the central organizing
concept of `domain/`. These gates keep illegal states unrepresentable.

- ALWAYS: the aggregate root is the only public entry point; internal types are
  module-private (`_`-prefixed). Consumers depend on the root, never on internals.
- ALWAYS: each aggregate STATE is a distinct frozen type. A transition is a pure
  function `(<State>, <Input>) -> Result[<NewState>, <Error>]` -- an imperative
  verb (`confirm_order`, `add_item`). Only valid transitions exist as signatures;
  there is no signature for an illegal one.
- NEVER: one mutable type carrying a `status: str`/enum plus conditionally-valid
  fields (`submitted_at: str | None`) -- that lets any state become any other at
  runtime. Model the states, not a status field.
- ALWAYS: every identity and quantity in the domain is a `NewType` or frozen value
  object, so a `Sku` passed where an `OrderId` belongs is a type error. Closing
  primitive obsession in the domain is a domain gate -- bare primitives survive
  only at the edge (see Boundaries) and are re-wrapped inward.
- ALWAYS: a smart constructor `make_*(raw) -> Result[T, ValidationError]` is the
  ONLY way to build a value object -- an invalid value cannot exist. The validation
  error carries the offending value (a pure VALIDATION failure, per Errors).
- ALWAYS: errors are explicit union members (frozen records), never exceptions;
  the domain names them, and exhaustive `match` at the edge handles them.
- ALWAYS: aggregate collections are `tuple[...]`, never `list` -- `frozen=True`
  blocks reassignment, not in-place `list` mutation; a transition returns a NEW
  aggregate.
- ALWAYS: after every command all aggregate invariants hold -- no partial or
  in-between state leaks out (pairs with the CQRS whole-aggregate atomic save).
- ALWAYS: names come from the domain's ubiquitous language; NEVER generic/technical
  names (`Entity`, `value`, `type`) or pattern-named modules (see `bounded-context.md`).

## Size

- NEVER: a module >~500 lines without a decomposition plan.
- NEVER: a function >~50 lines without an extraction plan.
- The decomposition plan IS the module -> package growth path in
  `.claude/rules/bounded-context.md` -- split by concept, `__init__.py` re-export.

## Composition Root

- ALWAYS: wiring lives ONLY in `infrastructure/config.py` (Settings + persistence)
  and `__main__.py` (handler wiring + dispatch). No DI container, no service
  locator, no module-level dep wiring in handler files.
- ALWAYS: a context's `__main__.py` wires its OWN `application` + `infrastructure`
  rings ONLY -- NEVER `domain` directly. `application` turns incoming primitives into
  domain objects (the anti-corruption boundary outside code talks to) and
  `infrastructure` persists/loads/supplies them; the composition root reaches the
  domain TRANSITIVELY through those two and never names it. A `__main__.py` that
  imports `domain` (e.g. injecting a domain constructor as a dependency) is a leak --
  every injected dep is an application handler or an infrastructure adapter, never a
  domain type. (A delivery `__main__`, e.g. `cli`, is exempt: it parses input into
  its OWN input model, which is its `domain`.)
- ALWAYS: cross-context edges (a producer's published port -> a consumer's ACL) are
  wired ONLY in the single top-level project `__main__.py` -- the sole cross-context
  composition root. A consumer takes foreign ports as injected callables via
  `compose(*, <foreign_port>)` and never imports the producer.

## Verify

- `make typecheck` (mypy) and `make lint` (ruff + `make lint-imports`) are the
  targets that exist today.
- Ring/import + bounded-context enforcement IS wired: `.importlinter` holds the
  contracts and `make lint-imports` runs them (folded into `make lint`). The
  CONTRACT is the gate, not grep. It enforces onion-ring direction per context AND
  cross-context isolation: a context may NOT import another context's `*.domain` or
  `*.application.{commands,queries,projections}`; the sole legal cross-context seam
  is the producer's `application/ports.py` (consumed via an ACL adapter -- see
  `.claude/rules/bounded-context.md`).
- Pre-existing cross-context leaks live in that contract's `ignore_imports` as
  ENUMERATED debt, deleted one entry per migration -- never a license to add new
  ones. Any NEW foreign import breaks the gate. Grep is only a secondary check.
- The composition-root domain seal is mechanically gated too: a `forbidden` contract
  (`allow_indirect_imports = True`) bars every core `__main__` from a DIRECT `*.domain`
  import -- transitive reach through `application`/`infrastructure` stays legal, a
  direct domain import does not. A producer's internals (`domain`, `infrastructure`)
  also get a `forbidden` seal so consumers cross in only through its `application`.

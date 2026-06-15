# ADR 0001 — Evolutionary, typed pipeline (DDD-lite + functional core / imperative shell)

- **Status:** Accepted
- **Scope:** `scripts/ingestions/`
- **Date:** 2026-05

## Context

`scripts/ingestions/` is a five-stage data-processing pipeline (fetch → clone →
digest → validate → inject), historically dominated by a 2,804-line `3_digest.py`
with mixed concerns. We need it maintainable and hard to break, but it is **not** a
rich behavioural domain: there are few competing domain rules, mostly transformations
of dataset metadata. Full Domain-Driven Design (aggregates, repositories, entities,
domain services everywhere) would add enterprise ceremony without payoff.

## Decision

Adopt **"Evolutionary, typed pipeline refactor using DDD-lite + functional core /
imperative shell"**, with four governing rules:

1. **DDD-lite, not full DDD.** Keep the strategic part — *ubiquitous language* and
   explicit *stage boundaries* (see [`../../CONTEXT.md`](../../CONTEXT.md) and
   [`../../models.py`](../../models.py)) — and drop tactical patterns (heavy
   aggregates, repository/service classes, ABC-everywhere). Prefer small free
   functions and `typing.Protocol` seams over inheritance.
2. **TypedDict for construction, Pydantic for boundaries.** `eegdash.schemas`
   deliberately co-defines TypedDicts (`Dataset`, `Record`, `Montage`) for cheap
   construction/loading of millions of records, Pydantic models
   (`DatasetModel`, `RecordModel`, `ManifestModel`) for the validation gate, and
   `create_dataset`/`create_record` factories as the single construction path. Do
   not Pydantic-ify every in-memory artifact; reserve it for real boundaries
   (config, the three core entities, the manifest, the inject write boundary).
3. **Validation returns a report, it does not throw.** `_validate.ValidationResult`
   accumulates errors/warnings/stats; normal validation failures are data, not
   exceptions. (A future `IngestionError`/`Recoverable`/`Fatal` hierarchy will
   distinguish dataset-level from config/infra failures — see roadmap.)
4. **Golden-master first.** Freeze current behaviour with characterization tests
   *before* refactoring it. New behaviour-touching work must keep the golden
   masters green or update them visibly in the same commit.

## Consequences

- Refactors proceed as Strangler-Fig / Branch-by-Abstraction increments, each
  releasable, each guarded by golden tests — never a big-bang rewrite.
- The codebase stays "boring, typed, explicit, testable": `config = Config.from_cli();
  bundles = [digest_dataset(ds, config) for ds in datasets]; reports = [validate(b) …]`.
- Some artifacts keep current names (`EnumerationResult`, `ValidationResult`) and
  will gain glossary aliases (`DigestBundle`, `ValidationReport`) with back-compat.

## Roadmap (deepening, not greenfield — most structure already exists)

P0 glossary + ADRs → **P1 complete golden net** → P2 converge vocabulary + tighten
inter-stage types → P3 isolate I/O behind injectable `Protocol`s + split pure cores
→ P4 `IngestionError` hierarchy → P5 (optional) scoped packages + sampled write-boundary
validation. See ADR 0002 for why P5 is scoped/optional.

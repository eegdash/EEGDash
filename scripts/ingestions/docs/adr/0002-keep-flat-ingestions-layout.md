# ADR 0002 — Keep the flat `scripts/ingestions/` layout

- **Status:** Accepted
- **Scope:** `scripts/ingestions/`
- **Date:** 2026-05

## Context

The methodology that informs ADR 0001 sketches an idealized package layout:

```
cli/  models.py  config.py  pipeline/  adapters/  parsers/  io/  tests/{golden,unit,integration}
```

It is tempting to chase that literal structure. We evaluated it against the current
flat layout (all modules at `scripts/ingestions/` top level, leading-underscore
naming for internals, digit-prefixed entry scripts).

## Decision

**Keep the flat layout.** Realize the *intent* of the target structure — clear seams,
isolated per-stage config, concern-split tests, a shared glossary — through naming
and module boundaries, **not** through `cli/pipeline/adapters/io/` subpackages.

Reasons:

1. **The entry scripts are not importable modules.** `2_clone.py`, `3_digest.py`,
   `4_validate_output.py`, `5_inject.py` start with a digit (not valid Python
   identifiers) and are loaded via `importlib` by `tests/_helpers` and
   `record_enumerator`. Moving them into a `cli/` package breaks those loaders.
   If named entry points are ever wanted, add `[project.scripts]` console-scripts
   that import the underscore modules, leaving the digit-prefixed files as thin
   wrappers.
2. **A big-bang move risks re-introducing import cycles.** The de-circularization
   that made the digest graph a DAG depends on lazy imports and a flat namespace;
   eager package `__init__.py` imports would re-create cycles.
3. **30+ flat `from _x import …` call sites** would need re-pointing or re-export
   shims; the cost/benefit is poor while the flat layout already reads clearly
   (see [`../../CONTEXT.md`](../../CONTEXT.md)).
4. **The intent is already met:** per-stage `pydantic-settings` configs, a `models.py`
   glossary, and concern-split `tests/{unit,digest,inject,parsers,acceptance,pipeline}`.

## When to revisit (scoped, optional — roadmap P5)

If navigation pain is real, migrate **only** the two genuinely cohesive clusters,
via Strangler-Fig with top-level re-export shims and **lazy** `__init__.py`:

- `parsers/` ← `_set_parser`, `_snirf_parser`, `_vhdr_parser`, `_mef3_parser`, `_format_parser_registry`
- `adapters/` ← `source_adapter`, `record_enumerator`, `_http`, `_github`

Run `tests/digest/test_snapshot.py` after each file move. Do **not** move the
digit-prefixed entry scripts.

## Consequences

- The flat layout is a deliberate, documented choice — not drift.
- Reviewers seeing "why isn't this in `pipeline/`?" are answered by this ADR.

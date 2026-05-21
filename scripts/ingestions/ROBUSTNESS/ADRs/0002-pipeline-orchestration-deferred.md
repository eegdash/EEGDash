# ADR 0002 — Defer the Pipeline-Orchestrator Module

**Status**: Accepted, 2026-05-22
**Context window**: ROADMAP P2.3 close-out.

## Context

The 5-stage ingestion pipeline (`1_fetch_sources/` → `2_clone.py`
→ `3_digest.py` → `4_validate_output.py` → `5_inject.py`) shares an
implicit JSON contract between stages. The 9 CI workflows
(`.github/workflows/1-fetch-*.yml`, `2-clone-digest.yml`, etc.)
duplicate the stage order across files.

Per LANGUAGE.md ("two Adapters = real seam"), this would normally be
an obvious deepening: define a `PipelineStage` Protocol, wrap each
numbered script as an Adapter, build an orchestrator that owns the
stage order + the inter-stage schemas.

We chose **not** to build the orchestrator now. This ADR explains
why, so a future architecture pass doesn't re-litigate.

## Decision

The Pipeline-Orchestrator Module is **deferred** until at least one
of the revisit triggers below fires.

The implicit contract is **documented** instead, in
`ROBUSTNESS/PIPELINE-CONTRACT.md` (this commit). The doc names each
stage's input + output JSON shape so a new contributor can read the
contract without tracing through 5 scripts.

## Reason

Today the 5 stages share a stable contract:
- Each writes to a per-dataset subdirectory.
- The JSON shapes are documented in `eegdash.schemas` (Dataset,
  Record, Montage).
- The 9 CI workflows are stable — they haven't been edited together
  in months.

A `PipelineStage` Protocol + orchestrator would add:
- A new top-level module (`pipeline.py` or similar)
- Adapter wrappers for the 5 scripts
- A registry of stages (probably a list of `PipelineStage` instances)
- Per-stage schema validators on the inter-stage JSON

Net effect today: more code, same behaviour, no new capabilities.
The 5 scripts run fine sequentially via CI; they're invoked one per
GitHub workflow job.

The driver for an orchestrator would be:

1. **A new stage is planned** (e.g., `3.5_anonymise.py` between
   digest and validate). Adding it today requires editing 9
   workflows; with an orchestrator, it's one config entry.

2. **Partial reprocess** — re-run digest + validate + inject without
   re-fetching. Today this requires 3 manual workflow_dispatch
   triggers in sequence.

3. **Pipeline resume** — recover from a partial failure at stage 3
   without re-running 1 and 2. Today: manual.

None of these are on the roadmap. Until they are, the orchestrator
is speculative.

## Anti-recommendations

A future architecture pass **should not** propose any of these until
the ADR is revisited:

- A `PipelineStage` `Protocol` / abstract base class.
- A `PipelineOrchestrator` class that owns the stage list.
- Per-stage `InputManifest` / `OutputManifest` TypedDicts above and
  beyond what `eegdash.schemas` already provides.
- A unified CLI (`eegdash-ingest run --stages=1-5`) that consolidates
  the 9 workflow YAMLs.
- A DAG library (Airflow / Prefect / Dagster) for orchestration.

All of these are correct designs **at scale**. None of them are
correct for a 5-stage, ~yearly-edited pipeline with 9 stable CI
workflows.

## What we DID do

Both partial closes of P2.3:

1. **Documented the contract** in `ROBUSTNESS/PIPELINE-CONTRACT.md`.
   New contributors can read the doc instead of tracing 5 scripts.

2. **Named the Seam** without inventing the Adapter classes. The
   stages communicate via per-dataset JSON in a documented shape.
   That IS the Seam, just expressed as files rather than callable
   Modules.

## Revisit when

This ADR should be revisited when **any one** of the following holds:

- A new stage is planned and enters CI. Editing 9 workflows for the
  new stage is the pain point; orchestration eliminates it.
- The pipeline grows past 5 stages (e.g., `3.5_anonymise` +
  `3.75_dedupe`). At ~7+ stages the workflow duplication becomes
  a maintenance burden.
- A partial-reprocess capability is needed (re-run only some
  stages without re-fetching). Today this is a manual procedure.
- A CVE in a workflow YAML requires updating all 9 at once. (Has
  not happened.)

At that point, this ADR is **superseded** by a follow-up that
designs the orchestrator against the actual driver, not against the
theoretical "wouldn't it be nice".

## Cross-references

- `ROBUSTNESS/PIPELINE-CONTRACT.md` — explicit input/output shapes
  for each stage. The doc IS the alternative to the orchestrator.
- `ROBUSTNESS/ROADMAP.md` P2.3 — marked deferred per this ADR.
- ADR 0001 — same deferral pattern for the source-listing Seam
  (1 production caller + 6 secondary).

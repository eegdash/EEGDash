# ADR 0003 - Treat the Ingestion Repository as the Daily CI Control Plane

**Status**: Proposed, 2026-05-22

## Context

`scripts/ingestions/` is being prepared to move out of the main EEGDash
repository into a separate repository. The moved repository must keep the
ingestion workflow running on scheduled CI, preferably daily, while still
supporting manual source refreshes and explicit production injection.

The current implementation is script-shaped:

- Stage 1 writes Source Catalogues.
- Stage 2 clones or mirrors selected Datasets.
- Stage 3 writes a Digest Corpus.
- Stage 4 validates the Digest Corpus.
- Stage 5 writes or dry-runs injection through the EEGDash Gateway.

The existing package metadata in `scripts/ingestions/pyproject.toml`
does not yet define a clean importable package. Attempting an editable
install from the current tree fails under setuptools because flat-layout
auto-discovery sees operational directories such as `ROBUSTNESS`,
`profiles`, `consolidated`, and `digestion_output` as top-level package
candidates.

There is also a design constraint from ADR 0002: do not introduce a
generic Pipeline-Orchestrator Module until there is a concrete driver
such as partial reprocess, resume, or frequent stage insertion.

## Decision

The separate ingestion repository should be designed as the daily CI
control plane for ingestion, not merely as a wheel containing the
current scripts.

Concretely:

- The first stable interface is the **Digest Corpus**, not a generic
  stage abstraction.
- Stage scripts remain as thin CLI compatibility adapters during the
  migration.
- Importable implementation moves under a real package namespace,
  expected to be `eegdash_ingestions`.
- Daily scheduled CI defaults to dry-run behavior: fetch, digest,
  validate, build an Injection Plan, and publish artifacts/reports.
- Production Gateway writes remain explicit and separately approved.
- The Dataset Listings Repository remains the durable data-artifact
  store until a separate artifact retention decision replaces it.

## Reason

A separate repository changes the failure modes:

1. **CI becomes the operational product.** The repository must answer
   "did today's ingestion still work?" without requiring a developer to
   inspect local scripts.

2. **Artifacts become contracts.** The Source Catalogue and Digest
   Corpus are what survive across jobs and repositories. Those are the
   interfaces that need validation, versioning, and test fixtures.

3. **The package namespace must create locality.** Moving files into
   `src/eegdash_ingestions/` only helps if logic stops depending on
   digit-prefixed scripts and loose top-level imports.

4. **A daily dry-run is safer than a daily write.** A scheduled job
   should detect drift and produce reviewable artifacts. Mutating
   production MongoDB should require an explicit trigger, environment,
   or approval.

## Target Module Shape

This ADR does not define final Python interfaces, but it sets the
module direction:

- `eegdash_ingestions.sources` owns Source Catalogue generation.
- `eegdash_ingestions.clone` owns Clone Workspace creation.
- `eegdash_ingestions.digest` owns Digest Corpus creation.
- `eegdash_ingestions.validate` owns Digest Corpus validation.
- `eegdash_ingestions.inject` owns Injection Plan creation and Gateway
  writes.
- `eegdash_ingestions.cli` owns command-line parsing and text output.

The numbered scripts (`2_clone.py`, `3_digest.py`, `4_validate_output.py`,
`5_inject.py`) can remain temporarily as adapters that call into
`eegdash_ingestions.cli`.

## Daily CI Contract

The daily scheduled workflow should produce, at minimum:

- the Source Catalogue revision used by the run;
- a Digest Corpus artifact;
- a validation report;
- an Injection Plan report;
- a clear skipped/changed/error summary;
- a link to logs for each Source.

The daily workflow should fail on schema errors, Digest Corpus contract
breaks, unexpected Gateway dry-run failures, or artifact publication
failure. It should not fail solely because a secondary Source is
temporarily unavailable unless that Source has been promoted to a
production-active Source.

## Anti-Recommendations

Do not start the repo split by:

- building a generic `PipelineStage` Protocol or orchestrator;
- making the package layout the main design conversation;
- preserving loose top-level imports as the long-term import style;
- treating `digestion_output/` or `consolidated/` as package data;
- letting scheduled CI perform production writes by default;
- moving operational docs out of the repository before the new daily CI
  contract is visible.

These are not all permanently wrong. They are wrong as the first
maturation step.

## Migration Sequence

1. Make editable install work from the ingestion root with explicit
   package discovery.
2. Create `src/eegdash_ingestions/` and move importable modules there.
3. Keep numbered scripts as compatibility adapters.
4. Replace `importlib.util` test loading of digit-prefixed scripts with
   imports from `eegdash_ingestions`.
5. Introduce Digest Corpus and Injection Plan modules before further
   splitting Stage 5 internals.
6. Convert CI to invoke package console commands.
7. Move workflows into the separate repository and make daily dry-run
   the default scheduled behavior.
8. Only then consider whether ADR 0002 should be revisited for resume or
   partial reprocess.

## Revisit When

Revisit this ADR when:

- the ingestion repository has run daily for at least two weeks;
- production injection is approved directly from that repository;
- partial reprocess or resume becomes a real operator need;
- Dataset Listings Repository is replaced by another durable artifact
  store;
- at least one secondary Source becomes production-active in daily CI.

## Cross-References

- `../PIPELINE-CONTRACT.md` - current Stage 1 through Stage 5 file
  contract.
- `0001-secondary-source-deferral.md` - Source listing seam remains
  deferred until more production-active adapters exist.
- `0002-pipeline-orchestration-deferred.md` - generic orchestration
  remains deferred.
- `../../CONTEXT.md` - ingestion domain vocabulary used by this ADR.

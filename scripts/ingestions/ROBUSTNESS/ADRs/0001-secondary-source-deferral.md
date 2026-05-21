# ADR 0001 — Defer the source-listing Seam until ≥ 2 production-active Adapters

**Status**: Accepted, 2026-05-21
**Context window**: This decision was crystallised during the
`improve-codebase-architecture` skill grilling session in May 2026.

## Context

`scripts/ingestions/_file_utils.py` currently defines **7 file-listing
Adapters** at an implicit Seam: `list_figshare_files`,
`list_zenodo_files`, `list_osf_files`, `list_scidb_files`,
`list_datarn_files`, `list_git_files`, `list_local_bids_files`.

Per LANGUAGE.md ("two Adapters = real seam"), this would normally be
an obvious deepening candidate: define a `SourceLister` Module with a
canonical schema, promote the 7 functions to Adapters at the same
Seam, fix the observable schema drift (Zenodo's `checksum` field is
silently dropped because `build_manifest` only reads `md5`).

We chose **not** to deepen at this Seam right now. This ADR explains
why, so a future architecture pass doesn't re-litigate the same
suggestion.

## Decision

The source-listing Seam is **deferred** until ≥ 2 Sources are
exercised in CI as regular production paths.

Concretely:

- **No `SourceLister` Protocol / shared schema dataclass is to be
  introduced in `_file_utils.py`.**
- **No abstraction over the 7 Adapters is to be added.**
- The 5 non-production Adapters
  (`list_figshare_files`, `list_zenodo_files`, `list_osf_files`,
  `list_scidb_files`, `list_datarn_files`) are tagged as
  **secondary** with a banner comment that says "best-effort, fix
  opportunistically, do not invest in depth until promoted".

The one known correctness bug (Zenodo `checksum` drop) is fixed
**in `build_manifest` only**, as a one-line accept-either change. The
Adapter shapes are not normalised.

## Reason

Today the production ingestion path uses two Sources:

- **OpenNeuro** — datasets are cloned via `git-annex`; the file listing
  comes from `list_git_files(clone_dir)`.
- **NEMAR** — git-mirrored; same path: `list_git_files(clone_dir)`.

So in production the source-listing Seam has **1 real Adapter**
(`list_git_files`). The other 6 are exercised on demand by manually
running `1_fetch_sources/1_fetch_figshare.py` / `.../zenodo.py` /
etc., and are not maintained as part of every release.

Per the maturity ladder we're working from, a Seam earns its name
when ≥ 2 callers exercise it. Investing in shared schema, Protocol,
and registry today would:

1. **Yak-shave on dead code.** The schema would be enforced against
   Adapters that no CI run exercises, so violations would be caught
   by humans, not gates.
2. **Preemptively constrain a future sprint.** When a non-OpenNeuro
   non-NEMAR Source enters production CI (Zenodo seems likely as the
   second candidate; SciDB as the third), the right schema design
   depends on what shape *that* Source has, not on retro-fitting the
   current 7.
3. **Anchor the wrong mental model.** A new contributor reading the
   codebase today would see the 7-Adapter Seam and assume all 7 are
   first-class; the secondary-tag banner makes the asymmetry
   explicit.

## Anti-recommendations

A future architecture review **should not** propose the following
until this ADR is revisited:

- A `SourceLister` `Protocol` / abstract base class.
- A `FileEntry` `TypedDict` / `dataclass` schema with required and
  optional fields.
- A registry mapping `source_name → SourceLister`.
- A pluggable `entry_points`-discoverable Source plug-in mechanism.
- Refactoring the 7 functions to a common pagination iterator
  pattern (lazy `Iterable[FileEntry]`).

All of these are correct designs at scale. None of them are correct
for **1 production caller + 6 secondary**.

## Revisit when

This ADR should be revisited when **any one** of the following holds:

- A second Source (Zenodo / SciDB / OSF / Figshare / DataRN) enters
  the regular CI ingest path (i.e., its workflow runs on every push,
  not just `workflow_dispatch`).
- A bug is reported that requires shared behaviour across ≥ 3
  Adapters (e.g., a CVE in a transitive dep requires patching all
  the URL builders).
- A new Source candidate (GIN, Brain-CODE, Donders, etc.) goes from
  "interesting" to "we want to ingest from this in the next sprint".

At that point, this ADR is **superseded** by a follow-up that
designs the schema against ≥ 2 concrete production Adapters, not
against the 7 theoretical ones.

## Side effects landed alongside this ADR

- `build_manifest` now accepts either `md5` or `checksum` from the
  Adapters (fixes a silent data-loss bug for Zenodo).
- Each of the 5 secondary Adapters has a banner comment pointing at
  this ADR.
- The `improve-codebase-architecture` candidate list in
  `ROBUSTNESS/PROGRESS-3.md` is updated to note this deferral.

## Cross-references

- `_file_utils.py:list_figshare_files` and the 4 other secondary
  Adapters carry the warning banner.
- `_file_utils.py:build_manifest` carries the checksum-or-md5 logic.
- `ROBUSTNESS/04-ROADMAP.md` § Phase 5 / 9 — original audit findings
  that proposed the deepening.
- Sister deepening that was NOT deferred:
  **storage descriptor** (the genuine 2-Adapter case between
  OpenNeuro CDN vs raw S3, and NEMAR annex-key resolution) is being
  grilled separately and will produce a real Module.

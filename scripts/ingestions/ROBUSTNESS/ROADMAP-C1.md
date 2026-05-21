# Cycle C1 — CI maturity

Anchor doc for the second cycle. The first cycle's ROADMAP (P0/P1/P2)
was about **what to build**. C1 is about **what to enforce in CI**
so the work persists.

## The opening signal

```
TOTAL                             5259   3539    33%   coverage
```

That's the current state on `record-enumerator-merge`. 33% line
coverage across 5,259 lines of business logic. The breakdown reveals
where the trust is:

| Module | LOC | Coverage |
|---|---:|---:|
| **New (Cycle 1) modules**: |  |  |
| `digest_telemetry.py` | 73 | **95%** ✅ |
| `record_enumerator.py` | 134 | **89%** ✅ |
| `source_adapter.py` | 98 | **86%** ✅ |
| `_format_parser_registry.py` | (new) | ~80% ✅ |
| **Legacy modules with no test investment**: |  |  |
| `_serialize.py` | 115 | **0%** 🔴 |
| `_validate.py` | 188 | **0%** 🔴 |
| `_montage.py` | 428 | 17% 🟠 |
| `_parser_utils.py` | 120 | 32% 🟠 |
| `_set_parser.py` | 117 | 36% 🟠 |
| `_snirf_parser.py` | 93 | 28% 🟠 |
| `_vhdr_parser.py` | 147 | 64% 🟡 |
| **One-off scripts** (per ADR 0001 banner): |  |  |
| `api_helper.py` | 159 | 0% (intentional) |
| `inject_nemar_citations.py` | 83 | 0% (intentional) |
| `patch_nemar_*.py` | 173 | 0% (intentional) |
| `time_openneuro_pipeline.py` | 82 | 0% (intentional) |

Two truths from this table:
1. **New modules are well-tested.** The Cycle-1 investments paid off.
2. **Legacy critical utilities are not.** `_serialize.py` and
   `_validate.py` are at 0% — these are the schema gate code paths
   used by stages 4 + 5.

The mutation-testing programme (P0.2 nightly) covers the parsers but
NOT `_serialize.py` or `_validate.py`. Those files have zero gates.

## Cycle goal

**Make CI a real safety net for the legacy modules**, not just the
new ones. Specifically: ratchet coverage upward, exercise the
end-to-end pipeline on the snapshot fixture in CI, add property-based
tests for the schema invariants, and unify per-source testing across
the 7 known sources.

## Tracks (priority order)

### C1.1 — Coverage gate in lint+test CI

**Driver**: today the lint+test workflow uploads `htmlcov/` but
doesn't gate on coverage. Coverage can silently drop on any PR.

**Outcome**:
- Add `--cov-fail-under=N` to the pytest invocation, where N is
  set to the current floor (~50% excluding one-off scripts).
- Exclude one-off ops scripts via `pyproject.toml`
  `[tool.coverage.run] omit = [...]`.
- Add a per-module coverage table to the workflow summary so PR
  reviewers can see which modules got worse / better.

**Estimated effort**: 1-2 hours.

**Definition of done**:
- Lint+test workflow fails a PR that drops coverage below the floor.
- The floor is set just-below-current-state so it ratchets up
  organically with each PR that adds tests.

### C1.2 — Cover the 0% dead zones (`_serialize.py`, `_validate.py`)

**Driver**: `_serialize.py` handles deterministic JSON serialization
(used by every digest output). `_validate.py` is the schema-pattern
gate used by `4_validate_output.py` and `5_inject.py`. Both at 0% is
dangerous — these are critical paths with no tests.

**Outcome**:
- Per-helper unit tests for every public function in `_serialize.py`
  (path setup, JSON sort, etc.).
- Tests for every storage-pattern in `_validate.py.VALID_STORAGE_PATTERNS`
  (positive + negative case per source).
- Both files reach >= 70% coverage.

**Estimated effort**: 3-4 hours.

**Definition of done**: `_serialize.py` >= 70%, `_validate.py` >= 70%.
The coverage gate auto-ratchets up.

### C1.3 — End-to-end integration smoke in CI

**Driver**: today the lint+test workflow only runs unit tests.
The snapshot test runs digest against the fixture, but the next
stages (validate, inject) don't run in CI on the same fixture. The
pipeline contract from `PIPELINE-CONTRACT.md` is **not enforced**.

**Outcome**:
- New CI job: run digest against `ds_snapshot_vhdr` AND
  `ds_snapshot_manifest`; then run `4_validate_output.py --strict`
  against the produced output; then run `5_inject.py --dry-run`
  to verify the records are inject-ready.
- The job fails if any stage rejects the output.
- Per-stage timing reported in the workflow summary.

**Estimated effort**: 3-4 hours.

**Definition of done**: a new GitHub job in `ingestions-lint-and-test.yml`
that exercises stages 3 → 4 → 5 against both snapshot fixtures.

### C1.4 — Property-based tests for schema invariants

**Driver**: today's tests are example-based (given fixture X,
output Y). They don't cover the SHAPE invariants: *every Record
has a `dataset` field*, *fingerprint is stable across re-runs*,
*manifest-path Records have no `_metadata_provenance`*.

**Outcome**:
- New file `tests/test_pipeline_invariants.py` using Hypothesis
  (already a dep) to generate Record / Dataset / manifest shapes
  and verify invariants.
- Tests for: required-field presence, type bounds (sfreq > 0, etc.),
  cross-Record consistency (dataset_id matches across all Records
  in a dataset), provenance source-name enumeration, byte-stability
  of `ingestion_fingerprint` across input-equivalent runs.

**Estimated effort**: 4-6 hours.

**Definition of done**: at least 15 property tests with default
Hypothesis examples; CI runs them with `--hypothesis-show-statistics`.

### C1.5 — Source transfer (parametrize tests across 7 sources)

**Driver**: per ADR 0001 we banner'd the 5 secondary sources but
they have ZERO tests beyond what slipped through. The user explicitly
asked "check what we can transfer for our source".

**Outcome**:
- Parametrize the source-listing tests across all 7 sources
  (`openneuro`, `nemar`, `gin`, `figshare`, `zenodo`, `osf`, `scidb`,
  `datarn`, plus `local_bids`). Mocked HTTP responses via respx.
- Each `list_X_files` adapter gets at least: happy-path, empty-
  response, HTTP-error, malformed-JSON.
- Document any drift between adapters in
  `findings-source-listing-tests.md`.

**Estimated effort**: 3-4 hours.

**Definition of done**: 30+ new parametrized tests; respx-mocked HTTP.

### C1.6 — Lighten where possible

**Driver**: user explicitly asked. Audit for code that can be
deleted or made lazier:
- The `extract_layout` MEG branch is 190 LOC — can it lazy-import
  MNE only when MEG hits?
- Are there `time_openneuro_pipeline.py` / `api_helper.py` /
  `patch_nemar_*.py` scripts that haven't been touched in months?

**Outcome**: identify candidates, delete with ADR if non-reversible,
mark `@pytest.mark.slow` for tests that don't need to run on every PR.

**Estimated effort**: 2-3 hours.

**Definition of done**: CI PR turnaround drops by 10%+ OR LOC
shrinks measurably; behaviour preserved.

## Order of execution

This cycle has a natural dependency chain:

```
C1.1 (coverage gate) ─┬─► C1.2 (cover dead zones) ──► C1.4 (property tests)
                      │
                      └─► C1.3 (e2e CI smoke) ──────► C1.5 (source transfer)

C1.6 (lighten) — can run in parallel with any of the above
```

C1.1 first because it's a 1-2 hour change that immediately
gates everything else. Without it, C1.2 / C1.4 / C1.5 land but
nothing **enforces** their gains.

C1.2 second because it has the highest leverage per hour: two files
at 0% coverage, both critical, both fixable with straightforward
unit tests.

C1.3 next: it tests the inter-stage contract that
`PIPELINE-CONTRACT.md` documents. Closes the gap between "we have
the contract" and "we enforce the contract".

C1.4 + C1.5 are independent rounds; pick by mood.

C1.6 is a continuous background task — any time we touch a legacy
file, check if it can be lighter.

## Working agreements

Same as cycle 1's ROADMAP:
- Snapshots stay byte-stable unless intentionally updated
- No `--no-verify`
- ADRs for deferrals
- Every track gets a PROGRESS-N doc when closed
- The coverage gate ratchets up — never down

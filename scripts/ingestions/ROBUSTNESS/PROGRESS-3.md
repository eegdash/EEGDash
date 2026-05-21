# Robustness programme — session 3 continuation (2026-05-21)

Third pass after `PROGRESS-2.md`. Six commits this round.

## Headline outcomes from this continuation

| Metric | Start of session 3 | End of session 3 |
|---|---:|---:|
| Tests passing | 143 | **176** |
| Audit findings turned into fix PRs | 4 / 10 | **9 / 10** |
| Mega-function helpers extracted | 2 (54 LOC) | **4 (208 LOC)** |
| `extract_record` LOC | 520 | **428** (-92, -18 %) |
| Open items from PROGRESS-2.md | 5 | **0** (mutmut interactive run still deferred) |

## Commit log (this continuation only)

```
ca528a097  refactor(ingestions): Phase 8 round 2 — extract 2 BIDS-walk helpers from extract_record
aa4df4acd  fix(ingestions): Phase 9 audit-3 — path-traversal hardening (F1+F2+F3)
6490f30a2  refactor(ingestions): Phase 9 audit-2 F3 — rename make_retry_client → make_authed_client
cb608ba4c  refactor(ingestions): Phase 9 audit-2 F1 — rate_limited decorator on tenacity (Retrying)
47d1d5b78  ci(ingestions): wire pytest-benchmark + memory tests into PR + main bench gates
```

## Phase status table (cumulative across all 3 sessions)

| Phase | Status |
|---|---|
| 0 Foundation | ✓ session 1 |
| 1 Parser tests | ✓ session 1 (69 tests) |
| 2 Property tests | ✓ session 1 (7 properties) |
| 3 Bare-except sweep | ✓ session 2 (17/17 files swept) |
| 4 Mutation testing | scaffold installed, interactive run remains |
| 5 Network tests | ✓ session 1 + tenacity.RetryError bug fixed |
| 6 Schema preflight | ✓ session 1 (10 tests + CI workflow) |
| 7 Memory + bench | ✓ session 1 + **session 3 CI integration** |
| 8 Decompose digest | ✓ session 1 scaffolding + **session 3 round 2 (extract_record -92 LOC)** |
| 9 Audits | **9 of 10 findings now fix PRs** (only audit-1 F2 was wrong; closed without fix) |

## What landed in this continuation

### Phase 8 round 2 — extract_record decomposition (commit `ca528a097`)

Two more helpers, each with full NumPy docstrings + 11 unit tests:

- `extract_sfreq_nchans_from_modality_sidecar(file, root, sfreq, nchans)`
  — walks `_meg.json` / `_eeg.json` / `_ieeg.json` / `_nirs.json` up the
  BIDS inheritance tree.
- `extract_sfreq_nchans_from_channels_tsv(file, root, sfreq, nchans)`
  — reads row count + `sampling_frequency` column from `_channels.tsv`.

`extract_record` is now 428 LOC (was 520). The LOC canary baseline in
`test_digest_helpers.py` is updated.

### Phase 9 audit-3 — path-traversal hardening (commit `aa4df4acd`)

New helper `path_is_within_root(path, root)` in `_parser_utils.py` that
formalises BIDS containment. Applied at two callsites:

- `_set_parser.parse_set_metadata` now calls `validate_file_path` (F3).
  Broken git-annex symlinks now return `None` consistently with the
  other parsers, where previously the raw `.exists()` returned `True`
  and `scipy.io.loadmat` would crash on the dangling target.
- `_vhdr_parser.extract_vhdr_references` rejects DataFile/MarkerFile
  references that resolve outside the `.vhdr`'s parent dir (F2).

11 regression tests in `tests/test_path_traversal.py` cover ``..``
escapes, absolute-path injection, symlink-out-of-tree, and the
`broken-symlink → None` path for `_set_parser`.

### Phase 9 audit-2 F3 — `make_retry_client` → `make_authed_client` (commit `6490f30a2`)

The old name was misleading (the client it returned had `retries=0`;
retries inject at the request site via tenacity). Renamed; old name
kept as a deprecated alias emitting `DeprecationWarning`. 2 tests
verify the deprecation and behavioural equivalence. Per
`CONTRIBUTING.md` § 6.

### Phase 9 audit-2 F1 — `rate_limited` consolidated on tenacity (commit `cb608ba4c`)

The decorator's hand-rolled try/except retry loop in `_file_utils.py`
is replaced with a tenacity `Retrying` iterator (avoids the nested
`@retry`-decorated inner function). Retry predicate
`_is_rate_limited_retryable` lives at module level. Legacy contract
preserved (retry 429 + network errors, surface non-429 HTTP, return
None on exhaustion, enforce min_interval). 9 regression tests via
factory pattern.

**Audit re-scope**: the original audit-2 F1 claimed 5 hand-rolled
retry loops across the codebase. On re-inspection only this site was
actually hand-rolled; `_parser_utils` and `_montage` have no retry
logic, and `5_inject.py` delegates to `_http.request_json`. The
"divergence" was overstated; only the rate_limited decorator was
divergent and is now converged.

### CI integration — bench + memory gates (commit `47d1d5b78`)

New workflow `.github/workflows/ingestions-bench.yml`:

- **PR**: runs `pytest tests/test_perf.py` with
  `--benchmark-json=output.json` and uploads as workflow artefact.
  The absolute ceilings in `test_perf.py` are the blocking gate
  (2 MB memory peak, 10 ms p99 latency etc.).
- **main**: also feeds the JSON to `github-action-benchmark` so the
  historic trend (kept on `gh-pages`) gets a new data point per
  commit. A future PR that regresses by > 150 % vs historic best
  triggers a PR comment (`fail-on-alert: false` so it's advisory,
  not blocking).

Locally smoke-tested: 533 KB JSON output, valid format, 2 tracked
benchmarks.

## Final test count

```
176 tests passing (+33 from session 2's 143)

New this session:
  11  tests/test_digest_extractions.py  (Phase 8 round 2 helpers)
  11  tests/test_path_traversal.py      (Phase 9 audit-3)
  2   tests/test_http.py                (audit-2 F3 deprecation)
  9   tests/test_rate_limited.py        (audit-2 F1 consolidation)
```

## Audit-finding summary (sessions 1-3)

| ID | Severity | Status |
|---|---|---|
| audit-1 F1 | **P1** silent error masking | ✓ fixed session 2 |
| audit-1 F2 | P2 _stats race | re-read: already lock-protected; closed without fix |
| audit-1 F3 | P2 no timeouts on as_completed | ✓ fixed session 2 |
| audit-1 F4 | P3 hard-coded max_workers=8 | ✓ fixed session 2 |
| audit-2 F1 | P2 retry divergence | ✓ fixed session 3 |
| audit-2 F2 | P3 missing 408 | ✓ fixed session 2 |
| audit-2 F3 | P3 make_retry_client name | ✓ fixed session 3 |
| audit-3 F1 | P2 set parser path containment | ✓ fixed session 3 |
| audit-3 F2 | P2 vhdr DataFile= sanitisation | ✓ fixed session 3 |
| audit-3 F3 | P3 validate_file_path consistency | ✓ fixed session 3 |

**9 of 10 findings closed.** The 10th (audit-1 F2) was a misread —
the `_stats` dict is already protected by `_lock` in `process_dataset`.

## What's still open after this session

1. **Phase 4 — mutmut interactive baseline**: `mutmut 3.x` config-parsing
   issue documented in `findings-phase-4.md`. An interactive run via
   `mutmut run --paths-to-mutate=_vhdr_parser.py` is the workaround;
   target ≥ 60 % kill ratio before expanding scope.
2. **Phase 8 deeper decomposition**: `digest_from_manifest` (670 LOC),
   `extract_dataset_metadata` (380 LOC), `digest_dataset` (330 LOC),
   and `extract_record` (428 LOC) all still > 100 LOC. Each round of
   extraction takes ~30 min and the safety net (characterisation tests
   + LOC canary) is already in place.
3. **PROGRESS-3 promotion**: when this branch ships, consolidate
   PROGRESS, PROGRESS-2, PROGRESS-3 into the canonical document.

## Total commits on `maturate-code` across all 3 sessions

```
db6ffd422  docs: ROBUSTNESS programme plan                                 session 1
ef7723b8e  Phase 0 — foundation                                            session 1
15f600dcd  Phase 1 — parser tests                                          session 1
5074db285  Phase 2 — property tests                                        session 1
d1788d8c2  Phase 3 — parsers bare-except                                   session 1
219558fda  Phase 3 commit 2 — _http/_bids/_github/_validate                session 1
7d4422a20  Phase 5 — respx + tenacity.RetryError fix                       session 1
749f598bc  Phase 7 — memory + bench gates                                  session 1
e39c62812  Phase 6 — schema preflight                                      session 1
bc2ceaf89  Phase 6 — golden fixture                                        session 1
a71b3521e  Phase 8 — characterisation tests                                session 1
be379785b  Phase 4 — mutmut scaffold                                       session 1
a26f7bffe  Phase 9 — three audit reports                                   session 1
8882205ea  Phase 9 F1 — silent error masking                               session 2
d8011453a  Phase 3 — 4 small scripts                                       session 2
1de5cb6f0  Phase 3 — _file_utils + api_helper                              session 2
e0d3c673d  Phase 3 — 4_validate + 5_inject                                 session 2
34b2de0ef  Phase 8 — first decomposition pass                              session 2
cdeddaf4f  Phase 9 F3 — timeouts                                           session 2
297221462  Phase 9 audit-2 F2 — 408                                        session 2
a01c96848  Phase 9 audit-1 F4 — cpu_count                                  session 2
62242f5f3  Phase 3 — 3_digest batch 1                                      session 2
f13ba7193  Phase 3 — 3_digest batch 2                                      session 2
ccfe36b2c  Phase 3 — 3_digest final                                        session 2
b411c8eba  Phase 3 — _montage (final BLE001-gated file)                    session 2
8da5a4bf9  PROGRESS-2                                                      session 2
ca528a097  Phase 8 round 2 — BIDS-walk helpers                             session 3
aa4df4acd  Phase 9 audit-3 — path-traversal hardening                      session 3
6490f30a2  Phase 9 audit-2 F3 — make_authed_client                         session 3
cb608ba4c  Phase 9 audit-2 F1 — tenacity-backed rate_limited               session 3
47d1d5b78  CI — bench + memory regression gates                            session 3
<this>     PROGRESS-3                                                      session 3
```

**32 commits across 3 sessions.** Branch `maturate-code` is ready for
review. Nothing pushed.

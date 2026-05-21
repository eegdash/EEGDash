# Cycle C2 — progress through tracks 1-3

ROADMAP-C2 has 5 tracks. This document covers the first 3
(C2.1 / C2.2 / C2.3 partial).

## Status

| Track | Status | Tests added |
|---|---|---:|
| **C2.1** _validate.py 20→76% | ✅ DONE | 29 |
| **C2.2** _bids.py 0→99% + _github.py 0→31% | ✅ DONE | 40 |
| **C2.3** parser direct tests | ✅ partial | 37 |
| **C2.4** _montage.py 17→50% | ⏳ open | — |
| **C2.5** eegdash.dataset lazy-load | ⏳ open (cross-package) | — |

**3 of 5 tracks closed.** 106 new tests this round (508 total).
Coverage 41% → 46%. Floor ratcheted 40 → 45.

## Coverage trajectory

| Round | Coverage | Tests | Gate floor |
|---|---:|---:|---:|
| Pre-C2 (end of C1) | 41% | 403 (+4 slow) | 40 |
| After C2.1 (_validate) | 43% | 432 | 42 |
| After C2.2 (_bids + _github) | 45% | 471 | 44 |
| After C2.3 (parser_utils + set tests) | **46%** | **508** | **45** |

## Per-module deltas this cycle

| Module | LOC | Pre-C2 | After C2 |
|---|---:|---:|---:|
| `_validate.py` | 188 | 20% | **76%** ✅ |
| `_bids.py` | 75 | 0% | **99%** ✅ |
| `_github.py` | 91 | 0% | 31% (pure helpers) |
| `_parser_utils.py` | 120 | 32% | **48%** |
| `_set_parser.py` | 117 | 36% | 36% (more tests but same lines hit) |

Remaining low-coverage modules for C2.4 + follow-up:
- `_montage.py` 17% — biggest untested file (428 LOC)
- `_set_parser.py` 36% — full extraction branches need real fixtures
- `_snirf_parser.py` 28%, `_mef3_parser.py` 30% — same fixture story

## Behaviour pinned by C2 tests

1. **Cross-source URL rejection across all 8 known sources** (C2.1 parametrized)
2. **Pydantic-error surfacing** for missing mandatory Record/Dataset fields
3. **Unknown-source warning + counter** in validate_dataset
4. **VALID_SOURCES is the canonical list** — every entry accepted, additions fire the test
5. **BIDS structure heuristics** for source-listing adapters (subject pattern, dataset_description, dataset zip patterns)
6. **count_bad_channels semantics** — None for missing/no-status (≠ 0 for "zero bad")
7. **GitHub token precedence** explicit > GITHUB_TOKEN > GH_TOKEN > anonymous
8. **PyGithub graceful degradation** — ImportError → None, REST fallback
9. **path_is_within_root** — the pre-PR-#327 path-traversal check pinned
10. **`build_s3_url`** — openneuro vs nemar URL shape, special-char encoding
11. **read_with_encoding_fallback** — UTF-8 happy + latin-1 fallback semantics

## Commits in this round

```
fee67572c  C2.3 — direct tests for _set_parser + _parser_utils (32→48%)
0b4f4e263  C2.2 — cover _bids.py (0→99%) + _github.py (0→31%)
8bb9c5a44  C2.1 — cover validate_record/_dataset/_digestion_output (20→76%)
```

3 code commits. ROADMAP-C2.md anchors the cycle.

## Branch position

```
maturate-code
  └── ingestion-phase4-and-8-deeper
        └── record-enumerator-merge   ← HEAD (45 commits)
              Cycle 1 (P0/P1/P2): 34 commits
              Cycle C1 (CI maturity): 8 commits
              Cycle C2 (legacy coverage): 3 commits (this round)
```

The ratchet pattern is visible: 4 successive PRs each raised the
coverage floor (35 → 38 → 40 → 42 → 44 → 45). Any future regression
fires CI.

## Remaining C2 tracks

**C2.4** — `_montage.py` 17% → 50%. The MEG branch (~190 LOC) +
EEG template fallback (~150 LOC) are the biggest untested code
paths. Harder than C2.1-C2.3 because MEG fixture data is large
and the template-matching code needs MNE's standard montages
available. Possible: use the IEEE fixtures in `tests/fixtures/`
+ mock MNE's montage builder.

**C2.5** — eegdash.dataset cold-import lazy-load. Out-of-scope
for `scripts/ingestions/`. Documented in PERFORMANCE.md with
revisit triggers. Probably best as a separate PR against the
main eegdash package.

If time permits: extend C2.3 with `_snirf_parser` and `_mef3_parser`
direct tests using the same conditional-extraction pattern as
`_set_parser`.

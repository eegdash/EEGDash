# Cycle C2 — tracks 1-4 closed (4 of 5)

PROGRESS-13 closed C2.1 + C2.2 + C2.3. This continues into C2.4
(_montage.py helpers) and parks C2.5 (cross-package lazy-load) as
out-of-scope per the original ROADMAP-C2.

## Status

| Track | Status | Tests added | Module gain |
|---|---|---:|---|
| **C2.1** _validate.py | ✅ DONE | 29 | 20% → 76% |
| **C2.2** _bids + _github | ✅ DONE | 40 | 0+0% → 99%+31% |
| **C2.3** parser direct tests | ✅ partial | 37 | _parser_utils 32→48% |
| **C2.4** _montage.py helpers | ✅ DONE (partial) | 37 | 17% → 38% |
| **C2.5** eegdash.dataset lazy-load | ⏸️ cross-package | — | (out of scope) |

**4 of 5 tracks closed.** 143 new tests across the cycle (403 → 546).
Total coverage 41% → **47%**. Floor 40 → **46**.

## Per-module coverage at C2 close

| Module | LOC | Pre-C1 | Pre-C2 | After C2 |
|---|---:|---:|---:|---:|
| `_serialize.py` | 115 | 0% | 93% | **93%** |
| `_validate.py` | 188 | 0% | 20% | **76%** |
| `_bids.py` | 75 | 0% | 0% | **99%** |
| `_github.py` | 91 | 0% | 0% | 31% |
| `_parser_utils.py` | 120 | 32% | 32% | **48%** |
| `_montage.py` | 428 | 17% | 17% | **38%** |
| `_file_utils.py` | 389 | 24% | 40% | **40%** |
| `_format_parser_registry.py` | 57 | new | 51% | 51% |
| `digest_telemetry.py` | 73 | new | 95% | 95% |
| `record_enumerator.py` | 134 | new | 89% | 89% |
| `source_adapter.py` | 98 | new | 86% | 86% |
| `_set_parser.py` | 117 | 36% | 36% | 36% (more tests, same lines) |
| `_snirf_parser.py` | 93 | 28% | 28% | 28% |
| `_mef3_parser.py` | 74 | 30% | 30% | 30% |
| `_vhdr_parser.py` | 147 | 64% | 64% | 64% |

## C2.4 highlights

The 37 new tests on `_montage.py` pin:

- **Hash determinism + invariants** — `_hash_sensors` modality-in-input
  prevents EEG-MEG collision, sub-mm jitter tolerance documented.
- **BIDS inheritance walks** — `_walk_up_find` walks parent dirs up to
  root + rejects out-of-root paths (security check).
- **Sidecar parsing** — `_parse_coordsystem_json` tries the 5 modality
  prefixes in order, units lowercased.
- **Template-matching** — `_score_template_match` tiebreaker chooses
  smaller templates (biosemi64 over standard_1005 for 64-channel data).
- **Dispatcher** — `extract_layout` aliases `'fnirs'` → `'nirs'`,
  returns None for unsupported types.

### Surprising behaviour pinned

`_parse_channels_tsv_for_eeg` drops rows where pandas reads the type
cell as NaN (the string "nan" isn't in EEG-types). The docstring says
"empty type is accepted" — divergence between docstring and code is
now visible in CI. Worth a follow-up fix if a real dataset breaks.

## Commits in this round

```
fa0db750f  C2.4 — _montage.py pure helpers (17→38%)
d224e5030  PROGRESS-13
fee67572c  C2.3 — _set_parser + _parser_utils direct tests
0b4f4e263  C2.2 — _bids.py (0→99%) + _github.py (0→31%)
8bb9c5a44  C2.1 — _validate.py (20→76%)
```

5 commits, 4 closed tracks, 143 new tests, 6 ratchets.

## Floor history (visible ratchet)

```
35 → 38 → 40 → 42 → 44 → 45 → 46
C1.1  C1.2  C1.5  C2.1  C2.2  C2.3  C2.4
```

Every step is pinned by a commit that raised coverage. The next PR
that touches a covered file MUST keep coverage ≥ 46% or CI fails.

## C2.5 — out of scope, documented

`eegdash/dataset/__init__.py` lazy-load lives outside
`scripts/ingestions/`. Touching it could break public API
(`from eegdash.dataset import DSXXXXX` relies on the eager
registration). Documented in `PERFORMANCE.md`:

- Cold-import cost ~4s, of which ~3.6s is braindecode → PyTorch
- Out of scope for ingestions/
- Revisit triggers documented (test suite startup becoming CI concern,
  package restructure for other reasons)

## What's still open (future C3 cycle candidates)

1. **_set / _snirf / _mef3 parsers 28-36% → 60%+** — needs fuller
   fixture data (real EEGLAB struct, real SNIRF HDF5, real MEF3
   blocks). Mutmut nightly (P0.2) catches some surviving mutants
   here.
2. **_github.py 31% → 60%** — the PyGithub iter_org_repos path
   needs mock pygithub object graph.
3. **_montage.py 38% → 55%** — MEG FIF streaming (190 LOC) +
   _load_mne_templates (110 LOC) + _extract_template_from_channels
   integration paths. Would need real MNE + maybe a small MEG fixture.
4. **_file_utils.py 40% → 60%** — SciDB, DataRN, git-based source
   adapters (per ADR 0001, secondary sources, low driver).
5. **C2.5 cross-package lazy-load** — when its revisit triggers fire.

## Branch position

```
maturate-code
  └── ingestion-phase4-and-8-deeper
        └── record-enumerator-merge   ← HEAD (46 commits)
              Cycle 1 (P0/P1/P2): 34 commits
              Cycle C1 (CI maturity): 8 commits
              Cycle C2 (legacy coverage): 4 commits + this doc
```

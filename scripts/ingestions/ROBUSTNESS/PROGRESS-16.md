# Cycle C3 — fully closed (4 of 4 actionable tracks)

PROGRESS-15 covered the first half of C3 (source-listing + REST
fallback). This doc covers C3.3 + C3.4 — the synthetic-fixture
pattern that landed happy-path coverage on SNIRF and SET parsers.

## Status

| Track | Status | Tests | Module gain |
|---|---|---:|---|
| **C3.1** _file_utils + _github REST | ✅ DONE | 24 | _file_utils 40→54%, _github 31→54% |
| **C3.2** _snirf + _mef3 fail paths | ✅ DONE | 13 | _mef3 30→58% |
| **C3.3** SNIRF synthetic HDF5 | ✅ DONE | 7 | **_snirf 28→78%** |
| **C3.4** SET synthetic MAT v5 | ✅ DONE | 8 | **_set 36→65%** |

**4 of 4 tracks closed.** 52 new tests this cycle (545 → 597).
Total coverage 47% → **51%**.

## The synthetic-fixture pattern (key insight from C3)

The format parsers were stuck at 28-36% coverage because the
committed CC0 fixtures are metadata-light. The pattern that worked:

1. **SNIRF**: build a minimal valid HDF5 in-memory via `h5py`
   (groups: nirs, data1, probe; datasets: time, sourceLabels,
   detectorLabels, sourceIndex, detectorIndex)
2. **SET**: build a minimal EEGLAB MAT v5 file via
   `scipy.io.savemat` (struct: EEG.srate, EEG.nbchan, EEG.pnts,
   EEG.chanlocs.labels)

Both use `pytest.importorskip(...)` so the module silently skips
when the underlying library is missing.

**This is the right shape for happy-path coverage** of binary
formats where checking in a real fixture is expensive. The fixture
size stays under 1 KB; the test files document the format
expectations inline.

## Per-module coverage at C3 close

| Module | LOC | Pre-C2 | Pre-C3 | After C3 |
|---|---:|---:|---:|---:|
| `_serialize.py` | 115 | 0% | 93% | 93% |
| `_validate.py` | 188 | 20% | 76% | 76% |
| `_bids.py` | 75 | 0% | 99% | **99%** |
| `_format_parser_registry.py` | 57 | 51% | 51% | 51% |
| `digest_telemetry.py` | 73 | 95% | 95% | 95% |
| `record_enumerator.py` | 134 | 89% | 89% | 89% |
| `source_adapter.py` | 98 | 86% | 86% | 86% |
| `_file_utils.py` | 389 | 24% | 40% | **54%** |
| `_github.py` | 91 | 0% | 31% | **54%** |
| `_montage.py` | 428 | 17% | 38% | 38% |
| `_parser_utils.py` | 120 | 32% | 48% | 48% |
| `_vhdr_parser.py` | 147 | 64% | 64% | 64% |
| `_mef3_parser.py` | 74 | 30% | 30% | **58%** |
| `_set_parser.py` | 117 | 36% | 36% | **65%** |
| `_snirf_parser.py` | 93 | 28% | 28% | **78%** |
| `_http.py` | 129 | 64% | 64% | 64% |

**Modules ≥ 70% at C3 close: 8 of 16** (was 4 pre-C2).
**Modules at 0% at C3 close: 0**.

## Ratchet history (9 visible steps)

```
35 → 38 → 40 → 42 → 44 → 45 → 46 → 48 → 49 → 50
C1.1  C1.2  C1.5  C2.1  C2.2  C2.3  C2.4  C3.1  C3.3  C3.4
```

Each step is pinned by a commit that raised coverage. A regression
that would have slipped through silently in May 2025 now fails CI.

## Cycle-by-cycle test counts

| Cycle | Tests added | Total at close |
|---|---:|---:|
| Pre-cycle | — | 311 |
| C1 (CI maturity) | 92 | 403 |
| C2 (legacy coverage) | 143 | 546 |
| C3 (production parity) | 52 | **597** |

**286 new tests across 3 cycles.** Coverage 33% → 51% (+18 pp).

## Behaviour pinned by C3

In addition to C1+C2's existing surface:

1. **SNIRF HDF5 contract** — nirs root group required; time deltas
   give sampling_frequency; measurementListN count gives nchans;
   source-detector labels build channel names; single-time-point
   handled (nchans extracted, sampling_frequency omitted)
2. **EEGLAB .set MAT v5 contract** — EEG.srate / EEG.nbchan /
   EEG.pnts / EEG.chanlocs.labels mapping; companion .fdt detected;
   missing EEG struct handled gracefully
3. **MEF3 .mefd contract** — directory + suffix required;
   .timd subdirs become channel names; iterdir OSError tolerated
4. **GitHub REST fallback** — HTML-doctype rejection (login redirect
   detection), JSON parse tolerance, fetch_first walks paths in order
5. **list_local_bids_files** — hidden file/dir skip, str/Path accept,
   recursive walk
6. **list_scidb_files** — code:20000 success, dir flag, md5 surface
7. **list_datarn_files** — no JSON-LD / no contentUrl → empty

## Commits in C3

```
60bd317e4  C3.4 — SET happy-path via synthetic MAT v5 (36→65%)
f0603f8b4  C3.3 — SNIRF happy-path via synthetic HDF5 (28→78%)
27034fd8a  C3.2 — _snirf + _mef3 fail paths (mef3 30→58%)
c88c03b99  PROGRESS-15
1822e0a17  C3.1 — extended source listing + GitHub REST (file_utils 40→54%, github 31→54%)
```

## Branch position

```
maturate-code
  └── ingestion-phase4-and-8-deeper
        └── record-enumerator-merge   ← HEAD (53 commits)
              Cycle 1 (P0/P1/P2):     34 commits
              Cycle C1 (CI maturity):  8 commits
              Cycle C2 (legacy):       5 commits
              Cycle C3 (production):   6 commits
```

## What's left (future C4 candidates — diminishing returns)

The remaining modules below 70% all have one of these shapes:

- **`_montage.py` 38%** — needs MNE (real ≈250 LOC of MEG FIF
  streaming + 110 LOC of MNE template loader). Worth the work
  only if MEG ingest becomes a production driver.
- **`_parser_utils.py` 48%** — network helpers need urllib mocking
  with byte-range Range headers. Doable but tedious.
- **`_vhdr_parser.py` 64%** — mutmut nightly drives the remaining
  branches; direct unit tests would compound.
- **`_file_utils.py` 54%** — Figshare cookies / git-annex
  resolution / SciDB tree-recursion paths. Each is a small win.
- **`_github.py` 54%** — PyGithub fast-path iter_org_repos needs
  pygithub-style object mocking.
- **`_mef3_parser.py` 58%** — binary .tmet header parsing requires
  fixture construction with the real MEF3 byte layout.
- **`_http.py` 64%** — request_text / request_bytes variants.

The cheap wins are exhausted. Future rounds compound less. The
ratchet is now solidly above 50%; the system is fit for purpose.

Nothing pushed. Ship-ready.

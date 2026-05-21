# Cycle C3 — production parity for secondary sources + parsers

C2 brought the highest-leverage legacy modules (`_validate`, `_bids`,
`_montage`, `_parser_utils`) up to a real coverage bar. C3 closes the
gap on the secondary source adapters and direct format-parser fail paths.

## Status

| Track | Status | Tests | Module gain |
|---|---|---:|---|
| **C3.1** _file_utils + _github REST fallback | ✅ DONE | 24 | _file_utils 40→54%, _github 31→54% |
| **C3.2** _snirf + _mef3 direct tests | ✅ DONE | 13 | _mef3 30→58%, _snirf fail paths |
| **C3.3** _vhdr_parser direct (already at 64%) | ⏳ open | — | mostly mutmut-driven |
| **C3.4** SciDB tree recursion / DataRN WebDAV PROPFIND | ⏳ open | — | needs deeper mock |

**2 of 4 candidate tracks closed.** 37 new tests across the round.

## Coverage trajectory

```
35 → 38 → 40 → 42 → 44 → 45 → 46 → 48
C1.1  C1.2  C1.5  C2.1  C2.2  C2.3  C2.4  C3.1
```

8 ratchet steps across 2 cycles. Floor went from "no gate" to **48%**.

## Per-module coverage at C3 close

| Module | LOC | Pre-C3 | After C3 |
|---|---:|---:|---:|
| `_serialize.py` | 115 | 93% | 93% |
| `_validate.py` | 188 | 76% | 76% |
| `_bids.py` | 75 | 99% | 99% |
| `_format_parser_registry.py` | 57 | 51% | 51% |
| `digest_telemetry.py` | 73 | 95% | 95% |
| `record_enumerator.py` | 134 | 89% | 89% |
| `source_adapter.py` | 98 | 86% | 86% |
| `_file_utils.py` | 389 | 40% | **54%** |
| `_github.py` | 91 | 31% | **54%** |
| `_montage.py` | 428 | 38% | 38% |
| `_parser_utils.py` | 120 | 48% | 48% |
| `_vhdr_parser.py` | 147 | 64% | 64% |
| `_mef3_parser.py` | 74 | 30% | **58%** |
| `_set_parser.py` | 117 | 36% | 36% |
| `_snirf_parser.py` | 93 | 28% | 28% (fail paths pinned) |
| `_http.py` | 129 | 64% | 64% |

Total: **49%**. Tests: **582**.

## Behaviour pinned by C3

1. **SciDB API contract** — `code: 20000` success marker, `dir` flag,
   version-prefix stripping, md5 surfacing
2. **DataRN WebDAV gate** — empty when JSON-LD distribution lacks
   contentUrl (no fallback)
3. **list_local_bids_files** — hidden file/dir skip, recursive walk,
   size reporting, str/Path acceptance
4. **GitHub REST fallback** — fetch_repo_file_text rejects HTML
   doctype responses (login redirect detection), 404 → None,
   fetch_repo_file_json tolerates malformed JSON
5. **fetch_first_repo_file_text** walks paths in order, returns first
   non-404
6. **MEF3 directory contract** — non-directory rejected, .mefd suffix
   required, .timd subdirs become channel names, iterdir OSError → None
7. **SNIRF fail paths** — missing file / broken symlink / garbage
   bytes / directory all → None

## Commits added this cycle

```
27034fd8a  C3.2 — direct tests for _snirf_parser + _mef3_parser (13 tests)
1822e0a17  C3.1 — extended source listing + GitHub REST fallback (24 tests)
```

## What's still open (future C4 candidates)

The remaining low-coverage modules and what each would need:

- **_set_parser 36% → 60%** — needs a real EEGLAB .set fixture with
  the full EEG struct (current CC0 fixture is metadata-light)
- **_snirf_parser 28% → 60%** — needs a real SNIRF (HDF5) fixture
  with a probe + measurementList; could build synthetically with h5py
- **_montage.py 38% → 55%** — MEG FIF streaming (190 LOC) + the
  MNE template loader (110 LOC); both need real MNE in CI
- **_github.py 54% → 80%** — PyGithub fast-path iter_org_repos
  (needs pygithub object graph mocking)
- **_file_utils.py 54% → 70%** — the helpers in lines 92-162 + 330-430
  (cookies / annex resolution / git remote URL helpers)
- **C2.5 cross-package lazy-load** — when revisit triggers fire

The pattern from C1+C2+C3 holds: each round picks one or two
modules that gain the most per test-hour, ratchets the floor, and
the gate keeps the gains.

## Branch position

```
maturate-code
  └── ingestion-phase4-and-8-deeper
        └── record-enumerator-merge   ← HEAD (50 commits)
              Cycle 1 (P0/P1/P2): 34 commits
              Cycle C1 (CI maturity): 8 commits
              Cycle C2 (legacy coverage): 5 commits
              Cycle C3 (production parity): 3 commits incl. this doc
```

Cycle C3 ship-ready. 8 ratchets visible in CI floor history.

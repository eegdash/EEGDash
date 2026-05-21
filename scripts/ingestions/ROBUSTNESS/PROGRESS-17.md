# Cycle C4 — diminishing-returns coverage round

C3 closed parser happy paths via synthetic fixtures. C4 hits the
remaining low-coverage modules with the techniques accumulated
across C1-C3:

- urllib monkeypatching (C4.1)
- PyGithub object-graph mocking via `sys.modules['github']` substitution (C4.2)
- BIDS-file / git-annex helper coverage (C4.3)

## Status

| Track | Status | Tests | Module gain |
|---|---|---:|---|
| **C4.1** _http + _parser_utils network | ✅ DONE | 28 | _http 64→84%, _parser_utils 48→80% |
| **C4.2** _github PyGithub fast-path | ✅ DONE | 4 | _github 54→85% |
| **C4.3** _file_utils helpers | ✅ DONE | 25 | _file_utils 54→60% |
| C4.4 _mef3 binary .tmet | ⏳ deferred (binary fixture construction) |
| C4.5 _montage MEG + MNE | ⏳ deferred (production driver gated) |

**3 of 5 tracks closed.** 57 new tests this round (597 → 654).
Total coverage 51% → 53%.

## Per-module coverage at C4 close

| Module | LOC | C3 end | C4 end |
|---|---:|---:|---:|
| `_serialize.py` | 115 | 93% | 93% |
| `_validate.py` | 188 | 76% | 76% |
| `_bids.py` | 75 | 99% | 99% |
| `_format_parser_registry.py` | 57 | 51% | 51% |
| `digest_telemetry.py` | 73 | 95% | 95% |
| `record_enumerator.py` | 134 | 89% | 89% |
| `source_adapter.py` | 98 | 86% | 86% |
| `_file_utils.py` | 389 | 54% | **60%** |
| `_github.py` | 91 | 54% | **85%** |
| `_montage.py` | 428 | 38% | 38% |
| `_parser_utils.py` | 120 | 48% | **80%** |
| `_vhdr_parser.py` | 147 | 64% | 64% |
| `_mef3_parser.py` | 74 | 58% | 58% |
| `_set_parser.py` | 117 | 65% | 65% |
| `_snirf_parser.py` | 93 | 78% | 78% |
| `_http.py` | 129 | 64% | **84%** |

**Modules ≥ 70%: 10 of 16** (was 8 at C3 end, 4 pre-C2).
**Modules ≥ 50%: 16 of 16.**

## Cumulative across 4 cycles

| Metric | Pre-C1 | C1 end | C2 end | C3 end | C4 end |
|---|---:|---:|---:|---:|---:|
| Tests | 311 | 403 | 546 | 597 | **654** |
| Coverage | 33% | 41% | 47% | 51% | **53%** |
| Modules at 0% | 7 | 2 | 0 | 0 | 0 |
| Modules ≥ 70% | 4 | 5 | 7 | 8 | **10** |
| Gate floor | none | 40% | 46% | 50% | **52%** |

## Ratchet history (12 visible steps)

```
35 → 38 → 40 → 42 → 44 → 45 → 46 → 48 → 49 → 50 → 51 → 52
C1.1  C1.2  C1.5  C2.1  C2.2  C2.3  C2.4  C3.1  C3.3  C3.4  C4.1  C4.2
```

## Behaviour pinned by C4

- **HTTP**: `request_text` (text+response tuple), `request_response`
  (raw httpx.Response), `build_headers` (User-Agent default + merge),
  `make_authed_client` (Bearer token), `make_retry_client` (deprecated
  but still working)
- **S3 byte-range fetch**: HTTPError/URLError/TimeoutError → None,
  servers ignoring Range and returning full body accepted
- **head_content_length**: int parsing, missing header → None,
  non-numeric → None
- **UTF-8 / latin-1 fallback**: `fetch_from_s3` graceful encoding
- **PyGithub fast-path**: iter_org_repos walks org.get_repos(),
  ImportError → REST fallback, Exception in PyGithub → REST fallback
- **fetch_repo_file_text**: PyGithub get_contents() decoded;
  directory (list-returned) → None
- **BIDS classification**: subject pattern, canonical extensions
  (.edf/.bdf/.vhdr/.set/.cnt/.nwb — NOT .fif or .snirf), root files
- **Git-annex resolution**: MD5E / SHA256E key parsing, symlink +
  smudged-pointer paths, oversized pointer rejection
- **Inline sidecar**: UTF-8 TSV/JSON/MD/TXT + README/CHANGES/LICENSE
  basenames, 5MB cap, symlinks rejected, empty file → "" not None

## Commits in this round

```
96e10b6b7  C4.3 — _file_utils helpers (54→60%)
1a88ce6be  C4.2 — PyGithub fast-path mocks (54→85%)
3a14d49f3  C4.1 — _http + _parser_utils network (28 tests)
```

## What's left (deferred)

- **`_mef3_parser` 58% → 75%** — needs synthetic .tmet binary file
  with the MEF3 header layout (well-documented but a few hundred
  bytes of struct construction). Worth doing if MEF3 ingest hits
  a real driver.
- **`_montage` 38% → 55%** — MEG FIF header streaming (190 LOC) +
  MNE template loader (110 LOC). Needs real MNE + maybe small MEG
  fixture; deferred per the same "no production driver" reasoning
  as PROGRESS-16.
- **C2.5 cross-package lazy-load** — when its revisit triggers fire.
- **`_format_parser_registry` 51% → 80%** — would compound on the
  existing parser tests; relatively cheap.

## Branch position

```
maturate-code
  └── ingestion-phase4-and-8-deeper
        └── record-enumerator-merge   ← HEAD (58 commits)
              Cycle 1 (P0/P1/P2):     34 commits
              Cycle C1 (CI maturity):  8 commits
              Cycle C2 (legacy):       5 commits
              Cycle C3 (production):   6 commits
              Cycle C4 (diminishing):  4 commits + this doc
```

The hardware-coverage stack is now: every module ≥ 50%. The two
remaining sub-60% modules (`_montage` at 38%, `_format_parser_registry`
at 51%) each have a clear next-step path documented above.

Nothing pushed. Ship-ready.

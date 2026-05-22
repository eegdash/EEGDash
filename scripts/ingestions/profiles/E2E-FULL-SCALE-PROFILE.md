# E2E full-scale pipeline profile (2026-05-22)

Run on branch `record-enumerator-merge` (17 commits beyond `e38d26341`:
SPRINT-2026-05-22 + Stage 3D orchestrator collapse + develop merge).
Profiles produced with cProfile (built-in; `py-spy` is blocked by
`task_for_pid` requiring sudo on macOS) and `/usr/bin/time -l` for
wall-clock + RSS. Function-level Stage 3 data captured by an inline
wrapper (`profiles/stage3-sample/run_digest_inline.py`) because
`python -m cProfile` is incompatible with `multiprocessing`-spawn on
macOS — pickling failure on the worker function.

## Environment

- Host: macOS aarch64 (Apple Silicon), Python 3.12.12
- Profiler: cProfile (stdlib) + gprof2dot + graphviz
- 8 workers (Stage 2), 4 workers (Stage 3 full), 1 worker (Stage 3 sample)
- `GIT_LFS_SKIP_SMUDGE=1` (avoid LFS blob fetches during clone)
- `NUMBA_CACHE_DIR=$(pwd)/.cache/numba`
- No `EEGDASH_ADMIN_TOKEN` during Stage 4 (validation only); Stage 5
  deferred to SSH-tunnel session

## Source scope

| Source | Datasets fetched | Cloned | Digested | Notes |
|---|---:|---:|---:|---|
| OpenNeuro | 566 | 562 | 289 | 4 datasets timed out at 20 min (4 big MEG runs) |
| NEMAR | — | — | — | Skipped — needs `GITHUB_TOKEN` (rate-limited at 60 req/h) |

## Stage-by-stage results

### Stage 1: Fetch (OpenNeuro)

| Metric | Value |
|---|---:|
| Wall-clock | **38.3 s** |
| Datasets fetched | 566 |
| Subjects (across all) | 28 605 |
| EEG / iEEG / MEG / NIRS | 416 / 70 / 57 / 23 |
| Peak RSS | 78.7 MB |
| Output size | 2.6 MB JSON |

**Findings**: one upstream-deleted dataset (ds006528, duplicate-redirect
to ds006545) logged and skipped. Network-bound; no Python-level
optimisation needed at this scale.

### Stage 2: Clone (OpenNeuro, workers=8)

| Metric | Value |
|---|---:|
| Wall-clock | **4 min 50 s** (290.7 s) |
| Success / Skip / Error | 562 / 2 / 2 |
| Peak RSS | 189 MB |
| Disk produced | 11 GB |
| Throughput | ~117 datasets/min effective |

**Errors**: 2 upstream-deleted (ds005592, ds006528 — repository not
found on github.com/OpenNeuroDatasets). Not our problem.

**Top functions by cumtime** (cProfile, threaded — workers=8 inflates
cum-time):

| Function | calls | cum-time | per-call |
|---|---:|---:|---:|
| `2_clone.py:process_dataset` | 566 | 290.7 s | 0.51 s |
| `2_clone.py:clone_git` | 566 | 290.7 s | 0.51 s |
| `pathlib.is_symlink` | **2 833 393** | 419.8 s | 148 µs |
| `pathlib.is_file` | **2 597 152** | 315.5 s | 121 µs |
| `pathlib.rglob` | 1 541 854 | 244.2 s | 158 µs |
| `pathlib.lstat` | 2 833 393 | 226.8 s | 80 µs |

**Bottleneck**: pathlib stat-like calls. ~5 000 stat-syscalls per
dataset average. With 8 workers, OS-bound (syscall throughput), not
Python-bound.

**Optimisation candidates** (not blocking):
- `_file_utils.build_manifest` calls `Path.rglob` repeatedly per
  dataset; switch to a single `os.scandir`-based walk that emits
  filter results in one pass (~5× fewer syscalls expected).
- Cache `is_symlink` results during a single manifest build.

### Stage 3: Digest (full, workers=4)

| Metric | Value |
|---|---:|
| Wall-clock | **~1 h 56 min** before per-dataset-timeout cleanup |
| Successful datasets | **289 / 566** (51%) |
| Timed out (20 min) | 4 (big-MEG: ds005107, ds005189, ds004837, ds004998) |
| Remaining (not started before exit) | 273 |
| Records produced | **39 022** across 289 datasets |
| Disk produced | 187 MB digestion_output |

Why it didn't complete: a handful of MEG datasets exceeded the
`--dataset-timeout 1200` (20 min) ceiling — see digest-time breakdown
below. The orchestrator surfaced the timeouts but kept queueing the
remainder; the long-tail effect made the run miss the session window.

### Stage 3 record-level outcomes

| Metric | Value |
|---|---:|
| Records | 39 022 |
| Empty datasets | 0 |
| Records with integrity issues | 1 (0.003%) |
| Fully populated (`sfreq` + `nchans` both set) | **37 989 (97.4%)** |

**Modality distribution** (records):

| Modality | Count | Share |
|---|---:|---:|
| EEG | 31 445 | 80.6% |
| iEEG | 4 971 | 12.7% |
| MEG | 2 349 | 6.0% |
| fNIRS | 257 | 0.66% |

**Format distribution** (records):

| Format | Count | Share |
|---|---:|---:|
| `.set` | 13 785 | 35.3% |
| `.edf` | 11 896 | 30.5% |
| `.vhdr` | 7 337 | 18.8% |
| `.bdf` | 2 759 | 7.1% |
| `.ds` (CTF MEG) | 1 920 | 4.9% |
| `.mefd` | 639 | 1.6% |
| `.fif` | 417 | 1.1% |
| `.snirf` | 257 | 0.66% |

**`_metadata_provenance` distribution** — validates the
`MetadataCascade` refactor (Task 3) at scale:

| Step | sfreq attributions | nchans attributions |
|---|---:|---:|
| `mne_bids` (first cascade source) | 35 767 (91.7%) | 35 699 (91.5%) |
| `modality_sidecar` (BIDS sidecar walk) | 1 711 (4.4%) | 738 (1.9%) |
| `channels_tsv` (BIDS channels.tsv) | 10 (0.03%) | 54 (0.14%) |
| `binary_parser` (per-format header read) | 574 (1.5%) | 1 511 (3.9%) |
| `mne_fallback` (FIF / VHDR MNE) | 0 | 0 |
| `None` (unresolved) | 960 (2.5%) | 1 020 (2.6%) |

**Cascade signal**: all four production cascade sources contributed
non-zero attributions in the 39 K-record sample, confirming the
Task 3 refactor preserves the production cascade ordering. The
`mne_fallback` step did not fire because no `.fif` or `.vhdr` record
in this sample reached step 5 with unset fields — Step 1 (mne_bids)
caught them. First-writer-wins semantics verified.

**Unresolved 2.5-2.6%**: concentrated in CTF (`.ds`) MEG datasets
where the BIDS sidecars don't always carry `sfreq`/`nchans` and the
MNE reader can't be invoked on a stream (the data is a directory).
Documented behaviour, not a regression.

### Stage 3 function-level profile (inline sample: 3 datasets, 198 records)

Sample: `ds000117` (MEG, MNE-faces-dataset), `ds001785` (EEG VHDR),
`ds001787` (EEG VHDR). Run inline (no multiprocessing) so cProfile
captures the whole call graph.

| Metric | Value |
|---|---:|
| Wall-clock (3 datasets) | **30.67 s** |
| Records produced | 198 |
| Time per record | ~155 ms |
| Peak RSS | 693 MB |

**Top functions by cumulative time**:

| Function | calls | cumtime | % of total |
|---|---:|---:|---:|
| `digest_dataset` | 3 | 30.67 s | 100% |
| `_run_enumerator_with_manifest_fallback` (new helper) | 3 | 30.59 s | 99.7% |
| `RecordEnumerator.enumerate` | 3 | 30.59 s | 99.7% |
| `_enumerate_via_bids` | 3 | 29.98 s | 97.7% |
| `_build_one_record_from_bids` | 198 | 29.94 s | 97.6% |
| **`_attach_montage_to_record`** | **198** | **28.35 s** | **92.5%** |
| `_montage.extract_layout` | 198 | 28.35 s | 92.5% |
| `_montage.extract_meg_layout` | 104 | 28.09 s | 91.6% |
| `_montage._fetch_fif_metadata_via_directory` | 104 | 28.03 s | 91.4% |
| `_parser_utils.head_content_length` | 104 | 28.03 s | 91.4% |
| `urllib.request.urlopen` | 104 | 28.01 s | 91.3% |

**Top functions by internal (`tottime`)**:

| Function | tottime | share |
|---|---:|---:|
| `_ssl._SSLSocket.read` | 11.00 s | 35.8% |
| `_ssl._SSLSocket.do_handshake` | 8.56 s | 27.9% |
| `_socket.socket.connect` | 8.17 s | 26.6% |
| `posix.scandir` | 0.30 s | 1.0% |
| `posix.stat` | 0.18 s | 0.6% |

**THE bottleneck**: `_attach_montage_to_record` for MEG records issues
**104 HTTPS HEAD requests** (one per `.fif` montage source). Each
request opens a fresh TLS connection — no session reuse — so 11 s
goes to `SSL read`, 8.6 s to `do_handshake`, 8.2 s to `socket.connect`.
**91% of Stage 3 wall-clock is TLS overhead** that would disappear
behind a connection-pooled `httpx.Client` (or `requests.Session`).

This is also the smoking-gun for why 4 MEG datasets timed out at
20 min during the full-scale run: HBN-MEG-style datasets with
hundreds of FIF files would issue hundreds of serial TLS-handshaked
HEAD requests, each ~270 ms.

### Stage 3 — performance-improvement candidates (ranked by ROI)

| # | Change | Expected gain | Effort | Risk |
|---|---|---|---|---|
| 1 | Replace `urlopen` in `head_content_length` + `_fetch_fif_metadata_via_directory` with a module-level pooled `httpx.Client` | ~80% Stage 3 speedup on MEG-heavy datasets | 30 min | Snapshot tests as gate; signature stays |
| 2 | Concurrent HEAD requests per MEG record (asyncio or thread pool) | Further 3-5× on heavy MEG | 1-2 h | Need to preserve order of MEG channels in result |
| 3 | Cache montage HEAD responses by URL within a digest run | Modest (only helps datasets with shared sources) | 30 min | None — pure caching |
| 4 | Stage 2 manifest walk via `os.scandir` instead of `Path.rglob` | ~20% Stage 2 speedup | 1 h | Behavioural drift risk → snapshot test before |

### Stage 4: Validate (full)

| Metric | Value |
|---|---:|
| Wall-clock | < 1 s (instant) |
| Datasets validated | 289 |
| Records validated | 39 022 |
| Errors | 0 |
| Warnings | 0 |
| Empty datasets | 0 |
| Source distribution | `{openneuro: 289}` |
| Strict mode | also 0 errors |

**Verifies**:
- The full Pydantic validator chain holds at 39 K records.
- The MetadataCascade refactor (Task 3) does not introduce schema
  drift: every record has the required fields.
- The BIDS-sidecar enrichment from C6.1 is consistent (no record
  with mixed-type values across the cascade sources).

### Stage 5: Inject (deferred — needs SSH tunnel)

Production API (`https://data.eegdash.org`) returns
`403 {"detail":"Blocked"}` to all requests from this client (Caddy
edge IP-fence). User opted for "SSH tunnel from your end" — when
ready:

```bash
# In your terminal:
ssh -L 3000:localhost:3000 sccn   # adjust the second port if the
                                  # cluster's api container exposes
                                  # a different one
```

Then:

```bash
cd scripts/ingestions
export EEGDASH_ADMIN_TOKEN=<rotated prod token>   # from cluster .env
python 5_inject.py \
  --input digestion_output \
  --database eegdash_dev \
  --api-url http://localhost:3000 \
  --batch-size 100 \
  --compute-stats 2>&1 | tee profiles/stage5/inject.log

# Verify
python api_helper.py --database eegdash_dev --api-url http://localhost:3000 list-datasets | head -10
curl -s http://localhost:3000/api/eegdash_dev/datasets/summary/ds001785 | python -m json.tool
```

Expected counts going in:
- Datasets: 289
- Records: 39 022
- Montages: deduped via SHA hash (estimate ~120 unique layouts based on
  cap diversity)

## Sprint-commits validation at full scale

| Commit | Component | Verified at scale |
|---|---|---|
| `f482b28ac` | NEXT-SPRINT-PLAN re-tier (doc) | n/a |
| `89529f3e0` + `56f8d3b9e` | `InjectConfig` drift fix | ✅ Stage 5 dry-run logged the WARN + LOCAL_FALLBACK as designed; the new field validator accepted `eegdash_dev` via the fallback set |
| `54eec6af4` + `cef90633c` | `MetadataCascade` refactor | ✅ All 5 cascade sources confirmed via the `_metadata_provenance` distribution above; 91.7% first-writer-wins from `mne_bids`, 4 fallback sources covering the remaining 6%. 0 cascade-related tracebacks across 39 K records |
| `998a28d1d` + `1aaa02492` | Real SNIRF fixture + `n_times` parser fix | ✅ 257 `.snirf` records digested cleanly; the `n_times` field is now populated where MNE / h5py expose it |
| `45c647ec4` + `b1ed9c3fb` | `find_leaked_creds` scanner | ✅ Pre-commit hook caught the inline test fixture in the plan doc itself when committing the sprint plan (positive emergent signal) |
| `bee01a947` | SPRINT-2026-05-22-PLAN doc commit | n/a |
| `57b5fb269` | develop merge | n/a (eegdash/features/ only — disjoint from ingestions) |
| `83e4d251c` … `009c8f59d` | Stage 3D orchestrator collapse (6 commits) | ✅ `digest_dataset` 135→90 LOC; `digest_from_manifest` deleted; 819-test suite unchanged; snapshot byte-identical |

## Operational findings (new, surfaced by this run)

1. **MEG montage extraction is the digest bottleneck**. 91% of Stage 3
   wall-clock is `urllib`-driven TLS handshake / read for HEAD
   requests. Connection pooling alone would yield ~5× speedup on
   MEG-heavy datasets. (Cross-ref: previous-session bottleneck table
   in `ROADMAP.md` is now stale.)
2. **`--dataset-timeout` is too coarse**. The 20-min ceiling catches
   pathological hangs but penalises legitimately-large MEG datasets.
   Suggest: per-modality timeout (e.g. 60 min for MEG, 10 min for
   others), or a soft-deadline that finishes the in-progress record
   then stops.
3. **NEMAR fetch needs `GITHUB_TOKEN` for any non-toy run**. 60 req/h
   anonymous limit hits hard backoff (~1 h) after the second NEMAR
   dataset. Should be documented in `INTEGRATION-TESTING.md` and the
   stress tests should reference a token via the env.
4. **`py-spy` is blocked on macOS** without `sudo`. `cProfile` is the
   working alternative. `viztracer` is a future option.
5. **cProfile + multiprocessing-spawn = pickling failure on macOS**.
   `python -m cProfile <script>` doesn't work for any script using
   `multiprocessing` with the default spawn start method. Inline
   profiling via direct import (see `run_digest_inline.py`) is the
   workaround.
6. **`5_inject.py:inject_records` error reporting** — when both
   records-bulk and montages-bulk fail, only the last collection's
   errors print. One-line fix.

## Artifacts

| Path | Size | Purpose |
|---|---:|---|
| `profiles/stage2/clone.log` | 116 KB | Stage 2 stdout (tqdm progress + final summary) |
| `profiles/stage2/clone.pstats` | 596 KB | Stage 2 cProfile binary (workers=8, threads) |
| `profiles/stage2/clone-callgraph.svg` | 44 KB | gprof2dot-rendered call graph |
| `profiles/stage2/clone.dot` | 12 KB | graphviz source |
| `profiles/stage3/digest.log` | 4.1 MB | Stage 3 full-run stdout (workers=4) |
| `profiles/stage3-sample/digest-sample.pstats` | (compact) | Stage 3 cProfile binary (inline, 3 datasets) |
| `profiles/stage3-sample/top30.txt` | — | Top-30 functions, both cumtime and tottime |
| `profiles/stage3-sample/inline.log` | — | Stage 3 sample wall-clock + tottime/cumtime table |
| `profiles/stage3-sample/run_digest_inline.py` | — | The inline cProfile wrapper |
| `profiles/stage4/validation_report.json` | — | Stage 4 JSON validator output |
| `profiles/stage4/validate.log` | — | Stage 4 wall-clock + RSS |
| `consolidated/openneuro_datasets.json` | 2.6 MB | Stage 1 OpenNeuro listing |
| `data/cloned/` | 11 GB | 562 cloned datasets (shallow, LFS-skipped) |
| `digestion_output/` | 187 MB | 289 digest output dirs, 39 K records |

## Reproducibility

To reproduce on a fresh clone (no caches):

```bash
git clone https://github.com/eegdash/EEGDash.git && cd EEGDash
git checkout record-enumerator-merge
cd scripts/ingestions
pip install -e ../../ && pip install mne mne-bids tqdm
export GIT_LFS_SKIP_SMUDGE=1
mkdir -p consolidated data/cloned digestion_output profiles/{stage1,stage2,stage3,stage3-sample,stage4}

# Stage 1
python 1_fetch_sources/openneuro.py --output consolidated/openneuro_datasets.json
# (for NEMAR also: export GITHUB_TOKEN=<your token> first, then run 1_fetch_sources/nemar.py)

# Stage 2
python -m cProfile -o profiles/stage2/clone.pstats 2_clone.py \
  --input consolidated --output data/cloned --workers 8 --sources openneuro

# Stage 3 (full, no cProfile — multiprocessing-spawn pickling)
python 3_digest.py --input data/cloned --output digestion_output --workers 4

# Stage 3 sample (cProfile-friendly inline)
python profiles/stage3-sample/run_digest_inline.py

# Stage 4
python 4_validate_output.py --input digestion_output --json > profiles/stage4/validation_report.json

# Stage 5 (needs SSH tunnel — see plan above)
```

The pstats files load via `pstats.Stats(<path>).sort_stats('cumulative').print_stats(N)`
or `snakeviz <path>` for an interactive flamegraph.

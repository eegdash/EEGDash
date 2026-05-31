# Efficient Remote Header Reader — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]`.

**Goal:** Close the shallow-clone `n_times` gap (MEF3 4.1%, SNIRF 48.9%, the EDF/BDF/SET
tail) in the **fewest bytes** — zero where the data layout allows, a few-KB ranged
header read otherwise — never touching signal, never shipping a guessed value.

**Spec:** `docs/superpowers/specs/2026-05-31-efficient-remote-header-reader-design.md`
(read it — the failure catalog is the contract).

**Correctness nuance found during design:** EDF+ files carry an extra *EDF Annotations*
signal with its own `samples_per_record`, so the pure `size/(nchans×dtype)` zero-fetch
is **unsafe for EDF/BDF** — the robust path is a **256-byte** main-header read. SET's
external `.fdt` *is* a clean zero-fetch (raw float32, no annotations). VHDR (shipped) is
the other clean zero-fetch.

**Run tests from** `scripts/ingestions/` (`python3 -m pytest … -q -p no:cacheprovider`).
Mark anything touching the network `@network` (and slow ranged reads `@slow`).

---

## Phase RH1 — SET external `.fdt` zero-fetch (0 bytes)

EEGLAB external `.set` stores continuous data in a companion `.fdt` as raw
`float32`, channels × points, no header: `n_times = fdt_size / (nchans × 4)`. The
`.fdt` size comes from its git-annex key — **zero fetch**, exactly like VHDR's `.eeg`.

**Files:** `scripts/ingestions/_set_parser.py`; test `tests/parsers/test_cheap_paths.py`.

### Task RH1.1: `.fdt` size arithmetic (TDD)
- [ ] **Step 1 — failing test** (`tests/parsers/test_cheap_paths.py`):
```python
def test_set_external_ntimes_from_fdt_size(tmp_path: Path):
    # External .set: small .set text (continuous, trials=1) + a .fdt whose SIZE encodes n_times.
    from _set_parser import parse_set_metadata
    from _helpers.builders import build_synthetic_set_v5  # writes a v5 .set
    set_path = build_synthetic_set_v5(tmp_path / "e.set", srate=250.0, nbchan=4, pnts=0,
                                      external_fdt=True)   # NEW builder kwarg: emit datfile ref
    (tmp_path / "e.fdt").write_bytes(b"\x00" * (4 * 1000 * 4))  # 4 ch × 1000 samples × float32
    out = parse_set_metadata(set_path)
    assert out["n_times"] == 1000 or out["n_samples"] == 1000
```
  (If extending the builder is heavy, instead unit-test a pure helper
  `_fdt_n_times(fdt_size, nchans) -> int | None` directly.)
- [ ] **Step 2 — run, expect fail.**
- [ ] **Step 3 — implement.** Add a module-level helper and call it when `has_fdt` and
  the `.set` declares continuous data (`EEG.trials == 1`, `EEG.data` is a string):
```python
def _fdt_n_times(fdt_path: Path, nchans: int | None) -> int | None:
    """EEGLAB external .fdt = raw float32 [nchans × n_times]; n_times from SIZE alone."""
    if not nchans or nchans <= 0:
        return None
    from _sizing import data_file_size
    size = data_file_size(fdt_path)            # annex key -> 0 bytes; or os.stat
    if not size or size % (nchans * 4) != 0:
        return None
    return size // (nchans * 4)
```
  Guard: only when `EEG.trials == 1` (continuous) — epoched `.set` (3-D) breaks the
  divide (failure-catalog A4). Wire after the scipy/h5py extraction: if `n_times`
  still absent and `has_fdt` and `nchans`, set it from `_fdt_n_times`.
- [ ] **Step 4 — run, expect pass; run `tests/parsers/test_set.py` (no regression).**
- [ ] **Step 5 — commit.**

---

## Phase RH2 — `RangeReader` + locate layer (the T2 substrate)

A budget-capped, 206-asserting, block-cached seekable file backed by HTTP Range,
plus the per-source URL derivation. Reuses `_parser_utils.fetch_bytes_from_s3`,
`head_content_length`, `extract_openneuro_info`/`build_s3_url`.

**Files:** create `scripts/ingestions/_remote_header.py`; tests
`tests/unit/test_remote_header.py` (use `respx` to mock S3 — already a dev dep).

### Task RH2.1: `RangeReader` (TDD, mocked transport)
- [ ] **Step 1 — failing tests** (respx-mocked):
  - a ranged GET returns 206 + the requested slice → `RangeReader.read(start,n)` yields it;
  - a 200 (Range ignored) on a large object → `RangeReader` raises `RangeUnsupported` (never buffers the body);
  - repeated reads within one 64-KB block hit the cache (one network call);
  - total bytes over budget → `ByteBudgetExceeded`.
- [ ] **Step 2 — run, expect fail.**
- [ ] **Step 3 — implement** `RangeReader(url, *, budget=128*1024, block=64*1024)` with
  `read(offset, length)`, a `{block_index: bytes}` cache, `bytes_fetched` counter,
  and a strict `206`/`Content-Range` assertion (abort on `200`). Provide a
  `.seekable file-like` adapter (`read`/`seek`/`tell`) so `h5py.File(reader)` and a MAT
  reader can consume it.
- [ ] **Step 4 — run, expect pass.** **Step 5 — commit.**

### Task RH2.2: `locate(record) -> (size:int|None, url:str|None)` (TDD)
- [ ] Resolve the annex-key **size** (T1, free) via `_file_utils` for any source.
- [ ] Resolve the **URL** (T2): OpenNeuro → `build_s3_url(dataset, relpath)`;
  NEMAR → mirror-if-mirrored else `None` (S3 closed, failure-catalog B8). Test both
  with fixtures; assert NEMAR-only returns `url=None` (→ caller drops to T3).
- [ ] **Commit.**

---

## Phase RH3 — EDF/BDF 256-byte robust header (annotation-safe)

`n_times = number_of_data_records × round(sfreq × duration_of_data_record)`, where
`number_of_data_records` (off 236) and `duration_of_data_record` (off 244) come from
the **256-byte main header** and `sfreq` from the sidecar. Annotation-safe (the
record count is independent of the annotations signal) and exact (verified on
`ds004577`: 37 × round(200×8) = 59,200).

**Files:** `scripts/ingestions/_edf_header.py` (new, pure); tests.

### Task RH3.1: pure 256-byte EDF parser (TDD, no network)
- [ ] **Step 1 — failing test** with a hand-built 256-byte EDF main header (set
  `number_of_data_records=37`, `duration_of_data_record=8`, `n_signals=19`):
  `edf_n_times_from_main_header(header_256, sfreq=200.0) == 59200`.
- [ ] **Step 2 — run, expect fail.**
- [ ] **Step 3 — implement** `edf_n_times_from_main_header(buf: bytes, sfreq: float|None) -> int|None`:
  parse the three ASCII fields; if `number_of_data_records > 0` and `sfreq` and
  `record_duration > 0` → `round(sfreq * record_duration) * number_of_data_records`;
  on unclean-stop (`-1`) or missing sfreq → `None` (caller falls back to T1 size-arith /
  T3). Never raises.
- [ ] **Step 4 — run, expect pass.** **Step 5 — commit.**

### Task RH3.2: wire EDF/BDF to RangeReader + verify vs MNE (`@network @slow`)
- [ ] Fetch the first 256 B via `RangeReader` for an annex-pointer EDF; feed
  `edf_n_times_from_main_header`. Validate **== `mne.io.read_raw_edf(...).n_times`**
  on ≥3 real OpenNeuro EDF/BDF (incl. one EDF+ with an annotations channel — the case
  that breaks naive size-arithmetic). Skip if offline.
- [ ] **Commit.**

---

## Phase RH4 — MEF3 + SNIRF ranged readers

### Task RH4.1: MEF3 `.tmet` over Range
- [ ] The shipped `_parse_tmet_number_of_samples` already reads a `bytes` buffer
  conceptually — refactor it to accept a `RangeReader` (or a whole ≤16-KB `.tmet`
  fetched via one ranged GET) and reuse the `number_of_blocks` consistency guard.
  `@network @slow` test vs the real corpus where the `.tmet` is annex'd.
- [ ] **Commit.**

### Task RH4.2: SNIRF `dataTimeSeries.shape` over Range
- [ ] `h5py.File(range_reader_filelike, "r")` → read `…/dataTimeSeries.shape[0]` only
  (HDF5 fetches just metadata blocks; the block cache bounds round-trips). Assert
  `bytes_fetched < budget`. `@network @slow` test on the real SNIRF.
- [ ] **Commit.**

(SET-embedded MAT field-skip and FIF head+dir are lower-volume — defer to RH4.3/4.4,
or T3 with a logged reason if the byte budget can't be met.)

---

## Phase RH5 — Cascade integration (opt-in tiers) + telemetry

### Task RH5.1: `SizeArithmeticStep` (T1) + `RemoteHeaderStep` (T2)
- [ ] Add two cascade steps **between** `BinaryParserStep` and `MneFallbackStep`:
  - `SizeArithmeticStep` (T1, 0 bytes): VHDR (already), SET-external `.fdt`, and
    EDF/BDF size-arith **only when provably safe** (no annotations evidence + uniform).
  - `RemoteHeaderStep` (T2): gated by a config flag `EEGDASH_REMOTE_HEADERS=1`
    (default off → local path unchanged, golden masters green). Uses `locate` +
    `RangeReader` + the RH3/RH4 readers; respects the per-dataset byte budget.
- [ ] Keep first-writer-wins + provenance: new sources `size_arithmetic`,
  `remote_header`. Update the provenance closed-set test + the documented enum.
- [ ] **Commit** (golden masters must stay green with the flag OFF).

### Task RH5.2: byte-cost telemetry + coverage reasons
- [ ] Emit `bytes_fetched` + `resolution_tier` on `record_built`; extend `_coverage.py`
  to report median/p99 bytes and to attribute each unresolved record to a
  failure-catalog reason (NEMAR-S3-closed, compressed, non-uniform, no-URL, …).
- [ ] **Commit.**

---

## Phase RH6 — Validate + re-measure at scale

- [ ] Re-run the full pipeline with `EEGDASH_REMOTE_HEADERS=1` over the cloned 685
  datasets; diff coverage vs the current `coverage_fullscale.json`. Expect SNIRF and
  EDF/BDF/SET tails to climb toward the high-90s; report the **bytes-fetched
  distribution** (target: median 0 B). Update the HTML report with an "after remote
  headers" panel and the byte-cost histogram.
- [ ] **Commit** the new coverage artifacts.

---

## Sequencing rationale (efficiency-first)
1. **RH1 + RH3** are the bulk of the gap and cost **0 / 256 bytes** — do them first.
2. **RH2** is the shared substrate for everything ranged.
3. **RH4** (SNIRF/MEF3) is lower-volume but needs RH2.
4. **RH5/RH6** integrate + prove the byte cost.

## Self-review
- Every tier returns `None` (→ T3) in ambiguous cases — no guessed `n_times`.
- T1 paths fetch 0 bytes by construction; T2 paths are budget-capped + 206-asserted.
- Remote reads opt-in; local/shallow path byte-identical when off (golden masters).
- Each failure-catalog item maps to a coverage reason — the residual gap is never hidden.

# Efficient Remote Header Reader — Design

**Status:** Design (drives a plan)
**Date:** 2026-05-31
**Motivation:** The full-scale run (172,432 records) exposed the shallow-clone wall:
when a recording's binary is a git-annex pointer, the header-struct parsers can't
read it, so MEF3 (4.1%), SNIRF (48.9%), and the long tail of EDF/BDF/SET fall back
to sidecar arithmetic — and exotic `.gz/.apr/.apx` get nothing.
**Constraint:** *Efficiency.* Extract the metadata in the **fewest possible bytes** —
ideally **zero** — and only ever touch a header, never signal.

---

## 1. The reframing (empirically verified)

The git-annex key encodes the exact file size for free (`MD5E-s<size>` /
`SHA256E-s<size>`). For any format that stores data as a **flat
`nchans × n_times × dtype` block**, `n_times` is pure arithmetic on that size — **zero
bytes fetched**, exactly like the VHDR win already shipped.

**Verified on a real OpenNeuro file** (`ds004577`, 19-ch EDF, annex size
2,254,720 B): `n_times = (2,254,720 − 256×(19+1)) / (19 × 2) = 59,200`, matching the
real EDF header (37 records × 1,600 samples/record). `nchans` came from the
**sidecar** — so the EDF itself contributed **0 bytes**. A confirming 256-byte HTTP
Range read against `s3://openneuro.org/...` returned **HTTP 206** (Range honored).

So the gap is mostly closable for **zero bytes**; only compressed/structured formats
need an actual read, and then only a few KB.

## 2. The cost-ordered resolution ladder

Every field flows through tiers, cheapest first; stop at the first that produces a
trustworthy value. Bytes-fetched is the cost metric.

| Tier | Cost | Mechanism | Formats |
|------|------|-----------|---------|
| **T0 Sidecar** | 0 B | `round(sfreq × RecordingDuration)`; channels.tsv | all (shipped) |
| **T1 Annex-size arithmetic** | **0 B** | `n_times = (size − header) / (nchans × dtype)` from the annex key + sidecar nchans | VHDR ✓, **EDF, BDF, SET-external(.fdt)** |
| **T2 Tiny ranged header** | ≤ budget | one or two HTTP Range reads of the header region | EDF/BDF (non-uniform), **MEF3 .tmet**, **SNIRF HDF5 shape**, SET-embedded, FIF |
| **T3 Unresolved** | 0 B | log + coverage metric | exotic / unreachable |

### Per-format byte budget

| Format | n_times source | Bytes fetched | Notes |
|--------|----------------|---------------|-------|
| **.vhdr** | `.eeg` annex size / (nchans×dtype) | **0** | shipped |
| **.edf** | `(size − 256×(nchans+1)) / (nchans×2)` | **0** | uniform sampling; verified |
| **.bdf** | `(size − 256×(nchans+1)) / (nchans×3)` | **0** | 24-bit samples |
| **.set (external .fdt)** | `fdt_size / (nchans×4)` | **0** | `.fdt` = raw float32, no header |
| **.edf/.bdf (non-uniform)** | first `256×(nchans+1)` B → per-signal samples block | ~`0.25×nchans` KB | only when channels differ; detect via 256-B main header |
| **.mefd** | `.tmet` `number_of_samples` @ sfreq_off+200 | **≤ 16 KB** (whole .tmet) or ~256 B ranged | `.tdat` never touched |
| **.snirf** | HDF5 `dataTimeSeries.shape[0]` | **few KB** (HDF5 metadata blocks) | via h5py over a Range-backed file |
| **.set (embedded)** | MAT struct field-skip to `EEG.pnts` | ~16–64 KB prefix | `pnts` is an early scalar field |
| **.fif** | tag-directory walk | 2 ranges (head + dir) | dir pointer near start, dir block near end |

**The headline:** EDF + BDF + SET-external ≈ the bulk of the remaining gap, and they
cost **0 bytes** — they just need to be wired into T1 like VHDR. Only `.snirf` /
`.mefd` / `.fif` / embedded-`.set` need T2, and their budgets are KB, not MB.

## 3. The remote-locate layer (where the bytes live)

T2 needs a URL for an annex-pointer file. T1 needs only the annex key (size), which
is in the symlink target / pointer text — already parsed by `_file_utils.parse_annex_size`.

| Source | Size (T1) | Header bytes (T2) |
|--------|-----------|-------------------|
| **OpenNeuro** | annex key (free) | `s3://openneuro.org/<ds>/<relpath>` — anonymous Range **verified 206** (also the existing `build_s3_url`/`fetch_from_s3` path) |
| **NEMAR (S3-closed)** | annex key (free) | **No S3 GetObject.** T1 still works (size only). T2 must use the GitHub-mirror-resolved annex special remote or `data.nemar.org`; many NEMAR datasets mirror OpenNeuro → use that S3. Otherwise **T3 (documented gap).** |

Existing primitives to reuse: `_parser_utils.fetch_bytes_from_s3(url, start, max_bytes)`
(sends `Range: bytes=`), `head_content_length(url)`, the pooled `_http_client()`,
`extract_openneuro_info` / `build_s3_url`.

## 4. Component decomposition

```
_remote_header/
├─ locate.py        annex-key size (T1, 0-fetch) + remote URL derivation per source
│                    (OpenNeuro S3; NEMAR-> mirror/closed). Returns (size, url|None).
├─ ranged_io.py     RangeReader: a seekable file-like backed by HTTP Range with a
│                    hard byte-budget + block cache; asserts 206 (aborts on 200 of a
│                    large object). Wraps fetch_bytes_from_s3. Feeds h5py/MAT readers.
├─ sizing_arith.py  T1: per-format size arithmetic (EDF/BDF/SET-external), reusing
│                    the dtype/header constants. Pure; 0 bytes.
├─ headers/         T2 readers that accept a RangeReader OR a local Path:
│   ├─ edf.py        256-B main header + (optional) signal-header samples block
│   ├─ mef3.py       .tmet number_of_samples (ranged or whole-16KB)
│   ├─ snirf.py      h5py over RangeReader -> dataTimeSeries.shape
│   ├─ set_embedded.py  MAT v5 field-skip / h5py v7.3 -> pnts
│   └─ fif.py        tag-directory walk (head + dir ranges)
└─ budget.py        per-file + per-dataset byte budget, telemetry (bytes_fetched).
```

This slots **between** the existing local header-struct tier and the (removed)
full-MNE tier in `_metadata_cascade`. It is **opt-in** (a config flag), so the pure
local/shallow path is unchanged when remote reads are disabled.

## 5. Efficiency guarantees (how "few bytes" is enforced)

1. **Zero-fetch first.** T1 always tried before T2. The resolver records the chosen
   tier; a record that resolved at T1 fetched 0 bytes by construction.
2. **Hard byte budget.** `RangeReader` caps total bytes per file (default 128 KB) and
   per dataset; exceeding it aborts to T3 rather than degrading into a full read.
3. **206-or-bust.** If an endpoint ignores `Range` and returns `200` for a large
   object, abort immediately (never stream a multi-MB body).
4. **Pooling + batching + parallel.** One pooled HTTPS client; T2 reads across a
   dataset's records issued concurrently (bounded). T1 needs no network at all.
5. **Telemetry.** Emit `bytes_fetched` and `resolution_tier` per record so the
   coverage report can prove the byte cost (target: median 0 B, p99 < budget).

## 6. FAILURE CATALOG — *every* way this fails, and the mitigation

### A. Arithmetic / format-truth failures (wrong value risk — the worst)
1. **EDF/BDF non-uniform sampling.** Channels with different `samples_per_record`
   (polysomnography: EEG 256 Hz + SpO₂ 1 Hz) break the flat divide → inflated/garbage
   `n_times`. **Mitigation:** T1 only when uniformity is evidenced (single sfreq in
   channels.tsv / sidecar); otherwise a 256-B main-header read checks the per-signal
   samples block; if non-uniform, sum the per-channel samples (still header-only) or
   drop to T3. **Never ship a non-uniform flat value.**
2. **EDF unclean stop** (`number_of_data_records = -1`). Header count invalid → must
   use file-size (which T1 already does — robust here), but `record_duration` may also
   be wrong. **Mitigation:** size-arithmetic is authoritative; cross-check vs
   `sfreq×RecordingDuration` within tolerance, log discrepancy.
3. **EDF+/BDF+ annotations channel.** The `EDF Annotations` signal occupies samples
   too; it's counted in nchans and its samples_per_record differs → flat divide wrong.
   **Mitigation:** detect the `EDF Annotations` label in the 256-B+signal header;
   exclude it from the data-channel count, or use header `number_of_data_records ×
   samples_per_record[data_channel]`.
4. **SET external `.fdt` with non-float32 / non-contiguous layout.** Rare EEGLAB
   variants store `int16` or epoched 3-D data → `fdt_size/(nchans×4)` wrong.
   **Mitigation:** trust only when the `.set` declares `EEG.data` is a `.fdt` string
   and 2-D continuous; else T2/T3. (The `.set` text is small/present.)
5. **Compressed recordings** (`.edf.gz`, `.bdf.gz`, gzipped data). Annex size is the
   *compressed* size → arithmetic invalid. **Mitigation:** extension/​magic-byte check;
   compressed → T2 ranged-decompress is infeasible cheaply → **T3** (documented).
6. **MEF3 coincidental `number_of_samples`.** The +200 offset relies on locating the
   `sampling_frequency` double; a wrong match yields garbage. **Mitigation:** already
   guarded by the `number_of_blocks` consistency check (shipped); keep it.
7. **HDF5 dtype ≠ assumed** (SNIRF `dataTimeSeries` could be float32). Only matters if
   we ever size-arithmetic HDF5 (we don't — we read `.shape`). **No risk** as designed.

### B. Locate / addressing failures
8. **NEMAR S3 closed.** T2 has no byte source for NEMAR-only datasets. **Mitigation:**
   T1 (size-only) still works for EDF/BDF/SET-external; for T2 formats, try the
   OpenNeuro mirror if the dataset is mirrored, else **T3 (documented limitation)**.
9. **OpenNeuro object addressing ambiguity.** The S3 object may be keyed by git-path
   *or* by annex content hash; a guessed URL can 404. **Mitigation:** reuse the proven
   `build_s3_url`/`extract_openneuro_info` path (already fetches sidecars); on 404 try
   the alternate addressing once, then T3.
10. **Renamed/moved/withdrawn datasets** → 403/404. **Mitigation:** catch, log, T3.
11. **Annex key without size** (bare `SHA256E--<hash>` pointer, or a non-annex LFS
    pointer). T1 has no free size. **Mitigation:** one `HEAD` for `Content-Length`
    (still 0 body bytes); if that fails, T2/T3.

### C. Protocol / transport failures
12. **Range ignored → HTTP 200 full body.** Some proxies/endpoints stream the whole
    file. **Mitigation:** assert `206` + `Content-Range`; on `200`, abort before
    reading the body (close the stream) → T3.
13. **Server caps Range / returns 416** (Requested Range Not Satisfiable) for tiny
    files. **Mitigation:** clamp the range to `Content-Length`; for files smaller than
    the budget, fetch whole (it's already within budget).
14. **Latency × volume.** Even few-byte reads are round-trips; ~17k of them is slow
    (the same wall-clock lesson as the sidecar fetches). **Mitigation:** T1 (0 network)
    handles the bulk; T2 only for the residual; pool + parallelize + per-dataset batch;
    cap concurrency to avoid throttling.
15. **S3 throttling / 503 SlowDown** under parallelism. **Mitigation:** bounded
    concurrency + ret/backoff (the existing `_http` retry); degrade to T3 on repeated
    503 rather than stalling.
16. **HDF5/​MAT readers issuing many small reads** over Range (h5py B-tree walks).
    **Mitigation:** `RangeReader` block cache (e.g. 64-KB blocks) so repeated seeks hit
    cache; budget still caps total.

### D. Structural / library failures
17. **FIF directory at end-of-file + split FIF.** Needs head (dir pointer) + tail (dir)
    ranges; split files reference siblings. **Mitigation:** two ranges; on split or
    missing continuation → T3 (FIF is already mostly sidecar-served).
18. **MAT field order not guaranteed / `pnts` beyond the prefix.** Embedded `.set`
    where `pnts` sits after a large early field. **Mitigation:** field-skip reader
    seeks past big fields using each field's tag size (no full read); if it would
    exceed budget → T3.
19. **Truncated / corrupt header** (partial upload). **Mitigation:** every reader
    returns `None` on parse failure (the `FormatParser` contract) → T3; never raises.
20. **Broken annex pointer with no derivable URL and no size.** **Mitigation:** T3.

### E. Scope gaps (out of this design, tracked separately)
21. **`.gz` / `.apr` / `.apx` (FIF.gz, Curry, ANT)** — no parser at all. **Out of
    scope here**; Phase-3 new-format work. Listed so coverage isn't silently truncated.

**Discipline:** in every ambiguous case the resolver returns **None (T3)**, never a
guessed value — a wrong `n_times` is worse than a missing one. Each T3 is counted in
the coverage report by reason, so the residual gap is always visible, never hidden.

## 7. Acceptance criteria
1. EDF/BDF/SET-external `n_times` resolve at **T1 (0 bytes)** wherever nchans is known
   and sampling is uniform — verified equal to MNE on a sample corpus.
2. SNIRF/MEF3/FIF/embedded-`.set` resolve at **T2** within the per-file byte budget;
   `bytes_fetched` telemetry proves median 0 B and p99 < budget.
3. No code path streams a full signal file; `200`-on-large aborts.
4. Every unresolved record is attributed to a named failure-catalog reason in the
   coverage report.
5. Remote reads are **opt-in**; the local/shallow path is byte-for-byte unchanged when
   disabled (golden masters green).

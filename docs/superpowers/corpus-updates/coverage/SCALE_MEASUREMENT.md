# Scale measurement — full OpenNeuro + NEMAR, new resolver

A real, end-to-end run of stages 1–4 over the **entire** catalogue with the new
cheap-exact resolver. Visual report: `coverage_report.html`. Raw data:
`coverage_fullscale.json` / `.txt`; catalogue: `catalog_openneuro_nemar.json`.

## What was run

1. **Enumerate (stage 1)** — OpenNeuro GraphQL (eeg/ieeg/meg/nirs) + NEMAR
   catalogue → **696 unique datasets** (OpenNeuro 571: eeg 421, ieeg 71, meg 59,
   nirs 26; NEMAR 125).
2. **Clone (stage 2)** — `git clone --depth 1` + `GIT_LFS_SKIP_SMUDGE=1`, 16
   workers → **694 cloned** (~2.5 min; no signal bytes).
3. **Digest (stage 3)** — cheap resolver, 12 workers → **685 datasets,
   172,432 records** (~22 min; bottleneck = S3 fetches of git-annex'd *sidecar*
   text, not signal — 34k+ small HTTPS GETs).
4. **Aggregate (stage 4)** — `coverage_report.py`.

## Full-corpus coverage (172,432 records)

| Field | Coverage | Resolved |
|-------|----------|----------|
| nchans | 97.31 % | 167,798 |
| sampling_frequency | 96.44 % | 166,302 |
| ch_names | 90.77 % | 156,510 |
| **duration_seconds** | **89.73 %** | 154,718 *(was 0 % — new field)* |
| **ntimes** | **88.85 %** | 153,204 |

### Where `ntimes` came from — the resolver fan-out

| Source | Records | Share |
|--------|---------|-------|
| `sidecar_arithmetic` (round(sfreq × RecordingDuration)) | 137,857 | 79.95 % |
| `binary_parser` (**.eeg size from the git-annex key, 0 bytes fetched**) | 15,347 | 8.90 % |
| unresolved | 19,228 | 11.15 % |

`duration_seconds`: 139,371 sidecar + 15,347 derived (= the exact-ntimes VHDR
records, consistency rule) + 17,714 unresolved.

## The headline

- **`duration_seconds` 0 % → 89.7 %** corpus-wide.
- **15,347 BrainVision `n_times` recovered with zero signal I/O** — from the
  `MD5E-s<size>` annex key. The old pipeline would have fetched each `.eeg` from
  S3 and run `read_raw_brainvision`; the new path fetches nothing and is byte-exact
  (MNE-certified), ~14× faster per file.

## Honest gaps (and why)

The shallow clone is the limiting factor for **binary-header** formats: when the
recording's binary is a git-annex pointer (not present locally), the header-struct
parsers can't read it. Only VHDR sidesteps this because it needs the companion's
*size*, not its contents.

| Format | n_times | Why |
|--------|---------|-----|
| `.vhdr` | 89.4 % | **15,347 via annex-key file-size** + sidecar |
| `.set` / `.edf` / `.bdf` / `.ds` | 76–97 % | sidecar arithmetic (binary annex'd) |
| `.snirf` | 48.9 % | annex'd HDF5 — header not local |
| `.mefd` | 4.1 % | annex'd `.tmet` — header not local |
| `.gz` / `.apr` / `.apx` | 0 % | no parser yet (FIF.gz, Curry/ANT) |

By modality: eeg 91.7 %, meg 83.1 %, ieeg 79.5 %, emg 76.9 % (duration 100 %),
fnirs 48.9 %.

## Closing the gap: the remote-header tier (shipped, opt-in) — full-corpus result

`EEGDASH_REMOTE_HEADERS=1` adds a ranged-header tier that resolves the annex-pointer
formats by fetching **only the header**, never the signal. Re-digested over the
**whole corpus** with `--workers 12 --n-jobs 6` (683 datasets, **171,644 records**,
~28 min, timeout-free). Raw: `coverage_after_remote_njobs.json`.

| Field | no-flag baseline → remote + n_jobs |
|-------|------------------------------------|
| **ntimes** | 88.85 % → **93.30 %** |
| **duration_seconds** | 89.73 % → **94.14 %** |

`ntimes` by source after: sidecar_arithmetic 87,675 (51.1 %) · **remote_header 44,109
(25.7 %)** · binary_parser 15,348 (8.9 %, VHDR) · size_arithmetic 13,015 (7.6 %, SET
`.fdt`) · missing 11,497 (6.7 %).

| Format | records | ntimes before → after | bytes / file |
|--------|---------|-----------------------|--------------|
| `.edf` | 45,934 | 85.5 → **94.3 %** | **256 B** |
| `.bdf` | 11,263 | 76.6 → **84.6 %** | **256 B** |
| `.mefd` | 640 | 4.1 → **100 %** | **16 KB** (`.tmet`) |
| `.snirf` | 1,998 | 48.9 → **98.2 %** | **~196 KB** (0.24 % of file) |
| `.set` | 80,922 | 96.8 → **97.9 %** | **0 B** (`.fdt` annex key) |
| `.vhdr` | 18,971 | 89.4 % (already) | **0 B** (`.eeg` annex key) |

All byte-exact, provenance-stamped, **zero signal bytes**. The `--n-jobs` thread pool
over per-record fetches ran it ~3× faster per dataset, recovering the records a serial
run dropped to timeouts. Design + 21-mode failure catalog:
`docs/superpowers/specs/2026-05-31-efficient-remote-header-reader-design.md`.

## Still open
- Wire `EEGDASH_REMOTE_HEADERS=1` into the full corpus re-digest to quantify the
  corpus-wide lift (projected `ntimes` 88.9 % → high-90s).
- CTF `.ds` / FIF header readers over Range; exotic `.gz` / `.apr` / `.apx` parsers.
- NEMAR-only datasets: S3 is closed, so the remote tier returns no URL (size-only T1
  still applies). Tracked in `docs/superpowers/plans/2026-05-31-cheap-metadata-resolver-phase3-remaining.md`
  and `…/2026-05-31-efficient-remote-header-reader-plan.md`.

*(The earlier `coverage_before.json` — 289 stale local datasets — is kept only as a
historical pre-resolver snapshot; this full run supersedes it.)*

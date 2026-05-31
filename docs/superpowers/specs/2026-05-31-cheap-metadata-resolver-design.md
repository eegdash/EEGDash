# Cheap-Exact Technical-Metadata Resolution at Scale — Design

**Status:** Approved (brainstorming → design)
**Date:** 2026-05-31
**Author:** Ingestion pipeline working session
**Related:** `scripts/ingestions/CONTEXT.md`, ADR 0001 (evolutionary typed pipeline), ADR 0002 (flat layout)

---

## 1. Problem

At ingestion, EEGDash must resolve the technical metadata of every recording —
`sampling_frequency`, `nchans`, `ntimes`, `ch_names`, and (new) `duration_seconds` —
for **all modalities** (EEG / MEG / iEEG / NIRS) and **all file formats**. BIDS
sidecars (`*_eeg.json`, `channels.tsv`, `*_scans.tsv`) carry most of this cheaply,
but they are frequently incomplete. When a field is missing, today's pipeline falls
through to **reading the raw signal** — `mne.io.read_raw_brainvision`, `read_raw_fif`,
`read_raw_edf`, or `scipy.io.loadmat` of an embedded `.set` — which is slow and, at
NEMAR / OpenNeuro scale (git-annex pointers), triggers **S3 fetches of large signal
files** just to learn how many samples a recording has.

The single worst offender is **`ntimes`**: it is the field most often absent from
sidecars, and the only cheap source in the current cascade (`MneBidsStep`) computes it
as `int(SamplingFrequency × RecordingDuration)` — but only when `mne_bids` recognizes
the file *and* `RecordingDuration` is present. Otherwise `ntimes` falls to a full Raw
construction. Records routinely ship `ntimes=None`.

## 2. Goal

Resolve **every** technical-metadata field using the **cheapest sufficient source**,
**never reading signal bytes in production**, while staying **≤ current pipeline cost**.
Concretely:

1. Eliminate every full-Raw / full-`loadmat` read from the hot path; they become a
   genuine last resort, reached only when no cheaper tier can produce the field.
2. Make `ntimes` (and derived `duration_seconds`) resolvable for **every format**
   either from sidecar arithmetic, a pure-bytes header read, or **file-size
   arithmetic on the git-annex key — fetching zero data bytes**.
3. Extend coverage to **all** BIDS electrophysiology formats, including those with no
   parser today (CTF `.ds`, KIT `.con`/`.sqd`, NWB `.nwb`, BTi/4D `.pdf`) and those
   silently dropped at enumeration (`.cnt`, `.cdt`, `.mff`, `.bin`, `.lay`).
4. Measure, at NEMAR + OpenNeuro scale across **all datasets**, exactly which fields
   are missing and which source produced each — a per-field coverage report — with no
   extra reads.
5. Prove no performance regression with a committed, baseline-relative benchmark.

## 3. Non-goals

- **Remote HTTP-Range header reads.** Out of scope by decision. We rely on the shallow
  clone (sidecars present) + git-annex key sizes + locally-present small files. NEMAR's
  `s3://nemar` GetObject is closed by design, which reinforces this.
- Decoding or validating signal content (filtering, artifacts, quality scores).
- Changing the 5-stage pipeline topology or the inject decision logic.
- Manifest-only sources (Zenodo / Figshare / OSF) getting remote header enrichment —
  they receive sidecar-arithmetic only, documented as a known limitation.

## 4. Locked decisions

| # | Decision | Choice | Consequence |
|---|----------|--------|-------------|
| D1 | **Accuracy contract** | Cheap-exact, MNE-validated once | Dims computed by header-struct / file-size arithmetic are byte-exact by construction; a **dev-only** harness validates every formula equals `mne.io.read_raw_*().n_times` within documented tolerances (EDF unclean-stop, FIF DATA_SKIP). Production never reads signal. |
| D2 | **Format scope** | Full coverage + new header parsers | Fix enumeration blind spots and add cheap header/sidecar readers for CTF / KIT / NWB / BTi. |
| D3 | **Scale read** | Shallow clone + git-annex key size | Zero signal bytes. `parse_annex_size` (the `MD5E-s<size>--<hash>` key) gives exact file size with no fetch; small files present in the clone get header-struct reads. No remote-range tier. |
| D4 | **Output / rollout** | Persist + provenance + `duration_seconds`; full re-digest | New persisted field `duration_seconds`; per-field provenance stamped on every record; rollout is a full corpus re-digest (golden masters protect byte stability). |

### Resolutions of the smaller open questions (defaults; adjustable)

- **R1 — First-writer wins.** The cheapest tier that produces a field owns it. When a
  later (more authoritative) tier produces a *different* value beyond tolerance, the
  first value is **kept** but a `metadata_discrepancy` telemetry warning is emitted
  (field, tiers, both values). This preserves byte-stability while surfacing drift.
- **R2 — Gating.** `nchans` and `sampling_frequency` remain the only data-quality
  gated fields (`_validate.py` `DATA_QUALITY_FIELDS`). `ntimes` / `ch_names` /
  `duration_seconds` stay **ungated** (some exotic formats legitimately cannot produce
  them cheaply) but are tracked by the coverage metric so gaps are visible.
- **R3 — Byte budget.** Header-struct reads are bounded to the header region only.
  EDF/BDF use a two-step read (256 B to learn `nchans`, then `(nchans+1)×256` for the
  signal-header). The one current full-read violator (`scipy.io.loadmat` on embedded
  `.set`) is fixed via `variable_names=['EEG']` / h5py.
- **R4 — Manifest-only.** Zenodo / Figshare / OSF records get sidecar-arithmetic only.

## 5. Architecture

### 5.1 The tiered `MetadataResolver`

Generalizes today's `MetadataCascade` (`scripts/ingestions/_metadata_cascade.py`). Same
public seam — `run(ctx) -> CascadeResult` with first-writer-wins semantics, per-field
provenance, and a short-circuit when all target fields are populated — but the steps are
re-expressed as **cost tiers** through which *every* format flows, cheapest first:

```
Tier 1  SIDECAR + ARITHMETIC      (no file read beyond text sidecars)
        - SamplingFrequency, *ChannelCount sums, channels.tsv rows/columns
        - ntimes = round(SamplingFrequency × RecordingDuration)   ← promoted into
          the JSON and channels.tsv steps (today only MneBidsStep does this)
        - duration_seconds from RecordingDuration directly when present
Tier 2  HEADER STRUCT             (pure-bytes header read; NO MNE Raw object)
        - EDF/BDF header struct · VHDR text (+ BinaryFormat / DataPoints / DataOrientation)
        - SET EEG.pnts via variable_names=['EEG'] · SNIRF dataset .shape
        - MEF3 .tmet number_of_samples · FIF tag-directory walk
        - NEW: CTF res4 · KIT param block · NWB hdf5 attrs
Tier 3  FILE-SIZE ARITHMETIC      (zero data bytes)
        - multiplexed binary: ntimes = data_bytes / (nchans × dtype_bytes)
        - data_bytes from parse_annex_size(key) | os.stat — covers VHDR/EDF/BDF remotely
Tier 4  MNE FALLBACK              (LAST RESORT)
        - read_raw_*(preload=False) only when Tiers 1-3 cannot produce a field
        - preserves today's coverage for genuinely unparseable files; also FIF split detection
```

`duration_seconds = ntimes / sampling_frequency` is derived in a pure-arithmetic step
after the tiers run (or taken from sidecar `RecordingDuration` when that is the source),
and is provenance-stamped.

### 5.2 Registry as single source of truth

A format becomes **one `formats/<fmt>.py` module + one registry entry** mapping
`ext -> (parser, capability bitset)`. The capability bitset declares which fields a
parser's header can yield (so the resolver knows whether Tier 2 can satisfy a missing
field before opening the file). The two special cases currently hard-coded in the
cascade — FIF metadata and VHDR-`ntimes` — are folded into `formats/fif.py` and
`formats/vhdr.py` + Tier 3. This is done **under golden-master protection** (ADR 0001):
freeze current digest byte-output first, refactor, prove identical output.

### 5.3 Module decomposition

Flat layout per ADR 0002 (or a subpackage with lazy `__init__` re-export shims). Target
files, each one responsibility, ≤ ~400 lines:

```
_resolver.py           MetadataResolver.run(ctx) -> CascadeResult  (replaces _metadata_cascade engine)
_resolver_tiers.py     Tier 1-4 step classes (SidecarTier, HeaderStructTier, FileSizeTier, MneFallbackTier)
_resolver_derive.py    pure arithmetic: ntimes<->duration, sfreq×duration, tolerance compare
_sizing.py             parse_annex_size / os.stat / size abstraction (one place)
_format_parser_registry.py   ext -> (parser, capabilities) — extended, single source of truth
formats/edf.py bdf via edf · set.py · vhdr.py · snirf.py · mef3.py · fif.py
formats/ctf.py kit.py nwb.py bti.py   (NEW header readers)
_coverage.py           aggregate metadata_provenance -> per-field coverage report
```

Existing `_set_parser.py` / `_vhdr_parser.py` / `_snirf_parser.py` / `_mef3_parser.py`
are refactored in place (or moved under `formats/` with back-compat re-exports) so their
existing tests keep importing the same symbols.

## 6. Per-format cheap-path matrix (drives parser work)

| Format | sfreq | nchans | ch_names | ntimes — cheapest exact path | Today's cost | Action |
|--------|-------|--------|----------|------------------------------|--------------|--------|
| `.vhdr` | `SamplingInterval`→Hz (text) | `NumberOfChannels` (text) | `[Channel Infos]` | **file-size**: `getsize(.eeg or annex key) // (nchans × dtype_bytes)`, dtype from `BinaryFormat`; or `[Common Infos] DataPoints` | full `read_raw_brainvision` | parse `BinaryFormat`/`DataPoints`/`DataOrientation`; add Tier-3 path |
| `.edf`/`.bdf` | header (256 B + 256 B/ch) | offset 252 (4 B ASCII) | 16 B label/ch | **header struct**: `n_records(@236) × samples_per_record`; or `(filesize − (nchans+1)×256) // record_bytes × samples_per_record` | full `read_raw_edf` Raw object | replace MNE with struct read |
| `.set` | `EEG.srate` | `EEG.nbchan` | `EEG.chanlocs.labels` | **scalar `EEG.pnts`** | `loadmat` of WHOLE file (embedded data = full read) | `variable_names=['EEG']` / h5py — never materialize signal |
| `.snirf` | `1/(t1−t0)` or MNE | measurementList groups | S#-D# labels | **HDF5 `dataTimeSeries.shape[0]`** | reads full time vector `[:]` | use `.shape`, not `[:]` |
| `.mefd` | `.tmet` scan | count `.timd` dirs | dir stem names | **`.tmet number_of_samples` (uint64)** | never extracted | add `number_of_samples` read |
| `.fif` | MEAS_INFO | MEAS_INFO | MEAS_INFO | **tag-directory walk** (payload-free) | `read_raw_fif(preload=False)` (already payload-free) | minor: skip annotation parse; handle `.fif.gz` |
| CTF `.ds` | res4 header | res4 | res4 | res4 `no_samples × no_trials`; else sidecar arithmetic | montage-only read | NEW `formats/ctf.py` |
| KIT `.con`/`.sqd` | param block | param block | param block | param block sample count; else sidecar arithmetic | montage-only read | NEW `formats/kit.py` |
| NWB `.nwb` | hdf5 attrs | hdf5 attrs | hdf5 attrs | `ElectricalSeries.data.shape[0]` (hdf5, no read) | none | NEW `formats/nwb.py` |
| BTi `.pdf` | header | header | header | header sample count; else sidecar | rejected as "PDF document" | NEW `formats/bti.py` + fix misclassification |
| `.cnt`/`.cdt`/`.mff`/`.bin`/`.lay` | varies | varies | varies | format-specific header or sidecar arithmetic | **never enumerated** | extend `ALLOWED_DATATYPE_EXTENSIONS` + parsers |

**Universal cheap fallback (all formats):** when the sidecar carries both
`SamplingFrequency` and `RecordingDuration`, `ntimes = round(sfreq × duration)`. This is
Tier 1 and covers every format including those with no header parser.

## 7. Scale measurement

"Check what is missing across ALL NEMAR + OpenNeuro datasets":

1. **Enumerate** via existing APIs — zero file reads. OpenNeuro GraphQL
   `latestSnapshot.summary` (modalities eeg/ieeg/meg/nirs); NEMAR
   `data.nemar.org/?format=json` (~291 datasets). No hard-coded counts.
2. **Shallow clone** each: `git clone --depth 1` + `GIT_LFS_SKIP_SMUDGE=1`
   (already how `2_clone.py` works) — text sidecars + annex pointers only, no signal.
3. **Digest** and aggregate the existing `metadata_provenance` telemetry
   (`record_built` event carries it) into a **per-field coverage report**
   (`_coverage.py`): for each field, per source / format / dataset, the fraction
   resolved and where it came from. This is a free, CI-gatable measurement of where the
   cheap path still fails — no extra reads.

Output artifact: a coverage JSON + human summary (counts of records with
`ntimes` from sidecar vs header vs filesize vs mne-fallback vs null, broken down by
format and modality and source).

## 8. Performance guardrails

- **Committed baseline harness.** `.benchmarks/` is empty today; add a committed
  baseline (pinned JSON) so "no regression" is **baseline-relative**, not just absolute
  ceilings. Benchmark `MetadataResolver.run` per format × {sidecar-present,
  sidecar-absent}.
- **Representative corpus.** Synthetic/fixture files per format
  (EDF / BDF / SET-embedded / SET-external / VHDR / SNIRF / FIF / MEF3 / CTF / KIT)
  with sidecar-present and sidecar-absent variants.
- **Header-only enforcement test.** Assert no parser materializes full signal (catches
  regressions reintroducing `loadmat`/`[:]`). Implemented by monkeypatching the read
  primitives and asserting bounded byte counts / no full-array allocation.
- Extend `tests/perf/` and `.github/workflows/ingestions-bench.yml`.

## 9. Schema & data model

- Add `duration_seconds: float | None` to the Record contract (`eegdash/schemas.py`
  `RecordModel` / `create_record`) — provenance-stamped, derived `ntimes / sfreq` or
  from sidecar `RecordingDuration`.
- Per-field provenance is already persisted (`_metadata_provenance`); extend its field
  set to include `duration_seconds` and the new tier source names
  (`sidecar_arithmetic`, `header_struct`, `filesize_arithmetic`, `mne_fallback`).
- Keep existing provenance source-name constants for byte-stability where the same
  source still applies; snapshot tests are updated deliberately when output changes.

## 10. Rollout

Full re-digest of the corpus (D4). Sequenced:
1. Land the resolver + cheap parsers behind golden-master tests (no output change yet
   except where a previously-`None` field is now filled — those snapshot deltas are
   reviewed and re-frozen deliberately).
2. Run the scale measurement to quantify the before/after coverage delta.
3. Re-digest and re-inject per the existing stage-5 plan logic (create/update/skip);
   updates flow naturally because newly-filled fields change the record fingerprint.

## 11. Testing strategy

- **Golden masters first** (ADR 0001): freeze `tests/digest/test_snapshot.py`,
  `tests/inject/test_inject_plan_golden.py`, `tests/unit/test_validate.py` before the
  refactor; review and re-freeze deliberately when cheap tiers fill new fields.
- **TDD per tier and per parser**: each cheap path gets a failing test (fixture with a
  known dim) → implementation → pass.
- **MNE-equivalence validation** (D1): dev-only/`slow`+`network`-marked tests that fetch
  one real file per format and assert tier output == `read_raw_*().n_times` within
  tolerance. Skipped in normal CI; run during development to certify formulas.
- **Property tests** (Hypothesis, already used) for the arithmetic
  (`ntimes ↔ duration` round-trips, file-size formula edge cases: zero-size, ragged
  records, odd byte counts).
- **Coverage/measurement** unit tests on synthetic provenance aggregates.
- **Idempotency / fingerprint** stability (`tests/acceptance/`).

## 12. Phasing (each phase = working, testable software)

| Phase | Deliverable | Gate |
|-------|-------------|------|
| **P1 Resolver core** | Tier refactor of the cascade; promote `ntimes = sfreq×duration` into sidecar tiers; add `duration_seconds` + provenance; fold FIF/VHDR special cases under the registry | golden masters green (deltas reviewed); cascade tests pass |
| **P2 Cheap parsers** | Kill all full reads for the 7 existing formats (VHDR file-size, EDF struct, SET `variable_names`, SNIRF `.shape`, MEF3 `number_of_samples`); MNE-equivalence validation harness | header-only enforcement test passes; equivalence harness certifies formulas |
| **P3 New formats** | `formats/ctf.py`, `kit.py`, `nwb.py`, `bti.py`; enumeration fix for `.cnt/.cdt/.mff/.bin/.lay`; `.fif.gz` handling; `.mef`/`.mefd` consistency | new-format parser tests pass; enumeration tests pass |
| **P4 Scale measurement** | `_coverage.py` + a measurement entry point that runs stages 1-3 over all NEMAR/OpenNeuro datasets and emits the coverage report | coverage report generated on a sample; aggregation unit-tested |
| **P5 Perf guardrails + rollout** | Committed benchmark baseline; CI wiring; full re-digest runbook | benchmark shows ≤ baseline cost; bench CI green |

## 13. Risks & mitigations

- **EDF samples-per-record is per-channel and may be ragged.** File-size arithmetic
  must use the per-channel signal-header sums, not a flat divide. Mitigation: header
  struct read (Tier 2) is authoritative for EDF; Tier-3 flat divide only when the
  header is unreachable and all channels share `samples_per_record` (validated by D1
  harness; emit discrepancy otherwise).
- **MNE unclean-stop / FIF DATA_SKIP** make naive arithmetic off by a record. Mitigation:
  D1 harness encodes the documented tolerances; discrepancies beyond tolerance are
  logged, not silently trusted.
- **Embedded `.set` with no `.fdt`** — `variable_names=['EEG']` still reads `EEG.data`
  field metadata; ensure scipy does not materialize the array (use `mat_dtype` /
  struct-only access, or h5py for v7.3 `.set`). Header-only test guards this.
- **Snapshot churn** from newly-filled fields. Mitigation: deliberate, reviewed
  re-freeze; each delta tied to a provenance source so it is auditable.
- **Refactor blast radius** across `_metadata_cascade.py`, `_format_parser_registry.py`,
  `_montage.py`, `_record_extractor.py`. Mitigation: golden masters + phase ordering;
  back-compat re-exports for moved symbols.

## 14. Acceptance criteria

1. No production code path constructs a full MNE Raw or `loadmat`s signal to obtain
   `ntimes`/`sfreq`/`nchans`/`ch_names` when a cheaper tier can produce it.
2. `ntimes` and `duration_seconds` resolvable for every supported format via sidecar
   arithmetic, header struct, or git-annex file-size — verified by the equivalence
   harness within tolerance.
3. CTF / KIT / NWB / BTi and `.cnt/.cdt/.mff/.bin/.lay` are enumerated and yield
   metadata (header or sidecar-arithmetic).
4. A per-field coverage report exists for all NEMAR + OpenNeuro datasets.
5. Benchmarks show `MetadataResolver.run` cost ≤ the committed baseline for every
   format.
6. Golden-master, idempotency, and full non-network test suite green.

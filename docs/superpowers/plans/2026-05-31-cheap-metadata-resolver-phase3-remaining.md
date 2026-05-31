# Cheap Metadata Resolver — Phase 3 Remaining (new formats & enumeration)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]`.

**Status:** MEF3 `n_times` (the worst-coverage format) is DONE (`_mef3_parser.py`,
commit `693905d99`). This doc covers the rest of Phase 3: enumerating formats that
are currently never listed, and header parsers for CTF/KIT/NWB/BTi.

**Goal:** Every BIDS electrophysiology format is (a) enumerated and (b) yields cheap
`n_times`/`sfreq`/`nchans` — header-exact where a reader exists, else sidecar
arithmetic (already universal from Phase 1).

**Blocker — fixtures:** the pinned `eegdash-testing-data` corpus has **no** CTF `.ds`,
KIT `.con/.sqd`, BTi `.pdf`, full-FIF, or `.cnt/.cdt/.mff/.lay` fixtures. New-format
parser tests need either synthetic builders (feasible for NWB hdf5) or new corpus
fixtures (CTF/KIT/BTi). Add fixtures to the corpus and bump `eegdash/testing.py`
VERSION/SHA, OR mark the tests `@network @slow` and skip-if-absent.

---

## Task 3R.1: Enumerate `.cnt/.cdt/.mff/.lay` on the BIDS-filesystem path

**Problem (✓ VERIFIED in recon):** `eegdash/dataset/bids_dataset.py:_init_bids_paths`
(lines ~208-221) iterates `ALLOWED_DATATYPE_EXTENSIONS` from mne_bids only, which lacks
`.cnt/.cdt/.mff/.bin/.lay`. A `.cnt`-only dataset yields **zero** files on the BIDS path.

**Files:** `eegdash/dataset/bids_dataset.py`; `scripts/ingestions/_constants.py` (add the
extra-extensions set); test in `tests/unit_tests/dataset/`.

- [ ] **Step 1: Failing test** — build a tmp BIDS tree with `sub-01/eeg/sub-01_task-x_eeg.cnt`,
  assert `EEGBIDSDataset(...).files` contains it. (Today it does not.)
- [ ] **Step 2: Add the extra-extension set** to `_constants.py`:
```python
# Neuro data formats not in mne_bids' ALLOWED_DATATYPE_EXTENSIONS but present in real
# OpenNeuro/NEMAR datasets. Enumerated in addition to the mne_bids table.
EXTRA_EEG_DATA_EXTENSIONS: tuple[str, ...] = (".cnt", ".cdt", ".mff", ".lay")
```
  (Deliberately NOT `.bin` — too generic, would misclassify non-data files.)
- [ ] **Step 3: Union them in `_init_bids_paths`** — after iterating the mne_bids
  extensions per modality, also glob the extra extensions under the modality dir
  (guard with the same `allow_symlinks`/dedup logic). Keep mne_bids enumeration first.
- [ ] **Step 4:** run the new test + the full `eegdash` dataset test suite; ensure no
  dataset that previously enumerated N files now enumerates a different N.
- [ ] **Step 5: Commit.**

**Note:** `is_neuro_data_file` (manifest path, `_bids_path.py`) already accepts these via
the `/eeg/` dir match — only the mne_bids BIDS-filesystem path needs the fix.

## Task 3R.2: BTi `.pdf` — do NOT blanket-declassify

**Decision:** `_bids_path.py:181-189` lists `.pdf` as a sidecar extension. Real BIDS
datasets contain genuine PDF documents (stimuli, docs). Removing `.pdf` blanket would
misclassify them. BTi/4D "pdf" data files are typically **extensionless** (`c,rfDC`),
not literally `.pdf`. **Recommendation:** leave `.pdf` as a sidecar; handle BTi only if a
real BTi dataset surfaces, keyed on the extensionless data-file convention + a
`coordsystem`/`*_meg.json` sidecar. Document as a known limitation. (No code change.)

## Task 3R.3: CTF `.ds` / KIT `.con,.sqd` metadata parser (reuse header-only MNE read)

`_montage.extract_meg_layout` already does `read_raw_ctf(preload=False)` /
`read_raw_kit(preload=False)` — header-only, signal-free. Wrap the same call to also
return `(sfreq, nchans, ch_names, n_times)`.

**Files:** new `scripts/ingestions/formats/ctf_kit.py` (or extend `_format_parser_registry`);
register `.ds`/`.con`/`.sqd`. Fixture: needs a real CTF/KIT recording in the corpus.

- [ ] **Step 1: Failing test** (`@network @slow`, skip if fixture absent): assert the
  parser returns `n_times == read_raw_ctf(...).n_times` for a CTF `.ds` fixture.
- [ ] **Step 2: Implement** a `parse_ctf_metadata(path)` / `parse_kit_metadata(path)` that
  calls `mne.io.read_raw_ctf/kit(preload=False)`, returns the 4 fields, **never raises**
  (return None on failure), and is safe on broken annex symlinks.
- [ ] **Step 3: Register** `.ds`/`.con`/`.sqd` in the format registry so the cascade
  `BinaryParserStep` uses them. Note: `.ds` is a directory — the registry/`get_parser_for_extension`
  must handle directory "extensions" (the cascade passes `ctx.ext` = suffix).
- [ ] **Step 4:** equivalence-validate against MNE; commit.

**Caveat:** these read via MNE (not "few bytes"), but header-only (no signal) and only
when the sidecar didn't already supply the value (Tier 1 still wins). For shallow clones
where the `.ds`/`.con` is an annex pointer, MNE can't read it → falls back to sidecar
arithmetic (Phase 1). So this tier only fires when the file is locally present.

## Task 3R.4: NWB `.nwb` hdf5 metadata parser (synthetic-fixture testable)

NWB is hdf5 — `ElectricalSeries.data.shape[0]` gives `n_times` without reading data;
`rate`/`starting_time` + `electrodes` give sfreq/nchans. Buildable synthetically with
h5py (no corpus fixture needed).

**Files:** new `scripts/ingestions/formats/nwb.py`; register `.nwb`; test with a synthetic
h5py NWB stub.

- [ ] **Step 1: Failing test** — build a minimal h5py file with
  `/acquisition/ts/data` (shape (1000, 8)), `/acquisition/ts/starting_time` attr `rate=200`;
  assert parser returns `n_times=1000, sampling_frequency=200, nchans=8`.
- [ ] **Step 2: Implement** `parse_nwb_metadata(path)` reading the hdf5 shape/attrs via h5py
  (use `pynwb` only if available; prefer raw h5py to stay dependency-light and signal-free),
  never raising.
- [ ] **Step 3: Register** `.nwb`; equivalence-check against `pynwb`/MNE if available.
- [ ] **Step 4: Commit.**

## Task 3R.5: Re-measure coverage

- [ ] After 3R.1/3R.3/3R.4 land, re-run `coverage_report.py` over a re-digest and confirm
  `.mefd`, `.ds`, `.nwb`, and the newly-enumerated formats moved up. Update
  `docs/superpowers/corpus-updates/coverage/`.

## Self-review checklist
- New parsers conform to the `FormatParser` Protocol (Path → Result|None, **never raise**,
  safe on broken annex symlinks).
- `.ds` directory handling in the registry (suffix `.ds`, but the path is a dir).
- No regression in `EEGBIDSDataset.files` counts for existing datasets (Task 3R.1).
- Each new format added in ONE place (registry entry + `formats/*.py`), per the
  registry-as-single-source-of-truth design.

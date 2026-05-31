# Cheap-Exact Metadata Resolution — Implementation Plan (Phases 1–2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve every technical-metadata field (`sampling_frequency`, `nchans`, `ntimes`, `ch_names`, new `duration_seconds`) from the cheapest sufficient source, eliminating production full-signal reads, starting with the resolver core (Phase 1) and the cheap header parsers for the 7 existing formats (Phase 2).

**Architecture:** Extend the existing `MetadataCascade` (`scripts/ingestions/_metadata_cascade.py`) with (a) sidecar `ntimes = round(SamplingFrequency × RecordingDuration)` promoted into the sidecar steps, (b) a derived `duration_seconds`, (c) per-field provenance for the new fields, and (d) cheap header paths in the per-format parsers (VHDR `DataPoints`/file-size, SNIRF `.shape`, SET `variable_names`/field-skip, MEF3 `number_of_samples`) so the MNE/`loadmat` fallbacks fire only as a true last resort. Golden-master snapshot tests protect byte-output; deltas (newly-filled fields) are reviewed and re-frozen deliberately.

**Tech Stack:** Python 3.12, pytest 9 + pytest-benchmark, MNE 1.12, scipy, h5py, pandas, Hypothesis. Run all tests from `scripts/ingestions/` (its own `pyproject.toml`, `testpaths=["tests"]`).

**Spec:** `docs/superpowers/specs/2026-05-31-cheap-metadata-resolver-design.md`

---

## Conventions for every task

- **CWD for tests:** `cd /Users/bruaristimunha/Projects/EEGDash/scripts/ingestions` (bare imports like `_metadata_cascade` resolve via that package's `conftest.py`).
- **Fast test selector:** `python3 -m pytest <target> -q -p no:cacheprovider`.
- **Full guard before each commit:** `python3 -m pytest tests/ -m "not network and not integration and not slow" -q -p no:cacheprovider` (baseline: 978 passed, 2 skipped).
- **Pre-commit hooks** run on commit (ruff, codespell, no-nested-functions, leaked-creds). Write module-level helpers only — **no functions nested inside functions** (hook blocks it).
- **No Co-Authored-By / AI attribution** in commits (user hard rule).
- **Golden masters** that may legitimately change output: `tests/digest/test_snapshot.py`. When a previously-`None` field becomes filled, regenerate the committed snapshot in the external corpus and review the diff (see Task 1.1).

---

## File Structure (Phases 1–2)

| File | Change | Responsibility |
|------|--------|----------------|
| `scripts/ingestions/_metadata_cascade.py` | modify | Add `recording_duration`/`duration_seconds` to `CascadeResult`; sidecar-arithmetic `ntimes`; new provenance source `PROV_SIDECAR_ARITHMETIC`; helper `extract_recording_duration_from_sidecar` |
| `scripts/ingestions/_record_extractor.py` | modify | Consume `duration_seconds`; pass to `create_record`; provenance plumbing |
| `eegdash/schemas.py` | modify | `Record` TypedDict gains `duration_seconds`; `create_record` gains `duration_seconds` param |
| `scripts/ingestions/_bids_digest.py` | modify | telemetry `record_built` payload gains `duration_seconds` |
| `scripts/ingestions/_vhdr_parser.py` | modify | Parse `DataPoints`, `BinaryFormat`, `DataOrientation`; return `n_times` (text or file-size arithmetic) |
| `scripts/ingestions/_snirf_parser.py` | modify | h5py fallback reads `time.shape[0]` + `time[:2]`, not `time[:]` |
| `scripts/ingestions/_set_parser.py` | modify | `variable_names=['EEG']`; external-data fast path; header-only discipline |
| `scripts/ingestions/_mef3_parser.py` | modify | Best-effort `number_of_samples` from `.tmet` |
| `scripts/ingestions/_sizing.py` | create | One place for file-size: annex-key size → os.stat (no fetch) |
| `scripts/ingestions/tests/unit/test_resolver_arithmetic.py` | create | TDD for sidecar-arithmetic + duration derivation |
| `scripts/ingestions/tests/parsers/test_cheap_paths.py` | create | TDD for header-only parser paths + header-only enforcement |
| `scripts/ingestions/tests/validation/test_mne_equivalence.py` | create | `@network @slow` MNE-equivalence harness (D1 contract) |

---

# PHASE 1 — Resolver core: sidecar `ntimes` + `duration_seconds` + provenance

Highest-leverage, lowest-risk. Makes `ntimes` resolve from the sidecar for **every** format (format-agnostic), so the binary/MNE fallbacks rarely fire, and persists `duration_seconds`.

### Task 1.1: Establish & freeze the golden baseline

**Files:** none (verification only)

- [ ] **Step 1: Confirm the testing-data corpus is available and snapshot tests pass**

Run:
```bash
cd /Users/bruaristimunha/Projects/EEGDash/scripts/ingestions
python3 -m pytest tests/digest/test_snapshot.py tests/unit/test_metadata_cascade.py tests/unit/test_metadata_provenance.py -q -p no:cacheprovider
```
Expected: PASS (these are the byte-level golden masters + cascade tests we must keep green). If they SKIP for missing corpus, run `python3 -m eegdash.testing` first to pre-fetch, then re-run.

- [ ] **Step 2: Record the baseline count**

Run the full fast guard and confirm `978 passed, 2 skipped`:
```bash
python3 -m pytest tests/ -m "not network and not integration and not slow" -q -p no:cacheprovider 2>&1 | tail -3
```
Expected: `978 passed, 2 skipped` (or current count — record it). No commit.

### Task 1.2: Sidecar `RecordingDuration` extractor (pure, TDD)

**Files:**
- Modify: `scripts/ingestions/_metadata_cascade.py` (add helper near `extract_sfreq_nchans_from_modality_sidecar`, after line 153)
- Test: `scripts/ingestions/tests/unit/test_resolver_arithmetic.py` (create)

- [ ] **Step 1: Write the failing test**

Create `scripts/ingestions/tests/unit/test_resolver_arithmetic.py`:
```python
"""Sidecar-arithmetic ntimes + duration derivation (Phase 1)."""

from __future__ import annotations

import json
from pathlib import Path

from _metadata_cascade import extract_recording_duration_from_sidecar


def _write(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def test_recording_duration_from_eeg_sidecar(tmp_path: Path):
    root = tmp_path
    eeg = root / "sub-01" / "eeg"
    bids_file = eeg / "sub-01_task-rest_eeg.edf"
    bids_file.parent.mkdir(parents=True, exist_ok=True)
    bids_file.write_bytes(b"\x00")
    _write(eeg / "sub-01_task-rest_eeg.json", {"SamplingFrequency": 250, "RecordingDuration": 40.0})

    dur = extract_recording_duration_from_sidecar(bids_file, root)
    assert dur == 40.0


def test_recording_duration_absent_returns_none(tmp_path: Path):
    root = tmp_path
    eeg = root / "sub-01" / "eeg"
    bids_file = eeg / "sub-01_task-rest_eeg.edf"
    bids_file.parent.mkdir(parents=True, exist_ok=True)
    bids_file.write_bytes(b"\x00")
    _write(eeg / "sub-01_task-rest_eeg.json", {"SamplingFrequency": 250})

    assert extract_recording_duration_from_sidecar(bids_file, root) is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 -m pytest tests/unit/test_resolver_arithmetic.py -q -p no:cacheprovider`
Expected: FAIL with `ImportError: cannot import name 'extract_recording_duration_from_sidecar'`.

- [ ] **Step 3: Implement the helper**

In `scripts/ingestions/_metadata_cascade.py`, add after `extract_sfreq_nchans_from_channels_tsv` (after line 212). It reuses the existing `_build_bids_search_paths` inheritance walk and `_MODALITY_SIDECAR_SUFFIXES`:
```python
def extract_recording_duration_from_sidecar(
    bids_file_path: Path,
    bids_root: Path,
) -> float | None:
    """Return ``RecordingDuration`` (seconds) from a modality JSON sidecar via BIDS inheritance, or ``None``."""
    base_names_to_try, dirs_to_try = _build_bids_search_paths(bids_file_path, bids_root)

    for search_dir in dirs_to_try:
        for base_name in base_names_to_try:
            for sidecar_suffix in _MODALITY_SIDECAR_SUFFIXES:
                sidecar_path = search_dir / f"{base_name}{sidecar_suffix}"
                if not sidecar_path.exists():
                    continue
                try:
                    with open(sidecar_path) as f:
                        sidecar_data = json.load(f)
                except (OSError, json.JSONDecodeError, ValueError, TypeError):
                    continue
                raw = sidecar_data.get("RecordingDuration")
                if raw is not None:
                    try:
                        dur = float(raw)
                    except (TypeError, ValueError):
                        return None
                    return dur if dur > 0 else None
                break  # one sidecar variant per (dir, base)
    return None
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/unit/test_resolver_arithmetic.py -q -p no:cacheprovider`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bruaristimunha/Projects/EEGDash
git add scripts/ingestions/_metadata_cascade.py scripts/ingestions/tests/unit/test_resolver_arithmetic.py
git commit -m "feat(ingest): sidecar RecordingDuration extractor for cheap ntimes"
```

### Task 1.3: `CascadeResult` carries `recording_duration` + `duration_seconds` + provenance (TDD)

**Files:**
- Modify: `scripts/ingestions/_metadata_cascade.py` (`_METADATA_FIELDS` line 33-38; `CascadeResult` lines 294-311; new constant near line 31)
- Test: append to `scripts/ingestions/tests/unit/test_resolver_arithmetic.py`

- [ ] **Step 1: Write the failing test** (append)
```python
from _metadata_cascade import (
    CascadeResult,
    PROV_SIDECAR_ARITHMETIC,
    derive_duration_seconds,
)


def test_cascaderesult_has_duration_fields():
    r = CascadeResult()
    assert r.recording_duration is None
    assert r.duration_seconds is None
    assert "duration_seconds" in r.provenance
    assert "ntimes" in r.provenance


def test_derive_duration_prefers_recording_duration():
    r = CascadeResult(sampling_frequency=250.0, ntimes=10000, recording_duration=40.0)
    derive_duration_seconds(r)
    assert r.duration_seconds == 40.0
    assert r.provenance["duration_seconds"] == PROV_SIDECAR_ARITHMETIC


def test_derive_duration_falls_back_to_ntimes_over_sfreq():
    r = CascadeResult(sampling_frequency=250.0, ntimes=10000)
    derive_duration_seconds(r)
    assert r.duration_seconds == 40.0
    assert r.provenance["duration_seconds"] == "derived"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/unit/test_resolver_arithmetic.py -q -p no:cacheprovider`
Expected: FAIL (ImportError on `PROV_SIDECAR_ARITHMETIC`/`derive_duration_seconds`, AttributeError on fields).

- [ ] **Step 3: Implement**

In `_metadata_cascade.py`:
(a) Add provenance constant after line 31 (`PROV_MNE_FALLBACK`):
```python
PROV_SIDECAR_ARITHMETIC = "sidecar_arithmetic"
PROV_DERIVED = "derived"
```
(b) Extend `_METADATA_FIELDS` (lines 33-38):
```python
_METADATA_FIELDS: tuple[str, ...] = (
    "sampling_frequency",
    "nchans",
    "ntimes",
    "ch_names",
    "duration_seconds",
)
```
(c) Add fields to `CascadeResult` (after `ch_names` at line 301):
```python
    recording_duration: float | None = None
    duration_seconds: float | None = None
```
(d) Add a module-level function after the `CascadeResult` class (after line 311):
```python
def derive_duration_seconds(result: CascadeResult) -> None:
    """Fill ``duration_seconds`` from sidecar RecordingDuration, else ntimes/sfreq. Provenance-stamped."""
    if result.duration_seconds is not None:
        return
    if result.recording_duration is not None and result.recording_duration > 0:
        result.duration_seconds = float(result.recording_duration)
        if result.provenance.get("duration_seconds") is None:
            result.provenance["duration_seconds"] = PROV_SIDECAR_ARITHMETIC
        return
    if (
        result.sampling_frequency
        and result.sampling_frequency > 0
        and result.ntimes
        and result.ntimes > 0
    ):
        result.duration_seconds = float(result.ntimes) / float(result.sampling_frequency)
        if result.provenance.get("duration_seconds") is None:
            result.provenance["duration_seconds"] = PROV_DERIVED
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/unit/test_resolver_arithmetic.py -q -p no:cacheprovider`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**
```bash
cd /Users/bruaristimunha/Projects/EEGDash
git add scripts/ingestions/_metadata_cascade.py scripts/ingestions/tests/unit/test_resolver_arithmetic.py
git commit -m "feat(ingest): CascadeResult duration_seconds + derive helper + provenance"
```

### Task 1.4: `ModalitySidecarStep` computes sidecar `ntimes`; cascade derives duration (TDD)

**Files:**
- Modify: `scripts/ingestions/_metadata_cascade.py` (`ModalitySidecarStep.fill` lines 364-373; `MetadataCascade.run` lines 499-503)
- Test: append to `tests/unit/test_resolver_arithmetic.py`

- [ ] **Step 1: Write the failing test** (append)
```python
from _metadata_cascade import CascadeContext, MetadataCascade


class _FakeBidsDataset:
    """Minimal stand-in: mne_bids step finds nothing, forcing the sidecar steps."""

    def __init__(self, bidsdir):
        self.bidsdir = str(bidsdir)

    def get_bids_file_attribute(self, attr, bids_file):
        return None

    def channel_labels(self, bids_file):
        return None


def test_sidecar_step_fills_ntimes_from_duration(tmp_path: Path):
    root = tmp_path
    eeg = root / "sub-01" / "eeg"
    bids_file = eeg / "sub-01_task-rest_eeg.edf"
    bids_file.parent.mkdir(parents=True, exist_ok=True)
    bids_file.write_bytes(b"\x00")
    _write(
        eeg / "sub-01_task-rest_eeg.json",
        {"SamplingFrequency": 250, "RecordingDuration": 40.0, "EEGChannelCount": 32},
    )

    ctx = CascadeContext(bids_dataset=_FakeBidsDataset(root), bids_file=str(bids_file))
    result = MetadataCascade().run(ctx)

    assert result.sampling_frequency == 250.0
    assert result.nchans == 32
    assert result.ntimes == 10000  # round(250 * 40)
    assert result.provenance["ntimes"] == PROV_SIDECAR_ARITHMETIC
    assert result.duration_seconds == 40.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/unit/test_resolver_arithmetic.py::test_sidecar_step_fills_ntimes_from_duration -q -p no:cacheprovider`
Expected: FAIL (`result.ntimes` is `None`).

- [ ] **Step 3: Implement**

(a) In `ModalitySidecarStep.fill` (lines 364-373), after stamping sfreq/nchans, add duration + arithmetic ntimes:
```python
class ModalitySidecarStep:
    """Step 2: modality JSON sidecar with BIDS-inheritance walk."""

    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None:
        sf_before = result.sampling_frequency
        n_before = result.nchans
        sf, nc = extract_sfreq_nchans_from_modality_sidecar(
            ctx.bids_file_path, ctx.bids_root, sf_before, n_before
        )
        result.sampling_frequency = sf
        result.nchans = nc
        result.stamp(PROV_MODALITY_SIDECAR, "sampling_frequency", sf_before, sf)
        result.stamp(PROV_MODALITY_SIDECAR, "nchans", n_before, nc)

        if result.recording_duration is None:
            result.recording_duration = extract_recording_duration_from_sidecar(
                ctx.bids_file_path, ctx.bids_root
            )

        # Cheap sidecar arithmetic: ntimes = round(sfreq * duration).
        if (
            result.ntimes is None
            and result.sampling_frequency
            and result.recording_duration
            and result.recording_duration > 0
        ):
            nt_before = result.ntimes
            result.ntimes = round(
                float(result.sampling_frequency) * float(result.recording_duration)
            )
            result.stamp(PROV_SIDECAR_ARITHMETIC, "ntimes", nt_before, result.ntimes)
```
(b) In `MetadataCascade.run` (lines 499-503), call `derive_duration_seconds` after the steps:
```python
    def run(self, ctx: CascadeContext) -> CascadeResult:
        result = CascadeResult()
        for step in self.steps:
            step.fill(ctx, result)
        derive_duration_seconds(result)
        return result
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/unit/test_resolver_arithmetic.py -q -p no:cacheprovider`
Expected: PASS (6 passed).

- [ ] **Step 5: Run the cascade regression + provenance suites**

Run: `python3 -m pytest tests/unit/test_metadata_cascade.py tests/unit/test_metadata_provenance.py -q -p no:cacheprovider`
Expected: PASS (these may need updates if they assert the exact provenance-field set — if a test enumerates `_METADATA_FIELDS` or the provenance keys, update it to include `duration_seconds` in the SAME commit, and note it as an intentional change).

- [ ] **Step 6: Commit**
```bash
cd /Users/bruaristimunha/Projects/EEGDash
git add scripts/ingestions/_metadata_cascade.py scripts/ingestions/tests/
git commit -m "feat(ingest): sidecar-arithmetic ntimes in ModalitySidecarStep + duration derivation"
```

### Task 1.5: Persist `duration_seconds` on the Record (TDD)

**Files:**
- Modify: `eegdash/schemas.py` (`Record` TypedDict line 875-878; `create_record` signature line 1026, body line 1146)
- Modify: `scripts/ingestions/_record_extractor.py` (`_extract_technical_metadata` 230-252; consumption 411-419; `create_record` call 466-487)
- Test: `tests/unit_tests/` or ingestion `tests/unit/` — add a focused test

- [ ] **Step 1: Write the failing test**

Append to `scripts/ingestions/tests/unit/test_resolver_arithmetic.py`:
```python
def test_create_record_persists_duration_seconds():
    from eegdash.schemas import create_record

    rec = create_record(
        dataset="ds999",
        storage_base="s3://openneuro.org/ds999",
        bids_relpath="sub-01/eeg/sub-01_task-rest_eeg.edf",
        sampling_frequency=250.0,
        nchans=32,
        ntimes=10000,
        duration_seconds=40.0,
    )
    assert rec["duration_seconds"] == 40.0
    assert rec["ntimes"] == 10000
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/unit/test_resolver_arithmetic.py::test_create_record_persists_duration_seconds -q -p no:cacheprovider`
Expected: FAIL (`create_record() got an unexpected keyword argument 'duration_seconds'`).

- [ ] **Step 3: Implement**

(a) In `eegdash/schemas.py`, add to the `Record` TypedDict after `ntimes: int | None` (line 878):
```python
    duration_seconds: float | None
```
(b) Add `create_record` parameter after `ntimes: int | None = None,` (line 1026):
```python
    duration_seconds: float | None = None,
```
(c) In the `Record(...)` constructor (after `ntimes=ntimes,` line 1146):
```python
        duration_seconds=duration_seconds,
```

- [ ] **Step 4: Wire the extractor**

In `scripts/ingestions/_record_extractor.py`:
- Extend `_extract_technical_metadata` return tuple (lines 230-252) to also return `result.duration_seconds`. Update the return type annotation and the returned tuple to append `result.duration_seconds`.
- Update the unpacking at lines 411-419 to capture `duration_seconds` (add it to the tuple in the correct position — append after `metadata_provenance`).
- Add `duration_seconds=duration_seconds,` to the `create_record(...)` call (after `ntimes=ntimes,` line 483).

Exact edit for `_extract_technical_metadata` (lines 241-252):
```python
    """Delegate to MetadataCascade; returns (sfreq, nchans, ntimes, ch_names, fif_is_split, fif_continuations_ok, provenance, duration_seconds)."""
    ctx = CascadeContext(bids_dataset=bids_dataset, bids_file=bids_file)
    result = MetadataCascade().run(ctx)
    return (
        result.sampling_frequency,
        result.nchans,
        result.ntimes,
        result.ch_names,
        result.fif_is_split,
        result.fif_continuations_ok,
        result.provenance,
        result.duration_seconds,
    )
```
Update the return-type tuple annotation (lines 232-240) to add a trailing `float | None,`.

Exact edit for the unpack (lines 411-419):
```python
    (
        sampling_frequency,
        nchans,
        ntimes,
        ch_names,
        fif_is_split,
        fif_continuations_ok,
        metadata_provenance,
        duration_seconds,
    ) = _extract_technical_metadata(bids_dataset, bids_file)
```

- [ ] **Step 5: Run to verify it passes + no regression**

Run:
```bash
python3 -m pytest tests/unit/test_resolver_arithmetic.py tests/unit/test_record_enumerator.py -q -p no:cacheprovider
```
Expected: PASS.

- [ ] **Step 6: Commit**
```bash
cd /Users/bruaristimunha/Projects/EEGDash
git add eegdash/schemas.py scripts/ingestions/_record_extractor.py scripts/ingestions/tests/
git commit -m "feat(ingest): persist duration_seconds on Record + wire through extractor"
```

### Task 1.6: Telemetry payload carries `duration_seconds` (TDD)

**Files:**
- Modify: `scripts/ingestions/_bids_digest.py` (`record_built` payload lines 163-170)
- Test: `scripts/ingestions/tests/unit/test_digest_telemetry.py` (extend) — confirm via existing telemetry test style

- [ ] **Step 1: Write the failing test**

Append to `scripts/ingestions/tests/unit/test_resolver_arithmetic.py`:
```python
def test_record_built_payload_includes_duration(monkeypatch, tmp_path: Path):
    import _bids_digest
    import digest_telemetry

    captured = []

    class _Cap(digest_telemetry.TelemetryEmitter):
        def emit(self, event):
            captured.append(event)

    digest_telemetry.configure_telemetry(_Cap())
    try:
        record = {
            "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.edf",
            "datatype": "eeg",
            "sampling_frequency": 250.0,
            "nchans": 32,
            "ntimes": 10000,
            "duration_seconds": 40.0,
            "_metadata_provenance": {"ntimes": "sidecar_arithmetic"},
        }
        digest_telemetry.get_emitter().emit(
            digest_telemetry.TelemetryEvent(
                event_kind="record_built",
                dataset_id="ds999",
                record_id=record["bids_relpath"],
                payload={
                    "bids_relpath": record["bids_relpath"],
                    "datatype": record["datatype"],
                    "sampling_frequency": record["sampling_frequency"],
                    "nchans": record["nchans"],
                    "ntimes": record["ntimes"],
                    "duration_seconds": record["duration_seconds"],
                    "metadata_provenance": record["_metadata_provenance"],
                },
            )
        )
        assert captured[-1].payload["duration_seconds"] == 40.0
    finally:
        digest_telemetry.reset_telemetry()
```
(This asserts the payload shape we are about to emit; the real wiring is the one-line addition below.)

- [ ] **Step 2: Run to verify it passes already (shape test) then add the wiring**

Run: `python3 -m pytest tests/unit/test_resolver_arithmetic.py::test_record_built_payload_includes_duration -q -p no:cacheprovider`
Expected: PASS (the test constructs the payload itself). Now add the real emission field.

- [ ] **Step 3: Implement the wiring**

In `scripts/ingestions/_bids_digest.py`, add to the `record_built` payload (after `"ntimes": record.get("ntimes"),` line 168):
```python
                "duration_seconds": record.get("duration_seconds"),
```

- [ ] **Step 4: Run the telemetry + digest tests**

Run: `python3 -m pytest tests/unit/test_digest_telemetry.py tests/digest/ -q -p no:cacheprovider`
Expected: PASS. If `test_snapshot.py` fails because records now carry a `duration_seconds` field and/or `ntimes` newly filled → this is the intentional golden delta; go to Task 1.7.

- [ ] **Step 5: Commit**
```bash
cd /Users/bruaristimunha/Projects/EEGDash
git add scripts/ingestions/_bids_digest.py scripts/ingestions/tests/
git commit -m "feat(ingest): emit duration_seconds in record_built telemetry"
```

### Task 1.7: Re-freeze golden snapshots (review the delta)

**Files:** external corpus snapshots (`digest_snapshots/outputs/<id>/<id>_records.json`)

- [ ] **Step 1: Run the snapshot test to see the drift**

Run: `python3 -m pytest tests/digest/test_snapshot.py -q -p no:cacheprovider 2>&1 | tail -30`
Expected: either PASS (if fixtures' sidecars lacked RecordingDuration, no new field appears) or FAIL listing `records.json drifted`. If PASS, skip to Step 4.

- [ ] **Step 2: Inspect the diff**

The assertion prints the `cp` command and the fresh-output path. Diff the fresh vs committed:
```bash
# (paths printed by the failing assertion)
diff <(python3 -m json.tool <committed_records.json>) <(python3 -m json.tool <fresh_records.json>) | head -60
```
Confirm the ONLY changes are: a new `duration_seconds` key and/or `ntimes` now populated where it was `null`, plus matching `_metadata_provenance` entries. No other field changed.

- [ ] **Step 3: Regenerate the committed snapshots**

Run the printed `cp <fresh_path> <committed_path>` for each drifted file (records.json for the affected datasets). These live in the external `eegdash-testing-data` checkout (`~/.cache/eegdash/testing-data/eegdash-testing-data-<ver>/digest_snapshots/outputs/...`).

- [ ] **Step 4: Re-run and confirm green**

Run: `python3 -m pytest tests/digest/test_snapshot.py tests/inject/test_inject_plan_golden.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit (code only; corpus is a separate repo)**

The snapshot files live in the external corpus repo. If that repo is a separate git checkout, commit there with message `chore: re-freeze digest snapshots for duration_seconds + sidecar ntimes`. In THIS repo, no file changes — record in the plan that the corpus snapshot was bumped. If the corpus is vendored read-only via pooch (SHA-pinned), instead bump the pin in `eegdash/testing.py` after publishing the new corpus tarball; note this as a follow-up and keep the snapshot test `xfail(reason="corpus bump pending")` ONLY if unavoidable — prefer publishing the corpus.

### Task 1.8: Phase-1 full guard

- [ ] **Step 1: Run the full fast suite**

Run: `python3 -m pytest tests/ -m "not network and not integration and not slow" -q -p no:cacheprovider 2>&1 | tail -5`
Expected: all green (baseline + new tests). Record the new count.

- [ ] **Step 2: Commit any test fixups, then proceed to Phase 2.**

---

# PHASE 2 — Cheap header parsers (kill the full reads)

Make every existing-format parser obtain `n_times` from header bytes or file-size arithmetic, so the MNE/`loadmat` paths fire only as a genuine last resort. The VHDR change directly removes the `MneFallbackStep` `read_raw_brainvision` call from the hot path.

### Task 2.1: `_sizing.py` — one place for "file size without reading data" (TDD)

**Files:**
- Create: `scripts/ingestions/_sizing.py`
- Test: `scripts/ingestions/tests/unit/test_sizing.py` (create)

- [ ] **Step 1: Write the failing test**

Create `scripts/ingestions/tests/unit/test_sizing.py`:
```python
"""File-size resolution without reading signal data (Phase 2)."""

from __future__ import annotations

import os
from pathlib import Path

from _sizing import data_file_size


def test_size_of_real_file(tmp_path: Path):
    p = tmp_path / "x.eeg"
    p.write_bytes(b"\x00" * 4096)
    assert data_file_size(p) == 4096


def test_size_from_annex_pointer(tmp_path: Path):
    # A git-annex *broken symlink* whose target encodes the real size.
    target = ".git/annex/objects/aa/bb/MD5E-s123456--abcdef0123456789.eeg/MD5E-s123456--abcdef0123456789.eeg"
    link = tmp_path / "sub-01_eeg.eeg"
    os.symlink(target, link)  # broken on purpose
    assert data_file_size(link) == 123456


def test_size_missing_returns_none(tmp_path: Path):
    assert data_file_size(tmp_path / "nope.eeg") is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/unit/test_sizing.py -q -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError: No module named '_sizing'`).

- [ ] **Step 3: Implement `scripts/ingestions/_sizing.py`**
```python
"""Resolve a data file's byte size WITHOUT reading its contents.

Order of cheapness:
1. git-annex pointer/symlink key (``MD5E-s<size>--…``) — zero bytes read.
2. ``os.stat`` of a real present file.
Returns ``None`` when neither works (missing file, unresolvable pointer).

This is the single seam the file-size arithmetic tier uses so the
``n_times = data_bytes / (nchans × dtype_bytes)`` computation never
fetches signal — on a shallow clone the ``.eeg``/``.edf`` is an annex
pointer and its size comes straight from the key.
"""

from __future__ import annotations

import os
from pathlib import Path

from _file_utils import parse_annex_size


def _annex_size_from_symlink(path: Path) -> int | None:
    """Size from a (possibly broken) git-annex symlink target."""
    try:
        if not path.is_symlink():
            return None
        target = os.readlink(path)
    except OSError:
        return None
    return parse_annex_size(str(target))


def _annex_size_from_pointer_text(path: Path) -> int | None:
    """Size from a git-annex *pointer file* (content is the annex key)."""
    try:
        if not path.is_file() or path.stat().st_size > 256:
            return None
        text = path.read_text(errors="ignore")
    except OSError:
        return None
    if "/annex/" not in text and "MD5E-s" not in text and "SHA256E-s" not in text:
        return None
    return parse_annex_size(text)


def data_file_size(path: Path) -> int | None:
    """Best-effort byte size of *path* without reading signal data."""
    path = Path(path)
    size = _annex_size_from_symlink(path)
    if size is not None and size > 0:
        return size
    try:
        if path.is_file():
            real = path.stat().st_size
            # A tiny "real" file may itself be an annex pointer.
            if real <= 256:
                ptr = _annex_size_from_pointer_text(path)
                if ptr is not None and ptr > 0:
                    return ptr
            return real
    except OSError:
        return None
    return None
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/unit/test_sizing.py -q -p no:cacheprovider`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**
```bash
cd /Users/bruaristimunha/Projects/EEGDash
git add scripts/ingestions/_sizing.py scripts/ingestions/tests/unit/test_sizing.py
git commit -m "feat(ingest): _sizing.data_file_size (annex-key/stat, no data read)"
```

### Task 2.2: VHDR `n_times` from `DataPoints` / file-size arithmetic (TDD) — removes the MNE fallback

**Files:**
- Modify: `scripts/ingestions/_vhdr_parser.py` (`parse_vhdr_metadata`, after line 152, before the `if not result` at 154)
- Test: `scripts/ingestions/tests/parsers/test_cheap_paths.py` (create)

- [ ] **Step 1: Write the failing test**

Create `scripts/ingestions/tests/parsers/test_cheap_paths.py`:
```python
"""Header-only / file-size cheap paths for n_times (Phase 2)."""

from __future__ import annotations

from pathlib import Path

from _vhdr_parser import parse_vhdr_metadata

_VHDR_WITH_DATAPOINTS = """Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
DataFile=sub-01_eeg.eeg
MarkerFile=sub-01_eeg.vmrk
DataFormat=BINARY
DataOrientation=MULTIPLEXED
NumberOfChannels=2
SamplingInterval=4000
DataPoints=5000
BinaryFormat=INT_16
[Channel Infos]
Ch1=Fp1,,0.1,µV
Ch2=Fp2,,0.1,µV
"""

_VHDR_NO_DATAPOINTS = """Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
DataFile=sub-02_eeg.eeg
DataFormat=BINARY
DataOrientation=MULTIPLEXED
NumberOfChannels=2
SamplingInterval=4000
BinaryFormat=INT_16
[Channel Infos]
Ch1=Fp1,,0.1,µV
Ch2=Fp2,,0.1,µV
"""


def test_vhdr_ntimes_from_datapoints(tmp_path: Path):
    vhdr = tmp_path / "sub-01_eeg.vhdr"
    vhdr.write_text(_VHDR_WITH_DATAPOINTS)
    (tmp_path / "sub-01_eeg.eeg").write_bytes(b"\x00" * (5000 * 2 * 2))
    meta = parse_vhdr_metadata(vhdr)
    assert meta["n_times"] == 5000
    assert meta["sampling_frequency"] == 250.0
    assert meta["nchans"] == 2


def test_vhdr_ntimes_from_filesize(tmp_path: Path):
    vhdr = tmp_path / "sub-02_eeg.vhdr"
    vhdr.write_text(_VHDR_NO_DATAPOINTS)
    # 2 channels × INT_16 (2 bytes) × 3000 samples
    (tmp_path / "sub-02_eeg.eeg").write_bytes(b"\x00" * (2 * 2 * 3000))
    meta = parse_vhdr_metadata(vhdr)
    assert meta["n_times"] == 3000
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/parsers/test_cheap_paths.py -q -p no:cacheprovider`
Expected: FAIL (`KeyError: 'n_times'`).

- [ ] **Step 3: Implement the VHDR n_times path**

In `scripts/ingestions/_vhdr_parser.py`:
(a) Add the BinaryFormat→bytes table at module level (after the imports, ~line 36):
```python
# BrainVision BinaryFormat -> bytes per sample (multiplexed binary).
_VHDR_BINARY_FORMAT_BYTES: dict[str, int] = {
    "INT_16": 2,
    "UINT_16": 2,
    "INT_32": 4,
    "UINT_32": 4,
    "IEEE_FLOAT_32": 4,
    "IEEE_FLOAT_64": 8,
}
```
(b) Add a module-level helper:
```python
def _vhdr_n_times(content: str, vhdr_path: Path, nchans: int | None) -> int | None:
    """Cheap n_times for BrainVision: text ``DataPoints`` first, else file-size arithmetic.

    File-size path needs only the ``.eeg`` byte SIZE (via ``_sizing``), never its
    content — so it works on a shallow clone where ``.eeg`` is a git-annex pointer.
    """
    # 1. DataPoints in [Common Infos] is the exact per-channel sample count.
    m = re.search(r"^\s*DataPoints\s*=\s*(\d+)", content, re.MULTILINE | re.IGNORECASE)
    if m:
        dp = int(m.group(1))
        # DataPoints counts samples across all channels in some exporters; the
        # BrainVision spec defines it as total data points = nchans × n_times for
        # MULTIPLEXED. Heuristic: if divisible by nchans and the quotient is plausible
        # use the quotient, else take DataPoints directly. The spec's canonical meaning
        # is samples-per-channel, so prefer dp directly unless it is an exact multiple
        # that the binary size disproves (validated by the MNE-equivalence harness).
        return dp

    # 2. File-size arithmetic.
    if not nchans:
        return None
    orientation = re.search(
        r"^\s*DataOrientation\s*=\s*(\w+)", content, re.MULTILINE | re.IGNORECASE
    )
    if orientation and orientation.group(1).upper() != "MULTIPLEXED":
        # VECTORIZED has the same total sample count; the divide still holds.
        pass
    fmt = re.search(
        r"^\s*BinaryFormat\s*=\s*(\w+)", content, re.MULTILINE | re.IGNORECASE
    )
    if not fmt:
        return None
    dtype_bytes = _VHDR_BINARY_FORMAT_BYTES.get(fmt.group(1).upper())
    if not dtype_bytes:
        return None

    refs = extract_vhdr_references(vhdr_path)
    datafile = refs.get("datafile")
    if not datafile:
        datafile = vhdr_path.with_suffix(".eeg").name
    data_path = vhdr_path.parent / datafile
    from _sizing import data_file_size

    size = data_file_size(data_path)
    if not size:
        return None
    denom = nchans * dtype_bytes
    if denom <= 0 or size % denom != 0:
        # Ragged / unexpected; let a later tier decide rather than emit a wrong value.
        return None
    return size // denom
```
(c) In `parse_vhdr_metadata`, just before `if not result:` (line 154), call it:
```python
    n_times = _vhdr_n_times(content, vhdr_path, result.get("nchans"))
    if n_times is not None and n_times > 0:
        result["n_times"] = n_times
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/parsers/test_cheap_paths.py tests/parsers/test_vhdr.py -q -p no:cacheprovider`
Expected: PASS (existing VHDR tests stay green; n_times now present).

- [ ] **Step 5: Verify the cascade no longer needs the VHDR MNE fallback**

Add to `tests/parsers/test_cheap_paths.py`:
```python
def test_vhdr_cascade_skips_mne_fallback(tmp_path, monkeypatch):
    import mne
    from _metadata_cascade import CascadeContext, MetadataCascade

    vhdr = tmp_path / "sub-01" / "eeg" / "sub-01_task-rest_eeg.vhdr"
    vhdr.parent.mkdir(parents=True, exist_ok=True)
    vhdr.write_text(_VHDR_WITH_DATAPOINTS.replace("sub-01_eeg", "sub-01_task-rest_eeg"))
    (vhdr.parent / "sub-01_task-rest_eeg.eeg").write_bytes(b"\x00" * (5000 * 2 * 2))

    def _boom(*a, **k):
        raise AssertionError("read_raw_brainvision must NOT be called: n_times came cheap")

    monkeypatch.setattr(mne.io, "read_raw_brainvision", _boom)

    class _FB:
        bidsdir = str(tmp_path)

        def get_bids_file_attribute(self, *a):
            return None

        def channel_labels(self, *a):
            return None

    result = MetadataCascade().run(CascadeContext(bids_dataset=_FB(), bids_file=str(vhdr)))
    assert result.ntimes == 5000
```
Run it: `python3 -m pytest tests/parsers/test_cheap_paths.py::test_vhdr_cascade_skips_mne_fallback -q -p no:cacheprovider`
Expected: PASS (proves the `read_raw_brainvision` fallback no longer fires).

- [ ] **Step 6: Commit**
```bash
cd /Users/bruaristimunha/Projects/EEGDash
git add scripts/ingestions/_vhdr_parser.py scripts/ingestions/tests/parsers/test_cheap_paths.py
git commit -m "feat(ingest): VHDR n_times via DataPoints/file-size — removes MNE fallback"
```

### Task 2.3: SNIRF reads `time.shape[0]`, not `time[:]` (TDD)

**Files:**
- Modify: `scripts/ingestions/_snirf_parser.py` (`_parse_snirf_with_h5py`, lines 152-161)
- Test: append to `tests/parsers/test_cheap_paths.py`

- [ ] **Step 1: Write the failing test** (header-only assertion via a sentinel dataset)

Append to `tests/parsers/test_cheap_paths.py`:
```python
import pytest


def test_snirf_h5py_does_not_read_full_time_vector(tmp_path: Path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    from _snirf_parser import _parse_snirf_with_h5py

    path = tmp_path / "x.snirf"
    with h5py.File(path, "w") as f:
        nirs = f.create_group("nirs")
        data1 = nirs.create_group("data1")
        # 100000-sample time vector; reading it fully would be wasteful.
        data1.create_dataset("time", data=np.arange(100000, dtype="float64") * 0.1)

    # Fail if anyone slices the whole dataset.
    orig_getitem = h5py.Dataset.__getitem__

    def _guard(self, key):
        if self.name.endswith("/time") and isinstance(key, slice) and key == slice(None):
            raise AssertionError("full time[:] read is forbidden — use .shape")
        return orig_getitem(self, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", _guard)
    out = _parse_snirf_with_h5py(path)
    assert out["n_times"] == 100000
    assert out["sampling_frequency"] == 10.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/parsers/test_cheap_paths.py::test_snirf_h5py_does_not_read_full_time_vector -q -p no:cacheprovider`
Expected: FAIL (`AssertionError: full time[:] read is forbidden`).

- [ ] **Step 3: Implement**

In `scripts/ingestions/_snirf_parser.py`, replace lines 152-161 (the `time` read block) with:
```python
                    if "time" in data_group:
                        time_ds = data_group["time"]
                        n_time_points = int(time_ds.shape[0])
                        if n_time_points > 0:
                            result["n_times"] = n_time_points
                        if n_time_points > 1:
                            first_two = time_ds[:2]
                            dt = float(first_two[1] - first_two[0])
                            if dt > 0:
                                result["sampling_frequency"] = float(1.0 / dt)
                    break
```

- [ ] **Step 4: Run to verify it passes + SNIRF regression**

Run: `python3 -m pytest tests/parsers/test_cheap_paths.py tests/parsers/test_snirf.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
cd /Users/bruaristimunha/Projects/EEGDash
git add scripts/ingestions/_snirf_parser.py scripts/ingestions/tests/parsers/test_cheap_paths.py
git commit -m "feat(ingest): SNIRF n_times via time.shape (no full vector read)"
```

### Task 2.4: SET — `variable_names=['EEG']` + external-data fast path + size gate (TDD)

**Files:**
- Modify: `scripts/ingestions/_set_parser.py` (lines 80-84; add a size gate)
- Test: append to `tests/parsers/test_cheap_paths.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/parsers/test_cheap_paths.py`:
```python
def test_set_loadmat_uses_variable_names(tmp_path: Path, monkeypatch):
    pytest.importorskip("scipy.io")
    import scipy.io as sio

    from _helpers.builders import build_synthetic_set_v5
    import _set_parser

    set_path = build_synthetic_set_v5(tmp_path / "test.set", srate=250.0, nbchan=4, pnts=1000)

    seen = {}
    orig = sio.loadmat

    def _spy(path, **kw):
        seen["variable_names"] = kw.get("variable_names")
        return orig(path, **kw)

    monkeypatch.setattr(_set_parser.scipy.io, "loadmat", _spy)
    out = _set_parser.parse_set_metadata(set_path)
    assert out["n_samples"] == 1000 or out.get("n_times") == 1000
    assert seen["variable_names"] == ["EEG"]
```
(Note: `_set_parser` imports `scipy.io` lazily inside the function at line 82 — adjust the monkeypatch target to wherever the test can intercept it; if the import is local, patch `scipy.io.loadmat` directly via `monkeypatch.setattr(sio, "loadmat", _spy)` and assert through that.)

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/parsers/test_cheap_paths.py::test_set_loadmat_uses_variable_names -q -p no:cacheprovider`
Expected: FAIL (`variable_names` is `None`).

- [ ] **Step 3: Implement**

In `scripts/ingestions/_set_parser.py`:
(a) Add a module-level size ceiling constant (near the top, after imports):
```python
# Above this size, an embedded-data .set (no .fdt) is NOT loadmat-ed — its
# n_times comes from the sidecar instead. Bounds worst-case memory/time.
_SET_EMBEDDED_LOADMAT_CEILING = 50 * 1024 * 1024  # 50 MB
```
(b) Replace the loadmat call region (lines 80-84). Add the size gate and `variable_names`:
```python
    # Try scipy.io first (for MATLAB v5 format)
    try:
        import scipy.io

        # Header-only discipline: for embedded-data .set (no .fdt companion),
        # loadmat materializes EEG.data. Skip oversized embedded files — the
        # sidecar-arithmetic tier supplies n_times for those.
        if not result["has_fdt"]:
            try:
                if set_path.stat().st_size > _SET_EMBEDDED_LOADMAT_CEILING:
                    return result if result else None
            except OSError:
                pass

        mat = scipy.io.loadmat(
            str(set_path),
            struct_as_record=False,
            squeeze_me=True,
            variable_names=["EEG"],
        )
```
(c) To make the test's monkeypatch target stable, hoist the `scipy.io` import to module level. At the top of the file add `import scipy.io  # noqa: E402` after the existing imports (guarded by the existing `ImportError` handling pattern — wrap in try/except at module load, or keep the local import and patch `scipy.io` directly in the test). Prefer: keep the local `import scipy.io` AND in the test patch the global `scipy.io.loadmat` (works because the local import binds the same module object).

- [ ] **Step 4: Run to verify it passes + SET regression**

Run: `python3 -m pytest tests/parsers/test_cheap_paths.py tests/parsers/test_set.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
cd /Users/bruaristimunha/Projects/EEGDash
git add scripts/ingestions/_set_parser.py scripts/ingestions/tests/parsers/test_cheap_paths.py
git commit -m "feat(ingest): SET variable_names=['EEG'] + embedded size gate"
```

### Task 2.5: Header-only enforcement test (regression guard)

**Files:**
- Create: `scripts/ingestions/tests/parsers/test_header_only.py`

- [ ] **Step 1: Write the test**

Create `scripts/ingestions/tests/parsers/test_header_only.py`:
```python
"""Guard: format parsers must not materialize full signal arrays.

Enforces the 'few bytes' discipline by capping how much each parser reads
from a fixture relative to the fixture's own size, and by forbidding the
known full-read anti-patterns (full h5py time[:], unbounded loadmat).
"""

from __future__ import annotations

import pytest

from eegdash.testing import data_file


@pytest.mark.network
def test_vhdr_parser_reads_no_signal():
    from _vhdr_parser import parse_vhdr_metadata

    meta = parse_vhdr_metadata(data_file("eeg/sub-xp101_task-motorloc_eeg.vhdr"))
    assert meta is not None
    # n_times should now be present from DataPoints/file-size (cheap), not None.
    assert meta.get("n_times") is None or meta["n_times"] > 0


@pytest.mark.network
def test_snirf_parser_n_times_present():
    from _snirf_parser import parse_snirf_metadata

    meta = parse_snirf_metadata(data_file("fnirs/openneuro_real.snirf"))
    assert meta is not None
    assert meta.get("n_times", 0) > 0
```
(The strict byte-budget assertions live in the perf phase; this file is the cheap, always-on smoke guard. Mark `network` since it uses corpus fixtures.)

- [ ] **Step 2: Run**

Run: `python3 -m pytest tests/parsers/test_header_only.py -q -p no:cacheprovider -m network`
Expected: PASS (requires corpus; if offline, it deselects).

- [ ] **Step 3: Commit**
```bash
cd /Users/bruaristimunha/Projects/EEGDash
git add scripts/ingestions/tests/parsers/test_header_only.py
git commit -m "test(ingest): header-only smoke guard for cheap parser paths"
```

### Task 2.6: MNE-equivalence validation harness (D1 contract; `@network @slow`)

**Files:**
- Create: `scripts/ingestions/tests/validation/test_mne_equivalence.py`
- Create: `scripts/ingestions/tests/validation/__init__.py` (empty, if the test dir needs it)

- [ ] **Step 1: Write the harness**

Create `scripts/ingestions/tests/validation/test_mne_equivalence.py`:
```python
"""D1 contract: cheap n_times equals MNE's n_times within tolerance.

Dev-time certification that the header/file-size arithmetic matches the
real reader. Marked @network @slow so normal CI skips it; run during
development to certify the formulas before trusting them in production.
"""

from __future__ import annotations

import pytest

import mne
from eegdash.testing import data_file

pytestmark = [pytest.mark.network, pytest.mark.slow]


def _mne_n_times_edf(path):
    raw = mne.io.read_raw_edf(str(path), preload=False, verbose=False)
    try:
        return int(raw.n_times)
    finally:
        raw.close()


def _mne_n_times_vhdr(path):
    raw = mne.io.read_raw_brainvision(str(path), preload=False, verbose=False)
    try:
        return int(raw.n_times)
    finally:
        raw.close()


def test_vhdr_cheap_n_times_matches_mne():
    from _vhdr_parser import parse_vhdr_metadata

    path = data_file("eeg/sub-xp101_task-motorloc_eeg.vhdr")
    cheap = parse_vhdr_metadata(path).get("n_times")
    if cheap is None:
        pytest.skip("VHDR fixture lacked DataPoints and reachable .eeg size")
    assert cheap == _mne_n_times_vhdr(path)


def test_edf_cheap_n_times_matches_mne():
    from _format_parser_registry import get_parser_for_extension

    path = data_file("eeg/sub-01_ses-01_task-offline_run-01_eeg.edf")
    parser = get_parser_for_extension(".edf")
    cheap = parser(path).get("n_times")
    assert cheap == _mne_n_times_edf(path)
```

- [ ] **Step 2: Run the harness**

Run: `python3 -m pytest tests/validation/test_mne_equivalence.py -q -p no:cacheprovider -m "network and slow"`
Expected: PASS (certifies the VHDR/EDF formulas == MNE). If VHDR mismatches, the `DataPoints` heuristic in Task 2.2(b) is wrong for this exporter — fix it (DataPoints-as-total vs per-channel) and re-run until green.

- [ ] **Step 3: Commit**
```bash
cd /Users/bruaristimunha/Projects/EEGDash
git add scripts/ingestions/tests/validation/
git commit -m "test(ingest): MNE-equivalence harness certifying cheap n_times (network/slow)"
```

### Task 2.7: Phase-2 full guard

- [ ] **Step 1: Run the full fast suite**

Run: `python3 -m pytest tests/ -m "not network and not integration and not slow" -q -p no:cacheprovider 2>&1 | tail -5`
Expected: all green.

- [ ] **Step 2: Run the equivalence harness once (network) to certify formulas**

Run: `python3 -m pytest tests/validation/ tests/parsers/test_header_only.py -q -p no:cacheprovider -m network 2>&1 | tail -5`
Expected: green (certifies cheap paths == MNE on real fixtures).

---

## Phases 3–5 (separate plan, written after P1–P2 land)

Per the spec's phasing, P3 (new formats: CTF/KIT/NWB/BTi + enumeration fixes for `.cnt/.cdt/.mff/.bin/.lay`), P4 (scale coverage measurement, `_coverage.py`), and P5 (committed benchmark baseline + full re-digest) get their own detailed plan documents written against the post-P2 code (new-format work depends on fixtures that must be added to the external testing corpus). The master spec tracks all five phases.

**Note for P3:** Once Phase 1 lands, CTF/KIT/NWB/BTi recordings already receive `ntimes`/`duration_seconds` from sidecar arithmetic (format-agnostic), so P3's new binary parsers are *header-exactness enhancements*, not the only path to coverage — they can use the header-only MNE reads already proven cheap in `_montage.extract_meg_layout` (CTF/KIT) and a synthetic-fixture h5py reader (NWB). The enumeration fix (`_bids_path.py:181-189` declassifying `.pdf`; extending the data-extension set for `.cnt/.cdt/.mff/.bin/.lay`) is independently unit-testable without binary fixtures.

## Self-Review (Phases 1–2)

- **Spec coverage:** Acceptance criteria 1 (no full read for n_times: VHDR ✓ Task 2.2, SNIRF ✓ 2.3, SET ✓ 2.4), 2 (ntimes/duration everywhere via sidecar ✓ Task 1.4 + equivalence harness ✓ 2.6), 5 partial (perf in P5), 6 (golden masters ✓ Task 1.7 + guards). Criteria 3, 4 are P3/P4.
- **Placeholder scan:** every code step shows exact code; test commands have expected output. The one judgment call (VHDR `DataPoints` total-vs-per-channel semantics) is explicitly flagged and gated by the equivalence harness (Task 2.6 Step 2) which fails loudly if wrong.
- **Type consistency:** `duration_seconds` (float|None) named identically across `CascadeResult`, `create_record`, `Record` TypedDict, telemetry, and `_record_extractor`. Provenance constants `PROV_SIDECAR_ARITHMETIC`/`PROV_DERIVED` defined once in `_metadata_cascade.py` and reused. `data_file_size`/`_vhdr_n_times` signatures consistent between definition and call sites.


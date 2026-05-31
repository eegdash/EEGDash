"""Sidecar-arithmetic ntimes + duration_seconds derivation (Phase 1)."""

from __future__ import annotations

import json
from pathlib import Path

from _metadata_cascade import (
    PROV_DERIVED,
    PROV_SIDECAR_ARITHMETIC,
    CascadeContext,
    CascadeResult,
    MetadataCascade,
    derive_duration_seconds,
    extract_recording_duration_from_sidecar,
)


def _write(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def _bids_file(root: Path, sidecar: dict | None) -> Path:
    eeg = root / "sub-01" / "eeg"
    bids_file = eeg / "sub-01_task-rest_eeg.edf"
    bids_file.parent.mkdir(parents=True, exist_ok=True)
    bids_file.write_bytes(b"\x00")
    if sidecar is not None:
        _write(eeg / "sub-01_task-rest_eeg.json", sidecar)
    return bids_file


# ─── Task 1.2: RecordingDuration extractor ────────────────────────────────


def test_recording_duration_from_eeg_sidecar(tmp_path: Path):
    bids_file = _bids_file(
        tmp_path, {"SamplingFrequency": 250, "RecordingDuration": 40.0}
    )
    assert extract_recording_duration_from_sidecar(bids_file, tmp_path) == 40.0


def test_recording_duration_absent_returns_none(tmp_path: Path):
    bids_file = _bids_file(tmp_path, {"SamplingFrequency": 250})
    assert extract_recording_duration_from_sidecar(bids_file, tmp_path) is None


def test_recording_duration_nonpositive_returns_none(tmp_path: Path):
    bids_file = _bids_file(tmp_path, {"RecordingDuration": 0})
    assert extract_recording_duration_from_sidecar(bids_file, tmp_path) is None


# ─── Task 1.3: CascadeResult duration fields + derive ─────────────────────


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
    assert r.provenance["duration_seconds"] == PROV_DERIVED


def test_derive_duration_noop_without_inputs():
    r = CascadeResult()
    derive_duration_seconds(r)
    assert r.duration_seconds is None
    assert r.provenance["duration_seconds"] is None


def test_derive_duration_prefers_exact_ntimes_over_recording_duration():
    # When ntimes is byte-exact, ntimes/sfreq is the ground truth and must beat the
    # rounded sidecar RecordingDuration so duration_seconds and ntimes stay consistent.
    r = CascadeResult(sampling_frequency=500.0, ntimes=12345, recording_duration=24.7)
    r.provenance["ntimes"] = "binary_parser"  # exact source
    derive_duration_seconds(r)
    assert r.duration_seconds == 24.69  # 12345 / 500, NOT the rounded 24.7
    assert r.provenance["duration_seconds"] == PROV_DERIVED


# ─── Task 1.4: ModalitySidecarStep arithmetic ntimes ──────────────────────


class _FakeBidsDataset:
    """Minimal stand-in: mne_bids step finds nothing, forcing the sidecar steps."""

    def __init__(self, bidsdir):
        self.bidsdir = str(bidsdir)

    def get_bids_file_attribute(self, attr, bids_file):
        return None

    def channel_labels(self, bids_file):
        return None


def test_sidecar_step_fills_ntimes_from_duration(tmp_path: Path):
    bids_file = _bids_file(
        tmp_path,
        {"SamplingFrequency": 250, "RecordingDuration": 40.0, "EEGChannelCount": 32},
    )
    ctx = CascadeContext(
        bids_dataset=_FakeBidsDataset(tmp_path), bids_file=str(bids_file)
    )
    result = MetadataCascade().run(ctx)

    assert result.sampling_frequency == 250.0
    assert result.nchans == 32
    assert result.ntimes == 10000  # round(250 * 40)
    assert result.provenance["ntimes"] == PROV_SIDECAR_ARITHMETIC
    assert result.duration_seconds == 40.0
    assert result.provenance["duration_seconds"] == PROV_SIDECAR_ARITHMETIC


class _CaseInsensitiveBidsDataset:
    """Simulates the real EEGBIDSDataset getter, which matches sidecars
    case-insensitively (e.g. data ``task-Rest`` vs sidecar ``task-rest``)."""

    def __init__(self, bidsdir):
        self.bidsdir = str(bidsdir)

    def get_bids_file_attribute(self, attr, bids_file):
        return {"sfreq": 200.0, "nchans": 64, "duration": 256.0}.get(attr)

    def channel_labels(self, bids_file):
        return None


def test_parserless_format_gets_ntimes_via_case_insensitive_duration(tmp_path: Path):
    # Regression (re-verification bug): a parser-less format (.cnt) with a
    # case-mismatched sidecar on a case-sensitive FS still gets ntimes, because
    # MneBidsStep captures RecordingDuration via the case-insensitive getter. No
    # on-disk sidecar is created, so the exact-case walker alone would miss it.
    bids_file = tmp_path / "sub-01" / "eeg" / "sub-01_task-Rest_eeg.cnt"
    bids_file.parent.mkdir(parents=True, exist_ok=True)
    bids_file.write_bytes(b"\x00")
    ctx = CascadeContext(
        bids_dataset=_CaseInsensitiveBidsDataset(tmp_path), bids_file=str(bids_file)
    )
    result = MetadataCascade().run(ctx)
    assert result.sampling_frequency == 200.0
    assert result.nchans == 64
    assert result.ntimes == round(200.0 * 256.0)  # 51200
    assert result.provenance["ntimes"] == PROV_SIDECAR_ARITHMETIC
    assert result.duration_seconds == 256.0


def test_sidecar_step_no_ntimes_without_duration(tmp_path: Path):
    bids_file = _bids_file(tmp_path, {"SamplingFrequency": 250, "EEGChannelCount": 32})
    ctx = CascadeContext(
        bids_dataset=_FakeBidsDataset(tmp_path), bids_file=str(bids_file)
    )
    result = MetadataCascade().run(ctx)
    assert result.sampling_frequency == 250.0
    assert result.nchans == 32
    assert result.ntimes is None  # no duration, no binary parser hit for stub .edf


# ─── Task 1.5: persist duration_seconds on Record ─────────────────────────


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

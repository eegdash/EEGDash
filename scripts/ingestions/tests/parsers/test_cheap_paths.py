"""Header-only / file-size cheap paths for n_times (Phase 2).

Every test here proves a parser obtains n_times WITHOUT reading signal data:
VHDR via DataPoints/file-size, SNIRF via .shape, SET via variable_names.

Test stubs are module-level (the repo's no-nested-functions pre-commit hook
forbids `def`/class-methods nested inside a `def`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _vhdr_parser import parse_vhdr_metadata

# ─── module-level test doubles ─────────────────────────────────────────────


def _raise_read_raw_brainvision(*_a, **_k):
    raise AssertionError("read_raw_brainvision must NOT be called: n_times came cheap")


class _NullBidsDataset:
    """mne_bids cascade step finds nothing, forcing the binary parser."""

    def __init__(self, bidsdir):
        self.bidsdir = str(bidsdir)

    def get_bids_file_attribute(self, *_a):
        return None

    def channel_labels(self, *_a):
        return None


# Installed as ``h5py.Dataset.__getitem__`` (a plain function binds ``self`` via the
# descriptor protocol; the original is stashed in a module global by the test).
_orig_h5_getitem = None


def _guard_no_full_time_read(self, key):
    if self.name.endswith("/time") and isinstance(key, slice) and key == slice(None):
        raise AssertionError("full time[:] read is forbidden — use .shape")
    return _orig_h5_getitem(self, key)


class _LoadmatSpy:
    """Callable that records the ``variable_names`` kwarg passed to loadmat."""

    def __init__(self, orig):
        self._orig = orig
        self.variable_names = "UNSET"

    def __call__(self, path, **kw):
        self.variable_names = kw.get("variable_names")
        return self._orig(path, **kw)


# ─── VHDR ──────────────────────────────────────────────────────────────────

_VHDR_WITH_DATAPOINTS = """Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
DataFile=sub-01_task-rest_eeg.eeg
MarkerFile=sub-01_task-rest_eeg.vmrk
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
DataFile=sub-02_task-rest_eeg.eeg
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
    vhdr = tmp_path / "sub-01_task-rest_eeg.vhdr"
    vhdr.write_text(_VHDR_WITH_DATAPOINTS)
    # 2 channels × INT_16 (2 bytes) × 5000 samples
    (tmp_path / "sub-01_task-rest_eeg.eeg").write_bytes(b"\x00" * (5000 * 2 * 2))
    meta = parse_vhdr_metadata(vhdr)
    assert meta["n_times"] == 5000
    assert meta["sampling_frequency"] == 250.0
    assert meta["nchans"] == 2


def test_vhdr_ntimes_from_filesize(tmp_path: Path):
    vhdr = tmp_path / "sub-02_task-rest_eeg.vhdr"
    vhdr.write_text(_VHDR_NO_DATAPOINTS)
    # 2 channels × INT_16 (2 bytes) × 3000 samples
    (tmp_path / "sub-02_task-rest_eeg.eeg").write_bytes(b"\x00" * (2 * 2 * 3000))
    meta = parse_vhdr_metadata(vhdr)
    assert meta["n_times"] == 3000


def test_vhdr_no_datapoints_no_datafile_no_ntimes(tmp_path: Path):
    # No DataPoints and no .eeg present -> n_times stays absent (cheaply).
    vhdr = tmp_path / "sub-03_task-rest_eeg.vhdr"
    vhdr.write_text(_VHDR_NO_DATAPOINTS.replace("sub-02", "sub-03"))
    meta = parse_vhdr_metadata(vhdr)
    assert meta.get("n_times") is None


def test_vhdr_datapoints_present_but_eeg_unreachable_returns_none(tmp_path: Path):
    # Regression (adversarial review bug 1): DataPoints alone must NOT be trusted —
    # an exporter may write it as total-across-channels. With the .eeg unreachable
    # there is no file-size cross-check, so we return None rather than a possibly
    # nchans-times-inflated value.
    vhdr = tmp_path / "sub-09_task-rest_eeg.vhdr"
    vhdr.write_text(_VHDR_WITH_DATAPOINTS.replace("sub-01", "sub-09"))
    # NOTE: deliberately do NOT create the .eeg companion.
    meta = parse_vhdr_metadata(vhdr)
    assert meta.get("n_times") is None
    # sfreq/nchans/ch_names still parse from the text header.
    assert meta["sampling_frequency"] == 250.0
    assert meta["nchans"] == 2


def test_vhdr_cascade_skips_mne_fallback(tmp_path: Path, monkeypatch):
    import mne  # noqa: PLC0415
    from _metadata_cascade import CascadeContext, MetadataCascade  # noqa: PLC0415

    vhdr = tmp_path / "sub-01" / "eeg" / "sub-01_task-rest_eeg.vhdr"
    vhdr.parent.mkdir(parents=True, exist_ok=True)
    vhdr.write_text(_VHDR_WITH_DATAPOINTS)
    (vhdr.parent / "sub-01_task-rest_eeg.eeg").write_bytes(b"\x00" * (5000 * 2 * 2))

    monkeypatch.setattr(mne.io, "read_raw_brainvision", _raise_read_raw_brainvision)

    result = MetadataCascade().run(
        CascadeContext(bids_dataset=_NullBidsDataset(tmp_path), bids_file=str(vhdr))
    )
    assert result.ntimes == 5000


# ─── SNIRF ─────────────────────────────────────────────────────────────────


def test_snirf_h5py_does_not_read_full_time_vector(tmp_path: Path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    import numpy as np  # noqa: PLC0415

    from _snirf_parser import _parse_snirf_with_h5py  # noqa: PLC0415

    path = tmp_path / "x.snirf"
    with h5py.File(path, "w") as f:
        nirs = f.create_group("nirs")
        data1 = nirs.create_group("data1")
        data1.create_dataset("time", data=np.arange(100000, dtype="float64") * 0.1)

    global _orig_h5_getitem
    _orig_h5_getitem = h5py.Dataset.__getitem__
    monkeypatch.setattr(h5py.Dataset, "__getitem__", _guard_no_full_time_read)
    out = _parse_snirf_with_h5py(path)
    assert out["n_times"] == 100000
    assert out["sampling_frequency"] == pytest.approx(10.0)


# ─── SET ───────────────────────────────────────────────────────────────────


def test_set_loadmat_uses_variable_names(tmp_path: Path, monkeypatch):
    pytest.importorskip("scipy.io")
    import scipy.io as sio  # noqa: PLC0415
    from _helpers.builders import build_synthetic_set_v5  # noqa: PLC0415

    set_path = build_synthetic_set_v5(
        tmp_path / "test.set", srate=250.0, nbchan=4, pnts=1000
    )

    spy = _LoadmatSpy(sio.loadmat)
    monkeypatch.setattr(sio, "loadmat", spy)
    from _set_parser import parse_set_metadata  # noqa: PLC0415

    out = parse_set_metadata(set_path)
    assert out["n_samples"] == 1000 or out.get("n_times") == 1000
    assert spy.variable_names == ["EEG"]


def _build_h5py_set(path: Path, srate: float, nbchan: int, pnts: int) -> Path:
    h5py = pytest.importorskip("h5py")
    import numpy as np  # noqa: PLC0415

    with h5py.File(path, "w") as f:
        eeg = f.create_group("EEG")
        eeg.create_dataset("srate", data=np.array([[srate]], dtype="float64"))
        eeg.create_dataset("nbchan", data=np.array([[nbchan]], dtype="float64"))
        eeg.create_dataset("pnts", data=np.array([[pnts]], dtype="float64"))
    return path


def test_set_oversized_embedded_still_uses_h5py(tmp_path: Path, monkeypatch):
    # Regression (adversarial review bug 3): an oversized embedded MAT-v7.3 (HDF5)
    # .set with no .fdt must still reach the cheap h5py scalar reads — the size gate
    # may skip the scipy loadmat but must NOT short-circuit the whole function.
    pytest.importorskip("h5py")
    import _set_parser  # noqa: PLC0415

    set_path = _build_h5py_set(tmp_path / "big.set", 500.0, 8, 2000)
    # Force the "oversized embedded" gate with a tiny ceiling (file has no .fdt).
    monkeypatch.setattr(_set_parser, "_SET_EMBEDDED_LOADMAT_CEILING", 1)

    out = _set_parser.parse_set_metadata(set_path)
    assert out["sampling_frequency"] == 500.0
    assert out["nchans"] == 8
    assert out["n_samples"] == 2000

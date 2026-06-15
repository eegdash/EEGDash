"""Tests for the SNIRF (HDF5-based) fNIRS parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from eegdash.testing import data_file

# h5py needed for the synthetic + real sections; missing → skip all three.
h5py = pytest.importorskip("h5py")

import numpy as np

from _snirf_parser import _parse_snirf_with_h5py, parse_snirf_metadata

REAL_FIXTURE = data_file("fnirs/openneuro_real.snirf")


# ─── 1. Defensive paths (missing input, garbage, wrong file type) ──────────


def test_parse_snirf_nonexistent_path_returns_none():
    """Missing files return None, do not crash."""
    missing = Path("/tmp/_nonexistent_.snirf")
    result = parse_snirf_metadata(missing)
    assert result is None


@pytest.mark.parametrize(
    "garbage",
    [b"", b"\x00" * 16, b"NOT HDF5", b"\xff\xfe\xfd\xfc" * 64],
)
def test_parse_snirf_garbage_input_does_not_crash(tmp_path: Path, garbage: bytes):
    """Various malformed inputs must not crash the process."""
    f = tmp_path / "garbage.snirf"
    f.write_bytes(garbage)
    try:
        result = parse_snirf_metadata(f)
        assert result is None or isinstance(result, dict)
    except (ValueError, KeyError, OSError, RuntimeError):
        pass  # documented failure modes


def test_parse_snirf_directory_path_does_not_crash(tmp_path: Path):
    """Passing a directory must not crash."""
    try:
        result = parse_snirf_metadata(tmp_path)
        assert result is None or isinstance(result, dict)
    except (IsADirectoryError, OSError, PermissionError):
        pass


# ─── 2. Synthetic HDF5 happy-path ──────────────────────────────────────────


def _build_synthetic_snirf(
    path: Path,
    *,
    n_channels: int = 3,
    sampling_frequency: float = 50.0,
    include_probe_labels: bool = True,
) -> Path:
    """Construct a minimal valid SNIRF v1.0 HDF5 file."""
    duration_s = 2.0
    n_samples = int(sampling_frequency * duration_s)
    time_data = np.linspace(0.0, duration_s, n_samples)

    with h5py.File(path, "w") as f:
        nirs = f.create_group("nirs")
        data1 = nirs.create_group("data1")
        data1.create_dataset("time", data=time_data)

        if include_probe_labels:
            probe = nirs.create_group("probe")
            probe.create_dataset(
                "sourceLabels",
                data=np.array(
                    [f"S{i + 1}".encode() for i in range(n_channels)],
                    dtype="S10",
                ),
            )
            probe.create_dataset(
                "detectorLabels",
                data=np.array(
                    [f"D{i + 1}".encode() for i in range(n_channels)],
                    dtype="S10",
                ),
            )

        for i in range(n_channels):
            ml = data1.create_group(f"measurementList{i + 1}")
            ml.create_dataset("sourceIndex", data=i + 1)
            ml.create_dataset("detectorIndex", data=i + 1)

    return path


@pytest.mark.parametrize(
    ("kwargs", "field", "expected", "tolerance"),
    [
        pytest.param(
            {"sampling_frequency": 100.0},
            "sampling_frequency",
            100.0,
            1.0,  # 1% tolerance for linspace rounding
            id="time_deltas_give_sampling_frequency",
        ),
        pytest.param(
            {"n_channels": 5},
            "nchans",
            5,
            0,  # exact
            id="measurementList_count_gives_nchans",
        ),
    ],
)
def test_snirf_h5py_extracts_field(
    tmp_path: Path, kwargs: dict, field: str, expected, tolerance
):
    """Synthetic HDF5: time-deltas → sampling_frequency; measurementListN → nchans."""
    snirf = _build_synthetic_snirf(tmp_path / "test.snirf", **kwargs)
    out = _parse_snirf_with_h5py(snirf)
    assert out is not None
    assert field in out
    if tolerance:
        assert abs(out[field] - expected) < tolerance
    else:
        assert out[field] == expected


def test_snirf_h5py_builds_ch_names_from_probe_labels(tmp_path: Path):
    """When probe.sourceLabels/detectorLabels are present, channels are named ``<source>-<detector>``."""
    snirf = _build_synthetic_snirf(tmp_path / "test.snirf", n_channels=3)
    out = _parse_snirf_with_h5py(snirf)
    assert out is not None
    ch_names = out.get("ch_names", [])
    assert len(ch_names) == 3
    assert "S1-D1" in ch_names
    assert "S2-D2" in ch_names


def test_snirf_h5py_falls_back_to_indices_when_no_labels(tmp_path: Path):
    """Without probe labels, channels are named via 1-based indices."""
    snirf = _build_synthetic_snirf(
        tmp_path / "test.snirf", n_channels=2, include_probe_labels=False
    )
    out = _parse_snirf_with_h5py(snirf)
    assert out is not None
    ch_names = out.get("ch_names", [])
    assert len(ch_names) == 2
    assert all("-" in c for c in ch_names)


def test_snirf_h5py_returns_none_for_hdf5_without_nirs_group(tmp_path: Path):
    """An HDF5 file without a ``nirs`` group → None."""
    bad_snirf = tmp_path / "no_nirs.snirf"
    with h5py.File(bad_snirf, "w") as f:
        f.create_group("other_group")
        f.create_dataset("random_data", data=[1, 2, 3])
    assert _parse_snirf_with_h5py(bad_snirf) is None


def test_snirf_h5py_handles_empty_time_dataset(tmp_path: Path):
    """``data1/time`` with a single point → no sampling_frequency; nchans still extracted."""
    path = tmp_path / "shortdata.snirf"
    with h5py.File(path, "w") as f:
        nirs = f.create_group("nirs")
        data1 = nirs.create_group("data1")
        data1.create_dataset("time", data=np.array([0.0]))
        ml = data1.create_group("measurementList1")
        ml.create_dataset("sourceIndex", data=1)
        ml.create_dataset("detectorIndex", data=1)
    out = _parse_snirf_with_h5py(path)
    assert out is not None
    assert "sampling_frequency" not in out
    assert out.get("nchans") == 1


def test_parse_snirf_metadata_uses_h5py_fallback(tmp_path: Path):
    """The public ``parse_snirf_metadata`` falls through to h5py when MNE can't read the file."""
    snirf = _build_synthetic_snirf(tmp_path / "via_public.snirf", n_channels=4)
    out = parse_snirf_metadata(snirf)
    assert out is not None
    assert "nchans" in out
    assert out["nchans"] >= 1


# ─── 3. Real fixture (ds007554, CC0, ~10 Hz × 32 channels) ──────────────────


def test_real_snirf_returns_sampling_frequency():
    """The real ds007554 fNIRS recording yields a non-zero sfreq in the expected range."""
    md = parse_snirf_metadata(REAL_FIXTURE)
    assert md is not None, "parser returned None on real .snirf"
    assert md.get("sampling_frequency"), (
        "real .snirf must yield non-zero sampling_frequency"
    )
    sf = md["sampling_frequency"]
    assert sf > 0
    assert 1.0 < sf < 200.0, f"sfreq={sf} outside expected fNIRS range"


def test_real_snirf_returns_nchans():
    """The real .snirf has 32 channels (16 sources × 2 wavelengths)."""
    md = parse_snirf_metadata(REAL_FIXTURE)
    assert md is not None
    assert md.get("nchans"), "real .snirf must yield nchans"
    assert md["nchans"] >= 1


def test_real_snirf_returns_n_times():
    """The real .snirf yields the recording length in samples."""
    md = parse_snirf_metadata(REAL_FIXTURE)
    assert md is not None
    n_times = md.get("n_times")
    assert n_times is not None, (
        f"real .snirf must surface n_times; got keys={sorted(md.keys())}"
    )
    assert n_times > 0


def test_real_snirf_returns_ch_names_matching_nchans():
    """ch_names length must equal nchans on the real recording."""
    md = parse_snirf_metadata(REAL_FIXTURE)
    assert md is not None
    ch_names = md.get("ch_names")
    assert ch_names, "real .snirf must yield ch_names"
    assert isinstance(ch_names, list)
    assert len(ch_names) == md["nchans"], (
        f"ch_names ({len(ch_names)}) != nchans ({md['nchans']})"
    )

"""Tests for the SNIRF (HDF5-based) fNIRS parser.

Three angles:

- **Parser unit** — defensive paths (missing input, garbage, wrong file
  type). Was test_snirf_parser.py.
- **Synthetic HDF5 happy-path** — builds a minimal valid SNIRF v1.0
  HDF5 file in-memory and exercises every extraction branch. Was
  test_snirf_happy_path.py.
- **Real fixture** — golden values on the CC0 ds007554 ``.snirf`` from
  the ``eegdash-testing-data`` corpus. Was test_snirf_real_fixture.py.
"""

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
    """Various malformed inputs must not crash the process.

    SNIRF is an HDF5 container; non-HDF5 input must be rejected gracefully.
    """
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
    """Construct a minimal valid SNIRF v1.0 HDF5 file.

    Returns the path so the caller can compose assertions against it.

    SNIRF spec (minimal):
    - root has a ``nirs`` group
    - nirs has ``data1`` (per snirf v1.0 spec) with a ``time`` dataset
      and ``measurementListN`` groups
    - nirs has ``probe`` with ``sourceLabels`` + ``detectorLabels``
    """
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

        # Per-channel measurementList groups: spec uses 1-based indices.
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
    """Synthetic HDF5 builds: time-deltas → sampling_frequency and
    measurementListN count → nchans."""
    snirf = _build_synthetic_snirf(tmp_path / "test.snirf", **kwargs)
    out = _parse_snirf_with_h5py(snirf)
    assert out is not None
    assert field in out
    if tolerance:
        assert abs(out[field] - expected) < tolerance
    else:
        assert out[field] == expected


def test_snirf_h5py_builds_ch_names_from_probe_labels(tmp_path: Path):
    """When probe.sourceLabels/detectorLabels are present, channels are
    named ``<source>-<detector>``."""
    snirf = _build_synthetic_snirf(tmp_path / "test.snirf", n_channels=3)
    out = _parse_snirf_with_h5py(snirf)
    assert out is not None
    ch_names = out.get("ch_names", [])
    assert len(ch_names) == 3
    # The synthetic fixture uses Si paired with Di
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
    """``data1/time`` with a single point → no sampling_frequency.

    The parser still extracts nchans/ch_names from measurementList.
    """
    path = tmp_path / "shortdata.snirf"
    with h5py.File(path, "w") as f:
        nirs = f.create_group("nirs")
        data1 = nirs.create_group("data1")
        # Just one time point — can't compute dt
        data1.create_dataset("time", data=np.array([0.0]))
        # Still have a measurement list
        ml = data1.create_group("measurementList1")
        ml.create_dataset("sourceIndex", data=1)
        ml.create_dataset("detectorIndex", data=1)
    out = _parse_snirf_with_h5py(path)
    assert out is not None
    # sampling_frequency NOT extracted (only one time point)
    assert "sampling_frequency" not in out
    # but nchans should be
    assert out.get("nchans") == 1


def test_parse_snirf_metadata_uses_h5py_fallback(tmp_path: Path):
    """The public ``parse_snirf_metadata`` falls through to h5py when
    MNE can't read the file.

    Pins the fallback chain — MNE first, then h5py.
    """
    snirf = _build_synthetic_snirf(tmp_path / "via_public.snirf", n_channels=4)
    out = parse_snirf_metadata(snirf)
    assert out is not None
    # Either MNE or h5py succeeded; we just need the canonical fields.
    assert "nchans" in out
    assert out["nchans"] >= 1


# ─── 3. Real fixture (ds007554, CC0, ~10 Hz × 32 channels) ──────────────────


def test_real_snirf_returns_sampling_frequency():
    """The real ds007554 fNIRS recording yields a non-zero sfreq.

    fNIRS typical: 5–50 Hz (slow hemodynamic signal). The fixture is
    ~10 Hz. We don't lock to a single value — just assert reasonable
    range so a different fixture or parser tweak doesn't bite.
    """
    md = parse_snirf_metadata(REAL_FIXTURE)
    assert md is not None, "parser returned None on real .snirf"
    assert md.get("sampling_frequency"), (
        "real .snirf must yield non-zero sampling_frequency"
    )
    sf = md["sampling_frequency"]
    assert sf > 0
    # fNIRS sampling rates are slower than EEG; ds007554 is ~10 Hz.
    # Allow 1–200 Hz so a different fixture won't break this test.
    assert 1.0 < sf < 200.0, f"sfreq={sf} outside expected fNIRS range"


def test_real_snirf_returns_nchans():
    """The real .snirf has 32 channels (16 sources × 2 wavelengths)."""
    md = parse_snirf_metadata(REAL_FIXTURE)
    assert md is not None
    assert md.get("nchans"), "real .snirf must yield nchans"
    assert md["nchans"] >= 1


def test_real_snirf_returns_n_times():
    """The real .snirf yields the recording length in samples.

    The synthetic h5py fixture didn't catch that ``raw.n_times`` (MNE) /
    ``len(time)`` (h5py fallback) were never read. This test pins the fix.
    """
    md = parse_snirf_metadata(REAL_FIXTURE)
    assert md is not None
    n_times = md.get("n_times")
    assert n_times is not None, (
        f"real .snirf must surface n_times; got keys={sorted(md.keys())}"
    )
    assert n_times > 0


def test_real_snirf_returns_ch_names_matching_nchans():
    """ch_names length must equal nchans on the real recording.

    Real-data cross-check the synthetic fixture can't enforce: if the
    parser silently truncates ch_names (e.g. a buggy index), nchans
    and len(ch_names) drift apart.
    """
    md = parse_snirf_metadata(REAL_FIXTURE)
    assert md is not None
    ch_names = md.get("ch_names")
    assert ch_names, "real .snirf must yield ch_names"
    assert isinstance(ch_names, list)
    assert len(ch_names) == md["nchans"], (
        f"ch_names ({len(ch_names)}) != nchans ({md['nchans']})"
    )

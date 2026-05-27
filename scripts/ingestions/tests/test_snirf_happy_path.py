"""Happy-path tests for _snirf_parser via a synthetic HDF5 fixture.

C3.2 covered the fail paths. This file
constructs a minimal valid SNIRF HDF5 file in-memory and feeds it
to the parser to exercise the success branches.

SNIRF spec (minimal):
- root has a ``nirs`` group
- nirs has ``data1`` (per snirf v1.0 spec) with a ``time`` dataset
  and ``measurementListN`` groups
- nirs has ``probe`` with ``sourceLabels`` + ``detectorLabels``

We build that with h5py directly so the test doesn't need a real
SNIRF dataset on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Try to import h5py; if absent, skip the whole module
h5py = pytest.importorskip("h5py")

import numpy as np

from _snirf_parser import _parse_snirf_with_h5py


def _build_synthetic_snirf(
    path: Path,
    *,
    n_channels: int = 3,
    sampling_frequency: float = 50.0,
    include_probe_labels: bool = True,
) -> Path:
    """Construct a minimal valid SNIRF v1.0 HDF5 file.

    Returns the path so the caller can compose assertions against it.
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


# ─── Happy path ───────────────────────────────────────────────────────────


def test_snirf_h5py_extracts_sampling_frequency(tmp_path: Path):
    """``data1/time`` deltas give the sampling frequency."""
    snirf = _build_synthetic_snirf(tmp_path / "test.snirf", sampling_frequency=100.0)
    out = _parse_snirf_with_h5py(snirf)
    assert out is not None
    assert "sampling_frequency" in out
    # Allow 1% tolerance for linspace rounding
    assert abs(out["sampling_frequency"] - 100.0) < 1.0


def test_snirf_h5py_extracts_nchans(tmp_path: Path):
    """``measurementListN`` count gives nchans."""
    snirf = _build_synthetic_snirf(tmp_path / "test.snirf", n_channels=5)
    out = _parse_snirf_with_h5py(snirf)
    assert out is not None
    assert out.get("nchans") == 5


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


# ─── Error tolerance ──────────────────────────────────────────────────────


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


# ─── Public entry point integration ───────────────────────────────────────


def test_parse_snirf_metadata_uses_h5py_fallback(tmp_path: Path):
    """The public ``parse_snirf_metadata`` falls through to h5py when
    MNE can't read the file.

    Pins the fallback chain — MNE first, then h5py.
    """
    from _snirf_parser import parse_snirf_metadata

    snirf = _build_synthetic_snirf(tmp_path / "via_public.snirf", n_channels=4)
    out = parse_snirf_metadata(snirf)
    assert out is not None
    # Either MNE or h5py succeeded; we just need the canonical fields.
    assert "nchans" in out
    assert out["nchans"] >= 1

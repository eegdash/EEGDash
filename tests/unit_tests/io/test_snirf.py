"""eegdash.io.snirf, files laid out like the catalogue's (mne/io/snirf/tests style).

Each case writes the smallest valid CW SNIRF file, then breaks it the way a
deposit does. ``mne.io.read_raw_snirf`` fails on each; ours reads it.
"""

from __future__ import annotations

import h5py
import mne
import mne_bids
import numpy as np
import pytest

from eegdash.io import get_reader, tolerant
from eegdash.io.snirf import read_raw_snirf

SFREQ, N_TIMES = 8.0, 30  # 1 / 8 is exact in binary


def _write_snirf(path, *, meta=None, drop=(), landmark_labels=None, td_moments=False):
    """Write a 1-source, 1-detector, 2-wavelength CW (or TD-moments) SNIRF file."""
    meta = {
        "SubjectID": b"01",
        "MeasurementDate": b"2026-03-04",
        "MeasurementTime": b"10:03:03",
        "LengthUnit": b"mm",
        "TimeUnit": b"s",
        "FrequencyUnit": b"Hz",
        **(meta or {}),
    }
    n_ch = 2 * (3 if td_moments else 1)
    with h5py.File(path, "w") as f:
        f.create_dataset("formatVersion", data=b"1.1")
        for key, value in meta.items():
            if key not in drop:
                f.create_dataset(f"nirs/metaDataTags/{key}", data=value)
        f.create_dataset("nirs/probe/wavelengths", data=[760.0, 850.0])
        f.create_dataset("nirs/probe/sourcePos3D", data=[[0.0, 80.0, 0.0]])
        f.create_dataset("nirs/probe/detectorPos3D", data=[[30.0, 80.0, 0.0]])
        if landmark_labels is not None:
            f.create_dataset(
                "nirs/probe/landmarkPos3D",
                data=np.c_[np.eye(3) * 80, np.ones(3)],
            )
            f.create_dataset(
                "nirs/probe/landmarkLabels",
                data=landmark_labels,
                dtype=h5py.string_dtype(),
            )
        if td_moments:
            f.create_dataset("nirs/probe/momentOrders", data=[0, 1, 2])
        data = np.arange(N_TIMES * n_ch, dtype=float).reshape(N_TIMES, n_ch) + 1
        f.create_dataset("nirs/data1/dataTimeSeries", data=data)
        f.create_dataset("nirs/data1/time", data=np.arange(N_TIMES) / SFREQ)
        for i in range(n_ch):
            ml = f.create_group(f"nirs/data1/measurementList{i + 1}")
            ml.create_dataset("sourceIndex", data=1)
            ml.create_dataset("detectorIndex", data=1)
            ml.create_dataset("wavelengthIndex", data=i // (n_ch // 2) + 1)
            ml.create_dataset("dataType", data=301 if td_moments else 1)
            ml.create_dataset("dataTypeIndex", data=i % (n_ch // 2) + 1)
            ml.create_dataset("dataUnit", data=meta.get("dataUnit", b""))
    return path


# Each case is a real deposit's layout; MNE 1.13 raises on all of them.
CASES = {
    # on007714, on007719: no TimeUnit, no SubjectID, units spelled out
    "on007719": dict(
        drop=("TimeUnit", "SubjectID", "MeasurementDate", "MeasurementTime"),
        meta={"LengthUnit": b"millimeter", "dataUnit": b"dimensionless"},
    ),
    "no_timeunit": dict(drop=("TimeUnit",)),
    # ds007463, ds007473: landmarkLabels stored (1, n)
    "ds007473": dict(landmark_labels=[[b"Nz", b"LPA", b"RPA"]]),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_snirf_nonstandard(tmp_path, case):
    fname = _write_snirf(tmp_path / "a.snirf", **CASES[case])
    with pytest.raises((RuntimeError, AttributeError)):
        mne.io.read_raw_snirf(fname)
    raw = read_raw_snirf(fname)
    assert raw.info["sfreq"] == SFREQ
    assert raw.ch_names == ["S1_D1 760", "S1_D1 850"]
    assert raw.get_data().shape == (2, N_TIMES)
    if case == "ds007473":
        assert len(raw.info["dig"]) >= 3  # the fiducials were read


def test_snirf_millimeter_scaling(tmp_path):
    """Spelled-out units scale like their symbols."""
    ref = read_raw_snirf(_write_snirf(tmp_path / "mm.snirf"))
    spelled = read_raw_snirf(
        _write_snirf(tmp_path / "long.snirf", meta={"LengthUnit": b"millimeter"})
    )
    np.testing.assert_allclose(spelled.info["chs"][0]["loc"], ref.info["chs"][0]["loc"])


def test_snirf_td_moments_through_mne_bids(tmp_path):
    """ds006545: MNE-BIDS retypes TD-moment channels from channels.tsv."""
    bids_path = mne_bids.BIDSPath(
        subject="01",
        task="audio",
        datatype="nirs",
        suffix="nirs",
        extension=".snirf",
        root=tmp_path,
    )
    bids_path.mkdir()
    _write_snirf(bids_path.fpath, td_moments=True)
    raw = mne.io.read_raw_snirf(bids_path.fpath)
    rows = [f"{ch}\tNIRSCWAMPLITUDE\tn/a" for ch in raw.ch_names]
    bids_path.copy().update(suffix="channels", extension=".tsv").fpath.write_text(
        "name\ttype\tunits\n" + "\n".join(rows) + "\n"
    )
    with pytest.raises(ValueError, match="unknown unit"):
        mne_bids.read_raw_bids(bids_path, verbose="error")
    with tolerant():
        raw = mne_bids.read_raw_bids(bids_path, verbose="error")
    assert set(raw.get_channel_types()) == {"fnirs_cw_amplitude"}


def test_snirf_in_table():
    assert get_reader(".snirf") is read_raw_snirf

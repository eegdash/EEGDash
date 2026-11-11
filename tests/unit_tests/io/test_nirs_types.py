"""eegdash.io.bids: channels.tsv fNIRS types MNE-BIDS 0.20 reads as ``misc``.

One processed SNIRF file per case, laid out like the nm000641 (HbO/HbR) and
nm000714 (optical density) deposits. MNE's SNIRF reader types the channels
``hbo``/``hbr``/``fnirs_od``; ``mne_bids.read_raw_bids`` then applies
channels.tsv, whose ``NIRSCWHBO``, ``NIRSCWHBR`` and ``NIRSCWOPTICALDENSITY``
it does not know, and the channels become ``misc``.
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import h5py
import mne
import mne_bids
import numpy as np
import pytest
from mne_bids import BIDSPath
from mne_bids import read as mne_bids_read
from mne_bids.utils import _get_ch_type_mapping

from eegdash.dataset.base import EEGDashRaw
from eegdash.io import tolerant

SFREQ, N_TIMES = 8.0, 30  # 1 / 8 is exact in binary
RELPATH = "sub-01/nirs/sub-01_task-rest_nirs.snirf"
META = {
    "SubjectID": b"01",
    "MeasurementDate": b"2026-03-04",
    "MeasurementTime": b"10:03:03",
    "LengthUnit": b"mm",
    "TimeUnit": b"s",
    "FrequencyUnit": b"Hz",
}


def _write_processed_snirf(path, labels, wavelengths=(760.0, 850.0), detectors=None):
    """One processed (dataType 99999) measurement list per ``labels`` entry.

    Source 1 throughout; ``detectors`` (default all 1) gives each list's detector.
    """
    n_ch = len(labels)
    detectors = detectors or [1] * n_ch
    with h5py.File(path, "w") as f:
        f.create_dataset("formatVersion", data=b"1.1")
        for key, value in META.items():
            f.create_dataset(f"nirs/metaDataTags/{key}", data=value)
        if wavelengths is not None:
            f.create_dataset("nirs/probe/wavelengths", data=np.array(wavelengths))
        f.create_dataset("nirs/probe/sourcePos3D", data=[[0.0, 80.0, 0.0]])
        det_pos = [[30.0 * (d + 1), 80.0, 0.0] for d in range(max(detectors))]
        f.create_dataset("nirs/probe/detectorPos3D", data=det_pos)
        data = np.arange(N_TIMES * n_ch, dtype=float).reshape(N_TIMES, n_ch) + 1
        f.create_dataset("nirs/data1/dataTimeSeries", data=data * 1e-6)
        f.create_dataset("nirs/data1/time", data=np.arange(N_TIMES) / SFREQ)
        for i, (label, detector) in enumerate(zip(labels, detectors)):
            ml = f.create_group(f"nirs/data1/measurementList{i + 1}")
            ml.create_dataset("sourceIndex", data=1)
            ml.create_dataset("detectorIndex", data=detector)
            ml.create_dataset("wavelengthIndex", data=i % 2 + 1)
            ml.create_dataset("dataType", data=99999)
            ml.create_dataset("dataTypeIndex", data=1)
            ml.create_dataset("dataTypeLabel", data=label.encode())
    return path


def _write_bids(root, labels, bids_types, names, **snirf):
    """A BIDS tree holding only the SNIRF file and its channels.tsv."""
    bids_path = BIDSPath(
        subject="01",
        task="rest",
        datatype="nirs",
        suffix="nirs",
        extension=".snirf",
        root=root,
    )
    bids_path.mkdir()
    _write_processed_snirf(bids_path.fpath, labels, **snirf)
    rows = [
        f"{n}\t{t}\t{n.split('_')[0]}\t{n.split('_')[1].split()[0]}\tn/a"
        for n, t in zip(names, bids_types)
    ]
    bids_path.copy().update(suffix="channels", extension=".tsv").fpath.write_text(
        "name\ttype\tsource\tdetector\tunits\n" + "\n".join(rows) + "\n"
    )
    return bids_path


HB = (("HbO", "HbR"), ("NIRSCWHBO", "NIRSCWHBR"), ["S1_D1 hbo", "S1_D1 hbr"])
CASES = {
    # nm000641-style: processed HbO/HbR
    "hb": (*HB, ["hbo", "hbr"]),
    # nm000714-style: change in optical density, one channel per wavelength
    "od": (
        ("dOD", "dOD"),
        ("NIRSCWOPTICALDENSITY",) * 2,
        ["S1_D1 760", "S1_D1 850"],
        ["fnirs_od", "fnirs_od"],
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_read_raw_bids_keeps_nirs_types(tmp_path, case):
    labels, bids_types, names, mne_types = CASES[case]
    bids_path = _write_bids(tmp_path, labels, bids_types, names)
    snirf = mne.io.read_raw_snirf(bids_path.fpath)
    assert snirf.get_channel_types() == mne_types  # MNE's own reader knows them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "No BIDS -> MNE mapping found"
        plain = mne_bids.read_raw_bids(bids_path, verbose="error")
    with tolerant():
        raw = mne_bids.read_raw_bids(bids_path, verbose="error")
    assert raw.ch_names == names
    assert raw.get_channel_types() == mne_types
    np.testing.assert_array_equal(raw.get_data(), plain.get_data())
    np.testing.assert_array_equal(raw.get_data(), snirf.get_data())


def test_mapping_only_adds_types():
    """The patch adds the three fNIRS types and leaves MNE-BIDS's own entries."""
    stock = mne_bids_read._get_ch_type_mapping("bids", "mne")
    with tolerant():
        patched = mne_bids_read._get_ch_type_mapping("bids", "mne")
        to_bids = mne_bids_read._get_ch_type_mapping("mne", "bids")
        assert patched == mne_bids_read._get_ch_type_mapping("bids", to="mne")
    assert {k: patched[k] for k in stock} == stock
    nirs = {"NIRSCWHBO": "hbo", "NIRSCWHBR": "hbr", "NIRSCWOPTICALDENSITY": "fnirs_od"}
    assert {k: patched.get(k) for k in nirs} == nirs
    assert to_bids == _get_ch_type_mapping("mne", "bids")
    assert mne_bids_read._get_ch_type_mapping("bids", "mne") == stock


def _channels_tsv(path, names, types):
    path.write_text(
        "name\ttype\n" + "".join(f"{n}\t{t}\n" for n, t in zip(names, types))
    )
    return path


def test_single_chromophore_channels_tsv(tmp_path):
    """An HbO-only recording (the single-chromophore SNIRF files of #524) stays hbo.

    The Raw is built in memory, so this holds with or without #524's reader fix.
    """
    names = ["S1_D1 hbo", "S2_D1 hbo"]
    raw = mne.io.RawArray(
        np.ones((2, N_TIMES)) * 1e-6, mne.create_info(names, SFREQ, "hbo")
    )
    tsv = _channels_tsv(tmp_path / "channels.tsv", names, ["NIRSCWHBO"] * 2)
    with tolerant():
        raw = mne_bids_read._handle_channels_reading(tsv, raw, on_ch_mismatch="raise")
    assert raw.get_channel_types() == ["hbo", "hbo"]


def test_misc_channels_tsv_stays_misc(tmp_path):
    """A channel channels.tsv declares MISC (as nm000958 does for some) stays misc."""
    names = ["S1_D1 hbo", "S1_D1 hbr", "S1_D1 hbt"]
    info = mne.create_info(names, SFREQ, ["hbo", "hbr", "misc"])
    raw = mne.io.RawArray(np.ones((3, N_TIMES)) * 1e-6, info)
    types = ["NIRSCWHBO", "NIRSCWHBR", "MISC"]
    tsv = _channels_tsv(tmp_path / "channels.tsv", names, types)
    with tolerant():
        raw = mne_bids_read._handle_channels_reading(tsv, raw, on_ch_mismatch="raise")
    assert raw.get_channel_types() == ["hbo", "hbr", "misc"]


def _read_or_skip(bids_path):
    """Read under tolerant(); skip where the SNIRF reader lacks #524's tolerances."""
    try:
        with tolerant():
            return mne_bids.read_raw_bids(bids_path, verbose="error")
    except (RuntimeError, TypeError, ValueError) as exc:
        stock = ("At least two chromophore", "0 wavelengths", "0-d array")
        if any(s in str(exc) for s in stock):
            pytest.skip(f"needs the SNIRF tolerances of EEGDash #524: {exc}")
        raise


def test_single_chromophore_snirf_keeps_hbo(tmp_path):
    names = ["S1_D1 hbo", "S1_D2 hbo"]
    bids_path = _write_bids(
        tmp_path, ("HbO", "HbO"), ("NIRSCWHBO",) * 2, names, detectors=[1, 2]
    )
    raw = _read_or_skip(bids_path)
    assert raw.ch_names == names
    assert raw.get_channel_types() == ["hbo", "hbo"]


@pytest.mark.parametrize("wavelengths", [None, ()], ids=["absent", "empty"])
def test_processed_without_wavelengths_keeps_hb(tmp_path, wavelengths):
    bids_path = _write_bids(tmp_path, *HB, wavelengths=wavelengths)
    raw = _read_or_skip(bids_path)
    assert raw.get_channel_types() == ["hbo", "hbr"]


def test_eegdashraw_restores_nirs_types(tmp_path):
    """EEGDash's own read path (local record, nothing downloaded) keeps hbo/hbr."""
    root = tmp_path / "ds-nirs"
    _write_bids(root, *HB)
    record = {
        "dataset": "ds-nirs",
        "bids_relpath": RELPATH,
        "datatype": "nirs",
        "suffix": "nirs",
        "extension": ".snirf",
        "storage": {"backend": "local", "base": str(root), "raw_key": RELPATH},
        "entities": {"subject": "01", "task": "rest"},
        "entities_mne": {"subject": "01", "task": "rest"},
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        raw = EEGDashRaw(record, cache_dir=str(tmp_path)).raw
    assert raw.get_channel_types() == ["hbo", "hbr"]
    expected = mne.io.read_raw_snirf(root / RELPATH).get_data()
    np.testing.assert_array_equal(raw.get_data(), expected)

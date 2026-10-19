"""eegdash.io.edf: EDF/BDF headers MNE 1.13's reader refuses, read under tolerant().

Each file is written byte by byte here, valid except for the start time, the way
on003555, on003775, on004381 and on004752 store it (``10.47.60``), or for the
header-size field (on003343).
"""

import json
import logging
import warnings
from datetime import UTC, datetime

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, read_raw_bids

from eegdash.io import get_reader, tolerant

LABELS = ["Cz", "Pz"]
N_RECORDS, N_SAMPLES = 3, 100  # 1-s records at 100 Hz


def _field(value, width):
    return str(value).ljust(width).encode("latin-1")[:width]


def _write(path, *, time="10.47.59", recording="X", header_bytes=None):
    """Write a 2-channel EDF (or BDF, from the suffix) with start ``29.10.24 time``."""
    bdf = path.suffix == ".bdf"
    ns, nbytes = len(LABELS), 3 if bdf else 2
    dmax = 2 ** (8 * nbytes - 1) - 1
    header = b"".join(
        [
            b"\xffBIOSEMI" if bdf else _field("0", 8),
            _field("X X X X", 80),
            _field(recording, 80),
            _field("29.10.24", 8),
            _field(time, 8),
            _field(header_bytes or 256 * (ns + 1), 8),
            _field("24BIT" if bdf else "", 44),
            _field(N_RECORDS, 8),
            _field(1, 8),
            _field(ns, 4),
        ]
    )
    header += b"".join(_field(label, 16) for label in LABELS)
    for width, value in [
        (80, ""),  # transducer
        (8, "uV"),
        (8, -3200),
        (8, 3200),
        (8, -dmax - 1),
        (8, dmax),
        (80, ""),  # prefiltering
        (8, N_SAMPLES),
        (32, ""),
    ]:
        header += _field(value, width) * ns
    samples = np.random.default_rng(0).integers(-1000, 1000, (N_RECORDS, ns, N_SAMPLES))
    data = samples.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :nbytes].tobytes()
    path.write_bytes(header + data)
    return path


@pytest.mark.parametrize("ext", [".edf", ".bdf"])
@pytest.mark.parametrize("recording", ["X", "Startdate 29-OCT-2024 X X X"])
def test_start_time_with_60_seconds(tmp_path, ext, recording):
    fname = _write(tmp_path / f"bad{ext}", time="10.47.60", recording=recording)
    good = _write(tmp_path / f"good{ext}", time="10.47.59", recording=recording)
    mne_reader = getattr(mne.io, f"read_raw_{ext[1:]}")
    with pytest.raises(ValueError, match="second must be in 0..59"):
        mne_reader(fname)
    ref = mne_reader(good, preload=True)
    assert ref.info["meas_date"] == datetime(2024, 10, 29, 10, 47, 59, tzinfo=UTC)

    with pytest.warns(RuntimeWarning, match="Invalid measurement time"):
        raw = get_reader(ext)(fname, preload=True)
    with tolerant(), pytest.warns(RuntimeWarning, match="Invalid measurement time"):
        raw_bids_path = mne_reader(fname, preload=True)  # what read_raw_bids calls

    for r in (raw, raw_bids_path):
        assert r.info["meas_date"] is None  # never a made-up date or time
        assert r.ch_names == LABELS
        np.testing.assert_array_equal(r.get_data(), ref.get_data())
    assert get_reader(ext)(good).info["meas_date"] == ref.info["meas_date"]


OVERLAP = 150  # header bytes written over the first data bytes (on003343: 768, 1024)


def _write_overlapped(path):
    """on003343's layout: the data starts at the header-size field (618 here), and
    the full 768-byte header was then written over its first 150 bytes.
    """
    good = _write(path.with_name(f"good{path.suffix}")).read_bytes()
    head, data = good[:768], good[768:]
    head = head[:184] + _field(768 - OVERLAP, 8) + head[192:]
    path.write_bytes(head + data[OVERLAP:])
    return path


@pytest.mark.parametrize("ext", [".edf", ".bdf"])
def test_header_written_over_data_reads_with_corrected_header(tmp_path, ext, caplog):
    fname = _write_overlapped(tmp_path / f"bad{ext}")
    mne_reader = getattr(mne.io, f"read_raw_{ext[1:]}")
    ref = mne_reader(tmp_path / f"good{ext}", preload=True)
    with pytest.raises(AssertionError):
        mne_reader(fname)

    match = "the header covers the first"
    with caplog.at_level(logging.WARNING, logger="eegdash"):
        raw = get_reader(ext)(fname, preload=True)
        assert caplog.text.count(match) == 1
        with tolerant():
            raw_bids_path = mne_reader(fname, preload=True)  # what read_raw_bids calls
        assert caplog.text.count(match) == 2

    n_bad = -(-OVERLAP // (3 if ext == ".bdf" else 2))  # samples of Cz under the header
    for r in (raw, raw_bids_path):
        assert r.n_times == N_RECORDS * N_SAMPLES  # every record, no padding
        np.testing.assert_array_equal(r.get_data()[1], ref.get_data()[1])
        np.testing.assert_array_equal(
            r.get_data()[0, n_bad:], ref.get_data()[0, n_bad:]
        )
        (bad,) = r.annotations
        assert bad["description"] == "BAD_header_overlap"
        assert bad["ch_names"] == ("Cz",)
        assert (bad["onset"], bad["duration"]) == (0, n_bad / 100)

    # The file NEMAR's PR (nemarDatasets/on003343#3) writes: same bytes, fixed header.
    fixed = tmp_path / f"fixed{ext}"
    head = fname.read_bytes()
    fixed.write_bytes(
        head[:184] + _field(768, 8) + head[192:768] + head[768 - OVERLAP :]
    )
    caplog.clear()
    with warnings.catch_warnings(), caplog.at_level(logging.WARNING, logger="eegdash"):
        warnings.simplefilter("error")
        raw_fixed = get_reader(ext)(fixed, preload=True)
    assert caplog.text == ""
    np.testing.assert_array_equal(raw_fixed.get_data(), raw.get_data())
    assert len(raw_fixed.annotations) == 0


def test_header_overlap_annotation_survives_read_raw_bids(tmp_path, caplog):
    eeg = tmp_path / "sub-01" / "eeg"
    eeg.mkdir(parents=True)
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"Name": "x", "BIDSVersion": "1.8.0"})
    )
    (eeg / "sub-01_task-x_eeg.json").write_text(
        json.dumps(
            {"TaskName": "x", "SamplingFrequency": 100, "PowerLineFrequency": 50}
        )
    )
    (eeg / "sub-01_task-x_events.tsv").write_text(
        "onset\tduration\ttrial_type\n1.0\t0.0\tgo\n"
    )
    _write_overlapped(eeg / "sub-01_task-x_eeg.bdf")
    bp = BIDSPath(
        subject="01",
        task="x",
        datatype="eeg",
        suffix="eeg",
        extension=".bdf",
        root=tmp_path,
    )
    with tolerant(), warnings.catch_warnings(), caplog.at_level(logging.WARNING):
        warnings.simplefilter("ignore")
        raw = read_raw_bids(bp, verbose="ERROR")
    assert sorted(raw.annotations.description) == ["BAD_header_overlap", "go"]
    assert "the header covers the first" in caplog.text  # seen at verbose="ERROR"


@pytest.mark.parametrize("ext", [".edf", ".bdf"])
def test_header_size_mismatch_says_why(tmp_path, ext):
    # Header-size field 512 for a 768-byte header, but the data starts at 768: the
    # file size does not show where the data starts, so the file is still refused.
    fname = _write(tmp_path / f"bad{ext}", header_bytes=512)
    mne_reader = getattr(mne.io, f"read_raw_{ext[1:]}")
    with pytest.raises(AssertionError):
        mne_reader(fname)

    match = "Bad EDF/BDF header: 2 signals take 768 bytes, but the header-size field says 512"
    with pytest.raises(ValueError, match=match):
        get_reader(ext)(fname)

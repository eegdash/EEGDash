"""eegdash.io.eeglab, files laid out like the catalogue's (mne/io/eeglab/tests style).

``mne.io.read_raw_eeglab`` fails on each file; ours reads it.
"""

from __future__ import annotations

import h5py
import mne
import numpy as np
import pytest
from scipy import io

from eegdash.io import get_reader
from eegdash.io.eeglab import _dol_to_lod, read_raw_eeglab

CH_NAMES = ["Fz", "Cz", "Pz"]
N_TIMES, SFREQ = 20, 128.0
DATA = np.arange(len(CH_NAMES) * N_TIMES, dtype=np.float32).reshape(-1, N_TIMES)


def _char(parent, name, text):
    """MATLAB v7.3 char array: uint16 codes stored (n, 1)."""
    d = parent.create_dataset(name, data=np.array([[ord(c)] for c in text], "u2"))
    d.attrs["MATLAB_class"] = np.bytes_("char")


def _double(parent, name, value):
    d = parent.create_dataset(name, data=np.array([[value]], float))
    d.attrs["MATLAB_class"] = np.bytes_("double")


def _write_set_v73(fname):
    """EEGLAB .set as MATLAB v7.3 (HDF5) with an external .fdt (on004166).

    ``chanlocs.type`` and ``chanlocs.ref`` are empty on every channel, which
    MATLAB stores as one empty array; pymatreader returns it as such.
    """
    DATA.T.astype("<f4").tofile(fname.with_suffix(".fdt"))
    with h5py.File(fname, "w", userblock_size=512) as f:
        for key, value in dict(
            nbchan=len(CH_NAMES), pnts=N_TIMES, trials=1, srate=SFREQ, xmin=0.0
        ).items():
            _double(f, key, value)
        _char(f, "data", fname.with_suffix(".fdt").name)
        refs = f.create_group("#refs#")
        chanlocs = f.create_group("chanlocs")
        chanlocs.attrs["MATLAB_class"] = np.bytes_("struct")
        fields = dict(
            labels=CH_NAMES, X=[80.0, 0.0, -80.0], Y=[0.0] * 3, Z=[40.0, 85.0, 40.0]
        )
        for field, values in fields.items():
            write = _char if field == "labels" else _double
            for i, value in enumerate(values):
                write(refs, f"{field}{i}", value)
            d = chanlocs.create_dataset(
                field,
                data=[[refs[f"{field}{i}"].ref] for i in range(len(values))],
                dtype=h5py.ref_dtype,
            )
            d.attrs["MATLAB_class"] = np.bytes_("cell")
        for field in ("type", "ref"):  # empty on every channel
            d = chanlocs.create_dataset(field, data=np.zeros(2, "u8"))
            d.attrs["MATLAB_class"] = np.bytes_("double")
            d.attrs["MATLAB_empty"] = 1
    with open(fname, "r+b") as fid:  # the header that marks v7.3 for scipy
        fid.write(b"MATLAB 7.3 MAT-file".ljust(116) + bytes(8) + b"\x00\x02IM")
    return fname


def _write_set_v5(fname, event):
    """EEGLAB .set as MATLAB v5 with embedded data (on008496)."""
    chanlocs = np.array(
        [(name.encode(),) for name in CH_NAMES], dtype=[("labels", "S10")]
    )
    eeg = dict(
        nbchan=len(CH_NAMES),
        pnts=N_TIMES,
        trials=1,
        srate=SFREQ,
        xmin=0.0,
        data=DATA,
        chanlocs=chanlocs,
        event=event,
        epoch=[],
    )
    io.savemat(fname, {"EEG": eeg}, appendmat=False, oned_as="row")
    return fname


def test_io_set_empty_chanlocs_fields(tmp_path):
    """on004166: chanlocs fields empty on every channel (v7.3)."""
    fname = _write_set_v73(tmp_path / "test_v73.set")
    try:  # MNE's outcome depends on the (hash-ordered) field order
        assert mne.io.read_raw_eeglab(fname).ch_names != CH_NAMES
    except IndexError:
        pass
    raw = read_raw_eeglab(fname)
    assert raw.ch_names == CH_NAMES
    assert raw.info["sfreq"] == SFREQ
    np.testing.assert_allclose(raw.get_data(), DATA * 1e-6)  # EEGLAB stores µV
    assert raw.get_montage() is not None  # X/Y/Z still read


@pytest.mark.parametrize("first", ["labels", "type"])
def test_dol_to_lod_empty_field_any_order(first):
    """Whichever field comes first, every channel keeps its label."""
    fields = {"labels": CH_NAMES, "type": np.array([])}
    chanlocs = _dol_to_lod({first: fields[first], **fields})
    assert [ch["labels"] for ch in chanlocs] == CH_NAMES
    assert all(ch["type"].size == 0 for ch in chanlocs)


def test_io_set_event_without_type(tmp_path):
    """on008496: one event, latency only."""
    fname = _write_set_v5(tmp_path / "test_event.set", {"latency": 5.0})
    with pytest.raises(AttributeError, match="type"):
        mne.io.read_raw_eeglab(fname)
    raw = read_raw_eeglab(fname)
    assert list(raw.annotations.description) == ["n/a"]
    np.testing.assert_allclose(raw.annotations.onset, [4.0 / SFREQ])


def test_io_set_typed_events_unchanged(tmp_path):
    """Files MNE already reads come out the same."""
    fname = _write_set_v5(
        tmp_path / "ok.set", {"latency": [5.0, 9.0], "type": ["a", "b"]}
    )
    ours, theirs = read_raw_eeglab(fname), mne.io.read_raw_eeglab(fname)
    assert list(ours.annotations.description) == ["a", "b"]
    assert list(ours.annotations.description) == list(theirs.annotations.description)
    np.testing.assert_array_equal(ours.annotations.onset, theirs.annotations.onset)


def _events(*events):
    """EEG.event as a MATLAB struct array; ``None`` leaves every field empty."""
    out = np.zeros(len(events), dtype=[("type", "O"), ("latency", "O")])
    for i, event in enumerate(events):
        out[i] = (np.array([]), np.array([])) if event is None else event
    return out


def test_io_set_event_empty_latency(tmp_path):
    """on004350: a trailing event with every field empty."""
    fname = _write_set_v5(tmp_path / "empty.set", _events(("a", 5.0), ("b", 9.0), None))
    with pytest.raises(ValueError, match="inhomogeneous"):
        mne.io.read_raw_eeglab(fname)
    with pytest.warns(RuntimeWarning, match="1 EEGLAB event"):
        raw = read_raw_eeglab(fname)
    assert list(raw.annotations.description) == ["a", "b"]
    np.testing.assert_allclose(raw.annotations.onset, [4.0 / SFREQ, 8.0 / SFREQ])


def test_io_set_event_negative_latency(tmp_path):
    """on004851: one event 1.3e9 samples before the recording."""
    fname = _write_set_v5(
        tmp_path / "neg.set", _events((-1, -1288972516.33), (7, 5.0), (8, 9.0))
    )
    with pytest.raises(ValueError, match="negative"):
        mne.io.read_raw_eeglab(fname)
    with pytest.warns(RuntimeWarning, match="1 EEGLAB event"):
        raw = read_raw_eeglab(fname)
    assert list(raw.annotations.description) == ["7", "8"]
    np.testing.assert_allclose(raw.get_data(), DATA * 1e-6)


def test_eeglab_in_table():
    assert get_reader(".set") is read_raw_eeglab

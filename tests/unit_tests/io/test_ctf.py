"""eegdash.io.ctf: the EEG position file of a CTF ``.ds`` (mne/io/ctf/tests style).

on004278 (30 runs): ``<name>.eeg`` inside the ``.ds`` holds only ``n/a``.
"""

import pytest
from mne.io.ctf import ctf as mne_ctf
from mne.io.ctf import eeg as mne_eeg

from eegdash.io import get_reader, tolerant


def _ds(tmp_path, content):
    ds = tmp_path / "sub-14_task-CuedRecall_run-1_meg.ds"
    ds.mkdir()
    (ds / "sub-14_task-CuedRecall_run-1_meg.eeg").write_bytes(content)
    return str(ds)


def test_na_position_file_is_read_as_absent(tmp_path):
    ds = _ds(tmp_path, b"n/a")
    with pytest.raises(RuntimeError, match="Illegal data in EEG position file"):
        mne_eeg._read_eeg(ds)  # MNE as released
    with tolerant():
        assert mne_ctf._read_eeg(ds) is None  # MNE's "file not present"
    assert mne_ctf._read_eeg is mne_eeg._read_eeg


def test_real_position_file_still_parsed(tmp_path):
    ds = _ds(tmp_path, b"1 EEG001 1.0 2.0 3.0\n")
    with tolerant():
        eeg = mne_ctf._read_eeg(ds)
        assert mne_ctf._read_eeg is not mne_eeg._read_eeg
    assert eeg["labels"] == ["EEG001"]


def test_other_illegal_content_still_raises(tmp_path):
    ds = _ds(tmp_path, b"garbage")
    with tolerant():
        assert mne_ctf._read_eeg is not mne_eeg._read_eeg
        with pytest.raises(RuntimeError, match="Illegal data"):
            mne_ctf._read_eeg(ds)


def test_direct_ctf_reader_runs_under_the_fix(tmp_path, monkeypatch):
    """``_load_raw_direct`` (no MNE-BIDS) gets the fix too."""
    import mne

    ds = _ds(tmp_path, b"n/a\n")
    seen = []
    monkeypatch.setattr(
        mne.io, "read_raw_ctf", lambda d, *a, **k: seen.append(mne_ctf._read_eeg(d))
    )
    get_reader(".ds")(ds, preload=False)
    assert seen == [None]

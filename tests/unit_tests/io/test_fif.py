"""eegdash.io.fif: a FIF file cut inside its last data buffer reads up to the cut.

The file is written by MNE (1-s buffers), then cut halfway through its last data
buffer, the way on004012 (sub-19 stim04, sub-22 stim02) and on004346 (sub-01
run-01) are stored: MNE opens it and counts the cut buffer, reading it fails.
"""

import struct
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

SFREQ, N_CH, N_SEC = 100.0, 3, 10
FIFF_DATA_BUFFER, FIFFT_FLOAT = 300, 4


def _data():
    return np.random.default_rng(0).standard_normal((N_CH, int(SFREQ * N_SEC))) * 1e-6


def _write(path):
    info = mne.create_info([f"EEG{i:03d}" for i in range(N_CH)], SFREQ, "eeg")
    mne.io.RawArray(_data(), info, verbose="error").save(
        path, buffer_size_sec=1.0, fmt="single", overwrite=True, verbose="error"
    )
    return path


def _cut_last_buffer(path):
    """Cut the file halfway through the data of its last FIFF_DATA_BUFFER tag."""
    blob = path.read_bytes()
    nbytes = int(SFREQ) * N_CH * 4
    pos = blob.rfind(struct.pack(">iii", FIFF_DATA_BUFFER, FIFFT_FLOAT, nbytes))
    assert pos > 0
    path.write_bytes(blob[: pos + 16 + nbytes // 2])
    return path


@pytest.fixture
def cut_fif(tmp_path):
    return _cut_last_buffer(_write(tmp_path / "sub-01_task-x_meg.fif"))


def test_cut_file_fails_on_its_last_sample(cut_fif):
    raw = mne.io.read_raw_fif(cut_fif, preload=False, verbose="error")
    assert raw.n_times == SFREQ * N_SEC  # MNE counts the cut buffer
    with pytest.raises(ValueError):
        raw.get_data(start=raw.n_times - 1, stop=raw.n_times)


def test_crop_keeps_the_complete_buffers(cut_fif):
    from eegdash.io.fif import crop_unreadable_tail

    raw = mne.io.read_raw_fif(cut_fif, preload=False, verbose="error")
    with pytest.warns(RuntimeWarning, match="runs past the end of the file"):
        out = crop_unreadable_tail(raw)
    assert out.n_times == SFREQ * (N_SEC - 1)
    np.testing.assert_allclose(
        out.get_data(), _data()[:, : out.n_times], rtol=1e-6, atol=1e-12
    )


def test_intact_or_non_fif_is_left_alone(tmp_path):
    from eegdash.io.fif import crop_unreadable_tail

    raw = mne.io.read_raw_fif(_write(tmp_path / "ok_meg.fif"), verbose="error")
    assert crop_unreadable_tail(raw) is None
    info = mne.create_info(["a"], SFREQ, "eeg")
    assert crop_unreadable_tail(mne.io.RawArray(np.zeros((1, 10)), info)) is None


def test_ensure_raw_reads_the_cut_file(tmp_path, cut_fif):
    """EEGDashRaw used to refuse the recording ("Data file unreadable")."""
    from eegdash.dataset.base import EEGDashRaw

    relpath = "sub-01/meg/sub-01_task-x_meg.fif"
    (tmp_path / "ds1" / "sub-01" / "meg").mkdir(parents=True)
    record = {
        "dataset": "ds1",
        "bids_relpath": relpath,
        "storage": {"backend": "local", "base": str(tmp_path / "ds1")},
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))
    ds._download_required_files = MagicMock()
    raw = mne.io.read_raw_fif(cut_fif, preload=False, verbose="error")
    ds._load_raw = MagicMock(return_value=raw)
    with pytest.warns(RuntimeWarning, match="runs past the end of the file"):
        ds._ensure_raw()
    assert ds._raw.n_times == SFREQ * (N_SEC - 1)
    last = ds._raw.get_data(start=ds._raw.n_times - int(SFREQ))
    assert np.isfinite(last).all()

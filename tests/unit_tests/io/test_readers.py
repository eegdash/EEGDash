"""eegdash.io.READERS: the one list of what EEGDash reads (mne/io/tests style)."""

from pathlib import Path

import pytest
from mne_bids.config import reader as mne_bids_reader

from eegdash.dataset.dataset import _VALID_DATA_EXTENSIONS
from eegdash.dataset.io import _load_raw_direct
from eegdash.io import READERS


def test_table_covers_mne_bids_and_feeds_the_client():
    assert {ext.lower() for ext in mne_bids_reader} | {".nwb"} == set(READERS)
    assert _VALID_DATA_EXTENSIONS == frozenset(READERS)
    assert {status for *_, status in READERS.values()} <= {
        "mne",
        "tolerant",
        "vendored",
    }


def test_vendored_reader_yields_to_mne(monkeypatch):
    import mne

    from eegdash.io import get_reader, nwb

    assert get_reader(".nwb") is nwb.read_raw_nwb
    monkeypatch.setattr(mne.io, "read_raw_nwb", len, raising=False)
    assert get_reader(".nwb") is len


def test_direct_reader_unknown_extension():
    with pytest.raises(ValueError, match="Supported: .*'.snirf'"):
        _load_raw_direct(Path("x.xyz"))

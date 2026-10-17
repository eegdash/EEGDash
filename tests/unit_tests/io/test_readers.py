"""eegdash.io.READERS: the one list of what EEGDash reads (mne/io/tests style)."""

import threading
from pathlib import Path

import pytest
from mne_bids.config import reader as mne_bids_reader

from eegdash.dataset.dataset import _VALID_DATA_EXTENSIONS
from eegdash.dataset.io import _load_raw_direct
from eegdash.io import READERS, tolerant
from eegdash.io import snirf as snirf_mod


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


def _hold_tolerant(entered, release):
    with tolerant():
        entered.set()
        release.wait()


def test_tolerant_overlapping_threads_restore_mne():
    """Overlapping readers (download_all threads): A enters, B enters, A leaves, B leaves.

    With plain nested mock.patch, A's exit restores MNE while B still reads, and
    B's exit then re-installs the patch for good.
    """
    from mne.io.snirf import _snirf  # noqa: PLC0415

    name = next(iter(snirf_mod._orig))
    original = getattr(_snirf, name)
    ev = {k: threading.Event() for k in ("a_in", "b_in", "a_go", "b_go")}
    a = threading.Thread(target=_hold_tolerant, args=(ev["a_in"], ev["a_go"]))
    b = threading.Thread(target=_hold_tolerant, args=(ev["b_in"], ev["b_go"]))
    a.start()
    ev["a_in"].wait(5)
    b.start()
    ev["b_in"].wait(5)
    try:
        ev["a_go"].set()
        a.join(5)
        assert getattr(_snirf, name) is not original  # B is still reading
    finally:
        ev["b_go"].set()
        b.join(5)
    assert getattr(_snirf, name) is original

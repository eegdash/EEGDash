"""Perf guardrail: the cheap n_times paths are no slower than the MNE reads.

The user requirement is "no slowness vs the existing end-to-end pipeline." The
cheap VHDR path (regex + a single ``stat`` for the ``.eeg`` size) replaces a full
``read_raw_brainvision`` construction, so it must be at least as fast — in
practice ~14x faster on the corpus fixture. Marked ``slow`` + ``network`` (timing
+ corpus fixture); the always-on absolute ceilings live in ``test_perf.py``.
"""

from __future__ import annotations

import statistics
import time

import pytest

import mne
from eegdash.testing import data_file

pytestmark = [pytest.mark.slow, pytest.mark.network]

_VHDR = "eeg/sub-xp101_task-motorloc_eeg.vhdr"


def _median_seconds(fn, rounds: int = 21) -> float:
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _cheap_vhdr_ntimes():
    from _vhdr_parser import parse_vhdr_metadata

    return parse_vhdr_metadata(data_file(_VHDR)).get("n_times")


def _mne_vhdr_ntimes():
    raw = mne.io.read_raw_brainvision(
        str(data_file(_VHDR)), preload=False, verbose=False
    )
    try:
        return int(raw.n_times)
    finally:
        raw.close()


def test_cheap_vhdr_ntimes_not_slower_than_mne():
    # Warm the corpus/file cache so the first-read penalty doesn't skew either path.
    _cheap_vhdr_ntimes()
    _mne_vhdr_ntimes()

    cheap = _median_seconds(_cheap_vhdr_ntimes)
    via_mne = _median_seconds(_mne_vhdr_ntimes)

    # Generous margin (cheap must merely not be slower); empirically ~14x faster.
    assert cheap <= via_mne, (
        f"cheap VHDR n_times ({cheap * 1e3:.3f} ms) is slower than the MNE read "
        f"({via_mne * 1e3:.3f} ms) — the few-bytes path regressed."
    )

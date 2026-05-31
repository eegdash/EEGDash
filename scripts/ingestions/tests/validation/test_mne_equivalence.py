"""D1 contract: cheap n_times equals MNE's n_times within tolerance.

Dev-time certification that the header/file-size arithmetic matches the real
reader. Marked ``network`` + ``slow`` so normal CI skips it; run during
development to certify the formulas before trusting them in production.
"""

from __future__ import annotations

import pytest

import mne
from eegdash.testing import data_file

pytestmark = [pytest.mark.network, pytest.mark.slow]


def _mne_n_times_edf(path) -> int:
    raw = mne.io.read_raw_edf(str(path), preload=False, verbose=False)
    try:
        return int(raw.n_times)
    finally:
        raw.close()


def _mne_n_times_bdf(path) -> int:
    raw = mne.io.read_raw_bdf(str(path), preload=False, verbose=False)
    try:
        return int(raw.n_times)
    finally:
        raw.close()


def _mne_n_times_vhdr(path) -> int:
    raw = mne.io.read_raw_brainvision(str(path), preload=False, verbose=False)
    try:
        return int(raw.n_times)
    finally:
        raw.close()


def test_vhdr_cheap_n_times_matches_mne():
    from _vhdr_parser import parse_vhdr_metadata

    path = data_file("eeg/sub-xp101_task-motorloc_eeg.vhdr")
    cheap = parse_vhdr_metadata(path).get("n_times")
    if cheap is None:
        pytest.skip("VHDR fixture lacked DataPoints and a reachable .eeg size")
    assert cheap == _mne_n_times_vhdr(path)


def test_ieeg_vhdr_cheap_n_times_matches_mne():
    from _vhdr_parser import parse_vhdr_metadata

    path = data_file("ieeg/sub-01_ses-iemu_task-film_acq-clinical_run-1_ieeg.vhdr")
    cheap = parse_vhdr_metadata(path).get("n_times")
    if cheap is None:
        pytest.skip("iEEG VHDR fixture lacked DataPoints and a reachable .eeg size")
    assert cheap == _mne_n_times_vhdr(path)


# EDF/BDF still use the MNE header reader (preload=False is already signal-free;
# a hand-rolled struct reader is a future optimization, and Phase 1 sidecar
# arithmetic already covers the shallow-clone case where the file is an annex
# pointer). So "cheap == MNE" is trivially true *unless* the fixture has real
# data — the current corpus EDF/BDF are header-only stubs (n_times == 0), which
# we skip rather than assert a degenerate equality on.


def test_edf_cheap_n_times_matches_mne():
    from _format_parser_registry import get_parser_for_extension

    path = data_file("eeg/sub-01_ses-01_task-offline_run-01_eeg.edf")
    mne_nt = _mne_n_times_edf(path)
    if mne_nt == 0:
        pytest.skip("EDF corpus fixture is a header-only stub (n_times == 0)")
    cheap = (get_parser_for_extension(".edf")(path) or {}).get("n_times")
    assert cheap == mne_nt


def test_bdf_cheap_n_times_matches_mne():
    from _format_parser_registry import get_parser_for_extension

    path = data_file("eeg/sub-001_ses-01_task-meditation_eeg.bdf")
    mne_nt = _mne_n_times_bdf(path)
    if mne_nt == 0:
        pytest.skip("BDF corpus fixture is a header-only stub (n_times == 0)")
    cheap = (get_parser_for_extension(".bdf")(path) or {}).get("n_times")
    assert cheap == mne_nt

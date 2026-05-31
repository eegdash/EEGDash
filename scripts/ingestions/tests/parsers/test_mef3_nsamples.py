"""MEF3 number_of_samples -> n_times (Phase 3).

MEF3 was the worst-covered format for ntimes (~3.9%). The time-series metadata
section 2 carries ``number_of_samples`` at a spec-fixed +200 bytes from the
sampling_frequency double; we validate it against ``number_of_blocks`` so a
coincidental sfreq match can't ship a wrong value.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eegdash.testing import data_file

_TMET = data_file("ieeg/EKG-000000.tmet")


@pytest.mark.network
def test_tmet_number_of_samples_from_real_fixture():
    from _mef3_parser import _parse_tmet_number_of_samples

    # Real MEF 3.0 fixture: sfreq 2048 Hz, 8_145_048 samples (~3977 s, 3978 blocks).
    assert _parse_tmet_number_of_samples(_TMET, 2048.0) == 8145048


@pytest.mark.network
def test_tmet_number_of_samples_rejects_wrong_sfreq():
    from _mef3_parser import _parse_tmet_number_of_samples

    # A sfreq that does not occur in the file -> no offset match -> None (never guesses).
    assert _parse_tmet_number_of_samples(_TMET, 999.0) is None


@pytest.mark.network
def test_mef3_parse_includes_n_times(tmp_path: Path):
    from _helpers.builders import build_mefd_around_real_tmet

    from _mef3_parser import parse_mef3_metadata

    mefd = build_mefd_around_real_tmet(tmp_path, _TMET, channels=["EKG"])
    out = parse_mef3_metadata(mefd)
    assert out["sampling_frequency"] == 2048.0
    assert out["n_times"] == 8145048
    assert out["nchans"] == 1

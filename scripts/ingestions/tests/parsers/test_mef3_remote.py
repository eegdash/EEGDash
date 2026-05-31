"""MEF3 .tmet parsing over a bytes buffer (remote-ready, Phase RH4.1).

The shipped path-based ``_parse_tmet_sampling_frequency`` /
``_parse_tmet_number_of_samples`` open the file and ``read()`` it. To make a
*remotely-fetched* ``.tmet`` (one ranged GET) parseable, the byte-scanning
logic is factored into ``tmet_sfreq_from_bytes`` / ``tmet_n_times_from_bytes``
that operate on the raw ``bytes`` directly. ``find_first_tmet`` locates the
first ``.tmet`` inside a ``.mefd`` tree without reading content (so it survives
broken git-annex symlinks).

The real fixture (``ieeg/EKG-000000.tmet``) is a production MEF 3.0 file:
sfreq 2048 Hz, 8_145_048 samples (~3977 s, 3978 blocks).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eegdash.testing import data_file

_TMET = data_file("ieeg/EKG-000000.tmet")


def _tmet_bytes() -> bytes:
    return Path(_TMET).read_bytes()


@pytest.mark.network
def test_tmet_sfreq_from_bytes_real_fixture():
    from _mef3_parser import tmet_sfreq_from_bytes

    assert tmet_sfreq_from_bytes(_tmet_bytes()) == 2048.0


@pytest.mark.network
def test_tmet_n_times_from_bytes_real_fixture():
    from _mef3_parser import tmet_n_times_from_bytes

    assert tmet_n_times_from_bytes(_tmet_bytes(), 2048.0) == 8145048


@pytest.mark.network
def test_tmet_n_times_from_bytes_rejects_wrong_sfreq():
    from _mef3_parser import tmet_n_times_from_bytes

    # A sfreq absent from the file -> no offset matches -> None (never guesses).
    assert tmet_n_times_from_bytes(_tmet_bytes(), 999.0) is None


@pytest.mark.network
def test_tmet_n_times_from_bytes_truncated_returns_none():
    from _mef3_parser import tmet_n_times_from_bytes

    # Truncated below the minimum header size -> None, never raises.
    truncated = _tmet_bytes()[:512]
    assert tmet_n_times_from_bytes(truncated, 2048.0) is None


@pytest.mark.network
def test_tmet_sfreq_from_bytes_truncated_returns_none():
    from _mef3_parser import tmet_sfreq_from_bytes

    truncated = _tmet_bytes()[:512]
    assert tmet_sfreq_from_bytes(truncated) is None


def test_tmet_n_times_from_bytes_none_sfreq():
    from _mef3_parser import tmet_n_times_from_bytes

    # None / non-positive sfreq -> None without touching the buffer.
    assert tmet_n_times_from_bytes(b"\x00" * 2000, None) is None
    assert tmet_n_times_from_bytes(b"\x00" * 2000, 0.0) is None


def test_find_first_tmet_locates_nested_path(tmp_path: Path):
    from _mef3_parser import find_first_tmet

    mefd = tmp_path / "sub-X_ieeg.mefd"
    seg = mefd / "EKG.timd" / "EKG-000000.segd"
    seg.mkdir(parents=True)
    tmet = seg / "EKG-000000.tmet"
    tmet.write_bytes(b"")  # content irrelevant; located by name, not read

    assert find_first_tmet(mefd) == tmet


def test_find_first_tmet_survives_broken_symlink(tmp_path: Path):
    from _mef3_parser import find_first_tmet

    mefd = tmp_path / "rec.mefd"
    seg = mefd / "C1.timd" / "C1-000000.segd"
    seg.mkdir(parents=True)
    tmet = seg / "C1-000000.tmet"
    # A broken git-annex-style symlink: target does not exist. find_first_tmet
    # must NOT read it, so it still returns the path.
    tmet.symlink_to(tmp_path / ".git" / "annex" / "objects" / "missing")

    assert find_first_tmet(mefd) == tmet


def test_find_first_tmet_returns_none_when_absent(tmp_path: Path):
    from _mef3_parser import find_first_tmet

    mefd = tmp_path / "empty.mefd"
    (mefd / "C1.timd" / "C1-000000.segd").mkdir(parents=True)

    assert find_first_tmet(mefd) is None


def test_find_first_tmet_never_raises_on_missing_dir(tmp_path: Path):
    from _mef3_parser import find_first_tmet

    assert find_first_tmet(tmp_path / "does-not-exist.mefd") is None

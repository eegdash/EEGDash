"""Real-data SNIRF parser test.

Mirrors ``test_mef3_real_fixture.py``: validates ``_snirf_parser`` against
a real BIDS ``.snirf`` file from OpenNeuro instead of a synthetic h5py
construction. See Lesson #1 in NEXT-SPRINT-PLAN.md.

The fixture comes from:
  ``s3://openneuro.org/ds007554/sub-001/ses-01/nirs/
    sub-001_ses-01_task-activemotor_nirs.snirf``

License: CC0 (OpenNeuro standard). Size: 748,624 bytes (~731 KB).
Real SNIRF v1.0 HDF5 container — fNIRS recording at ~10 Hz with 32
channels (16 sources × 2 wavelengths, 850 nm / 760 nm).

If the fixture file is missing (e.g., on a fresh checkout), the whole
module skips via ``pytestmark = pytest.mark.skipif(...)`` — same pattern
``test_mef3_real_fixture.py`` uses.

This test surfaced the C5.1 pattern: the synthetic h5py fixture
(``test_snirf_happy_path.py``) validated the parser against itself, but
the real-data file revealed that ``n_times`` was never extracted.
The parser fix lands in the same commit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# The real fixture lives at this path (downloaded from OpenNeuro
# ds007554, CC0). The whole module skips if it's absent.
FIXTURE = Path(__file__).parent / "fixtures" / "fnirs" / "openneuro_real.snirf"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason=(
        f"Real SNIRF fixture missing: {FIXTURE}. Recover with:\n"
        "  cd scripts/ingestions/tests/fixtures && mkdir -p fnirs && \\\n"
        "  curl -L -o fnirs/openneuro_real.snirf \\\n"
        "    'https://s3.amazonaws.com/openneuro.org/ds007554/sub-001/"
        "ses-01/nirs/sub-001_ses-01_task-activemotor_nirs.snirf'"
    ),
)

from _snirf_parser import parse_snirf_metadata

# ─── parse_snirf_metadata on the real fixture ─────────────────────────────


def test_real_snirf_returns_sampling_frequency():
    """The real ds007554 fNIRS recording yields a non-zero sfreq.

    fNIRS typical: 5–50 Hz (slow hemodynamic signal). The fixture is
    ~10 Hz. We don't lock to a single value — just assert reasonable
    range so a different fixture or parser tweak doesn't bite.
    """
    md = parse_snirf_metadata(FIXTURE)
    assert md is not None, "parser returned None on real .snirf"
    assert md.get("sampling_frequency"), (
        "real .snirf must yield non-zero sampling_frequency"
    )
    sf = md["sampling_frequency"]
    assert sf > 0
    # fNIRS sampling rates are slower than EEG; ds007554 is ~10 Hz.
    # Allow 1–200 Hz so a different fixture won't break this test.
    assert 1.0 < sf < 200.0, f"sfreq={sf} outside expected fNIRS range"


def test_real_snirf_returns_nchans():
    """The real .snirf has 32 channels (16 sources × 2 wavelengths)."""
    md = parse_snirf_metadata(FIXTURE)
    assert md is not None
    assert md.get("nchans"), "real .snirf must yield nchans"
    assert md["nchans"] >= 1


def test_real_snirf_returns_n_times():
    """The real .snirf yields the recording length in samples.

    the synthetic h5py fixture
    didn't catch that ``raw.n_times`` (MNE) / ``len(time)`` (h5py
    fallback) were never read. This test pins the fix.
    """
    md = parse_snirf_metadata(FIXTURE)
    assert md is not None
    n_times = md.get("n_times")
    assert n_times is not None, (
        "real .snirf must surface n_times (Task 4 C5.1 fix); "
        f"got keys={sorted(md.keys())}"
    )
    assert n_times > 0


def test_real_snirf_returns_ch_names_matching_nchans():
    """ch_names length must equal nchans on the real recording.

    Real-data cross-check the synthetic fixture can't enforce: if the
    parser silently truncates ch_names (e.g. a buggy index), nchans
    and len(ch_names) drift apart.
    """
    md = parse_snirf_metadata(FIXTURE)
    assert md is not None
    ch_names = md.get("ch_names")
    assert ch_names, "real .snirf must yield ch_names"
    assert isinstance(ch_names, list)
    assert len(ch_names) == md["nchans"], (
        f"ch_names ({len(ch_names)}) != nchans ({md['nchans']})"
    )

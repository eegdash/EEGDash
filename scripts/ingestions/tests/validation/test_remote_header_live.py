"""Live integration check for the opt-in remote header tier (EDF over HTTP Range).

Exercises the REAL path the cascade uses — dataset id from ``bids_root.name``,
``bids_relpath`` via ``relative_to``, ``_remote_header.locate`` URL derivation,
and a budget-capped ranged read — against a real OpenNeuro EDF. This is the
integration the mocked unit tests can't cover (and where the URL-derivation bug
hid). Marked ``network`` + ``slow``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.network, pytest.mark.slow]

# Real OpenNeuro file: 19-ch EDF, 2,254,720 B, 200 Hz, 59,200 samples (verified).
_DS = "ds004577"
_RELPATH = "sub-NORB00084/ses-1/eeg/sub-NORB00084_ses-1_task-EEG_eeg.edf"
_EXPECTED_NTIMES = 59200


class _BidsDatasetStub:
    """Minimal stand-in exposing only ``bidsdir`` (what CascadeContext needs)."""

    def __init__(self, bidsdir: str):
        self.bidsdir = bidsdir


def test_remote_edf_locate_and_ranged_read_few_bytes():
    import _edf_header as eh
    import _remote_header as rh

    size, url = rh.locate({"dataset": _DS, "bids_relpath": _RELPATH})
    assert url and "openneuro.org" in url
    reader = rh.RangeReader(url, block=eh.EDF_HEADER_LEN)
    buf = reader.read(0, eh.EDF_HEADER_LEN)
    # Efficiency contract: only the 256-byte main header is fetched.
    assert reader.bytes_fetched <= 1024
    assert eh.edf_n_times_from_main_header(buf, 200.0) == _EXPECTED_NTIMES


def test_remote_header_step_resolves_real_edf_via_cascade_path(
    tmp_path: Path, monkeypatch
):
    from _metadata_cascade import CascadeContext, CascadeResult, RemoteHeaderStep

    # Mirror the OpenNeuro layout so the cascade derives dataset=ds.../relpath=... and
    # _remote_header.locate builds the real S3 URL.
    bids_root = tmp_path / _DS
    edf = bids_root / _RELPATH
    edf.parent.mkdir(parents=True, exist_ok=True)
    edf.touch()  # local file is never read; the remote path is used.

    ctx = CascadeContext(
        bids_dataset=_BidsDatasetStub(str(bids_root)), bids_file=str(edf)
    )
    result = CascadeResult(sampling_frequency=200.0, nchans=19)

    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "1")
    RemoteHeaderStep().fill(ctx, result)

    assert result.ntimes == _EXPECTED_NTIMES
    assert result.provenance["ntimes"] == "remote_header"

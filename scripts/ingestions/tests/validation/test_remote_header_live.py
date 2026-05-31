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
    import _edf_header as eh  # noqa: PLC0415
    import _remote_header as rh  # noqa: PLC0415

    _size, url = rh.locate({"dataset": _DS, "bids_relpath": _RELPATH})
    assert url
    assert "openneuro.org" in url
    reader = rh.RangeReader(url, block=eh.EDF_HEADER_LEN)
    buf = reader.read(0, eh.EDF_HEADER_LEN)
    # Efficiency contract: only the 256-byte main header is fetched.
    assert reader.bytes_fetched <= 1024
    assert eh.edf_n_times_from_main_header(buf, 200.0) == _EXPECTED_NTIMES


def test_remote_header_step_resolves_real_edf_via_cascade_path(
    tmp_path: Path, monkeypatch
):
    from _metadata_cascade import (  # noqa: PLC0415
        CascadeContext,
        CascadeResult,
        RemoteHeaderStep,
    )

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


def test_remote_mef3_tmet_few_bytes_real():
    # MEF3: one ~16 KB .tmet ranged GET yields n_times; the .tdat is never touched.
    import _mef3_parser as mp  # noqa: PLC0415
    import _parser_utils as pu  # noqa: PLC0415
    import _remote_header as rh  # noqa: PLC0415

    relpath = (
        "ds004624/sub-c04/ses-Functional/ieeg/"
        "sub-c04_ses-Functional_task-Visual_acq-rec_run-9_ieeg.mefd/"
        "CH0.timd/CH0-000000.segd/CH0-000000.tmet"
    ).replace("ds004624/", "", 1)
    _size, url = rh.locate({"dataset": "ds004624", "bids_relpath": relpath})
    assert url
    data = pu.fetch_bytes_from_s3(url, max_bytes=20000)
    assert data
    assert len(data) <= 20000  # the whole .tmet is tiny
    sfreq = mp.tmet_sfreq_from_bytes(data)
    n_times = mp.tmet_n_times_from_bytes(data, sfreq)
    assert sfreq
    assert sfreq > 0
    assert n_times
    assert n_times > 0


def test_remote_snirf_shape_few_bytes_real():
    # SNIRF: h5py over a Range-backed file reads only metadata blocks — a tiny
    # fraction of the (multi-MB) file — to get the time-axis length.
    import _remote_header as rh  # noqa: PLC0415
    import _snirf_parser as sp  # noqa: PLC0415

    relpath = "sub-640/nirs/sub-640_task-BS_run-01_nirs.snirf"
    _size, url = rh.locate({"dataset": "ds006673", "bids_relpath": relpath})
    assert url
    reader = rh.RangeReader(url, budget=4 * 1024 * 1024)
    n_times = sp.snirf_n_times_from_fileobj(reader)
    assert n_times
    assert n_times > 0
    # Efficiency: well under 1 MB of HDF5 metadata fetched (no signal).
    assert reader.bytes_fetched < 1024 * 1024

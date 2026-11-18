"""MEF3 ``.mefd`` recordings on NEMAR that no other source sizes.

on004624 (614 ``.mefd`` iEEG recordings, NEMAR) had no ``ntimes``: the MEF
branch of :class:`RemoteHeaderStep` only asked ``_remote_header.locate``, which
knows OpenNeuro URLs, and was not read in ``missing`` mode at all. The EDF and
SNIRF branches already fall back to the public NEMAR object by annex key; MEF
now does the same, in the last-resort step (after the sidecar arithmetic), so
an exact local or sidecar source still wins.

Real check: the first ``.tmet`` of on004624 sub-c02 run-1, fetched by annex key,
gives 1000 Hz and 125,300 samples.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import _metadata_cascade as mc
from _metadata_cascade import (
    PROV_REMOTE_HEADER,
    CascadeContext,
    CascadeResult,
    RemoteHeaderStep,
)

_KEY = "SHA256E-s16384--" + "b" * 64 + ".tmet"
_MEFD = "sub-c02/ses-x/ieeg/sub-c02_ses-x_run-1_ieeg.mefd"


def _nemar_mefd(tmp_path):
    root = tmp_path / "on004624"
    seg = root / _MEFD / "CH0.timd" / "CH0-000000.segd"
    seg.mkdir(parents=True)
    os.symlink(
        f"../../../../../../.git/annex/objects/x/y/{_KEY}/{_KEY}",
        seg / "CH0-000000.tmet",
    )
    return CascadeContext(SimpleNamespace(bidsdir=str(root)), str(root / _MEFD))


def _wire(monkeypatch, n_times=125300):
    monkeypatch.setattr(mc._remote_header, "locate", lambda _rec: (None, None))
    fetch = MagicMock(return_value=b"\x01" * 16384)
    monkeypatch.setattr(mc._parser_utils, "fetch_bytes_from_s3", fetch)
    monkeypatch.setattr(mc._mef3_parser, "tmet_sfreq_from_bytes", lambda _d: 1000.0)
    monkeypatch.setattr(
        mc._mef3_parser, "tmet_n_times_from_bytes", lambda _d, _s: n_times
    )
    return fetch


def test_last_resort_reads_nemar_mef_by_annex_key(monkeypatch, tmp_path):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    fetch = _wire(monkeypatch)
    result = CascadeResult(sampling_frequency=1000.0)
    RemoteHeaderStep(last_resort=True).fill(_nemar_mefd(tmp_path), result)
    assert result.ntimes == 125300
    assert result.provenance["ntimes"] == PROV_REMOTE_HEADER
    url = fetch.call_args.args[0]
    assert url == f"https://nemar.s3.amazonaws.com/on004624/objects/{_KEY}"


def test_last_resort_leaves_sized_mef_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    fetch = _wire(monkeypatch)
    result = CascadeResult(sampling_frequency=1000.0, ntimes=42)
    RemoteHeaderStep(last_resort=True).fill(_nemar_mefd(tmp_path), result)
    assert result.ntimes == 42
    fetch.assert_not_called()


def test_mef_still_not_read_when_remote_headers_off(monkeypatch, tmp_path):
    monkeypatch.delenv("EEGDASH_REMOTE_HEADERS", raising=False)
    monkeypatch.setattr(
        mc._parser_utils,
        "fetch_bytes_from_s3",
        Mock(side_effect=AssertionError("fetched")),
    )
    result = CascadeResult(sampling_frequency=1000.0)
    RemoteHeaderStep(last_resort=True).fill(_nemar_mefd(tmp_path), result)
    assert result.ntimes is None

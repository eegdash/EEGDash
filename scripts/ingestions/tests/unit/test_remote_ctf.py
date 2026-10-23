"""CTF ``.ds`` recordings on NEMAR that no other source sizes.

A CTF dataset is a directory: ``<stem>.res4`` (~3 MB header: samples per
trial, trials, channels, rate) and ``<stem>.meg4`` (8-byte header + int32
samples). On a shallow clone both are annexed symlinks. The last-resort step
fetches only the ``.res4`` and accepts its ``nsamp x trials`` when every
``.meg4`` annex size holds a whole number of trials, the check
``mne.io.read_raw_ctf`` makes before reading (``_get_sample_info``).

Real values: on002761 sub-311 run-01, res4 nsamp 540,000 x 1 trial, 414
channels, 600 Hz; ``.meg4`` 894,240,008 B = 8 + 540,000 x 414 x 4. on004398 and
on002001 fail that check (MNE refuses them too) and stay unsized.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import _metadata_cascade as mc
from _metadata_cascade import (
    PROV_REMOTE_HEADER,
    CascadeContext,
    CascadeResult,
    RemoteHeaderStep,
)

_DS = "sub-311/meg/sub-311_task-mem_run-01_meg.ds"
_RES4 = {"nsamp": 540000, "nchan": 414, "sfreq": 600.0, "no_trials": 1}


def _link(path, key):
    os.symlink(f"../../../.git/annex/objects/x/y/{key}/{key}", path)


def _ctf(tmp_path, meg4_sizes):
    root = tmp_path / "on002761"
    d = root / _DS
    d.mkdir(parents=True)
    stem = d.name[: -len(".ds")]
    _link(d / f"{stem}.res4", "MD5E-s3341749--" + "d" * 32 + ".res4")
    for i, size in enumerate(meg4_sizes):
        name = f"{stem}.meg4" if i == 0 else f"{stem}.{i}_meg4"
        _link(d / name, f"MD5E-s{size}--" + "e" * 32 + ".meg4")
    return CascadeContext(SimpleNamespace(bidsdir=str(root)), str(d))


class _Fetch:
    """Records fetched URLs and returns a stand-in .res4 body."""

    def __init__(self):
        self.urls = []

    def __call__(self, url, **_kw):
        self.urls.append(url)
        return b"MEG41RS\x00" + b"\x00" * 64


def _wire(monkeypatch, res4=_RES4):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    monkeypatch.setattr(mc._remote_header, "locate", lambda _rec: (None, None))
    fetch = _Fetch()
    monkeypatch.setattr(mc._parser_utils, "fetch_bytes_from_s3", fetch)
    monkeypatch.setattr(mc, "_ctf_res4_from_bytes", lambda _data, _name: dict(res4))
    return fetch.urls


def test_ctf_sized_from_res4_when_meg4_holds_whole_trials(monkeypatch, tmp_path):
    urls = _wire(monkeypatch)
    ctx = _ctf(tmp_path, [8 + 540000 * 414 * 4])
    result = CascadeResult(sampling_frequency=600.0)
    RemoteHeaderStep(last_resort=True).fill(ctx, result)
    assert result.ntimes == 540000
    assert result.provenance["ntimes"] == PROV_REMOTE_HEADER
    assert urls == [
        "https://nemar.s3.amazonaws.com/on002761/objects/MD5E-s3341749--"
        + "d" * 32
        + ".res4"
    ]


def test_ctf_split_meg4_files_are_summed(monkeypatch, tmp_path):
    _wire(monkeypatch, {**_RES4, "nsamp": 1000, "no_trials": 3})
    ctx = _ctf(tmp_path, [8 + 2 * 1000 * 414 * 4, 8 + 1000 * 414 * 4])
    result = CascadeResult()
    RemoteHeaderStep(last_resort=True).fill(ctx, result)
    assert result.ntimes == 3000
    assert result.sampling_frequency == 600.0


def test_ctf_truncated_meg4_stays_unsized(monkeypatch, tmp_path):
    # on004398: 1,035,094,951 B is not 8 + k x (1,080,000 x 414 x 4).
    _wire(monkeypatch, {**_RES4, "nsamp": 1080000, "sfreq": 1200.0})
    ctx = _ctf(tmp_path, [1035094951])
    result = CascadeResult(sampling_frequency=1200.0)
    RemoteHeaderStep(last_resort=True).fill(ctx, result)
    assert result.ntimes is None


def test_ctf_not_read_when_flag_off(monkeypatch, tmp_path):
    _wire(monkeypatch)
    monkeypatch.delenv("EEGDASH_REMOTE_HEADERS")
    ctx = _ctf(tmp_path, [8 + 540000 * 414 * 4])
    result = CascadeResult()
    RemoteHeaderStep(last_resort=True).fill(ctx, result)
    assert result.ntimes is None

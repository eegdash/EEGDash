"""Regressions from the independent QA of #501 (QA-501.md, 2026-10-23).

1. An epoched EEGLAB ``.set`` (trials > 1) in either root layout must not
   publish one epoch's ``pnts`` as the recording's ``ntimes`` through the
   LOCAL parser/cascade path; rate and channel count are still kept.
2. CTF: ``.meg4`` byte divisibility is necessary but not sufficient. With the
   default ``system_clock='truncate'``, ``mne.io.read_raw_ctf`` stops at the
   first zero of the SCLK01 clock channel. The remote CTF sizing must give the
   same count as MNE's ``_get_sample_info`` on the same bytes.
"""

from __future__ import annotations

import os
import struct
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import scipy.io

import _metadata_cascade as mc
import _set_parser
from _metadata_cascade import (
    BinaryParserStep,
    CascadeContext,
    CascadeResult,
    RemoteHeaderStep,
    SizeArithmeticStep,
    derive_duration_seconds,
)
from mne.io.ctf.constants import CTF
from mne.io.ctf.ctf import _get_sample_info

# ---- 1. epoched root-layout .set, local path -------------------------------


def _v73_root(path, trials):
    with h5py.File(path, "w", userblock_size=512) as f:
        for name, value in (
            ("srate", 250.0),
            ("nbchan", 2.0),
            ("pnts", 1500.0),
            ("trials", float(trials)),
        ):
            f[name] = np.array([[value]])
        f["data"] = np.zeros((trials, 1500, 2), dtype="float32")


def _v5_root(path, trials):
    scipy.io.savemat(
        path,
        {
            "srate": 250.0,
            "nbchan": 2.0,
            "pnts": 1500.0,
            "trials": float(trials),
            "data": np.zeros((2, 1500, trials), dtype="float32"),
        },
    )


def _local_cascade(path):
    ctx = CascadeContext(SimpleNamespace(bidsdir=str(path.parent)), str(path))
    result = CascadeResult()
    BinaryParserStep().fill(ctx, result)
    derive_duration_seconds(result)
    return result


@pytest.mark.parametrize("writer", [_v73_root, _v5_root])
def test_epoched_root_set_keeps_rate_but_no_ntimes(tmp_path, writer):
    p = tmp_path / "sub-1_task-x_eeg.set"
    writer(p, trials=40)
    md = _set_parser.parse_set_metadata(p)
    assert md["trials"] == 40
    assert "n_samples" not in md
    assert "duration" not in md
    result = _local_cascade(p)
    assert result.ntimes is None
    assert result.duration_seconds is None
    assert result.sampling_frequency == 250.0
    assert result.nchans == 2


@pytest.mark.parametrize("writer", [_v73_root, _v5_root])
def test_continuous_root_set_still_sized(tmp_path, writer):
    p = tmp_path / "sub-1_task-x_eeg.set"
    writer(p, trials=1)
    result = _local_cascade(p)
    assert result.ntimes == 1500
    assert result.duration_seconds == 6.0


def test_binary_step_ignores_n_samples_of_epoched_parser_output(monkeypatch, tmp_path):
    # Shared boundary: any parser that reports trials > 1 does not size the recording.
    monkeypatch.setattr(
        mc,
        "get_parser_for_extension",
        lambda _e: (
            lambda _p: {
                "sampling_frequency": 250.0,
                "nchans": 2,
                "n_samples": 1500,
                "trials": 40,
            }
        ),
    )
    result = _local_cascade(tmp_path / "x.set")
    assert result.ntimes is None
    assert result.sampling_frequency == 250.0


# ---- 2. CTF system-clock truncation, compared with MNE ---------------------

_NAMES = ["MLC11-4408", CTF.SYSTEM_CLOCK_CH]


def _res4(nsamp):
    return {
        "nsamp": nsamp,
        "nchan": len(_NAMES),
        "sfreq": 600.0,
        "ch_names": list(_NAMES),
        "chs": [{"ch_name": n} for n in _NAMES],
    }


def _meg4_bytes(nsamp, clock_trials):
    """8-byte header, then per trial: each channel's nsamp big-endian int32 block."""
    out = b"MEG41CP\x00"
    for clock in clock_trials:
        out += struct.pack(f">{nsamp}i", *range(1, nsamp + 1))  # data channel
        out += struct.pack(f">{nsamp}i", *clock)  # SCLK01
    return out


def _ctf_dataset(tmp_path, payloads):
    root = tmp_path / "on002761"
    d = root / "sub-1" / "meg" / "sub-1_task-x_meg.ds"
    d.mkdir(parents=True)
    store = {}
    names = ["sub-1_task-x_meg.res4"] + [
        "sub-1_task-x_meg.meg4" if i == 0 else f"sub-1_task-x_meg.{i}_meg4"
        for i in range(len(payloads))
    ]
    sizes = [3000] + [len(p) for p in payloads]
    for name, size, data in zip(names, sizes, [b"", *payloads], strict=True):
        key = (
            f"MD5E-s{size}--"
            + format(len(store), "032x")
            + ("." + name.rsplit(".", 1)[-1] if "res4" in name else "")
        )
        os.symlink(f"../../../.git/annex/objects/x/y/{key}/{key}", d / name)
        store[f"https://nemar.s3.amazonaws.com/on002761/objects/{key}"] = data
    return CascadeContext(SimpleNamespace(bidsdir=str(root)), str(d)), store


class _Serve:
    def __init__(self, store):
        self.store, self.fetched = store, 0

    def __call__(self, url, *, start=0, max_bytes=262144, **_kw):
        data = (
            self.store[url][start : start + max_bytes] if self.store[url] else b"res4"
        )
        self.fetched += len(data)
        return data


def _mne_n_samp(tmp_path, nsamp, payloads):
    total = 0
    for i, data in enumerate(payloads):
        f = tmp_path / f"mne_{i}.meg4"
        f.write_bytes(data)
        n = _get_sample_info(str(f), _res4(nsamp), "truncate")["n_samp"]
        if n == 0:
            break
        total += n
    return total


def _remote_ntimes(monkeypatch, tmp_path, nsamp, payloads):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    monkeypatch.setattr(mc._remote_header, "locate", lambda _rec: (None, None))
    ctx, store = _ctf_dataset(tmp_path, payloads)
    monkeypatch.setattr(mc._parser_utils, "fetch_bytes_from_s3", _Serve(store))
    monkeypatch.setattr(mc, "_ctf_res4_from_bytes", lambda _d, _n: _res4(nsamp))
    result = CascadeResult()
    RemoteHeaderStep(last_resort=True).fill(ctx, result)
    return result.ntimes


@pytest.mark.parametrize(
    "clock_trials",
    [
        [[1, 2, 3, 0, 0]],  # QA case: MNE keeps 3 of 5
        [[1, 2, 3, 4, 5]],  # clean single trial
        [[1, 2, 3, 4, 5], [6, 7, 0, 0, 0]],  # zero in the second trial
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],  # two clean trials
    ],
)
def test_ctf_matches_mne_sample_info(monkeypatch, tmp_path, clock_trials):
    payload = _meg4_bytes(5, clock_trials)
    expected = _mne_n_samp(tmp_path, 5, [payload])
    assert _remote_ntimes(monkeypatch, tmp_path, 5, [payload]) == expected


def test_ctf_split_files_match_mne(monkeypatch, tmp_path):
    payloads = [_meg4_bytes(5, [[1, 2, 3, 4, 5]]), _meg4_bytes(5, [[6, 7, 0, 0, 0]])]
    assert (
        _remote_ntimes(monkeypatch, tmp_path, 5, payloads)
        == _mne_n_samp(tmp_path, 5, payloads)
        == 7
    )


def test_ctf_clock_zero_from_the_start_stays_unsized(monkeypatch, tmp_path):
    # MNE finds no valid sample and raises; never publish a guessed length.
    payload = _meg4_bytes(5, [[0, 0, 0, 0, 0]])
    assert _remote_ntimes(monkeypatch, tmp_path, 5, [payload]) is None


def test_ctf_clock_check_over_budget_stays_unsized(monkeypatch, tmp_path):
    monkeypatch.setattr(mc, "_CTF_CLOCK_MAX_BYTES", 8)
    payload = _meg4_bytes(5, [[1, 2, 3, 4, 5]])
    assert _remote_ntimes(monkeypatch, tmp_path, 5, [payload]) is None


# ---- 3. epoched .set with an external .fdt (QA round 2) --------------------


def _annexed_fdt(set_path, size):
    key = f"MD5E-s{size}--" + "f" * 32 + ".fdt"
    os.symlink(f".git/annex/objects/x/y/{key}/{key}", set_path.with_suffix(".fdt"))


def _parser_then_size_arithmetic(path):
    """The two adjacent production steps, then duration derivation."""
    ctx = CascadeContext(SimpleNamespace(bidsdir=str(path.parent)), str(path))
    result = CascadeResult()
    BinaryParserStep().fill(ctx, result)
    SizeArithmeticStep().fill(ctx, result)
    derive_duration_seconds(result)
    return result


@pytest.mark.parametrize("writer", [_v73_root, _v5_root])
def test_epoched_set_not_resized_from_fdt(tmp_path, writer):
    # 2 ch x 1500 pnts x 40 epochs x 4 B: the flat divide would give 60,000 / 240 s.
    p = tmp_path / "sub-1_task-x_eeg.set"
    writer(p, trials=40)
    _annexed_fdt(p, 2 * 1500 * 40 * 4)
    result = _parser_then_size_arithmetic(p)
    assert result.ntimes is None
    assert result.duration_seconds is None
    assert result.sampling_frequency == 250.0
    assert result.nchans == 2


def test_continuous_fdt_arithmetic_unchanged(monkeypatch, tmp_path):
    # Continuous file whose parser gives rate/channels but no length: .fdt sizes it.
    monkeypatch.setattr(
        mc,
        "get_parser_for_extension",
        lambda _e: lambda _p: {"sampling_frequency": 250.0, "nchans": 2, "trials": 1},
    )
    p = tmp_path / "sub-1_task-x_eeg.set"
    p.write_bytes(b"")
    _annexed_fdt(p, 2 * 6000 * 4)
    result = _parser_then_size_arithmetic(p)
    assert result.ntimes == 6000
    assert result.duration_seconds == 24.0


def test_ctf_split_cap_exhaustion_stays_unsized(monkeypatch, tmp_path):
    monkeypatch.setattr(mc, "_CTF_MAX_SPLITS", 1)
    payloads = [_meg4_bytes(5, [[1, 2, 3, 4, 5]]), _meg4_bytes(5, [[6, 7, 8, 9, 10]])]
    assert _remote_ntimes(monkeypatch, tmp_path, 5, payloads) is None


# ---- 4. full cascade: late sidecar arithmetic (QA round 3) -----------------


class _NoRateBids:
    """EEGBIDSDataset stand-in: the sidecar gives RecordingDuration, no rate."""

    def __init__(self, root):
        self.bidsdir = str(root)

    def get_bids_file_attribute(self, name, _file):
        return 6.0 if name == "duration" else None


def _full_cascade(tmp_path, writer, trials, monkeypatch):
    monkeypatch.delenv("EEGDASH_REMOTE_HEADERS", raising=False)
    root = tmp_path / "localds"
    eeg = root / "sub-1" / "eeg"
    eeg.mkdir(parents=True)
    p = eeg / "sub-1_task-x_eeg.set"
    writer(p, trials=trials)
    (eeg / "sub-1_task-x_eeg.json").write_text(
        '{"RecordingType": "%s", "RecordingDuration": 6}'
        % ("epoched" if trials > 1 else "continuous")
    )
    return mc.MetadataCascade().run(CascadeContext(_NoRateBids(root), str(p)))


@pytest.mark.parametrize("writer", [_v73_root, _v5_root])
def test_full_cascade_epoched_set_gets_no_sidecar_ntimes(tmp_path, writer, monkeypatch):
    # Rate only from the root .set (250 Hz), duration-only sidecar: no ntimes.
    result = _full_cascade(tmp_path, writer, 40, monkeypatch)
    assert result.ntimes is None
    assert result.sampling_frequency == 250.0
    assert result.recording_duration == 6.0  # sidecar metadata kept as is


@pytest.mark.parametrize("writer", [_v73_root, _v5_root])
def test_full_cascade_continuous_set_still_sized(tmp_path, writer, monkeypatch):
    result = _full_cascade(tmp_path, writer, 1, monkeypatch)
    assert result.ntimes == 1500

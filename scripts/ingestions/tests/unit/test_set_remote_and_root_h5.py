"""EEGLAB ``.set`` recordings on NEMAR that no other source sizes.

Two gaps left 8 BCIT datasets (on004105-on004123) and on007822 without ``ntimes``:

1. EEGLAB's MAT-v7.3 ``.set`` (HDF5) stores the EEG fields at the FILE ROOT
   (``/srate``, ``/nbchan``, ``/pnts``, ``/trials``, ``/data``), not under an
   ``EEG`` group; the h5py branch only looked for ``EEG``. on004105 sub-17:
   root ``pnts`` 2,836,480 at ``srate`` 1024, ``data`` (2836480, 74).
2. On a shallow clone the ``.set`` is an annexed symlink, so nothing read it.
   The last-resort remote step now reads it from the public object: HDF5 by
   HTTP range (~0.4 MB of a 674 MB file), a MAT-v5 file whole when it is under
   the loadmat ceiling (on007822: 4.1 MB).

Only continuous files get ``ntimes``: for an epoched ``.set`` (trials > 1)
``pnts`` is one epoch, which would record a single trial as the duration.
"""

from __future__ import annotations

import io
import os
from types import SimpleNamespace

import h5py
import numpy as np
import scipy.io

import _metadata_cascade as mc
import _set_parser
from _metadata_cascade import (
    PROV_REMOTE_HEADER,
    CascadeContext,
    CascadeResult,
    RemoteHeaderStep,
)

_REL = "sub-17/ses-01/eeg/sub-17_ses-01_task-Drive_run-1_eeg.set"


def _h5_root_set(path_or_buf, *, pnts=2836480, srate=1024.0, nbchan=74, trials=1):
    # As MATLAB writes v7.3: a 512-byte "MATLAB 7.3 MAT-file" text block, then HDF5.
    with h5py.File(path_or_buf, "w", userblock_size=512) as f:
        for name, value in (
            ("srate", srate),
            ("nbchan", nbchan),
            ("pnts", pnts),
            ("trials", trials),
        ):
            f[name] = np.array([[value]], dtype="float64")
    head = b"MATLAB 7.3 MAT-file, Platform: GLNXA64".ljust(512, b" ")
    if isinstance(path_or_buf, io.BytesIO):
        path_or_buf.seek(0)
        path_or_buf.write(head)
    else:
        with open(path_or_buf, "r+b") as fh:
            fh.write(head)


def _v5_set_bytes(*, pnts=54000, srate=300.0, nbchan=19, trials=1) -> bytes:
    eeg = {
        "srate": srate,
        "nbchan": nbchan,
        "pnts": pnts,
        "trials": trials,
        "data": np.zeros((nbchan, 2), dtype="float32"),
        "datfile": "",
    }
    buf = io.BytesIO()
    scipy.io.savemat(buf, {"EEG": eeg})
    return buf.getvalue()


def _nemar_set(tmp_path, size: int):
    root = tmp_path / "on004105"
    target = root / _REL
    target.parent.mkdir(parents=True)
    key = f"MD5E-s{size}--" + "c" * 32 + ".set"
    os.symlink(f"../../../../.git/annex/objects/x/y/{key}/{key}", target)
    return CascadeContext(SimpleNamespace(bidsdir=str(root)), str(target)), key


class _Replay:
    """A RangeReader stand-in that records URLs and replays one buffer."""

    def __init__(self, buf):
        self.buf, self.urls = buf, []

    def __call__(self, url, **_kw):
        self.urls.append(url)
        self.buf.seek(0)
        return self.buf


def _boom(*_a, **_k):
    raise AssertionError("must not be read")


# --- 1. local parser: root-level MAT-v7.3 fields --------------------------------


def test_parser_reads_root_level_v73_fields(tmp_path):
    p = tmp_path / "x.set"
    _h5_root_set(p)
    md = _set_parser.parse_set_metadata(p)
    assert md["sampling_frequency"] == 1024.0
    assert md["nchans"] == 74
    assert md["n_samples"] == 2836480
    assert md["trials"] == 1


# --- 2. remote last-resort read ---------------------------------------------------


def test_last_resort_sizes_remote_v73_set(monkeypatch, tmp_path):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    buf = io.BytesIO()
    _h5_root_set(buf)
    monkeypatch.setattr(mc._remote_header, "locate", lambda _rec: (None, None))
    reader = _Replay(buf)
    monkeypatch.setattr(mc._remote_header, "RangeReader", reader)
    ctx, key = _nemar_set(tmp_path, 673811826)
    result = CascadeResult(sampling_frequency=1024.0, nchans=74)
    RemoteHeaderStep(last_resort=True).fill(ctx, result)
    assert result.ntimes == 2836480
    assert result.provenance["ntimes"] == PROV_REMOTE_HEADER
    assert reader.urls == [f"https://nemar.s3.amazonaws.com/on004105/objects/{key}"]


def test_last_resort_skips_epoched_v73_set(monkeypatch, tmp_path):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    buf = io.BytesIO()
    _h5_root_set(buf, pnts=1500, srate=250.0, trials=40)
    monkeypatch.setattr(mc._remote_header, "locate", lambda _rec: (None, None))
    monkeypatch.setattr(mc._remote_header, "RangeReader", lambda *_a, **_k: buf)
    ctx, _key = _nemar_set(tmp_path, 1_000_000)
    result = CascadeResult(sampling_frequency=250.0)
    RemoteHeaderStep(last_resort=True).fill(ctx, result)
    assert result.ntimes is None


def test_last_resort_sizes_small_remote_v5_set(monkeypatch, tmp_path):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    data = _v5_set_bytes()
    monkeypatch.setattr(mc._remote_header, "locate", lambda _rec: (None, None))
    monkeypatch.setattr(
        mc._remote_header, "RangeReader", lambda *_a, **_k: io.BytesIO(data)
    )
    monkeypatch.setattr(mc._parser_utils, "fetch_bytes_from_s3", lambda _u, **_k: data)
    ctx, _key = _nemar_set(tmp_path, len(data))
    result = CascadeResult(sampling_frequency=300.0, nchans=19)
    RemoteHeaderStep(last_resort=True).fill(ctx, result)
    assert result.ntimes == 54000


def test_last_resort_does_not_fetch_large_v5_set(monkeypatch, tmp_path):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    monkeypatch.setattr(mc._remote_header, "locate", lambda _rec: (None, None))
    monkeypatch.setattr(
        mc._remote_header,
        "RangeReader",
        lambda *_a, **_k: io.BytesIO(b"MATLAB 5.0 MAT-file" + b" " * 200),
    )

    monkeypatch.setattr(mc._parser_utils, "fetch_bytes_from_s3", _boom)
    ctx, _key = _nemar_set(tmp_path, 60 * 1024 * 1024)
    result = CascadeResult(sampling_frequency=300.0)
    RemoteHeaderStep(last_resort=True).fill(ctx, result)
    assert result.ntimes is None


def test_set_not_read_remotely_when_flag_off(monkeypatch, tmp_path):
    monkeypatch.delenv("EEGDASH_REMOTE_HEADERS", raising=False)

    monkeypatch.setattr(mc._remote_header, "RangeReader", _boom)
    ctx, _key = _nemar_set(tmp_path, 1000)
    result = CascadeResult(sampling_frequency=300.0)
    RemoteHeaderStep(last_resort=True).fill(ctx, result)
    assert result.ntimes is None


def test_parser_reads_root_level_v5_fields(tmp_path):
    # on007822: MAT-v5 with srate/nbchan/pnts/trials as top-level variables.
    p = tmp_path / "x.set"
    scipy.io.savemat(
        p,
        {
            "srate": 300.0,
            "nbchan": 19.0,
            "pnts": 54000.0,
            "trials": 1.0,
            "data": np.zeros((19, 2), dtype="float32"),
        },
    )
    md = _set_parser.parse_set_metadata(p)
    assert md["sampling_frequency"] == 300.0
    assert md["nchans"] == 19
    assert md["n_samples"] == 54000
    assert md["trials"] == 1

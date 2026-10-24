"""The raw file's header is the truth, even when the sidecar says otherwise (Bruno, 2026-10-21).

Batch6 read-check, records whose sidecar contradicts a self-consistent file:

- on006465: EDF signal headers hold 1000 samples per 1 s record (MNE reads 1000 Hz,
  127 channels); the sidecar says 500 Hz and 126 channels. On a shallow clone the
  digest read only the 256-byte main header and multiplied the records by the
  sidecar rate.
- on005007: the EDF data record lasts 0.219 s (438 samples, 2000 Hz); the main-header
  formula parsed the duration as an integer and gave up.
- on005815: VHDR ``NumberOfChannels=30``; channels.tsv lists 31.
- on007554: SNIRF time vector steps 0.02 s (50 Hz, 10,751 points over 215 s); the
  sidecar says 10 Hz. SNIRF headers were read on a shallow clone only in mode ``1``.
"""

from __future__ import annotations

import io
import os
from types import SimpleNamespace

import numpy as np
import pytest

import _metadata_cascade as mc
from _metadata_cascade import (
    PROV_BINARY_PARSER,
    PROV_REMOTE_HEADER,
    BinaryParserStep,
    CascadeContext,
    CascadeResult,
    MetadataCascade,
    RemoteHeaderStep,
    SidecarArithmeticStep,
)


def _ctx(root, rel: str) -> CascadeContext:
    return CascadeContext(SimpleNamespace(bidsdir=str(root)), str(root / rel))


def _annexed(tmp_path, dataset: str, rel: str, ext: str) -> CascadeContext:
    key = "SHA256E-s1000--" + "e" * 64 + ext
    root = tmp_path / dataset
    (root / rel).parent.mkdir(parents=True)
    depth = "../" * (rel.count("/"))
    os.symlink(f"{depth}.git/annex/objects/Aa/Bb/{key}/{key}", root / rel)
    return _ctx(root, rel)


def _late_steps():
    return [
        s
        for s in MetadataCascade().steps
        if isinstance(s, (RemoteHeaderStep, SidecarArithmeticStep))
    ]


def _edf_header(n_records: int, record_s: str, signals: list[tuple[str, int]]) -> bytes:
    ns = len(signals)
    main = bytearray(b" " * 256)
    main[236:244] = f"{n_records:<8}".encode()
    main[244:252] = f"{record_s:<8}".encode()
    main[252:256] = f"{ns:<4}".encode()
    sig = bytearray(b" " * (256 * ns))
    for i, (label, spr) in enumerate(signals):
        sig[16 * i : 16 * i + 16] = f"{label:<16}".encode()
        off = 216 * ns + 8 * i
        sig[off : off + 8] = f"{spr:<8}".encode()
    return bytes(main + sig)


class _BlobReader:
    """RangeReader stand-in serving an in-memory file."""

    blob = b""

    def __init__(self, url, **_k):
        self.url = url

    def read(self, offset, length):
        return self.blob[offset : offset + length]

    read_once = read


def _serve(monkeypatch, blob: bytes):
    monkeypatch.setattr(_BlobReader, "blob", blob)
    monkeypatch.setattr(mc._remote_header, "RangeReader", _BlobReader)


@pytest.mark.parametrize(
    ("n_records", "record_s", "signals", "expected"),
    [
        pytest.param(
            1224,
            "1",
            [(f"EEG{i}", 1000) for i in range(127)],
            (1000.0, 1224000, 127),
            id="on006465-sidecar-500Hz",
        ),
        pytest.param(
            3900,
            "0.219",
            [(f"EEG{i}", 438) for i in range(69)] + [("EDF Annotations", 57)],
            (2000.0, 1708200, 69),
            id="on005007-fractional-record",
        ),
    ],
)
def test_edf_signal_headers_beat_sidecar(
    monkeypatch, tmp_path, n_records, record_s, signals, expected
):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    _serve(monkeypatch, _edf_header(n_records, record_s, signals))
    ctx = _annexed(tmp_path, "on006465", "sub-17/eeg/sub-17_eeg.edf", ".edf")
    result = CascadeResult(
        sampling_frequency=500.0, nchans=126, recording_duration=1629.0
    )
    for p in ("sampling_frequency", "nchans"):
        result.provenance[p] = "mne_bids"
    for step in _late_steps():
        step.fill(ctx, result)
    assert (result.sampling_frequency, result.ntimes, result.nchans) == pytest.approx(
        expected
    )
    assert result.provenance["sampling_frequency"] == PROV_REMOTE_HEADER
    assert result.provenance["ntimes"] == PROV_REMOTE_HEADER
    assert result.provenance["nchans"] == PROV_REMOTE_HEADER


def test_vhdr_header_channel_count_beats_channels_tsv(monkeypatch, tmp_path):
    monkeypatch.setattr(
        mc,
        "get_parser_for_extension",
        lambda ext: (
            lambda _p: {
                "sampling_frequency": 1000.0,
                "nchans": 30,
                "n_times": 86560,
            }
        ),
    )
    ctx = _ctx(tmp_path / "on005815", "sub-14/ses-1/eeg/sub-14_eeg.vhdr")
    result = CascadeResult(sampling_frequency=1000.0, nchans=31, ch_names=["x"] * 31)
    result.provenance["nchans"] = "mne_bids"
    BinaryParserStep().fill(ctx, result)
    assert result.nchans == 30
    assert result.provenance["nchans"] == PROV_BINARY_PARSER


def _snirf_bytes(n: int, step: float) -> bytes:
    h5py = pytest.importorskip("h5py")
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        d = f.create_group("nirs").create_group("data1")
        d["time"] = np.arange(n) * step
        d["dataTimeSeries"] = np.zeros((n, 2))
    return buf.getvalue()


def test_snirf_time_vector_beats_sidecar_in_missing_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    blob = _snirf_bytes(10751, 0.02)
    monkeypatch.setattr(
        mc._remote_header, "RangeReader", lambda url, **_k: io.BytesIO(blob)
    )
    ctx = _annexed(tmp_path, "on007554", "sub-022/nirs/sub-022_nirs.snirf", ".snirf")
    result = CascadeResult(sampling_frequency=10.0, recording_duration=215.02)
    result.provenance["sampling_frequency"] = "mne_bids"
    for step in _late_steps():
        step.fill(ctx, result)
    assert result.ntimes == 10751
    assert result.sampling_frequency == pytest.approx(50.0, rel=1e-9)
    assert result.provenance["sampling_frequency"] == PROV_REMOTE_HEADER
    assert result.provenance["ntimes"] == PROV_REMOTE_HEADER

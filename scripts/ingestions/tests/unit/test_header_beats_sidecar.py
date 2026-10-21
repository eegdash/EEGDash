"""The catalogue records what the data file holds: header values beat sidecar values.

Batch6 read-check (2026-10-21, develop e129ba5): 5,661 recordings where EEGDash reads a
different sampling rate or length than the catalogue. Two digest causes:

- on007605 (and on006850, on003846, on005397, on006065, on003029, on004980): the
  BrainVision ``SamplingInterval`` is 999.976914 us (1000.0231 Hz) and MNE reads that
  rate, but the cascade kept the sidecar's nominal ``SamplingFrequency: 1000`` because
  the sidecar step writes first. The header rate now replaces a sidecar rate.
  ``SamplingInterval`` is also written in exponent form (on005397 ``2.000018e+03``,
  on004017 ``4.882813e+02``); the parser read only the mantissa.
- on008082 (and ds/on007823, on004577, on007558, ds/on007667, on007666, ds/on007864,
  ds/on007865, on003194, on003195): an EDF holds whole data records of 6-8 s, the
  sidecar ``RecordingDuration`` is the original length (790.52 s), so
  ``sfreq x RecordingDuration`` (158,104) is up to one record more than the file holds
  (131 x 6 s = 157,200 samples). On a shallow clone the header was read only when no
  other source gave ntimes; it is now read before the arithmetic.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import _metadata_cascade as mc
import _parser_utils
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
from _vhdr_parser import parse_vhdr_metadata

_VHDR = """Brain Vision Data Exchange Header File Version 1.0

[Common Infos]
Codepage=UTF-8
DataFile={stem}.eeg
DataFormat=BINARY
DataOrientation=MULTIPLEXED
NumberOfChannels=2
SamplingInterval={interval}

[Binary Infos]
BinaryFormat=IEEE_FLOAT_32

[Channel Infos]
Ch1=Fz,,0.1,µV
Ch2=Cz,,0.1,µV
"""


def _ctx(root, rel: str) -> CascadeContext:
    return CascadeContext(SimpleNamespace(bidsdir=str(root)), str(root / rel))


def _write_vhdr(tmp_path, interval: str, n_samples: int) -> CascadeContext:
    root = tmp_path / "on007605"
    rel = "sub-BA86GM/eeg/sub-BA86GM_task-pvt_eeg.vhdr"
    vhdr = root / rel
    vhdr.parent.mkdir(parents=True)
    stem = vhdr.name[: -len(".vhdr")]
    vhdr.write_text(_VHDR.format(stem=stem, interval=interval), encoding="utf-8")
    (vhdr.parent / f"{stem}.eeg").write_bytes(b"\0" * (n_samples * 2 * 4))
    return _ctx(root, rel)


@pytest.mark.parametrize(
    ("interval", "sfreq"),
    [
        pytest.param("999.976914", 1e6 / 999.976914, id="on007605-decimal"),
        pytest.param("2.000018e+03", 1e6 / 2000.018, id="on005397-exponent"),
        pytest.param("4.882813e+02", 1e6 / 488.2813, id="on004017-exponent"),
        pytest.param("1000", 1000.0, id="integer"),
    ],
)
def test_vhdr_parser_reads_sampling_interval(tmp_path, interval, sfreq):
    ctx = _write_vhdr(tmp_path, interval, 50)
    md = parse_vhdr_metadata(ctx.bids_file_path)
    assert md["sampling_frequency"] == pytest.approx(sfreq, rel=1e-12)
    assert md["n_times"] == 50


def test_vhdr_header_rate_replaces_rounded_sidecar_rate(tmp_path):
    """on007605: sidecar 1000 Hz, header 999.976914 us; MNE reads 1000.0231 Hz."""
    ctx = _write_vhdr(tmp_path, "999.976914", 483540)
    result = CascadeResult(sampling_frequency=1000.0, nchans=2, ch_names=["Fz", "Cz"])
    result.provenance["sampling_frequency"] = "mne_bids"
    BinaryParserStep().fill(ctx, result)
    assert result.sampling_frequency == pytest.approx(1e6 / 999.976914, rel=1e-12)
    assert result.provenance["sampling_frequency"] == PROV_BINARY_PARSER
    assert result.ntimes == 483540
    mc.derive_duration_seconds(result)
    assert result.duration_seconds == pytest.approx(483540 * 999.976914e-6)


def test_nemar_annexed_vhdr_is_read_by_annex_key(monkeypatch, tmp_path):
    """On a NEMAR shallow clone the .vhdr is an annex pointer; fetch it by key."""
    root = tmp_path / "on007605"
    rel = "sub-BA86GM/eeg/sub-BA86GM_task-pvt_eeg.vhdr"
    (root / rel).parent.mkdir(parents=True)
    stem = "sub-BA86GM_task-pvt_eeg"
    vkey = "SHA256E-s5391--" + "c" * 64 + ".vhdr"
    ekey = f"SHA256E-s{483540 * 2 * 4}--" + "d" * 64 + ".eeg"
    for name, key in ((f"{stem}.vhdr", vkey), (f"{stem}.eeg", ekey)):
        os.symlink(
            f"../../.git/annex/objects/Aa/Bb/{key}/{key}",
            root / rel.rsplit("/", 1)[0] / name,
        )
    urls: list[str] = []
    text = _VHDR.format(stem=stem, interval="999.976914")
    monkeypatch.setattr(
        _parser_utils, "fetch_from_s3", lambda url, **_k: urls.append(url) or text
    )
    md = parse_vhdr_metadata(root / rel)
    assert md["sampling_frequency"] == pytest.approx(1e6 / 999.976914, rel=1e-12)
    assert md["n_times"] == 483540
    assert urls[0] == f"https://nemar.s3.amazonaws.com/on007605/objects/{vkey}"


def test_header_rate_equal_to_sidecar_keeps_provenance(tmp_path):
    ctx = _write_vhdr(tmp_path, "1000", 100)
    result = CascadeResult(sampling_frequency=1000.0, nchans=2, ch_names=["Fz", "Cz"])
    result.provenance["sampling_frequency"] = "mne_bids"
    BinaryParserStep().fill(ctx, result)
    assert result.sampling_frequency == 1000.0
    assert result.provenance["sampling_frequency"] == "mne_bids"


# ─── EDF/BDF on a shallow clone ───────────────────────────────────────────

_KEY = "SHA256E-s6922688--" + "b" * 64 + ".edf"
_REL = "sub-CUCOV091/eeg/sub-CUCOV091_task-COVID_eeg.edf"


def _edf_main_header(n_records: int, record_s: int, n_signals: int) -> bytes:
    buf = bytearray(b" " * 256)
    buf[236:244] = f"{n_records:<8}".encode()
    buf[244:252] = f"{record_s:<8}".encode()
    buf[252:256] = f"{n_signals:<4}".encode()
    return bytes(buf)


def _annexed_edf(tmp_path, dataset: str) -> CascadeContext:
    root = tmp_path / dataset
    (root / _REL).parent.mkdir(parents=True)
    os.symlink(f"../../.git/annex/objects/Xx/Yy/{_KEY}/{_KEY}", root / _REL)
    return _ctx(root, _REL)


def _late_steps():
    """The default cascade's steps after the local parsers, in their real order."""
    return [
        s
        for s in MetadataCascade().steps
        if isinstance(s, (RemoteHeaderStep, SidecarArithmeticStep))
    ]


@pytest.mark.parametrize(
    "seed",
    [
        pytest.param({"recording_duration": 790.52}, id="on008082-sidecar"),
        pytest.param(
            {"nemar_ntimes": 158104, "recording_duration": 790.52}, id="nemar-records"
        ),
    ],
)
def test_edf_header_beats_sidecar_arithmetic(monkeypatch, tmp_path, seed):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    reader = Mock()
    reader.read.side_effect = lambda off, n: (
        _edf_main_header(131, 6, 22) if off == 0 else b"EEG Fp1         " * 22
    )
    monkeypatch.setattr(mc._remote_header, "RangeReader", lambda url, **_k: reader)
    ctx = _annexed_edf(tmp_path, "on008082")
    result = CascadeResult(sampling_frequency=200.0, nchans=22, **seed)
    for step in _late_steps():
        step.fill(ctx, result)
    assert result.ntimes == 131 * 6 * 200
    assert result.provenance["ntimes"] == PROV_REMOTE_HEADER


def test_missing_mode_does_not_read_snirf_or_mef_before_arithmetic(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    monkeypatch.setattr(
        mc._remote_header, "RangeReader", Mock(side_effect=AssertionError("fetched"))
    )
    result = CascadeResult(sampling_frequency=10.0, recording_duration=100.0)
    ctx = _ctx(tmp_path / "on007554", "sub-1/nirs/sub-1_nirs.snirf")
    for step in _late_steps():
        step.fill(ctx, result)
    assert result.ntimes == 1000
    assert result.provenance["ntimes"] == "sidecar_arithmetic"


def test_exact_ntimes_is_not_refetched(monkeypatch, tmp_path):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    monkeypatch.setattr(
        mc._remote_header, "RangeReader", Mock(side_effect=AssertionError("fetched"))
    )
    ctx = _annexed_edf(tmp_path, "on008082")
    result = CascadeResult(sampling_frequency=200.0, ntimes=157200, nchans=22)
    for step in _late_steps():
        step.fill(ctx, result)
    assert result.ntimes == 157200

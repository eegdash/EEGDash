"""Last-resort EDF/BDF header read for recordings no other source sizes.

nm000281: 2,785 of 25,253 .bdf recordings have no ``RecordingDuration`` in
their sidecars and NEMAR serves no records.json for the dataset, so the digest
left ntimes empty; ds005841 (OpenNeuro, 288 .bdf) is the same. With
``EEGDASH_REMOTE_HEADERS=missing`` the cascade reads the 256-byte main header
of only those recordings, from the public NEMAR object (by annex key) or the
OpenNeuro S3 object.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import _metadata_cascade as mc
from _metadata_cascade import (
    PROV_REMOTE_HEADER,
    CascadeContext,
    CascadeResult,
    RemoteHeaderStep,
    SidecarArithmeticStep,
)

_KEY = "SHA256E-s13190486--" + "a" * 64 + ".bdf"
_REL = "sub-13/ses-02/emg/sub-13_ses-02_run-45_emg.bdf"


def _ctx(root, rel: str) -> CascadeContext:
    return CascadeContext(SimpleNamespace(bidsdir=str(root)), str(root / rel))


def _bdf_main_header(n_records: int) -> bytes:
    buf = bytearray(b" " * 256)
    buf[236:244] = f"{n_records:<8}".encode()
    buf[244:252] = b"1       "
    buf[252:256] = b"37  "
    return bytes(buf)


def _nemar_bdf(tmp_path) -> CascadeContext:
    root = tmp_path / "nm000281"
    (root / _REL).parent.mkdir(parents=True)
    os.symlink(f"../../../.git/annex/objects/Wv/kP/{_KEY}/{_KEY}", root / _REL)
    return _ctx(root, _REL)


def _reader_returning(header: bytes, urls: list[str]):
    reader = Mock()
    reader.read.return_value = header
    return lambda url, **_k: urls.append(url) or reader


def test_reads_nemar_bdf_header_when_nothing_else_sized_it(monkeypatch, tmp_path):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    urls: list[str] = []
    monkeypatch.setattr(
        mc._remote_header, "RangeReader", _reader_returning(_bdf_main_header(61), urls)
    )
    ctx = _nemar_bdf(tmp_path)
    result = CascadeResult(sampling_frequency=2000.0)
    SidecarArithmeticStep().fill(ctx, result)
    assert result.ntimes is None
    RemoteHeaderStep(last_resort=True).fill(ctx, result)
    assert result.ntimes == 122000
    assert result.provenance["ntimes"] == PROV_REMOTE_HEADER
    assert urls == [f"https://nemar.s3.amazonaws.com/nm000281/objects/{_KEY}"]


def test_reads_openneuro_bdf_header(monkeypatch, tmp_path):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    urls: list[str] = []
    monkeypatch.setattr(
        mc._remote_header, "RangeReader", _reader_returning(_bdf_main_header(90), urls)
    )
    rel = "sub-021/eeg/sub-021_task-signalvr_eeg.bdf"
    result = CascadeResult(sampling_frequency=512.0)
    RemoteHeaderStep(last_resort=True).fill(_ctx(tmp_path / "ds005841", rel), result)
    assert result.ntimes == 46080
    assert len(urls) == 1
    assert urls[0].endswith(f"ds005841/{rel}")


def test_keeps_a_value_another_source_gave(monkeypatch, tmp_path):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    monkeypatch.setattr(
        mc._remote_header, "RangeReader", Mock(side_effect=AssertionError("fetched"))
    )
    result = CascadeResult(sampling_frequency=2000.0, ntimes=1000, nchans=36)
    RemoteHeaderStep(last_resort=True).fill(_nemar_bdf(tmp_path), result)
    assert (result.ntimes, result.nchans) == (1000, 36)


@pytest.mark.parametrize("mode", [None, "0", "1"])
def test_needs_missing_mode(monkeypatch, tmp_path, mode):
    if mode is None:
        monkeypatch.delenv("EEGDASH_REMOTE_HEADERS", raising=False)
    else:
        monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", mode)
    monkeypatch.setattr(
        mc._remote_header, "RangeReader", Mock(side_effect=AssertionError("fetched"))
    )
    result = CascadeResult(sampling_frequency=2000.0)
    RemoteHeaderStep(last_resort=True).fill(_nemar_bdf(tmp_path), result)
    assert result.ntimes is None


def test_first_remote_step_reads_only_edf_in_missing_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    monkeypatch.setattr(
        mc._remote_header, "RangeReader", Mock(side_effect=AssertionError("fetched"))
    )
    result = CascadeResult(sampling_frequency=10.0)
    RemoteHeaderStep().fill(_ctx(tmp_path / "nm000281", "sub-1_nirs.snirf"), result)
    assert result.ntimes is None


def test_cascade_ends_with_last_resort_step():
    last = mc.MetadataCascade().steps[-1]
    assert isinstance(last, RemoteHeaderStep)
    assert last.last_resort


def test_digest_workflow_enables_missing_mode():
    wf = Path(mc.__file__).resolve().parents[2] / ".github/workflows/2-clone-digest.yml"
    text = wf.read_text()
    step = text[text.index("- name: Digest datasets") :]
    step = step[: step.index("- name:", 1)]
    assert "EEGDASH_REMOTE_HEADERS: missing" in step


def test_fills_nchans_from_labels_without_annotations(monkeypatch, tmp_path):
    """ds005841 has ntimes now but no nchans: count labels, minus the annotation signal."""
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    labels = b"".join(
        f"{n:<16}".encode() for n in [f"E{i}" for i in range(36)] + ["BDF Annotations"]
    )
    reader = Mock()
    reader.read.side_effect = lambda off, n: (
        _bdf_main_header(90) if off == 0 else labels[:n]
    )
    monkeypatch.setattr(mc._remote_header, "RangeReader", lambda url, **_k: reader)
    rel = "sub-021/eeg/sub-021_task-signalvr_eeg.bdf"
    result = CascadeResult(sampling_frequency=512.0, ntimes=46080)
    RemoteHeaderStep(last_resort=True).fill(_ctx(tmp_path / "ds005841", rel), result)
    assert (result.ntimes, result.nchans) == (46080, 36)
    assert result.provenance["nchans"] == PROV_REMOTE_HEADER

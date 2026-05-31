"""Tests for the SizeArithmeticStep (T1, 0-byte) and RemoteHeaderStep (opt-in T2).

Covers RH5 integration: the SET ``.fdt`` zero-fetch tier wired into the
cascade, the new provenance constants, the env-gated remote EDF tier, and the
default step-order including the two new steps.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import _metadata_cascade as mc
from _metadata_cascade import (
    PROV_REMOTE_HEADER,
    PROV_SIZE_ARITHMETIC,
    CascadeContext,
    CascadeResult,
    MetadataCascade,
    RemoteHeaderStep,
    SizeArithmeticStep,
)


def _ctx(bids_file: str) -> CascadeContext:
    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    bids_dataset.dataset = "ds000001"
    return CascadeContext(bids_dataset, bids_file)


# ─── provenance constants ─────────────────────────────────────────────────


def test_new_provenance_constants_exist_with_expected_values():
    assert PROV_SIZE_ARITHMETIC == "size_arithmetic"
    assert PROV_REMOTE_HEADER == "remote_header"


# ─── SizeArithmeticStep (T1, 0-byte SET .fdt) ─────────────────────────────


def test_size_arithmetic_step_fills_set_ntimes_from_fdt(monkeypatch):
    # nchans known, ntimes absent, .set ext -> _fdt_n_times is consulted.
    monkeypatch.setattr(mc._set_parser, "_fdt_n_times", lambda _fdt_path, nchans: 1000)
    ctx = _ctx("sub-01/eeg/sub-01_task-rest_eeg.set")
    result = CascadeResult(sampling_frequency=250.0, nchans=4, ntimes=None)

    SizeArithmeticStep().fill(ctx, result)

    assert result.ntimes == 1000
    assert result.provenance["ntimes"] == "size_arithmetic"


def test_size_arithmetic_step_noop_when_ntimes_already_set(monkeypatch):
    fdt_spy = MagicMock(return_value=999)
    monkeypatch.setattr(mc._set_parser, "_fdt_n_times", fdt_spy)
    ctx = _ctx("sub-01/eeg/sub-01_eeg.set")
    result = CascadeResult(sampling_frequency=250.0, nchans=4, ntimes=5000)

    SizeArithmeticStep().fill(ctx, result)

    assert result.ntimes == 5000  # untouched
    fdt_spy.assert_not_called()  # short-circuited before the helper


def test_size_arithmetic_step_noop_when_nchans_missing(monkeypatch):
    monkeypatch.setattr(mc._set_parser, "_fdt_n_times", lambda _fdt_path, nchans: 1000)
    ctx = _ctx("sub-01/eeg/sub-01_eeg.set")
    result = CascadeResult(sampling_frequency=250.0, nchans=None, ntimes=None)

    SizeArithmeticStep().fill(ctx, result)

    assert result.ntimes is None


def test_size_arithmetic_step_noop_for_non_set_extension(monkeypatch):
    fdt_spy = MagicMock(return_value=1000)
    monkeypatch.setattr(mc._set_parser, "_fdt_n_times", fdt_spy)
    ctx = _ctx("sub-01/eeg/sub-01_eeg.edf")
    result = CascadeResult(sampling_frequency=250.0, nchans=4, ntimes=None)

    SizeArithmeticStep().fill(ctx, result)

    assert result.ntimes is None
    fdt_spy.assert_not_called()


def test_size_arithmetic_step_never_raises_when_helper_returns_none(monkeypatch):
    monkeypatch.setattr(mc._set_parser, "_fdt_n_times", lambda _fdt_path, nchans: None)
    ctx = _ctx("sub-01/eeg/sub-01_eeg.set")
    result = CascadeResult(sampling_frequency=250.0, nchans=4, ntimes=None)

    SizeArithmeticStep().fill(ctx, result)  # must not raise

    assert result.ntimes is None
    assert result.provenance["ntimes"] is None


# ─── RemoteHeaderStep (opt-in, env-gated) ─────────────────────────────────


def test_remote_header_step_is_noop_when_flag_off(monkeypatch):
    monkeypatch.delenv("EEGDASH_REMOTE_HEADERS", raising=False)

    # locate must never be touched when the flag is off.
    sentinel = MagicMock(side_effect=AssertionError("locate called with flag off"))
    monkeypatch.setattr(mc._remote_header, "locate", sentinel)

    ctx = _ctx("sub-01/eeg/sub-01_eeg.edf")
    result = CascadeResult(sampling_frequency=200.0, nchans=19, ntimes=None)

    RemoteHeaderStep().fill(ctx, result)

    assert result.ntimes is None
    assert result.provenance["ntimes"] is None
    sentinel.assert_not_called()


def test_remote_header_step_fills_edf_ntimes_when_flag_on(monkeypatch):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "1")

    fake_reader = MagicMock()
    fake_reader.read.return_value = b"\x00" * 256

    monkeypatch.setattr(
        mc._remote_header, "locate", lambda _rec: (None, "s3://openneuro.org/x")
    )
    monkeypatch.setattr(mc._remote_header, "RangeReader", lambda url, **kw: fake_reader)
    monkeypatch.setattr(
        mc._edf_header,
        "edf_n_times_from_main_header",
        lambda _buf, sfreq: 59200,
    )

    ctx = _ctx("sub-01/eeg/sub-01_eeg.edf")
    result = CascadeResult(sampling_frequency=200.0, nchans=19, ntimes=None)

    RemoteHeaderStep().fill(ctx, result)

    assert result.ntimes == 59200
    assert result.provenance["ntimes"] == "remote_header"
    fake_reader.read.assert_called_once_with(0, 256)


def test_remote_header_step_noop_when_no_url(monkeypatch):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "1")
    monkeypatch.setattr(
        mc._remote_header,
        "locate",
        lambda _rec: (None, None),  # NEMAR closed
    )
    ctx = _ctx("sub-01/eeg/sub-01_eeg.edf")
    result = CascadeResult(sampling_frequency=200.0, nchans=19, ntimes=None)

    RemoteHeaderStep().fill(ctx, result)

    assert result.ntimes is None


def _rangereader_boom(url, **_kw):
    raise RuntimeError("network exploded")


def test_remote_header_step_never_raises_on_reader_failure(monkeypatch):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "1")
    monkeypatch.setattr(
        mc._remote_header, "locate", lambda _rec: (None, "s3://openneuro.org/x")
    )

    monkeypatch.setattr(mc._remote_header, "RangeReader", _rangereader_boom)

    ctx = _ctx("sub-01/eeg/sub-01_eeg.edf")
    result = CascadeResult(sampling_frequency=200.0, nchans=19, ntimes=None)

    RemoteHeaderStep().fill(ctx, result)  # must not raise

    assert result.ntimes is None


def test_remote_header_step_noop_for_non_edf_when_flag_on(monkeypatch):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "1")
    sentinel = MagicMock(side_effect=AssertionError("locate called for .set"))
    monkeypatch.setattr(mc._remote_header, "locate", sentinel)

    ctx = _ctx("sub-01/eeg/sub-01_eeg.set")
    result = CascadeResult(sampling_frequency=250.0, nchans=4, ntimes=None)

    RemoteHeaderStep().fill(ctx, result)

    assert result.ntimes is None
    sentinel.assert_not_called()


# ─── default step ordering ────────────────────────────────────────────────


def test_default_steps_include_size_and_remote_in_correct_position():
    cascade = MetadataCascade()
    names = [type(s).__name__ for s in cascade.steps]
    assert names == [
        "MneBidsStep",
        "ModalitySidecarStep",
        "ChannelsTsvStep",
        "BinaryParserStep",
        "SizeArithmeticStep",
        "MneFallbackStep",
        "RemoteHeaderStep",
        "SidecarArithmeticStep",
    ]


def test_default_path_remote_step_does_not_fire(monkeypatch):
    # End-to-end smoke: with the flag off, a full cascade run never touches
    # the remote locate path even for an EDF file.
    monkeypatch.delenv("EEGDASH_REMOTE_HEADERS", raising=False)
    sentinel = MagicMock(side_effect=AssertionError("remote locate fired"))
    monkeypatch.setattr(mc._remote_header, "locate", sentinel)

    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    bids_dataset.dataset = "ds000001"
    bids_dataset.get_bids_file_attribute.side_effect = lambda key, _f: None
    bids_dataset.channel_labels.return_value = None

    monkeypatch.setattr(
        mc,
        "extract_sfreq_nchans_from_modality_sidecar",
        lambda _p, _r, sf, nc: (sf, nc),
    )
    monkeypatch.setattr(
        mc, "extract_sfreq_nchans_from_channels_tsv", lambda _p, _r, sf, nc: (sf, nc)
    )
    monkeypatch.setattr(mc, "get_parser_for_extension", lambda _ext: None)
    monkeypatch.setattr(
        mc, "extract_recording_duration_from_sidecar", lambda _p, _r: None
    )

    ctx = CascadeContext(bids_dataset, "sub-01/eeg/sub-01_eeg.edf")
    MetadataCascade().run(ctx)  # must not raise / must not call locate
    sentinel.assert_not_called()

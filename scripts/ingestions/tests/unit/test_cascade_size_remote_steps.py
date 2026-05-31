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


def _fetch_boom(_url, **_kw):
    raise RuntimeError("network exploded")


def _rangereader_budget_boom(url, **_kw):
    raise mc._remote_header.ByteBudgetExceeded("budget blown reading h5 metadata")


class _RecordSfreqNTimes:
    """Module-level fake for ``tmet_n_times_from_bytes`` that records sfreq.

    A callable class (not a nested function) so the no-nested-functions
    pre-commit hook is satisfied while still capturing the sfreq the step
    forwards into the parser.
    """

    def __init__(self, n_times: int) -> None:
        self.n_times = n_times
        self.seen_sfreq = None

    def __call__(self, _data, sfreq):
        self.seen_sfreq = sfreq
        return self.n_times


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
    sentinel = MagicMock(side_effect=AssertionError("locate called for .vhdr"))
    monkeypatch.setattr(mc._remote_header, "locate", sentinel)

    ctx = _ctx("sub-01/eeg/sub-01_eeg.vhdr")
    result = CascadeResult(sampling_frequency=250.0, nchans=4, ntimes=None)

    RemoteHeaderStep().fill(ctx, result)

    assert result.ntimes is None
    sentinel.assert_not_called()


# ─── RemoteHeaderStep: MEF3 .mefd branch (mocked) ─────────────────────────


def _fake_tmet_path(tmp_root: str = "/tmp/bids/ds000001"):
    # A .tmet inside a .mefd dir tree (path only — never opened in these tests).
    return mc.Path(f"{tmp_root}/sub-01/ieeg/sub-01_ieeg.mefd/ch.timd/seg.segd/seg.tmet")


def test_remote_header_step_fills_mefd_ntimes_when_flag_on(monkeypatch):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "1")

    tmet_path = _fake_tmet_path()
    monkeypatch.setattr(mc._mef3_parser, "find_first_tmet", lambda _d: tmet_path)
    monkeypatch.setattr(
        mc._remote_header, "locate", lambda _rec: (16384, "s3://openneuro.org/x.tmet")
    )
    fetch_spy = MagicMock(return_value=b"\x01" * 16384)
    monkeypatch.setattr(mc._parser_utils, "fetch_bytes_from_s3", fetch_spy)
    monkeypatch.setattr(mc._mef3_parser, "tmet_sfreq_from_bytes", lambda _data: 8720.0)
    monkeypatch.setattr(
        mc._mef3_parser,
        "tmet_n_times_from_bytes",
        lambda _data, _sfreq: 123456,
    )

    ctx = _ctx("sub-01/ieeg/sub-01_ieeg.mefd")
    # sfreq known from sidecar -> should be reused, not re-derived from bytes.
    result = CascadeResult(sampling_frequency=8720.0, nchans=32, ntimes=None)

    RemoteHeaderStep().fill(ctx, result)

    assert result.ntimes == 123456
    assert result.provenance["ntimes"] == "remote_header"
    # the .tmet is ~16 KB — fetched with max_bytes=20000.
    _args, kwargs = fetch_spy.call_args
    assert kwargs.get("max_bytes") == 20000


def test_remote_header_step_mefd_derives_sfreq_from_bytes_when_missing(monkeypatch):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "1")

    tmet_path = _fake_tmet_path()
    monkeypatch.setattr(mc._mef3_parser, "find_first_tmet", lambda _d: tmet_path)
    monkeypatch.setattr(
        mc._remote_header, "locate", lambda _rec: (16384, "s3://openneuro.org/x.tmet")
    )
    monkeypatch.setattr(
        mc._parser_utils, "fetch_bytes_from_s3", lambda _u, **_k: b"\x01" * 16384
    )
    sfreq_spy = MagicMock(return_value=5000.0)
    monkeypatch.setattr(mc._mef3_parser, "tmet_sfreq_from_bytes", sfreq_spy)

    n_times_recorder = _RecordSfreqNTimes(999)
    monkeypatch.setattr(mc._mef3_parser, "tmet_n_times_from_bytes", n_times_recorder)

    ctx = _ctx("sub-01/ieeg/sub-01_ieeg.mefd")
    result = CascadeResult(sampling_frequency=None, nchans=32, ntimes=None)

    RemoteHeaderStep().fill(ctx, result)

    assert result.ntimes == 999
    sfreq_spy.assert_called_once()  # had to derive sfreq from bytes
    assert n_times_recorder.seen_sfreq == 5000.0


def test_remote_header_step_mefd_noop_when_no_tmet(monkeypatch):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "1")
    monkeypatch.setattr(mc._mef3_parser, "find_first_tmet", lambda _d: None)
    sentinel = MagicMock(side_effect=AssertionError("locate called without a tmet"))
    monkeypatch.setattr(mc._remote_header, "locate", sentinel)

    ctx = _ctx("sub-01/ieeg/sub-01_ieeg.mefd")
    result = CascadeResult(sampling_frequency=8720.0, nchans=32, ntimes=None)

    RemoteHeaderStep().fill(ctx, result)

    assert result.ntimes is None
    sentinel.assert_not_called()


def test_remote_header_step_mefd_noop_when_no_url(monkeypatch):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "1")
    monkeypatch.setattr(
        mc._mef3_parser, "find_first_tmet", lambda _d: _fake_tmet_path()
    )
    monkeypatch.setattr(mc._remote_header, "locate", lambda _rec: (None, None))
    fetch_spy = MagicMock(side_effect=AssertionError("fetch called without a url"))
    monkeypatch.setattr(mc._parser_utils, "fetch_bytes_from_s3", fetch_spy)

    ctx = _ctx("sub-01/ieeg/sub-01_ieeg.mefd")
    result = CascadeResult(sampling_frequency=8720.0, nchans=32, ntimes=None)

    RemoteHeaderStep().fill(ctx, result)

    assert result.ntimes is None
    fetch_spy.assert_not_called()


def test_remote_header_step_mefd_noop_when_fetch_returns_none(monkeypatch):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "1")
    monkeypatch.setattr(
        mc._mef3_parser, "find_first_tmet", lambda _d: _fake_tmet_path()
    )
    monkeypatch.setattr(
        mc._remote_header, "locate", lambda _rec: (16384, "s3://openneuro.org/x.tmet")
    )
    monkeypatch.setattr(mc._parser_utils, "fetch_bytes_from_s3", lambda _u, **_k: None)
    tmet_spy = MagicMock(side_effect=AssertionError("parsed bytes that were None"))
    monkeypatch.setattr(mc._mef3_parser, "tmet_n_times_from_bytes", tmet_spy)

    ctx = _ctx("sub-01/ieeg/sub-01_ieeg.mefd")
    result = CascadeResult(sampling_frequency=8720.0, nchans=32, ntimes=None)

    RemoteHeaderStep().fill(ctx, result)

    assert result.ntimes is None
    tmet_spy.assert_not_called()


def test_remote_header_step_mefd_never_raises(monkeypatch):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "1")
    monkeypatch.setattr(
        mc._mef3_parser, "find_first_tmet", lambda _d: _fake_tmet_path()
    )
    monkeypatch.setattr(
        mc._remote_header, "locate", lambda _rec: (16384, "s3://openneuro.org/x.tmet")
    )
    monkeypatch.setattr(mc._parser_utils, "fetch_bytes_from_s3", _fetch_boom)

    ctx = _ctx("sub-01/ieeg/sub-01_ieeg.mefd")
    result = CascadeResult(sampling_frequency=8720.0, nchans=32, ntimes=None)

    RemoteHeaderStep().fill(ctx, result)  # must not raise

    assert result.ntimes is None


# ─── RemoteHeaderStep: SNIRF .snirf branch (mocked) ───────────────────────


def test_remote_header_step_fills_snirf_ntimes_when_flag_on(monkeypatch):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "1")

    fake_reader = MagicMock()
    monkeypatch.setattr(
        mc._remote_header, "locate", lambda _rec: (50000, "s3://openneuro.org/x.snirf")
    )
    reader_spy = MagicMock(return_value=fake_reader)
    monkeypatch.setattr(mc._remote_header, "RangeReader", reader_spy)
    monkeypatch.setattr(
        mc._snirf_parser,
        "snirf_n_times_from_fileobj",
        lambda _fileobj: 8000,
    )

    ctx = _ctx("sub-01/nirs/sub-01_nirs.snirf")
    result = CascadeResult(sampling_frequency=10.0, nchans=40, ntimes=None)

    RemoteHeaderStep().fill(ctx, result)

    assert result.ntimes == 8000
    assert result.provenance["ntimes"] == "remote_header"
    reader_spy.assert_called_once()  # RangeReader is the file-like


def test_remote_header_step_snirf_noop_when_no_url(monkeypatch):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "1")
    monkeypatch.setattr(mc._remote_header, "locate", lambda _rec: (None, None))
    reader_spy = MagicMock(side_effect=AssertionError("RangeReader without a url"))
    monkeypatch.setattr(mc._remote_header, "RangeReader", reader_spy)

    ctx = _ctx("sub-01/nirs/sub-01_nirs.snirf")
    result = CascadeResult(sampling_frequency=10.0, nchans=40, ntimes=None)

    RemoteHeaderStep().fill(ctx, result)

    assert result.ntimes is None
    reader_spy.assert_not_called()


def test_remote_header_step_snirf_noop_when_parser_returns_none(monkeypatch):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "1")
    monkeypatch.setattr(
        mc._remote_header, "locate", lambda _rec: (50000, "s3://openneuro.org/x.snirf")
    )
    monkeypatch.setattr(mc._remote_header, "RangeReader", lambda url, **kw: MagicMock())
    monkeypatch.setattr(
        mc._snirf_parser, "snirf_n_times_from_fileobj", lambda _fileobj: None
    )

    ctx = _ctx("sub-01/nirs/sub-01_nirs.snirf")
    result = CascadeResult(sampling_frequency=10.0, nchans=40, ntimes=None)

    RemoteHeaderStep().fill(ctx, result)

    assert result.ntimes is None
    assert result.provenance["ntimes"] is None


def test_remote_header_step_snirf_never_raises(monkeypatch):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "1")
    monkeypatch.setattr(
        mc._remote_header, "locate", lambda _rec: (50000, "s3://openneuro.org/x.snirf")
    )
    monkeypatch.setattr(mc._remote_header, "RangeReader", _rangereader_budget_boom)

    ctx = _ctx("sub-01/nirs/sub-01_nirs.snirf")
    result = CascadeResult(sampling_frequency=10.0, nchans=40, ntimes=None)

    RemoteHeaderStep().fill(ctx, result)  # must not raise

    assert result.ntimes is None


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

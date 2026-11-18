"""ntimes precedence: exact header sources beat NEMAR records.json and arithmetic.

Values from the 2026-10-17 Expanse diagnostic (1,704 MNE reads): NEMAR's
records.json ntimes and ``sfreq x RecordingDuration`` were one sample short
(MNE-BIDS writes ``RecordingDuration = (n - 1) / sfreq``), and BDF/EDF sidecars
can also drop the padded last data record.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import _metadata_cascade as mc
import _nemar_records as nr
from _metadata_cascade import (
    CascadeContext,
    CascadeResult,
    MetadataCascade,
    SidecarArithmeticStep,
    correct_arithmetic_ntimes,
)


@pytest.mark.parametrize(
    ("ntimes", "sfreq", "ext", "expected"),
    [
        pytest.param(2652999, 1000.0, ".bdf", 2653000, id="nm000242-bdf-n-1"),
        pytest.param(772839, 1000.0, ".bdf", 773000, id="nm000160-padded-record"),
        pytest.param(88827, 2000.0, ".bdf", 90000, id="nm000281-padded-record"),
        pytest.param(5611519, 512.0, ".edf", 5611520, id="on006446-edf"),
        pytest.param(28672, 2048.0, ".bdf", 28672, id="already-whole"),
        pytest.param(1015999, 1000.0, ".fif", 1016000, id="on005356-fif"),
        pytest.param(299136, 1000.0, ".set", 299136, id="set-not-whole-second"),
        pytest.param(622592, 1024.0, ".set", 622592, id="set-whole-second"),
        pytest.param(844139, 499.9, ".vhdr", 844139, id="fractional-rate-kept"),
    ],
)
def test_correct_arithmetic_ntimes(ntimes, sfreq, ext, expected):
    assert correct_arithmetic_ntimes(ntimes, sfreq, ext) == expected


def _ctx(root, rel: str) -> CascadeContext:
    return CascadeContext(SimpleNamespace(bidsdir=str(root)), str(root / rel))


def test_sidecar_arithmetic_corrects_n_minus_one(tmp_path):
    root = tmp_path / "nm000242"
    result = CascadeResult(sampling_frequency=1000.0, recording_duration=2652.999)
    SidecarArithmeticStep().fill(_ctx(root, "sub-1/eeg/sub-1_eeg.bdf"), result)
    assert result.ntimes == 2653000
    assert result.provenance["ntimes"] == "sidecar_arithmetic"


def test_exact_parser_beats_nemar_records(monkeypatch, tmp_path):
    """A header count must win over NEMAR's n-1 (nm000242 run-0)."""
    monkeypatch.setattr(
        nr,
        "signal_summary",
        Mock(
            return_value={
                "nchans": 32,
                "ntimes": 2652999,
                "recording_duration": 2652.999,
            }
        ),
    )
    monkeypatch.setattr(
        mc, "get_parser_for_extension", lambda _e: lambda _p: {"n_times": 2653000}
    )
    bids = SimpleNamespace(
        bidsdir=str(tmp_path / "nm000242"),
        get_bids_file_attribute=lambda key, _f: 1000.0 if key == "sfreq" else None,
        channel_labels=lambda _f: None,
    )
    ctx = CascadeContext(bids, str(tmp_path / "nm000242/sub-1/eeg/sub-1_eeg.bdf"))
    result = MetadataCascade().run(ctx)
    assert result.ntimes == 2653000
    assert result.provenance["ntimes"] == "binary_parser"


def test_nemar_ntimes_without_sfreq_is_kept(monkeypatch, tmp_path):
    """No rate to correct with: NEMAR's count is still better than nothing."""
    monkeypatch.setattr(
        nr, "signal_summary", Mock(return_value={"nchans": 256, "ntimes": 28672})
    )
    ctx = _ctx(tmp_path / "nm000108", "sub-01/emg/sub-01_emg.bdf")
    result = CascadeResult()
    mc.NemarRecordsStep().fill(ctx, result)
    SidecarArithmeticStep().fill(ctx, result)
    assert result.ntimes == 28672
    assert result.provenance["ntimes"] == "nemar_records"

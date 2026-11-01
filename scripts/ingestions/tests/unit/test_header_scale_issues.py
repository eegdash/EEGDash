"""The digest flags EDF/BDF channels MNE reads at an implausible scale.

nm000281 (1.0.2 to 1.0.4): joint angles stored as rad x 1e6 under a ``rad``
header, so MNE read them up to 2e6 rad. MNE scales a header range only for
µV/uV and mV, and ignores channels.tsv.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import _metadata_cascade as mc
from _edf_header import edf_scale_issues
from _metadata_cascade import (
    CascadeContext,
    CascadeResult,
    RemoteHeaderStep,
    channels_tsv_units,
)
from _validate import ValidationResult, validate_record

_KEY = "SHA256E-s13190486--" + "a" * 64 + ".bdf"
_REL = "sub-13/ses-02/emg/sub-13_ses-02_task-emg2pose_run-45_emg.bdf"


def _signals(channels) -> bytes:
    """The 256-per-signal block for ``(name, unit, physical_min, physical_max)``."""
    columns = [
        (16, [c[0] for c in channels]),
        (80, [""] * len(channels)),
        (8, [c[1] for c in channels]),
        (8, [c[2] for c in channels]),
        (8, [c[3] for c in channels]),
        (8, [-8388608] * len(channels)),
        (8, [8388607] * len(channels)),
        (80, [""] * len(channels)),
        (8, [2000] * len(channels)),
        (32, [""] * len(channels)),
    ]
    return b"".join(
        str(v).ljust(width)[:width].encode("latin-1")
        for width, values in columns
        for v in values
    )


@pytest.mark.parametrize(
    ("unit", "low", "high", "declared", "flagged"),
    [
        ("rad", -1272543, 2001499, "rad", True),  # nm000281 1.0.2-1.0.4
        ("rad", -1.27254, 2.001499, "rad", False),  # nm000281 1.0.5
        ("uV", -1272543, 2001499, "rad", False),  # nm000281 1.0.1: MNE x 1e-6
        ("", -3276, 3276, "µV", True),  # blank unit: MNE reads 3276 V
        ("µV", -3276, 3276, "µV", False),
        ("nV", -3e6, 3e6, "nV", True),  # MNE reads nV as V
        ("degC", -1e6, 1e6, "degC", False),  # quantity not judged
    ],
)
def test_scale_issues(unit, low, high, declared, flagged):
    sig = _signals([("emg0", "uV", -2116.78, 1761.673), ("joint0", unit, low, high)])
    issues = edf_scale_issues(sig, 2, {"joint0": declared})
    assert bool(issues) is flagged
    assert all(issue.startswith("joint0: ") for issue in issues)


def test_unparsable_range_is_skipped():
    sig = _signals([("joint0", "rad", "x", 2e6)])
    assert edf_scale_issues(sig, 1, {}) == []


def test_channels_tsv_units_inherit(tmp_path):
    (tmp_path / "task-emg2pose_channels.tsv").write_text(
        "name\ttype\tunits\nemg0\tEMG\tµV\njoint0\tMISC\trad\n"
    )
    rec = tmp_path / _REL
    rec.parent.mkdir(parents=True)
    assert channels_tsv_units(rec, tmp_path) == {"emg0": "µV", "joint0": "rad"}
    assert (
        channels_tsv_units(tmp_path / "sub-1" / "sub-1_task-x_eeg.edf", tmp_path) == {}
    )


def test_remote_header_step_records_issues(monkeypatch, tmp_path):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    root = tmp_path / "nm000281"
    (root / _REL).parent.mkdir(parents=True)
    os.symlink(f"../../../.git/annex/objects/Wv/kP/{_KEY}/{_KEY}", root / _REL)
    (root / _REL.replace("_emg.bdf", "_channels.tsv")).write_text(
        "name\ttype\tunits\nemg0\tEMG\tuV\njoint0\tMISC\trad\n"
    )
    main = bytearray(b" " * 256)
    main[236:244], main[244:252], main[252:256] = b"61      ", b"1       ", b"2   "
    reader = Mock()
    reader.read.return_value = bytes(main)
    reader.read_once.return_value = _signals(
        [("emg0", "uV", -2116.78, 1761.673), ("joint0", "rad", -1272543, 2001499)]
    )
    monkeypatch.setattr(mc._remote_header, "RangeReader", lambda url, **_k: reader)
    result = CascadeResult(sampling_frequency=2000.0)
    ctx = CascadeContext(SimpleNamespace(bidsdir=str(root)), str(root / _REL))
    RemoteHeaderStep().fill(ctx, result)
    assert result.ntimes == 122000
    assert result.header_issues == ["joint0: reads up to 2.0015e+06 rad (header 'rad')"]


def test_validation_warns_on_header_issues():
    result = ValidationResult()
    record = {
        "dataset": "nm000281",
        "bids_relpath": _REL,
        "storage": {"backend": "nemar", "base": "s3://nemar/nm000281"},
        "recording_modality": ["emg"],
        "_header_issues": ["joint0: reads up to 2.0015e+06 rad (header 'rad')"],
    }
    validate_record(record, "nm000281", "nemar", result, record_idx=1)
    assert result.stats["header_scale_issues"] == 1
    assert any("implausible channel scale" in str(w) for w in result.warnings)

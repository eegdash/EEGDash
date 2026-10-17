"""Sampling-rate sources for NEMAR EMG deposits (nm000108, 1,514 .bdf records).

The digest left ``sampling_frequency`` empty on every nm000108 record: NEMAR's
records.json carries the rate under ``signal_properties`` (not
``signal_summary``), and the rate's sidecar is a root ``task-<x>_emg.json``,
a suffix the modality-sidecar walk did not know.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import _nemar_records as nr
from _metadata_cascade import (
    PROV_NEMAR_RECORDS,
    CascadeContext,
    CascadeResult,
    ModalitySidecarStep,
    NemarRecordsStep,
)

_REL = "sub-01/ses-1/emg/sub-01_ses-1_task-gesture01_emg.bdf"
_REC = {
    "bids_relpath": _REL,
    "signal_summary": {
        "nchans": 256,
        "ntimes": 28672,
        "recording_duration": 14,
        "channel_type_counts": {"EMG": 256},
    },
    "signal_properties": {"sampling_frequency": 2048},
}


@pytest.fixture(autouse=True)
def _clear_index_cache():
    nr._records_index.cache_clear()
    yield
    nr._records_index.cache_clear()


def _ctx(root, rel: str) -> CascadeContext:
    return CascadeContext(SimpleNamespace(bidsdir=str(root)), str(root / rel))


def test_records_json_signal_properties_rate(monkeypatch, tmp_path):
    monkeypatch.setattr(
        nr,
        "request_json",
        Mock(return_value=([_REC], SimpleNamespace(status_code=200))),
    )
    result = CascadeResult()
    NemarRecordsStep().fill(_ctx(tmp_path / "nm000108", _REL), result)
    assert result.sampling_frequency == 2048
    assert result.provenance["sampling_frequency"] == PROV_NEMAR_RECORDS


def test_signal_summary_rate_wins_over_signal_properties(monkeypatch):
    rec = {
        **_REC,
        "signal_summary": {**_REC["signal_summary"], "sampling_frequency": 1000},
    }
    monkeypatch.setattr(
        nr, "request_json", Mock(return_value=([rec], SimpleNamespace(status_code=200)))
    )
    assert nr.signal_summary("nm000108", _REL)["sampling_frequency"] == 1000


def test_root_emg_sidecar_gives_rate_and_channels(tmp_path):
    root = tmp_path / "nm000108"
    emg = root / "sub-01/ses-1/emg"
    emg.mkdir(parents=True)
    (emg / "sub-01_ses-1_task-gesture01_emg.json").write_text(
        '{"RecordingDuration": 14}'
    )
    (root / "task-gesture01_emg.json").write_text(
        '{"SamplingFrequency": 2048, "EMGChannelCount": 256}'
    )
    result = CascadeResult()
    ModalitySidecarStep().fill(_ctx(root, _REL), result)
    assert (result.sampling_frequency, result.nchans) == (2048.0, 256)

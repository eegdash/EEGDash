"""Unit tests for the NEMAR ``records.json`` signal-summary fast-path.

Covers ``_nemar_records.signal_summary`` (the cached control-plane fetch) and
``_metadata_cascade.NemarRecordsStep`` (the first cascade step that seeds from
it). The network is never touched — ``request_json`` is replaced with a ``Mock``.

Pytest-style throughout: fixtures + parametrization, no test-helper classes. A
pre-commit hook forbids functions/classes nested inside a ``def``, so the
stand-in "later step" is a module-level function.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import _nemar_records as nr
from _metadata_cascade import (
    PROV_NEMAR_RECORDS,
    CascadeContext,
    CascadeResult,
    NemarRecordsStep,
    derive_duration_seconds,
)

_REL = "sub-01/eeg/sub-01_task-rest_eeg.set"
_SS = {
    "nchans": 33,
    "ntimes": 622592,
    "recording_duration": 608.0,
    "sampling_frequency": 1024.0,
}
_REC = {"bids_relpath": _REL, "signal_summary": _SS}


@pytest.fixture(autouse=True)
def _clear_index_cache():
    """Each test starts and ends with an empty per-dataset fetch cache."""
    nr._records_index.cache_clear()
    yield
    nr._records_index.cache_clear()


@pytest.fixture
def nm_ctx() -> CascadeContext:
    """A CascadeContext for ``nm000132`` whose absolute bids_file is <root>/<_REL>."""
    bids_dataset = SimpleNamespace(bidsdir="/data/nm000132")
    return CascadeContext(bids_dataset=bids_dataset, bids_file=f"/data/nm000132/{_REL}")


def _later_step_sets_ntimes(ctx: CascadeContext, result: CascadeResult) -> None:
    """Stand-in for a later cascade step (module-level: no class, no nested def)."""
    if result.ntimes is None:
        result.ntimes = 999


# ─── _nemar_records.signal_summary ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("dataset_id", "relpath", "payload", "status_code", "expected", "fetches"),
    [
        pytest.param("ds004660", _REL, [_REC], 200, None, 0, id="non-nemar-no-fetch"),
        pytest.param("nm000132", _REL, [_REC], 200, _SS, 1, id="match"),
        pytest.param("nm000132", "sub-9/x.set", [_REC], 200, None, 1, id="unmatched"),
        pytest.param("nm000132", _REL, None, 404, None, 1, id="http-404"),
        pytest.param("nm000132", _REL, {"oops": 1}, 200, None, 1, id="non-list-body"),
        pytest.param("nm000132", _REL, None, None, None, 1, id="request-raised"),
    ],
)
def test_signal_summary(
    monkeypatch, dataset_id, relpath, payload, status_code, expected, fetches
):
    response = None if status_code is None else SimpleNamespace(status_code=status_code)
    mock = Mock(return_value=(payload, response))
    monkeypatch.setattr(nr, "request_json", mock)
    assert nr.signal_summary(dataset_id, relpath) == expected
    assert mock.call_count == fetches


def test_signal_summary_fetches_latest_records_url(monkeypatch):
    mock = Mock(return_value=([_REC], SimpleNamespace(status_code=200)))
    monkeypatch.setattr(nr, "request_json", mock)
    nr.signal_summary("nm000132", _REL)
    method, url = mock.call_args.args
    assert method == "GET"
    assert url == "https://data.nemar.org/nm000132/latest/records.json"


def test_kill_switch_disables_without_fetching(monkeypatch):
    mock = Mock(return_value=([_REC], SimpleNamespace(status_code=200)))
    monkeypatch.setattr(nr, "request_json", mock)
    monkeypatch.setenv("EEGDASH_NEMAR_RECORDS", "0")
    assert nr.signal_summary("nm000132", _REL) is None
    assert mock.call_count == 0


def test_index_fetched_once_per_dataset(monkeypatch):
    mock = Mock(return_value=([_REC], SimpleNamespace(status_code=200)))
    monkeypatch.setattr(nr, "request_json", mock)
    nr.signal_summary("nm000132", _REL)
    nr.signal_summary("nm000132", _REL)
    nr.signal_summary("nm000132", "sub-02/eeg/other_eeg.set")
    assert mock.call_count == 1, "records.json must be fetched once per dataset"


# ─── NemarRecordsStep ──────────────────────────────────────────────────────


def test_step_seeds_nonnull_fields_with_provenance(monkeypatch, nm_ctx):
    monkeypatch.setattr(nr, "signal_summary", Mock(return_value=_SS))
    result = CascadeResult()
    NemarRecordsStep().fill(nm_ctx, result)
    assert (
        result.sampling_frequency,
        result.nchans,
        result.ntimes,
        result.recording_duration,
    ) == (1024.0, 33, 622592, 608.0)
    assert all(
        result.provenance[f] == PROV_NEMAR_RECORDS
        for f in ("sampling_frequency", "nchans", "ntimes")
    )
    # recording_duration feeds derive_duration_seconds; it is not provenance-tracked.
    assert "recording_duration" not in result.provenance


def test_step_passes_dataset_id_and_relpath_from_ctx(monkeypatch, nm_ctx):
    mock = Mock(return_value=_SS)
    monkeypatch.setattr(nr, "signal_summary", mock)
    NemarRecordsStep().fill(nm_ctx, CascadeResult())
    assert mock.call_args == call("nm000132", _REL)


def test_step_skips_null_fields(monkeypatch, nm_ctx):
    monkeypatch.setattr(
        nr, "signal_summary", Mock(return_value={**_SS, "ntimes": None})
    )
    result = CascadeResult()
    NemarRecordsStep().fill(nm_ctx, result)
    assert result.ntimes is None
    assert result.provenance["ntimes"] is None
    assert result.nchans == 33  # sibling fields still seeded


@pytest.mark.parametrize("summary", [None, {}], ids=["none", "empty"])
def test_step_noop_when_summary_absent(monkeypatch, nm_ctx, summary):
    monkeypatch.setattr(nr, "signal_summary", Mock(return_value=summary))
    result = CascadeResult()
    NemarRecordsStep().fill(nm_ctx, result)
    assert result == CascadeResult()


def test_step_does_not_overwrite_filled_field(monkeypatch, nm_ctx):
    monkeypatch.setattr(nr, "signal_summary", Mock(return_value=_SS))
    result = CascadeResult(ntimes=111)  # an exact local value already present
    NemarRecordsStep().fill(nm_ctx, result)
    assert result.ntimes == 111
    assert result.provenance["ntimes"] is None


def test_nemar_value_survives_a_later_step(monkeypatch, nm_ctx):
    monkeypatch.setattr(nr, "signal_summary", Mock(return_value=_SS))
    result = CascadeResult()
    NemarRecordsStep().fill(nm_ctx, result)
    _later_step_sets_ntimes(nm_ctx, result)
    assert result.ntimes == 622592


def test_nemar_ntimes_is_exact_for_duration(monkeypatch, nm_ctx):
    monkeypatch.setattr(nr, "signal_summary", Mock(return_value=_SS))
    result = CascadeResult()
    NemarRecordsStep().fill(nm_ctx, result)
    derive_duration_seconds(result)
    assert result.duration_seconds == pytest.approx(622592 / 1024.0)

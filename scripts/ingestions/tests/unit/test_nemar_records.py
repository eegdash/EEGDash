"""Unit tests for the NEMAR ``records.json`` signal-summary fast-path.

Covers ``_nemar_records.signal_summary`` (the cached control-plane fetch) and
``_metadata_cascade.NemarRecordsStep`` (the first cascade step that seeds from
it). The network is never touched — ``request_json`` is replaced with a ``Mock``.

Pytest-style: fixtures + parametrization, no test-helper classes. A pre-commit
hook forbids functions/classes nested inside a ``def``, so the stand-in "later
step" is a module-level function.
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
_OK = SimpleNamespace(status_code=200)


@pytest.fixture(autouse=True)
def _clear_index_cache():
    """Each test starts and ends with an empty per-dataset fetch cache."""
    nr._records_index.cache_clear()
    yield
    nr._records_index.cache_clear()


@pytest.fixture
def nm_ctx() -> CascadeContext:
    """A CascadeContext for ``nm000132`` whose absolute bids_file is <root>/<_REL>."""
    ds = SimpleNamespace(bidsdir="/data/nm000132")
    return CascadeContext(bids_dataset=ds, bids_file=f"/data/nm000132/{_REL}")


def _later_step_sets_ntimes(ctx: CascadeContext, result: CascadeResult) -> None:
    """Module-level stand-in later step (no class / nested def — pre-commit forbids)."""
    if result.ntimes is None:
        result.ntimes = 999


# ─── _nemar_records.signal_summary ─────────────────────────────────────────


def test_happy_path_fetches_latest_url_once_per_dataset(monkeypatch):
    mock = Mock(return_value=([_REC], _OK))
    monkeypatch.setattr(nr, "request_json", mock)
    assert nr.signal_summary("nm000132", _REL) == _SS
    nr.signal_summary("nm000132", _REL)  # cached
    nr.signal_summary("nm000132", "sub-02/eeg/other_eeg.set")  # cached
    assert mock.call_count == 1
    assert mock.call_args.args == (
        "GET",
        "https://data.nemar.org/nm000132/latest/records.json",
    )


@pytest.mark.parametrize(
    ("dataset_id", "relpath", "payload", "response", "fetches"),
    [
        pytest.param("ds004660", _REL, [_REC], _OK, 0, id="non-nemar-skips-fetch"),
        pytest.param("nm000132", "sub-9/x.set", [_REC], _OK, 1, id="unmatched-relpath"),
        pytest.param(
            "nm000132", _REL, None, SimpleNamespace(status_code=404), 1, id="http-404"
        ),
        pytest.param("nm000132", _REL, {"x": 1}, _OK, 1, id="non-list-body"),
        pytest.param("nm000132", _REL, None, None, 1, id="request-raised"),
    ],
)
def test_signal_summary_returns_none(
    monkeypatch, dataset_id, relpath, payload, response, fetches
):
    mock = Mock(return_value=(payload, response))
    monkeypatch.setattr(nr, "request_json", mock)
    assert nr.signal_summary(dataset_id, relpath) is None
    assert mock.call_count == fetches


def test_kill_switch_disables_without_fetching(monkeypatch):
    mock = Mock(return_value=([_REC], _OK))
    monkeypatch.setattr(nr, "request_json", mock)
    monkeypatch.setenv("EEGDASH_NEMAR_RECORDS", "0")
    assert nr.signal_summary("nm000132", _REL) is None
    assert mock.call_count == 0


# ─── NemarRecordsStep ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        pytest.param(_SS, (1024.0, 33, 622592, 608.0), id="full"),
        pytest.param(
            {**_SS, "ntimes": None}, (1024.0, 33, None, 608.0), id="null-skipped"
        ),
        pytest.param(None, (None, None, None, None), id="missing"),
        pytest.param({}, (None, None, None, None), id="empty"),
    ],
)
def test_step_seeds_present_fields_with_provenance(
    monkeypatch, nm_ctx, summary, expected
):
    monkeypatch.setattr(nr, "signal_summary", Mock(return_value=summary))
    result = CascadeResult()
    NemarRecordsStep().fill(nm_ctx, result)
    got = (
        result.sampling_frequency,
        result.nchans,
        result.ntimes,
        result.recording_duration,
    )
    assert got == expected
    # seeded scalars carry NEMAR provenance; null/absent ones stay unstamped.
    names = ("sampling_frequency", "nchans", "ntimes")
    for name, value in zip(names, expected, strict=False):
        assert result.provenance[name] == (
            PROV_NEMAR_RECORDS if value is not None else None
        )
    assert "recording_duration" not in result.provenance


def test_step_passes_ctx_dataset_id_and_relpath(monkeypatch, nm_ctx):
    mock = Mock(return_value=_SS)
    monkeypatch.setattr(nr, "signal_summary", mock)
    NemarRecordsStep().fill(nm_ctx, CascadeResult())
    assert mock.call_args.args == ("nm000132", _REL)


def test_step_is_first_writer(monkeypatch, nm_ctx):
    """An existing value is never overwritten; a NEMAR value survives a later step."""
    monkeypatch.setattr(nr, "signal_summary", Mock(return_value=_SS))

    prefilled = CascadeResult(ntimes=111)
    NemarRecordsStep().fill(nm_ctx, prefilled)
    assert prefilled.ntimes == 111
    assert prefilled.provenance["ntimes"] is None

    seeded = CascadeResult()
    NemarRecordsStep().fill(nm_ctx, seeded)
    _later_step_sets_ntimes(nm_ctx, seeded)
    assert seeded.ntimes == 622592


def test_step_ntimes_is_exact_for_duration(monkeypatch, nm_ctx):
    monkeypatch.setattr(nr, "signal_summary", Mock(return_value=_SS))
    result = CascadeResult()
    NemarRecordsStep().fill(nm_ctx, result)
    derive_duration_seconds(result)
    assert result.duration_seconds == pytest.approx(622592 / 1024.0)

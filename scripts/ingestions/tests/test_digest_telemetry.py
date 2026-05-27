"""Unit + integration tests for the DigestTelemetry Module (P1.1).

The Module emits structured per-Dataset and per-Record events to a
pluggable backend. Default emitter is :class:`NullEmitter` so digest
behaviour is unchanged when telemetry is disabled.

These tests cover:
- :class:`TelemetryEvent` shape + dict round-trip
- :class:`NullEmitter` / :class:`NDJSONEmitter` behaviour
- ``get_emitter`` / ``configure_telemetry`` / ``reset_telemetry`` semantics
- ``auto_configure_from_env`` opt-in via ``$EEGDASH_TELEMETRY_PATH``
- Integration: BIDS-fs digest run emits the expected event stream
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

from eegdash.testing import data_file

from digest_telemetry import (
    NDJSONEmitter,
    NullEmitter,
    TelemetryEvent,
    auto_configure_from_env,
    configure_telemetry,
    get_emitter,
    reset_telemetry,
)


def _load_digest():
    """Lazy-load 3_digest.py (digit-prefixed filename)."""
    spec = importlib.util.spec_from_file_location(
        "_digest_telemetry_target", _INGEST_DIR / "3_digest.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── TelemetryEvent ────────────────────────────────────────────────────────


def test_event_required_fields():
    e = TelemetryEvent(
        event_kind="record_built",
        dataset_id="ds-1",
        payload={"foo": "bar"},
    )
    assert e.event_kind == "record_built"
    assert e.dataset_id == "ds-1"
    assert e.payload == {"foo": "bar"}
    assert e.record_id is None
    # Default timestamp is set on construction.
    assert isinstance(e.timestamp, str)
    assert "T" in e.timestamp  # ISO 8601


def test_event_to_dict_has_stable_field_order():
    """Field order in to_dict() is stable across calls."""
    e = TelemetryEvent(
        event_kind="record_built",
        dataset_id="ds-1",
        payload={"foo": "bar"},
        record_id="sub-01/eeg/sub-01_eeg.edf",
        timestamp="2026-05-22T12:00:00+00:00",
    )
    d = e.to_dict()
    assert list(d.keys()) == [
        "timestamp",
        "event_kind",
        "dataset_id",
        "record_id",
        "payload",
    ]
    assert d["payload"] == {"foo": "bar"}


# ─── NullEmitter ───────────────────────────────────────────────────────────


def test_null_emitter_is_noop():
    """NullEmitter.emit returns None without side effects."""
    e = NullEmitter()
    result = e.emit(TelemetryEvent(event_kind="x", dataset_id="ds-1", payload={}))
    assert result is None  # explicitly None


# ─── NDJSONEmitter ─────────────────────────────────────────────────────────


def test_ndjson_emitter_writes_one_line_per_event(tmp_path: Path):
    out = tmp_path / "events.ndjson"
    emitter = NDJSONEmitter(out)
    emitter.emit(
        TelemetryEvent(
            event_kind="dataset_started",
            dataset_id="ds-1",
            payload={"src": "openneuro"},
        )
    )
    emitter.emit(
        TelemetryEvent(
            event_kind="dataset_finished",
            dataset_id="ds-1",
            payload={"record_count": 5},
        )
    )
    emitter.close()

    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2
    e0 = json.loads(lines[0])
    e1 = json.loads(lines[1])
    assert e0["event_kind"] == "dataset_started"
    assert e1["event_kind"] == "dataset_finished"
    assert e1["payload"]["record_count"] == 5


def test_ndjson_emitter_appends_not_truncates(tmp_path: Path):
    """Re-opening the same path doesn't truncate; events accumulate."""
    out = tmp_path / "events.ndjson"
    e1 = NDJSONEmitter(out)
    e1.emit(TelemetryEvent(event_kind="first", dataset_id="ds-1", payload={}))
    e1.close()
    e2 = NDJSONEmitter(out)
    e2.emit(TelemetryEvent(event_kind="second", dataset_id="ds-1", payload={}))
    e2.close()
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["event_kind"] == "first"
    assert json.loads(lines[1])["event_kind"] == "second"


def test_ndjson_emitter_tolerates_unserializable_payload(tmp_path: Path):
    """A non-JSON-serializable payload is logged and dropped, not raised."""
    out = tmp_path / "events.ndjson"
    emitter = NDJSONEmitter(out)

    # Set is not JSON-serializable by default; the emitter should warn
    # and continue rather than crash the caller.
    emitter.emit(
        TelemetryEvent(
            event_kind="record_built",
            dataset_id="ds-1",
            payload={"weird": {1, 2, 3}},
        )
    )
    emitter.close()

    # File exists (header line is fine even if event was dropped).
    assert out.exists()


def test_ndjson_emitter_creates_parent_directory(tmp_path: Path):
    """The emitter creates intermediate directories if they don't exist."""
    out = tmp_path / "deep" / "nested" / "events.ndjson"
    emitter = NDJSONEmitter(out)
    emitter.emit(TelemetryEvent(event_kind="x", dataset_id="ds-1", payload={}))
    emitter.close()
    assert out.exists()


# ─── get_emitter / configure_telemetry / reset_telemetry ──────────────────


def test_default_emitter_is_null():
    """Before any configure_telemetry call, get_emitter() returns NullEmitter."""
    reset_telemetry()
    assert isinstance(get_emitter(), NullEmitter)


def test_configure_telemetry_replaces_emitter(tmp_path: Path):
    out = tmp_path / "events.ndjson"
    configure_telemetry(NDJSONEmitter(out))
    assert isinstance(get_emitter(), NDJSONEmitter)
    reset_telemetry()
    assert isinstance(get_emitter(), NullEmitter)


def test_reset_telemetry_restores_null(tmp_path: Path):
    configure_telemetry(NDJSONEmitter(tmp_path / "events.ndjson"))
    reset_telemetry()
    assert isinstance(get_emitter(), NullEmitter)


# ─── auto_configure_from_env ───────────────────────────────────────────────


def test_auto_configure_no_env_var_keeps_null(monkeypatch):
    monkeypatch.delenv("EEGDASH_TELEMETRY_PATH", raising=False)
    reset_telemetry()
    auto_configure_from_env()
    assert isinstance(get_emitter(), NullEmitter)


def test_auto_configure_installs_ndjson_when_env_set(tmp_path: Path, monkeypatch):
    out = tmp_path / "events.ndjson"
    monkeypatch.setenv("EEGDASH_TELEMETRY_PATH", str(out))
    reset_telemetry()
    auto_configure_from_env()
    try:
        assert isinstance(get_emitter(), NDJSONEmitter)
        assert get_emitter().path == out
    finally:
        reset_telemetry()


def test_auto_configure_idempotent_for_same_path(tmp_path: Path, monkeypatch):
    """Calling auto_configure_from_env twice for the same path doesn't
    re-open the file."""
    out = tmp_path / "events.ndjson"
    monkeypatch.setenv("EEGDASH_TELEMETRY_PATH", str(out))
    reset_telemetry()
    auto_configure_from_env()
    first = get_emitter()
    auto_configure_from_env()
    second = get_emitter()
    try:
        assert first is second  # same instance, no re-open
    finally:
        reset_telemetry()


# ─── Integration: digest_dataset emits the expected event stream ─────────


def test_digest_dataset_emits_event_stream(tmp_path_factory, monkeypatch):
    """Running digest_dataset against the BIDS-fs snapshot fixture
    produces dataset_started + record_built + dataset_finished events
    in that order."""
    events_path = tmp_path_factory.mktemp("telemetry_run") / "events.ndjson"
    monkeypatch.setenv("EEGDASH_TELEMETRY_PATH", str(events_path))

    digest = _load_digest()
    # Re-install the env-driven emitter (the module was loaded once at
    # import; the monkeypatched env var was set after).
    auto_configure_from_env()
    try:
        tmp_output = tmp_path_factory.mktemp("digest_run_telemetry")
        summary = digest.digest_dataset(
            "ds_snapshot_vhdr",
            data_file("digest_snapshots/inputs"),
            tmp_output,
        )
        assert summary["status"] == "success"
        # Flush — close the emitter so all lines are on disk.
        get_emitter().close()

        lines = events_path.read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        kinds = [e["event_kind"] for e in events]

        assert kinds[0] == "dataset_started"
        assert kinds[-1] == "dataset_finished"
        assert "record_built" in kinds
        # ds_snapshot_vhdr has 1 successful record → exactly one record_built
        assert sum(1 for k in kinds if k == "record_built") == 1
        # Provenance was passed through to the event payload.
        record_event = next(e for e in events if e["event_kind"] == "record_built")
        assert "metadata_provenance" in record_event["payload"]
        prov = record_event["payload"]["metadata_provenance"]
        assert set(prov.keys()) == {
            "sampling_frequency",
            "nchans",
            "ntimes",
            "ch_names",
        }
        # dataset_finished carries the summary totals.
        finished_event = events[-1]
        assert finished_event["payload"]["status"] == "success"
        assert finished_event["payload"]["record_count"] == 1
    finally:
        reset_telemetry()
        monkeypatch.delenv("EEGDASH_TELEMETRY_PATH", raising=False)

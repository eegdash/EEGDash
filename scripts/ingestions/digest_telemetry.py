"""Structured per-Record / per-Dataset event stream for the digest pipeline.

closes the operational-visibility gap.

Before this Module the only forensic tool for a 1000-dataset ingest run
was grep through stdout. Questions like "Which datasets had Records
with ``sampling_frequency = None``?" or "What fraction of Records get
their nchans from the binary parser vs the sidecar?" required
re-running the ingest. Now they're queries over a structured event
stream.

The event payloads use the ``_metadata_provenance`` dict landed in
P0.1 — each ``record_built`` event carries the per-field provenance
so support can ask: "for each Record where sampling_frequency was
filled by the binary_parser, which datasets were they in?"

Design
------

- :class:`TelemetryEvent` — frozen dataclass. Five fields:
  ``event_kind``, ``dataset_id``, ``record_id`` (optional),
  ``payload`` (dict), ``timestamp``.
- :class:`TelemetryEmitter` — abstract base. Concrete:

  - :class:`NDJSONEmitter` — appends one JSON line per event to a file.
    The format is queryable by ``jq`` / ``pandas.read_json(lines=True)``.
  - :class:`NullEmitter` — no-op. The default; preserves the
    pre-telemetry behaviour exactly when no path is configured.

- :func:`get_emitter` — process-global emitter. Set once via
  :func:`configure_telemetry` (typically from the CLI / orchestrator
  start-up); read by the digest helpers via ``get_emitter().emit(...)``.

Event kinds
-----------

- ``"dataset_started"`` — once per ``digest_dataset`` call.
  Payload: ``{"source": ..., "dataset_dir": ...}``.

- ``"dataset_finished"`` — once per ``digest_dataset`` call.
  Payload: ``{"status": ..., "record_count": N, "error_count": N,
  "digest_method": ...}``.

- ``"record_built"`` — one per successful ``extract_record`` call.
  Payload: ``{"bids_relpath": ..., "datatype": ...,
  "metadata_provenance": {...}, "sampling_frequency": ...,
  "nchans": ..., "ntimes": ...}``.

- ``"record_failed"`` — one per ``extract_record`` failure (the
  ``errors`` list).
  Payload: ``{"bids_file": ..., "error": ...}``.

Future kinds (not implemented in this commit) can be added without
schema migration — consumers should branch on ``event_kind``.

See Also
--------
-  P1.1 — driver + definition-of-done
-  — provenance (the payload for ``record_built``)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TelemetryEvent:
    """One structured event in the digest event stream.

    Attributes
    ----------
    event_kind : str
        ``"dataset_started"`` / ``"dataset_finished"`` /
        ``"record_built"`` / ``"record_failed"``.
    dataset_id : str
        Always present — every event belongs to a Dataset.
    payload : dict
        Event-kind-specific fields; see the Module docstring.
    record_id : str or None
        Optional — the per-Record events carry it (e.g.
        ``bids_relpath``); the per-Dataset events leave it None.
    timestamp : str
        ISO 8601 with tz. Defaults to ``datetime.now(tz=UTC)`` at
        construction.
    """

    event_kind: str
    dataset_id: str
    payload: dict[str, Any]
    record_id: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation (field order is stable)."""
        return {
            "timestamp": self.timestamp,
            "event_kind": self.event_kind,
            "dataset_id": self.dataset_id,
            "record_id": self.record_id,
            "payload": self.payload,
        }


class TelemetryEmitter(ABC):
    """Where events go. Implementations decide the backend (file / DB / null)."""

    @abstractmethod
    def emit(self, event: TelemetryEvent) -> None:
        """Record one event. Must not raise on best-effort errors —
        telemetry failures should not crash the ingest pipeline.
        """

    def close(self) -> None:  # noqa: B027 — intentional default no-op
        """Optional cleanup hook. Subclasses with open files / DB
        handles should override.
        """


class NullEmitter(TelemetryEmitter):
    """No-op. The default before telemetry is configured.

    Preserves the pre-P1.1 behaviour exactly — no extra I/O, no extra
    memory. Any caller can ``get_emitter().emit(...)`` unconditionally.
    """

    def emit(self, event: TelemetryEvent) -> None:
        return None


class NDJSONEmitter(TelemetryEmitter):
    """Append one JSON line per event to a file.

    Thread-safe: each ``emit`` acquires a lock around the write so
    concurrent digesting (via multiprocessing or threading) doesn't
    interleave half-events. Each line is one self-contained JSON
    object — readable by ``jq`` / ``pandas.read_json(lines=True)`` /
    ``sqlite3 -json`` after import.

    The output file is opened in append mode so a long-running ingest
    job that's interrupted can resume; the file isn't truncated on
    re-open.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Append mode so re-runs accumulate; the lock is per-instance,
        # so a fresh emitter is needed in each subprocess.
        self._fh = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def emit(self, event: TelemetryEvent) -> None:
        try:
            line = json.dumps(event.to_dict(), separators=(",", ":"))
        except (TypeError, ValueError) as e:
            # Best-effort: a malformed payload shouldn't crash digest.
            logger.warning(
                "telemetry: failed to serialize %s event: %s",
                event.event_kind,
                e,
            )
            return
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except OSError:
                pass


# ─── Process-global emitter ───────────────────────────────────────────


_EMITTER: TelemetryEmitter = NullEmitter()
_EMITTER_LOCK = threading.Lock()


def get_emitter() -> TelemetryEmitter:
    """Return the process-global emitter (defaults to :class:`NullEmitter`).

    Callers should treat this as cheap — emitting goes through the
    NullEmitter when no real emitter is configured, so unconditional
    calls cost ~1 attribute lookup.
    """
    return _EMITTER


def configure_telemetry(emitter: TelemetryEmitter) -> None:
    """Install ``emitter`` as the process-global emitter.

    The previous emitter (if any) is closed. Call once at orchestrator
    start-up; for tests, call at fixture teardown to restore the
    NullEmitter default.
    """
    global _EMITTER
    with _EMITTER_LOCK:
        old = _EMITTER
        _EMITTER = emitter
    # Close the old one outside the lock so a slow close() doesn't
    # block other threads.
    if old is not emitter:
        try:
            old.close()
        except OSError:
            pass


def reset_telemetry() -> None:
    """Restore the default :class:`NullEmitter`. Test-only convenience."""
    configure_telemetry(NullEmitter())


# ─── Environment-driven auto-configuration ────────────────────────────


_ENV_VAR = "EEGDASH_TELEMETRY_PATH"


def auto_configure_from_env() -> None:
    """If ``$EEGDASH_TELEMETRY_PATH`` is set, install an :class:`NDJSONEmitter`.

    Idempotent — calling multiple times only opens the file once
    (subsequent calls find the existing :class:`NDJSONEmitter` and
    return). Call from the orchestrator's start-up so CI / production
    runs can opt in without code changes.
    """
    path = os.environ.get(_ENV_VAR)
    if not path:
        return
    current = get_emitter()
    if isinstance(current, NDJSONEmitter) and current.path == Path(path):
        return  # already configured for this path
    configure_telemetry(NDJSONEmitter(path))


__all__ = [
    "NDJSONEmitter",
    "NullEmitter",
    "TelemetryEmitter",
    "TelemetryEvent",
    "auto_configure_from_env",
    "configure_telemetry",
    "get_emitter",
    "reset_telemetry",
]

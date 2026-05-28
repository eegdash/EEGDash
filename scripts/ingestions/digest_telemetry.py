"""Structured per-Record / per-Dataset event stream for the digest pipeline.

Emits four event kinds: ``dataset_started``, ``dataset_finished``,
``record_built`` (carries ``metadata_provenance``), and ``record_failed``.
Events are queryable via ``jq`` / ``pandas.read_json(lines=True)``.
Use :func:`configure_telemetry` at orchestrator start-up or set
``$EEGDASH_TELEMETRY_PATH`` to enable :class:`NDJSONEmitter`; the
default :class:`NullEmitter` is a no-op.
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
    """Frozen dataclass representing one event in the digest event stream."""

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
        """Record one event; must not raise on best-effort errors."""

    def close(self) -> None:  # noqa: B027 — intentional default no-op
        """Optional cleanup hook; override in subclasses with open handles."""


class NullEmitter(TelemetryEmitter):
    """No-op default emitter; zero I/O overhead."""

    def emit(self, event: TelemetryEvent) -> None:
        return None


class NDJSONEmitter(TelemetryEmitter):
    """Append one JSON line per event to a file; thread-safe, append-mode."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Append mode so re-runs accumulate; lock is per-instance (new emitter per subprocess).
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
    """Return the process-global emitter (defaults to :class:`NullEmitter`)."""
    return _EMITTER


def configure_telemetry(emitter: TelemetryEmitter) -> None:
    """Install ``emitter`` as the process-global emitter, closing the previous one."""
    global _EMITTER
    with _EMITTER_LOCK:
        old = _EMITTER
        _EMITTER = emitter
    # Close outside the lock so a slow close() doesn't block other threads.
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
    """Install an :class:`NDJSONEmitter` if ``$EEGDASH_TELEMETRY_PATH`` is set; idempotent."""
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

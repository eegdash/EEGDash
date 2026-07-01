"""Thin client for the server-side ``/aggregations/*`` endpoints.

Every chart in this package now reads pre-computed cross-tabs from the
EEGDash server instead of recomputing them client-side. This module is
the single seam that owns:

* URL construction (``api_base`` + ``database`` from
  :class:`DatasetSnapshot` or the same env vars the snapshot honors).
* On-disk caching at ``.eegdash_cache/aggregations/{endpoint}-{hash}.json``
  with a 24-hour TTL, mirroring the helper pattern in
  :mod:`eegdash.dataset.snapshot`.
* Provenance tagging (``"live" | "cached" | "error"``) so the docs build
  log can report exactly where each chart's data came from.

Failure mode: if the live endpoint *and* the disk cache both fail, the
helper raises ``AggregationFetchError``. Each chart catches that and
falls back to the legacy CSV path so a local docs build with no network
keeps rendering.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# Mirror DatasetSnapshot's defaults so the two clients stay in lockstep
# when the server URL or database name changes.
_DEFAULT_API_BASE = "https://data.eegdash.org/api"
_DEFAULT_DATABASE = "eegdash"

# 24-hour TTL. The aggregations endpoints are derived from the same
# nightly MongoDB snapshot DatasetSnapshot consumes, so a tighter TTL
# would just thrash the on-disk cache during a docs build that calls
# all 4 charts back-to-back.
_DEFAULT_TTL_SECONDS = 24 * 60 * 60


class AggregationFetchError(RuntimeError):
    """Raised when the live endpoint and the disk cache both fail."""


@dataclass(frozen=True)
class AggregationResponse:
    """A JSON payload with provenance attached."""

    payload: dict[str, Any]
    source: str  # "live" | "cached"
    url: str


def _cache_root() -> Path:
    """Return the ``.eegdash_cache/aggregations/`` directory.

    Honors ``EEGDASH_CACHE_DIR`` first (so CI can pin it), falls back to
    a per-repo ``.eegdash_cache`` directory at the project root so the
    docs build stays self-contained.
    """
    env = os.environ.get("EEGDASH_CACHE_DIR")
    if env:
        base = Path(env)
    else:
        base = Path(__file__).resolve().parents[2] / ".eegdash_cache"
    out = base / "aggregations"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cache_key(endpoint: str, params: Mapping[str, Any]) -> str:
    """Produce a stable filename ``{endpoint}-{sha1}.json`` for a request.

    Hash includes the api_base + database so two consumers pointed at
    different shards never collide on disk — the same bug DatasetSnapshot
    keys around for the in-memory cache.
    """
    canonical = json.dumps(dict(params), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
    safe_endpoint = endpoint.strip("/").replace("/", "_") or "agg"
    return f"{safe_endpoint}-{digest}.json"


def _read_disk_cache(path: Path, ttl_seconds: float) -> dict[str, Any] | None:
    """Return the cached payload if present and fresh, else ``None``."""
    if not path.exists():
        return None
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return None
    if age > ttl_seconds:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("aggregations disk cache unreadable at %s: %s", path, exc)
        return None


def _write_disk_cache(path: Path, payload: Mapping[str, Any]) -> None:
    """Best-effort write; never raises (the live response is in hand)."""
    try:
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    except OSError as exc:
        logger.debug("aggregations disk cache write failed at %s: %s", path, exc)


def _resolve_endpoint(api_base: str, database: str, endpoint: str) -> str:
    """Build a full ``{api_base}/{database}/aggregations/{endpoint}`` URL."""
    base = api_base.rstrip("/")
    db = database.strip("/")
    suffix = endpoint.lstrip("/")
    return f"{base}/{db}/aggregations/{suffix}"


def _http_get_json(url: str, *, timeout: int = 30) -> dict[str, Any]:
    """Single GET → JSON; mirrors ``snapshot._http_get_json``."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def fetch_aggregation(
    endpoint: str,
    params: Mapping[str, Any] | None = None,
    *,
    api_base: str | None = None,
    database: str | None = None,
    ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    force_refresh: bool = False,
) -> AggregationResponse:
    """Fetch ``/aggregations/{endpoint}`` with disk caching.

    Parameters
    ----------
    endpoint
        Endpoint under ``/aggregations/`` (e.g. ``"sankey"``).
    params
        Query parameters merged into the URL (``levels=source,modality``,
        ``bucket=year``, etc.).
    api_base, database
        Override the URL components. Defaults read from
        ``EEGDASH_API_BASE`` / ``EEGDASH_DATABASE`` env vars, falling
        back to the production server.
    ttl_seconds
        Maximum disk cache age. Set to 0 to always re-fetch.
    force_refresh
        Skip the disk cache lookup but still write the response on
        success.

    Returns
    -------
    AggregationResponse
        Payload + provenance. Raises :class:`AggregationFetchError` when
        both the live call and the disk cache miss.

    """
    api_base = api_base or os.environ.get("EEGDASH_API_BASE", _DEFAULT_API_BASE)
    database = database or os.environ.get("EEGDASH_DATABASE", _DEFAULT_DATABASE)
    params = dict(params or {})

    url = _resolve_endpoint(api_base, database, endpoint)
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"

    cache_params = dict(params)
    cache_params["__api"] = api_base.rstrip("/")
    cache_params["__db"] = database
    cache_path = _cache_root() / _cache_key(endpoint, cache_params)

    if not force_refresh and ttl_seconds > 0:
        cached = _read_disk_cache(cache_path, ttl_seconds)
        if cached is not None:
            return AggregationResponse(payload=cached, source="cached", url=url)

    live_error: Exception | None = None
    try:
        payload = _http_get_json(url)
        if not payload.get("success", False):
            raise AggregationFetchError(f"{endpoint} responded success=False at {url}")
        _write_disk_cache(cache_path, payload)
        return AggregationResponse(payload=payload, source="live", url=url)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        live_error = exc

    # Last-resort: serve a stale cache rather than fail the docs build
    # outright. The legacy CSV fallback in each chart picks up from
    # here when even this fails.
    cached = _read_disk_cache(cache_path, ttl_seconds=float("inf"))
    if cached is not None:
        logger.warning(
            "aggregations live fetch failed; serving stale cache (%s)", live_error
        )
        return AggregationResponse(payload=cached, source="cached", url=url)

    raise AggregationFetchError(
        f"aggregations endpoint {url} unreachable and no cache: {live_error}"
    )


def api_base_from_snapshot(snapshot: Any) -> tuple[str, str]:
    """Read ``(api_base, database)`` off a :class:`DatasetSnapshot`.

    The snapshot doesn't expose these as public attributes today, but
    the manifest carries enough to reconstruct them. Falls back to the
    env-var defaults when called with ``None`` or an unrecognised
    object so the helper stays usable in tests and standalone scripts.
    """
    if snapshot is None:
        return (
            os.environ.get("EEGDASH_API_BASE", _DEFAULT_API_BASE),
            os.environ.get("EEGDASH_DATABASE", _DEFAULT_DATABASE),
        )
    manifest = getattr(snapshot, "manifest", None) or {}
    api_base = manifest.get("api_base") or os.environ.get(
        "EEGDASH_API_BASE", _DEFAULT_API_BASE
    )
    database = manifest.get("database") or os.environ.get(
        "EEGDASH_DATABASE", _DEFAULT_DATABASE
    )
    return api_base, database


__all__ = [
    "AggregationFetchError",
    "AggregationResponse",
    "api_base_from_snapshot",
    "fetch_aggregation",
]

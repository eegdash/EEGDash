"""Single data-access seam for the docs build.

:class:`DatasetSnapshot` owns everything between the EEGDash server and
any docs-build consumer: HTTP fetch, in-memory cache, disk cache,
package-CSV fallback, and the provenance that tells callers apart "live
API" from "stale cache" from "silent failure". One
``chart-data?include=montages,metadata`` call returns rows, montages,
and per-dataset metadata (authors / keywords / versions / description)
— the server assembles the metadata, so there is no separate NEMAR
round-trip. New consumers should import :class:`DatasetSnapshot`
directly.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Mapping

import pandas as pd

from ..paths import get_default_cache_dir

logger = logging.getLogger(__name__)

# Provenance tag the snapshot self-reports. CI gates on
# ``source == "live"`` before publishing the docs site.
Source = Literal["live", "cached", "package-csv"]


# Package-shipped registry CSV — the final fallback. A constant so tests
# can monkeypatch the path.
PACKAGE_CSV_PATH: Path = Path(__file__).resolve().parent / "dataset_summary.csv"


# Process-wide cache of built snapshots, keyed by ``(api_base, database,
# limit)`` so consumers on different shards never read each other's data.
# The lock guards concurrent first-builds; the fetch still happens once.
_INSTANCE_CACHE: dict[tuple[str, str, int], "DatasetSnapshot"] = {}
_INSTANCE_CACHE_LOCK = Lock()


# Server's ``/snapshots/{date}`` route shape — ISO ``YYYY-MM-DD`` only.
_SNAPSHOT_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Per-dataset metadata shapes
#
# Parsed from the ``datasets[i].metadata`` sub-object the server emits on
# ``chart-data?include=metadata``. The docs ``dataset_page`` renderers
# consume these dataclasses; this is the same shape the retired in-package
# NEMAR HTTP client used, minus the network/cache machinery.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NemarAuthor:
    """One author: ``name`` plus an optional bare 16-digit ``orcid``."""

    name: str
    orcid: str | None = None


@dataclass(frozen=True)
class NemarKeyword:
    """One keyword: free-text ``term`` plus optional controlled-vocab links."""

    term: str
    scheme: str | None = None
    value_uri: str | None = None


@dataclass(frozen=True)
class NemarVersion:
    """One entry in a dataset's version history (newest-first)."""

    version: str
    doi: str
    created_at: datetime


@dataclass(frozen=True)
class NemarMetadata:
    """Per-dataset metadata for the docs page; ``versions`` newest-first."""

    description: str | None
    license: str | None
    authors: tuple[NemarAuthor, ...]
    keywords: tuple[NemarKeyword, ...]
    versions: tuple[NemarVersion, ...]


class DatasetSnapshot:
    """A frozen view of the dataset catalog used by one docs build.

    Three entry points, one per lifecycle:

    - :meth:`build` — fetch the current catalog from the live API (disk
      / package-CSV fallback).
    - :meth:`load` — re-hydrate a snapshot previously written to disk.
    - :meth:`pin` — fetch a specific dated snapshot from
      ``/snapshots/{date}`` for reproducible builds.

    All three return the same object; provenance is self-reported via
    :attr:`source`, :attr:`fetched_at`, :attr:`api_errors`, and
    :attr:`pinned_at`. Read through the accessor methods (:meth:`rows`,
    :meth:`aggregations`, :meth:`montage`, :meth:`metadata`), not the
    underlying fields.
    """

    # ``__slots__`` keeps instances small. ``source``/``fetched_at``/
    # ``api_errors``/``manifest`` are class-level sentinels below (so
    # ``hasattr(DatasetSnapshot, 'source')`` is True for surface checks)
    # and are shadowed per-instance via ``__dict__``.
    __slots__ = ("_rows", "_aggregations", "_montages", "_metadata", "__dict__")

    # Class-level sentinels so ``hasattr(cls, '<name>')`` is True for the
    # surface check; instances shadow them with real values.
    source: Source = "package-csv"
    fetched_at: datetime | None = None
    api_errors: list[str] = []  # noqa: RUF012 — overridden per-instance
    manifest: dict[str, Any] = {}  # noqa: RUF012 — overridden per-instance
    pinned_at: str | None = None

    def __init__(
        self,
        *,
        rows: pd.DataFrame,
        aggregations: dict[str, Any],
        montages: Mapping[str, Mapping[str, Any]],
        source: Source,
        fetched_at: datetime,
        api_errors: list[str] | None = None,
        manifest: dict[str, Any] | None = None,
        pinned_at: str | None = None,
        metadata: Mapping[str, "NemarMetadata"] | None = None,
    ) -> None:
        # Frozen. Not a @dataclass because dataclass fields don't surface
        # in ``hasattr(cls, ...)`` without literal defaults, which the
        # surface check relies on.
        object.__setattr__(self, "_rows", rows)
        object.__setattr__(self, "_aggregations", aggregations)
        object.__setattr__(self, "_montages", montages)
        object.__setattr__(self, "_metadata", dict(metadata or {}))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "fetched_at", fetched_at)
        object.__setattr__(self, "api_errors", list(api_errors or []))
        object.__setattr__(self, "manifest", dict(manifest or {}))
        object.__setattr__(self, "pinned_at", pinned_at)

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: D401
        raise AttributeError(f"DatasetSnapshot is immutable; cannot assign to {name!r}")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"DatasetSnapshot(source={self.source!r}, "
            f"dataset_count={self.dataset_count}, "
            f"fetched_at={self.fetched_at.isoformat() if self.fetched_at else None!r}, "
            f"api_errors={len(self.api_errors)})"
        )

    # ----- accessors -----------------------------------------------------

    def rows(self) -> pd.DataFrame:
        """Return the per-dataset records as a DataFrame.

        Returns a copy so a consumer mutating the frame cannot leak
        state into the process-wide instance cache.
        """
        return self._rows.copy()

    def aggregations(self) -> dict[str, Any]:
        """Return the server-side pre-computed totals.

        Empty dict when a fallback fired (cached / package-csv build).
        """
        return dict(self._aggregations)

    def montage(self, dataset_id: str) -> Mapping[str, Any] | None:
        """Return the top montage dict for one dataset (case-insensitive), or ``None``."""
        if not dataset_id:
            return None
        # Try the exact key first (live path), then fall back to a
        # case-folded match (covers the docs consumer that lowercases
        # before lookup).
        result = self._montages.get(dataset_id) or self._montages.get(
            dataset_id.lower()
        )
        if result is None:
            return None
        # Return a shallow copy so callers cannot mutate the cached
        # mapping from underneath the snapshot.
        return dict(result)

    def metadata(self, dataset_id: str) -> "NemarMetadata | None":
        """Return per-dataset :class:`NemarMetadata` (case-insensitive), or ``None``.

        Populated from ``chart-data?include=metadata`` on a live build.
        Cached / package-CSV builds carry no metadata, so this returns
        ``None`` there — consumers render the empty placeholder, the same
        graceful-degradation contract :meth:`montage` follows.
        """
        if not dataset_id:
            return None
        # NemarMetadata is frozen, so no defensive copy is needed.
        return self._metadata.get(dataset_id) or self._metadata.get(dataset_id.lower())

    @property
    def dataset_count(self) -> int:
        """Number of dataset rows in this snapshot."""
        return int(len(self._rows))

    @property
    def schema_version(self) -> str | None:
        """Server-reported ``schema_version`` from the manifest, or ``None``."""
        value = self.manifest.get("schema_version")
        return value if isinstance(value, str) else None

    # ----- builders ------------------------------------------------------

    @classmethod
    def build(
        cls,
        api_base: str = "https://data.eegdash.org/api",
        database: str = "eegdash",
        *,
        limit: int | None = None,
        force_refresh: bool = False,
    ) -> "DatasetSnapshot":
        """Fetch / cache / fallback in one call.

        Idempotent across a process: two calls with the same
        ``(api_base, database, limit)`` triple return the same instance
        unless ``force_refresh`` is set.

        Resolution order (each level falls through on failure, and
        records the failure on :attr:`api_errors`):

        1. Live HTTP GET to ``{api_base}/{database}/datasets/chart-data``
           ``?limit={limit}&include=montages,metadata``.
        2. Disk cache at ``{get_default_cache_dir()}/snapshot_{db}.json``.
        3. Package CSV at :data:`PACKAGE_CSV_PATH`.

        Parameters
        ----------
        api_base : str
            Base URL of the EEGDash server, without a trailing slash.
        database : str
            Database / shard name.
        limit : int | None
            Max datasets to request; defaults to ``EEGDASH_DOC_LIMIT`` or ``1000``.
        force_refresh : bool
            Bypass in-memory and disk caches on entry; still falls back on failure.

        """
        resolved_limit = (
            limit
            if limit is not None
            else int(os.environ.get("EEGDASH_DOC_LIMIT", 1000))
        )
        cache_key = (api_base.rstrip("/"), database, resolved_limit)

        if not force_refresh:
            with _INSTANCE_CACHE_LOCK:
                cached = _INSTANCE_CACHE.get(cache_key)
            if cached is not None:
                return cached

        snapshot = _build_uncached(
            api_base=cache_key[0],
            database=database,
            limit=resolved_limit,
            force_refresh=force_refresh,
        )

        with _INSTANCE_CACHE_LOCK:
            # Another concurrent caller may have populated the slot
            # while we were fetching. Prefer the live result over a
            # racing fallback, but otherwise keep the first winner so
            # callers see one consistent instance.
            existing = _INSTANCE_CACHE.get(cache_key)
            if existing is None or (
                snapshot.source == "live" and existing.source != "live"
            ):
                _INSTANCE_CACHE[cache_key] = snapshot
            else:
                snapshot = existing

        _log_provenance(snapshot)
        return snapshot

    @classmethod
    def load(cls, manifest_path: str | Path) -> "DatasetSnapshot":
        """Re-hydrate a previously-built snapshot from disk.

        ``manifest_path`` is a manifest JSON file; the rows ride in a
        sibling ``*.rows.json`` written by :meth:`build`. Metadata is not
        persisted to disk, so a loaded snapshot carries none.
        """
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        rows_path = manifest_path.with_suffix(".rows.json")
        if rows_path.exists() and rows_path.stat().st_size > 0:
            rows = pd.read_json(rows_path, orient="records")
        else:
            rows = pd.DataFrame()

        fetched_at_raw = manifest.get("fetched_at")
        if fetched_at_raw:
            fetched_at = datetime.fromisoformat(fetched_at_raw)
        else:
            fetched_at = _utcnow()

        return cls(
            rows=rows,
            aggregations=manifest.get("aggregations") or {},
            montages=manifest.get("montages") or {},
            source=manifest.get("source", "cached"),
            fetched_at=fetched_at,
            api_errors=list(manifest.get("api_errors") or []),
            manifest=manifest,
        )

    @classmethod
    def pin(
        cls,
        date: str,
        *,
        api_base: str | None = None,
        database: str | None = None,
    ) -> "DatasetSnapshot":
        """Pin to the server's dated ``/snapshots/{date}`` for reproducible builds.

        Reconstructs a snapshot from the frozen ``chart_data`` +
        ``build_manifest`` payload. Does NOT touch the disk cache or the
        in-process ``_INSTANCE_CACHE`` used by :meth:`build`. ``source``
        is ``"live"``, ``fetched_at`` is the payload's creation time, and
        ``pinned_at`` is the requested date.

        Parameters
        ----------
        date : str
            ISO ``YYYY-MM-DD``; anything else raises ``ValueError``.
        api_base : str | None
            Server base URL; defaults to ``https://data.eegdash.org/api``.
        database : str | None
            Database / shard name; defaults to ``eegdash``.

        Raises
        ------
        ValueError
            If ``date`` does not match the ``YYYY-MM-DD`` ISO shape.
        LookupError
            If the server returns 404 for the requested date.

        """
        if not isinstance(date, str) or not _SNAPSHOT_DATE_RE.match(date):
            raise ValueError(
                f"date must match YYYY-MM-DD; got {date!r}. "
                "Use DatasetSnapshot.build() for the current state."
            )

        resolved_api_base = (api_base or "https://data.eegdash.org/api").rstrip("/")
        resolved_database = database or "eegdash"

        url = f"{resolved_api_base}/{resolved_database}/snapshots/{date}"
        try:
            payload = _http_get_json(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise LookupError(f"snapshot for {date} not found") from exc
            raise

        chart_data = payload.get("chart_data") or {}
        build_manifest = payload.get("build_manifest") or {}
        created_at_raw = payload.get("created_at")
        if not isinstance(created_at_raw, str):
            raise ValueError(
                f"snapshot payload for {date} missing or malformed 'created_at'"
            )
        # ``datetime.fromisoformat`` in 3.10 doesn't accept the
        # trailing ``Z``; normalise to ``+00:00`` so the parse always
        # round-trips.
        fetched_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))

        datasets = chart_data.get("datasets") or []
        rows = _rows_from_chart_data(datasets)
        aggregations = chart_data.get("aggregations") or {}
        montages = _montages_from_chart_data(datasets)
        metadata = _metadata_from_chart_data(datasets)

        return cls(
            rows=rows,
            aggregations=aggregations,
            montages=montages,
            source="live",
            fetched_at=fetched_at,
            api_errors=[],
            manifest=dict(build_manifest),
            pinned_at=date,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _resolve_author_year(
    *,
    name_source: str,
    raw_aliases: list[str] | None,
    explicit: object = None,
) -> str | None:
    """Pick a single ``FirstAuthorSurnameYear`` from catalog metadata.

    Explicit ``author_year`` wins; else the first alias when
    ``name_source == "author_year"``; else ``None``.
    """
    explicit_str = str(explicit).strip() if explicit else ""
    if explicit_str:
        return explicit_str
    if name_source and name_source.strip().lower() == "author_year" and raw_aliases:
        first = str(raw_aliases[0]).strip()
        return first or None
    return None


def _disk_cache_path(database: str) -> Path:
    """Where the JSON disk cache for ``database`` lives."""
    cache_dir = get_default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize the database name; we never accept input here that
    # could contain path separators, but defence in depth never hurts.
    safe = "".join(c for c in database if c.isalnum() or c in {"_", "-"})
    return cache_dir / f"snapshot_{safe or 'default'}.json"


def _montages_sidecar_path(database: str) -> Path:
    """Where the per-dataset montage cache for ``database`` lives (JSON)."""
    cache_dir = get_default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in database if c.isalnum() or c in {"_", "-"})
    return cache_dir / f"snapshot_{safe or 'default'}_montages.json"


def _http_get_json(url: str, *, timeout: int = 30) -> dict[str, Any]:
    """Single GET → JSON helper. Raises on any failure."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _try_fetch_chart_data(
    api_base: str, database: str, limit: int, *, errors: list[str]
) -> (
    tuple[
        pd.DataFrame,
        dict[str, Any],
        dict[str, dict[str, Any]],
        dict[str, "NemarMetadata"],
    ]
    | None
):
    """Attempt the rich chart-data endpoint (rows, aggregations, montages, metadata).

    Requests ``?include=montages,metadata`` so the top montage and the
    per-dataset metadata sub-object ride along in one round-trip. Returns
    ``None`` on failure.
    """
    url = (
        f"{api_base}/{database}/datasets/chart-data"
        f"?limit={limit}&include=montages,metadata"
    )
    try:
        data = _http_get_json(url)
    except urllib.error.HTTPError as exc:
        errors.append(f"chart-data {exc.code} at {url}: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001 — network library raises broadly
        errors.append(f"chart-data error at {url}: {exc}")
        return None

    if not data.get("success"):
        errors.append(f"chart-data returned success=False at {url}")
        return None

    datasets = data.get("datasets", []) or []
    rows = _rows_from_chart_data(datasets)
    aggregations = data.get("aggregations") or {}
    montages = _montages_from_chart_data(datasets)
    metadata = _metadata_from_chart_data(datasets)
    return rows, aggregations, montages, metadata


def _try_fetch_summary(
    api_base: str, database: str, limit: int, *, errors: list[str]
) -> pd.DataFrame | None:
    """Attempt the legacy summary endpoint. Returns ``None`` on failure.

    A compatibility fallback for old/partial server deployments that
    don't yet serve ``chart-data``. It carries no montage or metadata, so
    those accessors return ``None`` until chart-data comes back online.
    """
    url = f"{api_base}/{database}/datasets/summary?limit={limit}"
    try:
        data = _http_get_json(url)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"summary error at {url}: {exc}")
        return None

    if not data.get("success"):
        errors.append(f"summary returned success=False at {url}")
        return None

    return _rows_from_summary(data.get("data", []) or [])


def _read_disk_cache(path: Path, *, errors: list[str]) -> pd.DataFrame | None:
    """Re-hydrate a JSON disk cache. Returns ``None`` on failure/absence."""
    if not path.exists():
        return None
    try:
        if path.stat().st_size == 0:
            return None
        return pd.read_json(path, orient="records")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"disk cache read failed at {path}: {exc}")
        return None


def _read_package_csv(path: Path, *, errors: list[str]) -> pd.DataFrame | None:
    """Final-resort: read the registry CSV shipped with the package."""
    if not path.exists():
        errors.append(f"package CSV missing at {path}")
        return None
    try:
        return pd.read_csv(path, comment="#", skip_blank_lines=True)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"package CSV read failed at {path}: {exc}")
        return None


def _write_disk_cache(df: pd.DataFrame, path: Path) -> None:
    """Persist a freshly-fetched payload as JSON. Best-effort; failures swallowed."""
    try:
        path.write_text(df.to_json(orient="records"), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.debug("snapshot disk cache write failed at %s: %s", path, exc)


def _write_montages_sidecar(
    montages: Mapping[str, Mapping[str, Any]], path: Path
) -> None:
    """Persist the parsed montages dict as JSON. Best-effort."""
    try:
        path.write_text(
            json.dumps(dict(montages), sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("snapshot montages sidecar write failed at %s: %s", path, exc)


def _read_montages_sidecar(
    path: Path, *, errors: list[str]
) -> dict[str, dict[str, Any]]:
    """Re-hydrate the montages sidecar JSON. Empty dict on any failure.

    Called from the disk-cache fallback path. A missing or corrupt
    sidecar is downgraded to "no montages this build" — the consumer
    renders the empty placeholder rather than crashing.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"montages sidecar read failed at {path}: {exc}")
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)}


def _build_uncached(
    *,
    api_base: str,
    database: str,
    limit: int,
    force_refresh: bool,
) -> "DatasetSnapshot":
    """The real fetch/fallback chain. Always returns a snapshot.

    Resolution order: live chart-data → disk cache → package CSV. The
    legacy ``/datasets/summary`` endpoint is no longer consulted; the
    server's chart-data endpoint is the single live source.
    """
    errors: list[str] = []

    # 1. Live fetch — the rich chart-data endpoint (rows + montages + metadata).
    chart_result = _try_fetch_chart_data(api_base, database, limit, errors=errors)
    if chart_result is not None:
        rows, aggregations, montages, metadata = chart_result
        if not rows.empty:
            snapshot = _live_snapshot(
                rows,
                aggregations,
                montages,
                metadata,
                errors,
                api_base=api_base,
                database=database,
            )
            _write_disk_cache(rows, _disk_cache_path(database))
            _write_montages_sidecar(montages, _montages_sidecar_path(database))
            return snapshot
        errors.append("chart-data returned 0 datasets")

    # Legacy summary fallback — for old/partial server deployments that
    # don't serve chart-data. No montage or metadata rides along, so those
    # accessors return None on this path.
    summary_rows = _try_fetch_summary(api_base, database, limit, errors=errors)
    if summary_rows is not None and not summary_rows.empty:
        snapshot = _live_snapshot(
            summary_rows, {}, {}, {}, errors, api_base=api_base, database=database
        )
        _write_disk_cache(summary_rows, _disk_cache_path(database))
        return snapshot
    if summary_rows is not None and summary_rows.empty:
        errors.append("summary returned 0 datasets")

    # 2. Disk cache fallback (skipped when force_refresh=True, but only
    # after the live attempt above). On any fallback path the live API was
    # unreachable, so ``/build-manifest`` is skipped and the local
    # projection truthfully reports ``source != "live"``. Cached rows carry
    # no metadata (it is not persisted), so :meth:`metadata` returns None.
    if not force_refresh:
        disk_path = _disk_cache_path(database)
        cached_rows = _read_disk_cache(disk_path, errors=errors)
        if cached_rows is not None and not cached_rows.empty:
            cached_at = _mtime(disk_path)
            cached_montages = _read_montages_sidecar(
                _montages_sidecar_path(database), errors=errors
            )
            return DatasetSnapshot(
                rows=cached_rows,
                aggregations={},
                montages=cached_montages,
                source="cached",
                fetched_at=cached_at,
                api_errors=list(errors),
                manifest=_local_manifest_projection(
                    source="cached",
                    dataset_count=len(cached_rows),
                    fetched_at=cached_at,
                ),
            )

    # 3. Package CSV — the floor. We still return a snapshot even if this
    # fails; the caller (CI) can inspect ``api_errors`` and refuse to
    # publish.
    csv_rows = _read_package_csv(PACKAGE_CSV_PATH, errors=errors)
    if csv_rows is None:
        csv_rows = pd.DataFrame()
    fetched_at = _mtime(PACKAGE_CSV_PATH) if PACKAGE_CSV_PATH.exists() else _utcnow()

    return DatasetSnapshot(
        rows=csv_rows,
        aggregations={},
        montages={},
        source="package-csv",
        fetched_at=fetched_at,
        api_errors=list(errors),
        manifest=_local_manifest_projection(
            source="package-csv",
            dataset_count=len(csv_rows),
            fetched_at=fetched_at,
        ),
    )


def _live_snapshot(
    rows: pd.DataFrame,
    aggregations: dict[str, Any],
    montages: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, "NemarMetadata"],
    errors: list[str],
    *,
    api_base: str,
    database: str,
) -> "DatasetSnapshot":
    """Build a ``source="live"`` snapshot.

    Also hits ``/build-manifest`` and surfaces its response verbatim on
    :attr:`DatasetSnapshot.manifest`; the local projection is the
    fallback when that endpoint is missing or erroring.
    """
    fetched_at = _utcnow()
    dataset_count = len(rows)

    server_manifest = _try_fetch_build_manifest(
        api_base=api_base, database=database, errors=errors
    )
    if server_manifest is not None:
        # If the server's dataset_count disagrees with the rows we loaded,
        # surface the divergence on ``api_errors`` (a benign ingestion-window
        # skew is possible) rather than crashing.
        server_count = server_manifest.get("dataset_count")
        if isinstance(server_count, int) and server_count != dataset_count:
            errors.append(
                f"build-manifest dataset_count={server_count} "
                f"but snapshot.rows()={dataset_count}"
            )
        manifest = dict(server_manifest)
    else:
        manifest = _local_manifest_projection(
            source="live",
            dataset_count=dataset_count,
            fetched_at=fetched_at,
        )

    return DatasetSnapshot(
        rows=rows,
        aggregations=aggregations,
        montages=montages,
        source="live",
        fetched_at=fetched_at,
        # A missing ``/build-manifest`` endpoint is recorded as context,
        # not a build failure.
        api_errors=list(errors),
        manifest=manifest,
        metadata=metadata,
    )


def _try_fetch_build_manifest(
    api_base: str,
    database: str,
    *,
    timeout: float = 5.0,
    errors: list[str] | None = None,
) -> dict[str, Any] | None:
    """Fetch the server's ``/build-manifest`` JSON, or ``None`` on failure.

    Best-effort: any failure returns ``None`` and appends the error text
    to ``errors`` (when provided). Short timeout so a manifest outage
    can't stall the docs build.
    """
    url = f"{api_base}/{database}/build-manifest"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)
    except urllib.error.HTTPError as exc:
        if errors is not None:
            errors.append(f"build-manifest {exc.code} at {url}: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001 — network/JSON libraries raise broadly
        if errors is not None:
            errors.append(f"build-manifest error at {url}: {exc}")
        return None


def _local_manifest_projection(
    *, source: Source, dataset_count: int, fetched_at: datetime
) -> dict[str, Any]:
    """Locally-projected manifest dict, used when the server is unreachable.

    Keeps :attr:`DatasetSnapshot.manifest` from ever being ``None``.
    """
    return {
        "source": source,
        "dataset_count": dataset_count,
        "fetched_at": fetched_at.isoformat(),
    }


def _mtime(path: Path) -> datetime:
    """Return the mtime of ``path`` as a tz-aware UTC datetime."""
    try:
        ts = path.stat().st_mtime
    except OSError:
        return _utcnow()
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _log_provenance(snapshot: "DatasetSnapshot") -> None:
    """Emit the provenance line (format is grepped by CI; do not change)."""
    logger.info(
        "DatasetSnapshot source=%s dataset_count=%d fetched_at=%s",
        snapshot.source,
        snapshot.dataset_count,
        snapshot.fetched_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# Metadata mappers
# ---------------------------------------------------------------------------


def _clean_optional(value: object) -> str | None:
    """Return ``None`` for empty / ``None`` / explicit-null strings."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "nan"}:
        return None
    return text


def _clean_orcid(value: object) -> str | None:
    """Normalise an ORCID to the bare 16-digit id (no URL prefix), or ``None``."""
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    for prefix in ("https://orcid.org/", "http://orcid.org/"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text or None


def _parse_iso(raw: object) -> datetime:
    """Parse an ISO-8601 timestamp (``Z`` tolerated); ``utcnow`` on failure."""
    if isinstance(raw, str) and raw.strip():
        try:
            return datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
        except ValueError:
            pass
    return _utcnow()


def _metadata_from_chart_data(
    datasets: list[dict[str, Any]],
) -> dict[str, "NemarMetadata"]:
    """Pull ``datasets[i].metadata`` into a ``{lowercased id: NemarMetadata}`` map.

    Datasets without a usable ``metadata`` sub-object are silently
    skipped, so a missing key means "no metadata served for this dataset"
    (the consumer renders the empty placeholder).
    """
    out: dict[str, NemarMetadata] = {}
    for ds in datasets:
        ds_id = str(ds.get("dataset_id") or "").strip()
        if not ds_id:
            continue
        meta = ds.get("metadata")
        if not isinstance(meta, dict) or not meta:
            continue
        out[ds_id.lower()] = _build_metadata(ds, meta)
    return out


def _build_metadata(ds: Mapping[str, Any], meta: Mapping[str, Any]) -> "NemarMetadata":
    """Construct a :class:`NemarMetadata` from one chart-data row + its metadata.

    Authors are dicts (``{name, orcid}``) per the server contract.
    """
    authors = tuple(
        NemarAuthor(
            name=str(a.get("name") or "").strip(), orcid=_clean_orcid(a.get("orcid"))
        )
        for a in (meta.get("authors") or [])
        if isinstance(a, dict) and str(a.get("name") or "").strip()
    )
    keywords = tuple(
        NemarKeyword(
            term=str(k.get("term") or "").strip(),
            scheme=_clean_optional(k.get("scheme")),
            value_uri=_clean_optional(k.get("value_uri")),
        )
        for k in (meta.get("keywords") or [])
        if isinstance(k, dict) and str(k.get("term") or "").strip()
    )
    versions = sorted(
        (
            NemarVersion(
                version=str(v.get("version") or "").strip(),
                doi=str(v.get("doi") or "").strip(),
                created_at=_parse_iso(v.get("created_at")),
            )
            for v in (meta.get("versions") or [])
            if isinstance(v, dict) and str(v.get("version") or "").strip()
        ),
        key=lambda v: v.created_at,
        reverse=True,
    )
    return NemarMetadata(
        description=_clean_optional(meta.get("description")),
        license=_clean_optional(ds.get("license")),
        authors=authors,
        keywords=keywords,
        versions=tuple(versions),
    )


# ---------------------------------------------------------------------------
# Row mappers
# ---------------------------------------------------------------------------


def _human_readable_size(num_bytes: int | float | None) -> str:
    """Convert bytes to human-readable string."""
    if num_bytes is None or num_bytes == 0:
        return "Unknown"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def _normalize_tag_value(val: Any) -> str:
    if isinstance(val, list):
        return ", ".join(val) if val else ""
    return val or ""


# Pretty modality label used in the per-page caption.
_MONTAGE_MODALITY_LABEL = {
    "eeg": "EEG",
    "ieeg": "iEEG",
    "meg": "MEG",
    "nirs": "fNIRS",
    "emg": "EMG",
}


def _project_montage(montage: Mapping[str, Any]) -> dict[str, Any]:
    """Project a server montage into the viewer-friendly dict.

    Superset of the server fields plus the legacy ``electrode-layouts``
    aliases (``label``, ``n_channels``, ``montage_id``).
    """
    modality = str(montage.get("modality") or "").strip().lower()
    n_sensors = int(montage.get("n_sensors") or 0)
    mod_label = _MONTAGE_MODALITY_LABEL.get(modality, modality.upper() or "Sensors")
    label = f"{mod_label} · {n_sensors} sensors" if n_sensors else mod_label

    return {
        # Server fields, passed through unchanged.
        "hash": str(montage.get("hash") or "").strip(),
        "subject_count": int(montage.get("subject_count") or 0),
        "modality": modality or None,
        "n_sensors": n_sensors,
        "space_declared": montage.get("space_declared"),
        "units_declared": montage.get("units_declared"),
        "channel_names": list(montage.get("channel_names") or []),
        # Viewer-friendly aliases. The retired build script used these
        # exact key names; we keep them so the section formatter is a
        # straight find-and-replace migration.
        "label": label,
        "n_channels": n_sensors,
        "montage_id": str(montage.get("hash") or "").strip(),
    }


def _montages_from_chart_data(
    datasets: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Pull ``datasets[i].montage`` into a ``{lowercased id: dict}`` map.

    Datasets without a usable top montage are silently skipped.
    """
    montages: dict[str, dict[str, Any]] = {}
    for ds in datasets:
        ds_id = str(ds.get("dataset_id") or "").strip().lower()
        if not ds_id:
            continue
        montage = ds.get("montage")
        if not isinstance(montage, dict) or not montage:
            continue
        # Require at least one informative field; a future server could
        # emit only the hash without the joined registry doc.
        if not (montage.get("hash") or montage.get("n_sensors")):
            continue
        montages[ds_id] = _project_montage(montage)
    return montages


def _rows_from_chart_data(datasets: list[dict[str, Any]]) -> pd.DataFrame:
    """Map the chart-data response shape into the canonical DataFrame.

    Mirrors the legacy ``fetch_chart_data_from_api`` columns exactly.
    """
    rows: list[dict[str, Any]] = []
    for ds in datasets:
        ds_id = ds.get("dataset_id", "").strip()

        demographics = ds.get("demographics") or {}
        tags = ds.get("tags") or {}
        clinical = ds.get("clinical") or {}
        paradigm = ds.get("paradigm") or {}
        timestamps = ds.get("timestamps") or {}

        recording_modality = ds.get("recording_modality") or []
        if isinstance(recording_modality, str):
            recording_modality = [recording_modality]

        type_subject = _normalize_tag_value(tags.get("pathology"))
        if not type_subject and clinical.get("is_clinical"):
            type_subject = clinical.get("purpose") or "Clinical"
        elif not type_subject and clinical.get("is_clinical") is False:
            type_subject = "Healthy"
        elif not type_subject:
            type_subject = "Unknown"

        modality_of_exp = _normalize_tag_value(tags.get("modality"))
        if not modality_of_exp:
            modality_of_exp = paradigm.get("modality", "")

        type_of_exp = _normalize_tag_value(tags.get("type"))
        if not type_of_exp:
            type_of_exp = paradigm.get("cognitive_domain", "")

        canonical_list = ds.get("canonical_name") or []
        name_source = (ds.get("name_source") or "").strip()
        author_year_value = (
            _resolve_author_year(
                name_source=name_source,
                raw_aliases=canonical_list,
                explicit=ds.get("author_year"),
            )
            or ""
        )
        rows.append(
            {
                "dataset": ds_id,
                "canonical_name": json.dumps(canonical_list),
                "name_source": name_source,
                "author_year": author_year_value,
                "dataset_title": ds.get("computed_title") or ds.get("name", ""),
                "n_subjects": demographics.get("subjects_count") or 0,
                "n_records": ds.get("total_files") or 0,
                "n_tasks": len(ds.get("tasks") or []),
                "tasks": json.dumps(ds.get("tasks") or []),
                "n_sessions": len(ds.get("sessions") or []),
                "record_modality": ", ".join(recording_modality),
                "recording_modality": ", ".join(recording_modality),
                "modality of exp": modality_of_exp,
                "type of exp": type_of_exp,
                "Type Subject": type_subject,
                "size_bytes": ds.get("size_bytes") or 0,
                "size": ds.get("size_human")
                or _human_readable_size(ds.get("size_bytes")),
                "source": ds.get("source") or "unknown",
                "license": ds.get("license", ""),
                "doi": ds.get("dataset_doi", ""),
                "nchans_set": json.dumps(ds.get("nchans_counts") or []),
                "sampling_freqs": json.dumps(ds.get("sfreq_counts") or []),
                "dataset_created_at": timestamps.get("dataset_created_at", ""),
                "nemar_citation_count": ds.get("nemar_citation_count"),
                "duration_hours_total": (ds.get("total_duration_s") or 0) / 3600
                or None,
            }
        )

    return pd.DataFrame(rows)


def _rows_from_summary(datasets: list[dict[str, Any]]) -> pd.DataFrame:
    """Map the legacy ``/datasets/summary`` response.

    Same column set as the chart-data path so consumers see one
    canonical schema regardless of which endpoint produced the data.
    """
    rows: list[dict[str, Any]] = []
    for ds in datasets:
        ds_id = ds.get("dataset_id", "").strip()

        nchans_list = ds.get("nchans_counts") or []
        sfreq_list = ds.get("sfreq_counts") or []

        demographics = ds.get("demographics", {}) or {}
        recording_modality = ds.get("recording_modality", []) or []
        if isinstance(recording_modality, str):
            recording_modality = [recording_modality]

        tags = ds.get("tags", {}) or {}
        clinical = ds.get("clinical", {}) or {}
        paradigm = ds.get("paradigm", {}) or {}

        pathology_list = tags.get("pathology", [])
        if pathology_list and isinstance(pathology_list, list):
            type_subject = ", ".join(pathology_list)
        elif clinical.get("is_clinical"):
            type_subject = clinical.get("purpose") or "Unspecified Clinical"
        elif clinical.get("is_clinical") is False:
            type_subject = "Healthy"
        else:
            type_subject = ""

        modality_list = tags.get("modality", [])
        if modality_list and isinstance(modality_list, list):
            paradigm_modality = ", ".join(modality_list)
        else:
            paradigm_modality = paradigm.get("modality") or ""

        type_list = tags.get("type", [])
        if type_list and isinstance(type_list, list):
            cognitive_domain = ", ".join(type_list)
        else:
            cognitive_domain = paradigm.get("cognitive_domain") or ""

        canonical_list = ds.get("canonical_name") or []
        name_source = (ds.get("name_source") or "").strip()
        author_year_value = (
            _resolve_author_year(
                name_source=name_source,
                raw_aliases=canonical_list,
                explicit=ds.get("author_year"),
            )
            or ""
        )
        rows.append(
            {
                "dataset": ds_id,
                "canonical_name": json.dumps(canonical_list),
                "name_source": name_source,
                "author_year": author_year_value,
                "n_subjects": demographics.get("subjects_count", 0) or 0,
                "n_records": ds.get("total_files", 0) or 0,
                "n_tasks": len(ds.get("tasks", []) or []),
                "tasks": json.dumps(ds.get("tasks", []) or []),
                "modality of exp": paradigm_modality,
                "type of exp": cognitive_domain,
                "Type Subject": type_subject,
                "duration_hours_total": (ds.get("total_duration_s") or 0) / 3600
                or None,
                "size_bytes": ds.get("size_bytes") or 0,
                "size": ds.get("size_human")
                or _human_readable_size(ds.get("size_bytes")),
                "source": ds.get("source") or "unknown",
                "dataset_title": ds.get("computed_title") or ds.get("name", ""),
                "record_modality": ", ".join(recording_modality),
                "nchans_set": json.dumps(nchans_list),
                "sampling_freqs": json.dumps(sfreq_list),
                "license": ds.get("license", ""),
                "doi": ds.get("dataset_doi", ""),
                "nemar_citation_count": ds.get("nemar_citation_count"),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test affordance
# ---------------------------------------------------------------------------


def _reset_instance_cache_for_testing() -> None:
    """Wipe the process-wide snapshot cache (used by tests / force-refresh)."""
    with _INSTANCE_CACHE_LOCK:
        _INSTANCE_CACHE.clear()


__all__ = [
    "DatasetSnapshot",
    "NemarAuthor",
    "NemarKeyword",
    "NemarMetadata",
    "NemarVersion",
    "PACKAGE_CSV_PATH",
    "Source",
]

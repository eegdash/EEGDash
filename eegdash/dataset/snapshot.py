"""Single data-access seam for the docs build.

The :class:`DatasetSnapshot` module owns *everything* between the
EEGDash server and any docs-build consumer: HTTP fetch, retry,
in-memory cache, disk cache, package-CSV fallback, and the provenance
that lets callers (and CI) tell apart "live API" from "stale cache" from
"silent failure."

Before this module landed there were three near-duplicate paths to the
same data:

1. ``fetch_datasets_from_api`` in :mod:`eegdash.dataset.registry`
   (DataFrame; disk cache; 5-level fallback; silent on failure).
2. ``fetch_chart_data_from_api`` in :mod:`eegdash.dataset.registry`
   ((DataFrame, aggregations) tuple; no disk cache; falls back to
   ``fetch_datasets_from_api`` on 404).
3. ``_load_dataset_summary_from_api`` in ``docs/source/conf.py``, which
   memoised the first DataFrame in a module-level global *not keyed by
   API URL or database* — so any future parameterisation silently
   returned stale data.

Those three are now thin compatibility shims over this module. New
consumers should import :class:`DatasetSnapshot` directly.

Cross-references:

- ``docs_pipeline_architecture_review.md`` §3 B1 (the surface), B2
  (tagged failure modes).
- ``docs_pipeline_validation_plan.md`` §3 step 5 (B1) and §3 step 6 (B2).
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Mapping

import pandas as pd

from ..paths import get_default_cache_dir
from ._excluded import EXCLUDED_DATASETS  # noqa: F401 — re-exported

logger = logging.getLogger(__name__)

# Provenance tag the snapshot self-reports. CI gates on
# ``source == "live"`` before publishing the docs site.
Source = Literal["live", "cached", "package-csv"]


# Package-shipped registry CSV — used as the final fallback. Kept in a
# constant so tests can monkeypatch the path without touching the
# package layout.
PACKAGE_CSV_PATH: Path = Path(__file__).resolve().parent / "dataset_summary.csv"


# Process-wide cache of built snapshots. Keyed by ``(api_base, database,
# limit)`` so two consumers hitting different shards never read each
# other's data — the explicit bug ``_DATASET_SUMMARY_CACHE`` had in
# ``conf.py``. The lock guards against concurrent first-builds from
# parallel Sphinx workers; the actual fetch still happens once.
_INSTANCE_CACHE: dict[tuple[str, str, int], "DatasetSnapshot"] = {}
_INSTANCE_CACHE_LOCK = Lock()


# Match the server's ``/snapshots/{date}`` route shape — ISO ``YYYY-MM-DD``
# only. Reject anything else early with a ``ValueError`` so a typo
# doesn't reach the network as a wildcard.
_SNAPSHOT_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class DatasetSnapshot:
    """A frozen view of the dataset catalog used by one docs build.

    There are three equal entry points, one per lifecycle:

    - :meth:`build` — fetch the *current* catalog from the live API
      (with disk / package-CSV fallback). Use this for normal docs
      builds that always want the freshest data.
    - :meth:`load` — re-hydrate a snapshot that was previously written
      to disk (manifest JSON + parquet rows). Use this in tests, or
      when round-tripping a snapshot through CI artefacts.
    - :meth:`pin` — fetch a *specific dated* snapshot from the server's
      ``/snapshots/{date}`` endpoint, byte-for-byte the same payload
      that was frozen on that date. Use this for reproducible docs
      builds (CI pinned to a fixed date) or for diffing two builds.

    All three return the same kind of object; consumers should never
    have to know which classmethod produced their snapshot. The
    :attr:`source`, :attr:`fetched_at`, and :attr:`pinned_at`
    properties self-report the provenance honestly.

    All consumers should read through the accessor methods rather than
    poking at the underlying fields directly — the field set is
    allowed to grow (see :attr:`manifest` for A2 and :attr:`pinned_at`
    for A6).

    Attributes
    ----------
    source : {"live", "cached", "package-csv"}
        Where this snapshot's data came from. Never silent on failure —
        if the live API throws, :attr:`api_errors` carries the
        exception text and :attr:`source` falls back to ``"cached"``
        or ``"package-csv"``.
    fetched_at : datetime
        Wall-clock instant the snapshot's underlying data was actually
        produced. For ``source == "live"`` this is the moment the HTTP
        call returned; for ``"cached"`` it is the disk cache's mtime;
        for ``"package-csv"`` it is the package CSV's mtime. Always
        UTC.
    dataset_count : int
        ``len(self.rows())``, exposed as a property so consumers can
        ask "is this build healthy?" without materialising the
        DataFrame.
    api_errors : list[str]
        Exception strings collected during fetch / fallback. Empty when
        ``source == "live"`` and the call succeeded; non-empty whenever
        a fallback fired.
    manifest : dict[str, Any]
        The server's ``/build-manifest`` response when reachable; falls
        back to a locally-projected ``{"dataset_count", "source",
        "fetched_at"}`` dict when the endpoint is offline, returns an
        error, or the snapshot resolved from the disk cache / package
        CSV without a live API call. Either way the field is never
        ``None``, so downstream consumers can read it unconditionally.

        When the server manifest is present it carries — at minimum —
        ``dataset_count``, ``schema_version``, ``last_ingested_at``,
        ``last_stats_computed_at``, ``git_sha``, and ``name_coverage``.
        The schema is owned by the server (review §3 A2); this snapshot
        re-exposes it verbatim.
    pinned_at : str | None
        The ISO date (``YYYY-MM-DD``) this snapshot was pinned to via
        :meth:`pin`, or ``None`` when the snapshot came from
        :meth:`build` or :meth:`load`. Lets a consumer tell apart a
        snapshot pinned to ``2026-04-01`` from a live build on the
        same source tree.

    """

    # ``__slots__`` keeps instances small and prevents accidental
    # attribute creation. We deliberately don't list ``source``,
    # ``fetched_at``, ``api_errors``, ``manifest`` here — they live as
    # class-level placeholders below (so ``hasattr(DatasetSnapshot,
    # 'source')`` returns True for the surface check in the
    # validation plan) and as real per-instance values written via
    # ``__dict__``.
    __slots__ = ("_rows", "_aggregations", "_montages", "__dict__")

    # Class-level sentinels so ``hasattr(DatasetSnapshot, '<name>')``
    # returns True for the post-condition surface check in
    # ``docs_pipeline_validation_plan.md`` §3 step 5. Instances shadow
    # them with real values via their own ``__dict__``.
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
    ) -> None:
        # Snapshots are conceptually frozen. We don't use @dataclass
        # because dataclass fields don't surface in ``dir(cls)`` /
        # ``hasattr(cls, ...)`` without literal defaults — and the
        # post-condition check in the validation plan expects every
        # named member to be discoverable on the class.
        object.__setattr__(self, "_rows", rows)
        object.__setattr__(self, "_aggregations", aggregations)
        object.__setattr__(self, "_montages", montages)
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

        Empty dict when the underlying call was the legacy summary
        endpoint (which has no aggregations block) or a fallback fired.
        """
        return dict(self._aggregations)

    def montage(self, dataset_id: str) -> Mapping[str, Any] | None:
        """Return the top montage for one dataset, or ``None``.

        Lookup is case-insensitive on the catalog ``dataset_id``
        (e.g. ``"ds002718"`` or ``"DS002718"``). Returns ``None`` when
        no montage was joined into this snapshot (e.g. the server
        omitted one, or the snapshot resolved from the disk-cache /
        package-CSV fallback path and never received montage data in
        the first place).

        Population is driven server-side by ``?include=montages`` on
        the ``/datasets/chart-data`` request (see arch #5). The shape
        is the registry montage doc plus a couple of viewer-friendly
        derived fields:

        - ``hash`` — 16-char montage registry id
        - ``modality`` — ``"eeg"``, ``"meg"``, ``"ieeg"``, ...
        - ``n_sensors`` — channel count for this montage
        - ``space_declared`` / ``units_declared`` — BIDS coords schema
        - ``channel_names`` — list of strings (truncated server-side)
        - ``label`` — pretty caption (``"EEG · 64 sensors"``)
        - ``n_channels`` — alias of ``n_sensors`` for backward compat
        - ``montage_id`` — alias of ``hash`` for backward compat
        """
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

    @property
    def dataset_count(self) -> int:
        """Number of dataset rows in this snapshot."""
        return int(len(self._rows))

    @property
    def schema_version(self) -> str | None:
        """The server-reported ``schema_version`` from ``/build-manifest``.

        Returns ``None`` when the snapshot fell back to the local
        manifest projection (i.e. the server wasn't reached). Convenience
        accessor — equivalent to ``self.manifest.get("schema_version")``.
        """
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

        1. Live HTTP GET to
           ``{api_base}/{database}/datasets/chart-data?limit={limit}``.
        2. Live HTTP GET to ``/datasets/summary?limit={limit}`` (legacy
           shape; no ``aggregations`` block).
        3. Disk cache at
           ``{get_default_cache_dir()}/snapshot_{db}.parquet``.
        4. Package CSV at :data:`PACKAGE_CSV_PATH`.

        Parameters
        ----------
        api_base : str
            Base URL of the EEGDash server, *without* a trailing slash.
        database : str
            Database / shard name.
        limit : int | None
            Maximum number of datasets to request. Defaults to
            ``int(os.environ["EEGDASH_DOC_LIMIT"])`` if set, else
            ``1000``.
        force_refresh : bool
            Bypass both the in-memory and disk caches and force a live
            fetch. Falls back exactly like a normal build if the live
            fetch fails; this only invalidates entry into the cache,
            not exit from it.

        Returns
        -------
        DatasetSnapshot
            A snapshot whose :attr:`source`, :attr:`fetched_at`, and
            :attr:`api_errors` honestly describe how the data arrived.

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

        ``manifest_path`` points at a JSON file that was written next
        to a parquet payload by :meth:`build` (under the on-disk cache
        directory). Used by A6 (dated snapshots) and by tests that want
        to round-trip a snapshot without monkeypatching the network.
        """
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        rows_path = manifest_path.with_suffix(".parquet")
        if rows_path.exists():
            rows = pd.read_parquet(rows_path)
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
        """Pin to the server's dated snapshot for reproducible builds.

        Fetches ``GET {api_base}/{database}/snapshots/{date}`` and
        reconstructs a snapshot from the frozen ``chart_data`` +
        ``build_manifest`` payload. The server's snapshot endpoint
        emits ``Cache-Control: public, max-age=31536000, immutable``,
        so the HTTP layer (or any future CDN) handles caching — this
        method deliberately does NOT touch the disk cache or the
        in-process ``_INSTANCE_CACHE`` used by :meth:`build`.

        Use case: a docs CI pipeline pins to a fixed date so two
        successive builds against the same source tree produce
        byte-identical artefacts even when the live catalog ingests
        new datasets between runs. Diffing two builds reduces to a
        pair of pinned snapshots with different dates.

        Parameters
        ----------
        date : str
            ISO calendar date, ``YYYY-MM-DD``. Anything else raises
            :class:`ValueError` before any network call is attempted.
        api_base : str | None
            Base URL of the EEGDash server. Defaults to the same value
            :meth:`build` uses (``https://data.eegdash.org/api``).
        database : str | None
            Database / shard name. Defaults to ``eegdash``.

        Returns
        -------
        DatasetSnapshot
            A snapshot whose :attr:`source` is ``"live"`` (the
            server's snapshot is authoritative live data, just
            frozen), :attr:`fetched_at` is the original creation
            time from the payload, and :attr:`pinned_at` is the
            requested date.

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

        return cls(
            rows=rows,
            aggregations=aggregations,
            montages=montages,
            source="live",
            fetched_at=fetched_at,
            api_errors=[],
            manifest=dict(build_manifest),
            pinned_at=date,
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _disk_cache_path(database: str) -> Path:
    """Where the disk cache for ``database`` lives.

    Kept inside ``get_default_cache_dir()`` so the package-managed
    cache and the docs-build cache share one directory (and one
    ``.gitignore`` entry).
    """
    cache_dir = get_default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize the database name; we never accept input here that
    # could contain path separators, but defence in depth never hurts.
    safe = "".join(c for c in database if c.isalnum() or c in {"_", "-"})
    return cache_dir / f"snapshot_{safe or 'default'}.parquet"


def _montages_sidecar_path(database: str) -> Path:
    """Where the per-dataset montage cache for ``database`` lives.

    Sits next to the parquet rows cache (same directory, same naming)
    so the two artefacts share one cache invalidation lifecycle. JSON
    rather than parquet because the payload is shallow (~500 small
    dicts at the production scale) and the docs-build consumer is
    happiest with plain dicts.
    """
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
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]] | None:
    """Attempt the rich chart-data endpoint. Returns ``None`` on failure.

    Requests ``?include=montages`` so the top montage per dataset rides
    along with the catalog rows in one round-trip (arch #5). The
    parsed montage objects are returned as a third tuple element keyed
    by lowercased ``dataset_id``; callers store it on
    :attr:`DatasetSnapshot._montages`. When the server doesn't
    populate a montage for a given dataset (or the field is absent
    entirely on an old deployment) the per-dataset value is simply
    missing from the dict — never ``None`` and never a partial doc.
    """
    url = f"{api_base}/{database}/datasets/chart-data?limit={limit}&include=montages"
    try:
        data = _http_get_json(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            errors.append(f"chart-data 404 at {url}; trying summary")
            return None
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
    return rows, aggregations, montages


def _try_fetch_summary(
    api_base: str, database: str, limit: int, *, errors: list[str]
) -> pd.DataFrame | None:
    """Attempt the legacy summary endpoint. Returns ``None`` on failure."""
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
    """Re-hydrate a parquet disk cache. Returns ``None`` on failure/absence."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
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
    """Persist a freshly-fetched payload. Best-effort; failures are swallowed."""
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("snapshot disk cache write failed at %s: %s", path, exc)


def _write_montages_sidecar(
    montages: Mapping[str, Mapping[str, Any]], path: Path
) -> None:
    """Persist the parsed montages dict. Best-effort.

    Sibling to :func:`_write_disk_cache` — writes a small JSON file
    next to the parquet so a subsequent ``cached`` resolution still
    has montages to serve. An empty dict is written when no montages
    came through; that's expected when the server responds without
    ``?include=montages`` support.
    """
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
    """The real fetch/fallback chain. Always returns a snapshot."""
    errors: list[str] = []

    # 1. Live fetch — try the rich chart-data endpoint first.
    chart_result = _try_fetch_chart_data(api_base, database, limit, errors=errors)
    if chart_result is not None:
        rows, aggregations, montages = chart_result
        if not rows.empty:
            snapshot = _live_snapshot(
                rows,
                aggregations,
                montages,
                errors,
                api_base=api_base,
                database=database,
            )
            _write_disk_cache(rows, _disk_cache_path(database))
            _write_montages_sidecar(montages, _montages_sidecar_path(database))
            return snapshot
        errors.append("chart-data returned 0 datasets")

    # Fall back to the legacy summary endpoint when chart-data is
    # absent OR returned zero rows (the latter happens during early
    # ingestion windows and during tests that stub one endpoint at a
    # time). The summary endpoint has no montage data, so the
    # snapshot's :attr:`montage` accessor will return ``None`` until
    # chart-data comes back online.
    summary_rows = _try_fetch_summary(api_base, database, limit, errors=errors)
    if summary_rows is not None and not summary_rows.empty:
        snapshot = _live_snapshot(
            summary_rows, {}, {}, errors, api_base=api_base, database=database
        )
        _write_disk_cache(summary_rows, _disk_cache_path(database))
        return snapshot
    if summary_rows is not None and summary_rows.empty:
        errors.append("summary returned 0 datasets")

    # 2. Disk cache fallback (skipped when force_refresh=True, but only
    # after the live attempt above — that's the contract callers expect).
    #
    # When we reach a fallback path the live API was unreachable for
    # this build, so the server's ``/build-manifest`` is by definition
    # not relevant: we are explicitly serving stale or shipped data and
    # the local projection truthfully says so (``source != "live"``).
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
    errors: list[str],
    *,
    api_base: str,
    database: str,
) -> "DatasetSnapshot":
    """Build a ``source="live"`` snapshot.

    After rows have loaded successfully we ALSO hit the server's
    ``/build-manifest`` endpoint and surface its response verbatim on
    :attr:`DatasetSnapshot.manifest`. The local re-projection is only
    used as a fallback when the manifest endpoint is missing or
    erroring (offline build, old server deployment, transient 5xx).
    """
    fetched_at = _utcnow()
    dataset_count = len(rows)

    server_manifest = _try_fetch_build_manifest(
        api_base=api_base, database=database, errors=errors
    )
    if server_manifest is not None:
        # Mismatch detection — if the server's reported dataset_count
        # disagrees with the rows we just loaded, surface the divergence
        # on ``api_errors`` so the B2 CI gate can refuse to publish.
        # Don't crash: a benign skew is possible during an ingestion
        # window (server has finished a batch but the chart-data shard
        # hasn't caught up, or vice-versa).
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
        # Keep any partial errors collected before success — e.g. a
        # chart-data 404 followed by a successful summary call records
        # the 404 as context, not as a failure. The same convention
        # applies to a missing ``/build-manifest`` endpoint: it's a
        # warning but not a build failure.
        api_errors=list(errors),
        manifest=manifest,
    )


def _try_fetch_build_manifest(
    api_base: str,
    database: str,
    *,
    timeout: float = 5.0,
    errors: list[str] | None = None,
) -> dict[str, Any] | None:
    """Fetch the server's ``/build-manifest`` JSON, or ``None`` on failure.

    The endpoint is treated as best-effort: any failure (timeout, 404,
    JSON decode error, transport-level exception) returns ``None`` and
    appends the error text to ``errors`` (when provided) so the caller
    can decide whether to surface it via :attr:`DatasetSnapshot.api_errors`.

    Parameters
    ----------
    api_base : str
        Base URL of the EEGDash server, *without* a trailing slash.
    database : str
        Database / shard name.
    timeout : float
        Request timeout in seconds. Shorter than the rows-fetch
        timeout because the docs build should not be held up by a
        ``/build-manifest`` outage — the local projection is fine as
        fallback.
    errors : list[str] | None
        Optional list to append error strings to. Caller decides
        whether to attach these to ``snapshot.api_errors``.

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
    """Locally-projected manifest dict — only used when the server is unreachable.

    The preferred manifest source is the server's ``/build-manifest``
    response (see :func:`_try_fetch_build_manifest`). This projection
    exists so that :attr:`DatasetSnapshot.manifest` is never ``None``:
    when the live API isn't reached (offline build, disk-cache
    fallback, package-CSV fallback) or when ``/build-manifest`` itself
    fails, we surface what we already know.
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
    """Emit the I8 invariant line.

    Format is contractual — ``docs_pipeline_validation_plan.md`` §2
    grep's the build log for exactly this shape.
    """
    logger.info(
        "DatasetSnapshot source=%s dataset_count=%d fetched_at=%s",
        snapshot.source,
        snapshot.dataset_count,
        snapshot.fetched_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# Row mappers
#
# These were lifted intact from ``eegdash.dataset.registry`` so the
# legacy compatibility shims see byte-identical DataFrame shapes. Any
# evolution of the catalog schema goes here, not in the (deprecated)
# shims.
# ---------------------------------------------------------------------------


# ``EXCLUDED_DATASETS`` is imported at the top of this module from
# :mod:`eegdash.dataset._excluded`. Both this module and
# :mod:`eegdash.dataset.registry` read from that single source of
# truth — a B1-era snapshot-local copy had silently drifted 16 entries
# away from the registry's curated list and become the only filter
# effectively running in the docs build.


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


# Pretty modality label used in the per-page caption. Lifted from the
# retired ``docs/build_electrode_layouts.py`` so the rendered ``label``
# string is byte-identical to the legacy JSON output.
_MONTAGE_MODALITY_LABEL = {
    "eeg": "EEG",
    "ieeg": "iEEG",
    "meg": "MEG",
    "nirs": "fNIRS",
    "emg": "EMG",
}


def _project_montage(montage: Mapping[str, Any]) -> dict[str, Any]:
    """Project a server montage payload into the viewer-friendly dict.

    Output shape — superset of the registry doc to keep both the
    server fields (``hash``, ``modality``, ``n_sensors``,
    ``channel_names``, ``space_declared``, ``units_declared``,
    ``subject_count``) AND the legacy ``electrode-layouts.json`` keys
    (``label``, ``n_channels``, ``montage_id``). The duplication is
    cheap (~40 bytes per dataset) and keeps the dataset_page consumer
    a no-op rename.
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
    """Pull the ``datasets[i].montage`` field into a ``{id: dict}`` map.

    Keyed by lowercased ``dataset_id`` because the docs consumer
    lowercases before lookup. Datasets without a top montage (server
    returned ``null`` or omitted the field) are silently skipped, so a
    missing key in this dict has the same meaning as the legacy
    ``electrode-layouts.json``: "no scalp electrode layout currently
    indexed for this dataset".
    """
    montages: dict[str, dict[str, Any]] = {}
    for ds in datasets:
        ds_id = str(ds.get("dataset_id") or "").strip().lower()
        if not ds_id or ds_id.upper() in EXCLUDED_DATASETS:
            continue
        montage = ds.get("montage")
        if not isinstance(montage, dict) or not montage:
            continue
        # Defensive: a future server version could emit only the hash
        # without the joined registry doc. Require at least one
        # informative field before we surface it.
        if not (montage.get("hash") or montage.get("n_sensors")):
            continue
        montages[ds_id] = _project_montage(montage)
    return montages


def _rows_from_chart_data(datasets: list[dict[str, Any]]) -> pd.DataFrame:
    """Map the chart-data response shape into the canonical DataFrame.

    Mirrors the legacy ``fetch_chart_data_from_api`` implementation
    exactly — same columns, same exclusions, same fallback rules — so
    existing consumers see no schema drift during the migration.
    """
    # Lazy import to avoid a circular ``registry`` ↔ ``snapshot`` cycle.
    from .registry import _resolve_author_year  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    for ds in datasets:
        ds_id = ds.get("dataset_id", "").strip()
        if ds_id.upper() in EXCLUDED_DATASETS:
            continue

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
    from .registry import _resolve_author_year  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    for ds in datasets:
        ds_id = ds.get("dataset_id", "").strip()
        # Filter test datasets and excluded ones (mirror legacy
        # fetch_datasets_from_api semantics).
        if (
            ds_id.lower() in ("test", "test_dataset")
            or ds_id.upper() in EXCLUDED_DATASETS
        ):
            continue

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
    """Wipe the process-wide snapshot cache.

    Called from pytest fixtures and the legacy compatibility shims when
    the docs build asks for a force-refresh. Not part of the public
    API, but lives in this module so tests don't have to monkeypatch
    a private name from outside the package.
    """
    with _INSTANCE_CACHE_LOCK:
        _INSTANCE_CACHE.clear()


__all__ = [
    "DatasetSnapshot",
    "PACKAGE_CSV_PATH",
    "Source",
]

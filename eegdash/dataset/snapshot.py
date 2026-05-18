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


class DatasetSnapshot:
    """A frozen view of the dataset catalog used by one docs build.

    Build it with :meth:`build` (network) or :meth:`load` (disk). All
    consumers should read through the accessor methods rather than
    poking at the underlying fields directly — the field set is
    allowed to grow (see :attr:`manifest` placeholder for A2).

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
        Placeholder for the future ``/build-manifest`` server response
        (review §3 A2). Today filled with ``{"dataset_count",
        "source", "fetched_at"}`` so downstream consumers can read it
        unconditionally; the schema will widen when A2 ships.

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

        The lookup is keyed exactly on the catalog ``dataset_id``
        (e.g. ``"ds002718"``). Returns ``None`` when no montage has
        been joined into this snapshot. A3 will populate it
        server-side via ``?include=montages``; today this is always
        ``None`` and consumers must fall back to the per-dataset
        ``/datasets/{id}/montages`` endpoint.
        """
        if not dataset_id:
            return None
        result = self._montages.get(dataset_id)
        if result is None:
            return None
        # Return a shallow copy so callers cannot mutate the cached
        # mapping from underneath the snapshot.
        return dict(result)

    @property
    def dataset_count(self) -> int:
        """Number of dataset rows in this snapshot."""
        return int(len(self._rows))

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


def _http_get_json(url: str, *, timeout: int = 30) -> dict[str, Any]:
    """Single GET → JSON helper. Raises on any failure."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _try_fetch_chart_data(
    api_base: str, database: str, limit: int, *, errors: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Attempt the rich chart-data endpoint. Returns ``None`` on failure."""
    url = f"{api_base}/{database}/datasets/chart-data?limit={limit}"
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

    rows = _rows_from_chart_data(data.get("datasets", []) or [])
    aggregations = data.get("aggregations") or {}
    return rows, aggregations


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
        rows, aggregations = chart_result
        if not rows.empty:
            snapshot = _live_snapshot(rows, aggregations, errors)
            _write_disk_cache(rows, _disk_cache_path(database))
            return snapshot
        errors.append("chart-data returned 0 datasets")

    # Fall back to the legacy summary endpoint when chart-data is
    # absent OR returned zero rows (the latter happens during early
    # ingestion windows and during tests that stub one endpoint at a
    # time).
    summary_rows = _try_fetch_summary(api_base, database, limit, errors=errors)
    if summary_rows is not None and not summary_rows.empty:
        snapshot = _live_snapshot(summary_rows, {}, errors)
        _write_disk_cache(summary_rows, _disk_cache_path(database))
        return snapshot
    if summary_rows is not None and summary_rows.empty:
        errors.append("summary returned 0 datasets")

    # 2. Disk cache fallback (skipped when force_refresh=True, but only
    # after the live attempt above — that's the contract callers expect).
    if not force_refresh:
        disk_path = _disk_cache_path(database)
        cached_rows = _read_disk_cache(disk_path, errors=errors)
        if cached_rows is not None and not cached_rows.empty:
            cached_at = _mtime(disk_path)
            return DatasetSnapshot(
                rows=cached_rows,
                aggregations={},
                montages={},
                source="cached",
                fetched_at=cached_at,
                api_errors=list(errors),
                manifest=_manifest_from(
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
        manifest=_manifest_from(
            source="package-csv",
            dataset_count=len(csv_rows),
            fetched_at=fetched_at,
        ),
    )


def _live_snapshot(
    rows: pd.DataFrame,
    aggregations: dict[str, Any],
    errors: list[str],
) -> "DatasetSnapshot":
    fetched_at = _utcnow()
    return DatasetSnapshot(
        rows=rows,
        aggregations=aggregations,
        montages={},
        source="live",
        fetched_at=fetched_at,
        # Keep any partial errors collected before success — e.g. a
        # chart-data 404 followed by a successful summary call records
        # the 404 as context, not as a failure.
        api_errors=list(errors),
        manifest=_manifest_from(
            source="live",
            dataset_count=len(rows),
            fetched_at=fetched_at,
        ),
    )


def _manifest_from(
    *, source: Source, dataset_count: int, fetched_at: datetime
) -> dict[str, Any]:
    """Minimal manifest dict. Widened when A2 (/build-manifest) ships."""
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

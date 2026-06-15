"""Single data-access seam for the docs build.

:class:`DatasetSnapshot` fetches the dataset catalog from the EEGDash
server in one ``chart-data`` call (rows, montages, and per-dataset
metadata are all shaped server-side and ride along), caches it on disk,
and falls back to the package CSV — self-reporting provenance so callers
can tell "live" from "stale" from "silent failure".
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

# Provenance tag; CI gates on ``source == "live"`` before publishing.
Source = Literal["live", "cached", "package-csv"]

# Package-shipped registry CSV — the final fallback. A constant so tests
# can monkeypatch the path.
PACKAGE_CSV_PATH: Path = Path(__file__).resolve().parent / "dataset_summary.csv"

# Process-wide cache of built snapshots, keyed by (api_base, database, limit).
_INSTANCE_CACHE: dict[tuple[str, str, int], "DatasetSnapshot"] = {}
_INSTANCE_CACHE_LOCK = Lock()

# Server ``/snapshots/{date}`` route shape — ISO ``YYYY-MM-DD`` only.
_SNAPSHOT_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Per-dataset metadata shapes (parsed from datasets[i].metadata, which the
# server already emits clean: bare ORCID, nulled optionals, ISO created_at).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NemarAuthor:
    name: str
    orcid: str | None = None


@dataclass(frozen=True)
class NemarKeyword:
    term: str
    scheme: str | None = None
    value_uri: str | None = None


@dataclass(frozen=True)
class NemarVersion:
    version: str
    doi: str
    created_at: str  # ISO-8601 string from the server


@dataclass(frozen=True)
class NemarMetadata:
    description: str | None
    license: str | None
    authors: tuple[NemarAuthor, ...]
    keywords: tuple[NemarKeyword, ...]
    versions: tuple[NemarVersion, ...]


class DatasetSnapshot:
    """A frozen view of the dataset catalog for one docs build.

    Entry points: :meth:`build` (live + disk/CSV fallback), :meth:`load`
    (re-hydrate from disk), :meth:`pin` (a dated ``/snapshots/{date}``).
    Read through the accessors; provenance is on :attr:`source` /
    :attr:`fetched_at` / :attr:`api_errors` / :attr:`pinned_at`.
    """

    __slots__ = ("_rows", "_aggregations", "_montages", "_metadata", "__dict__")

    # Class-level sentinels so ``hasattr(cls, name)`` holds; instances shadow.
    source: Source = "package-csv"
    fetched_at: datetime | None = None
    api_errors: list[str] = []  # noqa: RUF012
    manifest: dict[str, Any] = {}  # noqa: RUF012
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

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DatasetSnapshot(source={self.source!r}, "
            f"dataset_count={self.dataset_count}, "
            f"api_errors={len(self.api_errors)})"
        )

    # ----- accessors -----------------------------------------------------

    def rows(self) -> pd.DataFrame:
        """Per-dataset records as a DataFrame (a copy, so callers can't leak state)."""
        return self._rows.copy()

    def aggregations(self) -> dict[str, Any]:
        """Server-side totals; empty on a fallback (cached / package-csv) build."""
        return dict(self._aggregations)

    def montage(self, dataset_id: str) -> Mapping[str, Any] | None:
        """Top montage dict for one dataset (case-insensitive), or ``None``."""
        if not dataset_id:
            return None
        result = self._montages.get(dataset_id) or self._montages.get(
            dataset_id.lower()
        )
        return dict(result) if result is not None else None

    def metadata(self, dataset_id: str) -> "NemarMetadata | None":
        """Per-dataset :class:`NemarMetadata` (case-insensitive), or ``None``.

        Only present on a live build; cached / package-csv builds carry none.
        """
        if not dataset_id:
            return None
        return self._metadata.get(dataset_id) or self._metadata.get(dataset_id.lower())

    @property
    def dataset_count(self) -> int:
        return int(len(self._rows))

    @property
    def schema_version(self) -> str | None:
        """Server ``schema_version`` from the manifest, or ``None``."""
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
        """Fetch / cache / fallback in one call, memoised per process.

        Resolution order (each records its failure on :attr:`api_errors`):
        live ``chart-data`` → disk cache → package CSV. ``limit`` defaults
        to ``EEGDASH_DOC_LIMIT`` or 1000; ``force_refresh`` bypasses the
        in-memory and disk caches on entry.
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
            # Prefer a live result over a racing fallback; else keep the
            # first winner so callers see one consistent instance.
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
        """Re-hydrate from a manifest JSON + sibling ``*.rows.json``.

        Metadata/montages are not persisted, so a loaded snapshot has none.
        """
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        rows_path = manifest_path.with_suffix(".rows.json")
        if rows_path.exists() and rows_path.stat().st_size > 0:
            rows = pd.read_json(rows_path, orient="records")
        else:
            rows = pd.DataFrame()

        fetched_at_raw = manifest.get("fetched_at")
        fetched_at = (
            datetime.fromisoformat(fetched_at_raw) if fetched_at_raw else _utcnow()
        )
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
        """Pin to a dated ``/snapshots/{date}`` for reproducible builds.

        ``source`` is ``"live"`` (frozen authoritative data), ``fetched_at``
        is the payload's creation time, ``pinned_at`` is the requested date.
        Does not touch the disk / in-process caches.

        Raises ``ValueError`` on a non ``YYYY-MM-DD`` date (before any
        network call) and ``LookupError`` on a server 404.
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
        created_at_raw = payload.get("created_at")
        if not isinstance(created_at_raw, str):
            raise ValueError(
                f"snapshot payload for {date} missing or malformed 'created_at'"
            )
        # ``fromisoformat`` on 3.10 rejects a trailing ``Z``.
        fetched_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))

        datasets = chart_data.get("datasets") or []
        return cls(
            rows=_rows_from_chart_data(datasets),
            aggregations=chart_data.get("aggregations") or {},
            montages=_montages_from_chart_data(datasets),
            source="live",
            fetched_at=fetched_at,
            api_errors=[],
            manifest=dict(payload.get("build_manifest") or {}),
            pinned_at=date,
            metadata=_metadata_from_chart_data(datasets),
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _disk_cache_path(database: str) -> Path:
    cache_dir = get_default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in database if c.isalnum() or c in {"_", "-"})
    return cache_dir / f"snapshot_{safe or 'default'}.json"


def _http_get_json(url: str, *, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


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
    """Live chart-data fetch (rows + montages + metadata in one round-trip).

    ``?include=montages,metadata,rows`` — all three are shaped server-side.
    Returns ``None`` on any failure.
    """
    url = (
        f"{api_base}/{database}/datasets/chart-data"
        f"?limit={limit}&include=montages,metadata,rows"
    )
    try:
        data = _http_get_json(url)
    except Exception as exc:  # noqa: BLE001 — network/JSON raise broadly
        errors.append(f"chart-data error at {url}: {exc}")
        return None

    if not data.get("success"):
        errors.append(f"chart-data returned success=False at {url}")
        return None

    datasets = data.get("datasets", []) or []
    return (
        _rows_from_chart_data(datasets),
        data.get("aggregations") or {},
        _montages_from_chart_data(datasets),
        _metadata_from_chart_data(datasets),
    )


def _read_disk_cache(path: Path, *, errors: list[str]) -> pd.DataFrame | None:
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
    if not path.exists():
        errors.append(f"package CSV missing at {path}")
        return None
    try:
        return pd.read_csv(path, comment="#", skip_blank_lines=True)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"package CSV read failed at {path}: {exc}")
        return None


def _write_disk_cache(df: pd.DataFrame, path: Path) -> None:
    """Best-effort JSON persist; failures are swallowed."""
    try:
        path.write_text(df.to_json(orient="records"), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.debug("snapshot disk cache write failed at %s: %s", path, exc)


def _build_uncached(
    *,
    api_base: str,
    database: str,
    limit: int,
    force_refresh: bool,
) -> "DatasetSnapshot":
    """The real fetch/fallback chain: live chart-data → disk cache → CSV."""
    errors: list[str] = []

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
            return snapshot
        errors.append("chart-data returned 0 datasets")

    # Disk cache (rows only — montage/metadata are not persisted, so those
    # accessors return None on a cached build). Skipped on force_refresh.
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
                manifest=_local_manifest_projection(
                    source="cached",
                    dataset_count=len(cached_rows),
                    fetched_at=cached_at,
                ),
            )

    # Package CSV — the floor. Always returns a snapshot; CI inspects
    # ``api_errors`` and refuses to publish on a non-live source.
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
    """Build a ``source="live"`` snapshot, surfacing ``/build-manifest`` verbatim."""
    fetched_at = _utcnow()
    server_manifest = _try_fetch_build_manifest(
        api_base=api_base, database=database, errors=errors
    )
    if server_manifest is not None:
        # A server/rows dataset_count skew is a benign ingestion-window
        # warning, not a failure — surface it on api_errors for the gate.
        server_count = server_manifest.get("dataset_count")
        if isinstance(server_count, int) and server_count != len(rows):
            errors.append(
                f"build-manifest dataset_count={server_count} "
                f"but snapshot.rows()={len(rows)}"
            )
        manifest = dict(server_manifest)
    else:
        manifest = _local_manifest_projection(
            source="live", dataset_count=len(rows), fetched_at=fetched_at
        )

    return DatasetSnapshot(
        rows=rows,
        aggregations=aggregations,
        montages=montages,
        source="live",
        fetched_at=fetched_at,
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
    """Best-effort ``/build-manifest`` fetch; ``None`` on any failure."""
    url = f"{api_base}/{database}/build-manifest"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        if errors is not None:
            errors.append(f"build-manifest error at {url}: {exc}")
        return None


def _local_manifest_projection(
    *, source: Source, dataset_count: int, fetched_at: datetime
) -> dict[str, Any]:
    """Local manifest stand-in so :attr:`manifest` is never ``None``."""
    return {
        "source": source,
        "dataset_count": dataset_count,
        "fetched_at": fetched_at.isoformat(),
    }


def _mtime(path: Path) -> datetime:
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
# Server-shaped sub-objects (rows / montages / metadata) — read, don't map.
# The server owns the shaping (_build_docs_row / montage projection /
# _build_dataset_metadata); the client just lifts them off each dataset.
# ---------------------------------------------------------------------------


def _rows_from_chart_data(datasets: list[dict[str, Any]]) -> pd.DataFrame:
    """Docs DataFrame from server-shaped ``datasets[i]['row']``.

    Datasets without a ``row`` (a snapshot pinned before ``include=rows``
    shipped) are skipped — that build degrades rather than re-mapping.
    """
    return pd.DataFrame([d["row"] for d in datasets if isinstance(d.get("row"), dict)])


def _montages_from_chart_data(
    datasets: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """``{lowercased id: montage}`` from server-shaped ``datasets[i]['montage']``."""
    return {
        str(d.get("dataset_id") or "").strip().lower(): d["montage"]
        for d in datasets
        if isinstance(d.get("montage"), dict) and d["montage"]
    }


def _metadata_from_chart_data(
    datasets: list[dict[str, Any]],
) -> dict[str, "NemarMetadata"]:
    """``{lowercased id: NemarMetadata}`` from ``datasets[i]['metadata']``."""
    out: dict[str, NemarMetadata] = {}
    for ds in datasets:
        ds_id = str(ds.get("dataset_id") or "").strip()
        meta = ds.get("metadata")
        if ds_id and isinstance(meta, dict) and meta:
            out[ds_id.lower()] = _build_metadata(ds, meta)
    return out


def _build_metadata(ds: Mapping[str, Any], meta: Mapping[str, Any]) -> "NemarMetadata":
    """Wrap the server's already-clean metadata sub-object in dataclasses."""
    authors = tuple(
        NemarAuthor(name=str(a.get("name") or ""), orcid=a.get("orcid") or None)
        for a in (meta.get("authors") or [])
        if isinstance(a, dict) and a.get("name")
    )
    keywords = tuple(
        NemarKeyword(
            term=str(k.get("term") or ""),
            scheme=k.get("scheme") or None,
            value_uri=k.get("value_uri") or None,
        )
        for k in (meta.get("keywords") or [])
        if isinstance(k, dict) and k.get("term")
    )
    versions = tuple(
        NemarVersion(
            version=str(v.get("version") or ""),
            doi=str(v.get("doi") or ""),
            created_at=str(v.get("created_at") or ""),
        )
        for v in (meta.get("versions") or [])
        if isinstance(v, dict) and v.get("version")
    )
    return NemarMetadata(
        description=meta.get("description") or None,
        license=ds.get("license") or None,
        authors=authors,
        keywords=keywords,
        versions=versions,
    )


def _reset_instance_cache_for_testing() -> None:
    """Wipe the process-wide snapshot cache (tests / force-refresh)."""
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

"""Single data-access seam for the docs build.

:class:`DatasetSnapshot.build` fetches the dataset catalog from the
EEGDash server in one ``chart-data`` call — rows, montages, and
per-dataset metadata are all shaped server-side and ride along — caches
it on disk, and falls back to the package CSV, self-reporting provenance
so callers can tell "live" from "stale" from "silent failure".
"""

from __future__ import annotations

import json
import logging
import os
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


# Per-dataset metadata shapes, parsed from ``datasets[i].metadata`` (the
# server emits it clean: bare ORCID, nulled optionals, ISO created_at).


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

    :meth:`build` fetches live (with disk / package-CSV fallback). Read
    through the accessors; provenance is on :attr:`source` /
    :attr:`fetched_at` / :attr:`api_errors`.
    """

    __slots__ = ("_rows", "_aggregations", "_montages", "_metadata", "__dict__")

    # Class-level sentinels so ``hasattr(cls, name)`` holds; instances shadow.
    source: Source = "package-csv"
    fetched_at: datetime | None = None
    api_errors: list[str] = []  # noqa: RUF012
    manifest: dict[str, Any] = {}  # noqa: RUF012

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
        """Per-dataset metadata (case-insensitive), or ``None``.

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

    # ----- builder -------------------------------------------------------

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

        logger.info(  # format is grepped by CI — do not change
            "DatasetSnapshot source=%s dataset_count=%d fetched_at=%s",
            snapshot.source,
            snapshot.dataset_count,
            snapshot.fetched_at.isoformat(),
        )
        return snapshot


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _disk_cache_path(database: str) -> Path:
    cache_dir = get_default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in database if c.isalnum() or c in {"_", "-"})
    return cache_dir / f"snapshot_{safe or 'default'}.json"


def _read_package_csv(path: Path, *, errors: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        errors.append(f"package CSV missing at {path}")
        return None
    try:
        return pd.read_csv(path, comment="#", skip_blank_lines=True)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"package CSV read failed at {path}: {exc}")
        return None


def _try_fetch_build_manifest(
    api_base: str, database: str, *, errors: list[str] | None = None
) -> dict[str, Any] | None:
    """Best-effort ``/build-manifest`` fetch; ``None`` on any failure."""
    url = f"{api_base}/{database}/build-manifest"
    try:
        with urllib.request.urlopen(url, timeout=5.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        if errors is not None:
            errors.append(f"build-manifest error at {url}: {exc}")
        return None


def _metadata_from_chart_data(
    datasets: list[dict[str, Any]],
) -> dict[str, "NemarMetadata"]:
    """``{lowercased id: NemarMetadata}`` from each ``datasets[i].metadata``.

    The server already emits the contract shape clean, so this just wraps it.
    """
    out: dict[str, NemarMetadata] = {}
    for ds in datasets:
        ds_id = str(ds.get("dataset_id") or "").strip()
        meta = ds.get("metadata")
        if not (ds_id and isinstance(meta, dict) and meta):
            continue
        out[ds_id.lower()] = NemarMetadata(
            description=meta.get("description") or None,
            license=ds.get("license") or None,
            authors=tuple(
                NemarAuthor(name=str(a.get("name") or ""), orcid=a.get("orcid") or None)
                for a in (meta.get("authors") or [])
                if isinstance(a, dict) and a.get("name")
            ),
            keywords=tuple(
                NemarKeyword(
                    term=str(k.get("term") or ""),
                    scheme=k.get("scheme") or None,
                    value_uri=k.get("value_uri") or None,
                )
                for k in (meta.get("keywords") or [])
                if isinstance(k, dict) and k.get("term")
            ),
            versions=tuple(
                NemarVersion(
                    version=str(v.get("version") or ""),
                    doi=str(v.get("doi") or ""),
                    created_at=str(v.get("created_at") or ""),
                )
                for v in (meta.get("versions") or [])
                if isinstance(v, dict) and v.get("version")
            ),
        )
    return out


def _build_uncached(
    *, api_base: str, database: str, limit: int, force_refresh: bool
) -> "DatasetSnapshot":
    """The fetch/fallback chain: live chart-data → disk cache → package CSV."""
    errors: list[str] = []

    # 1. Live: one chart-data call carries server-shaped rows + montages +
    # metadata. Each is lifted off the response verbatim (no client mapping).
    url = (
        f"{api_base}/{database}/datasets/chart-data"
        f"?limit={limit}&include=montages,metadata,rows"
    )
    datasets: list[dict[str, Any]] | None = None
    aggregations: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        if data.get("success"):
            datasets = data.get("datasets") or []
            aggregations = data.get("aggregations") or {}
        else:
            errors.append(f"chart-data returned success=False at {url}")
    except Exception as exc:  # noqa: BLE001 — network/JSON raise broadly
        errors.append(f"chart-data error at {url}: {exc}")

    if datasets:
        rows = pd.DataFrame(
            [d["row"] for d in datasets if isinstance(d.get("row"), dict)]
        )
        if not rows.empty:
            montages = {
                str(d.get("dataset_id") or "").strip().lower(): d["montage"]
                for d in datasets
                if isinstance(d.get("montage"), dict) and d["montage"]
            }
            now = datetime.now(timezone.utc)
            manifest = _try_fetch_build_manifest(
                api_base=api_base, database=database, errors=errors
            )
            if manifest is not None:
                # A server/rows dataset_count skew is a benign ingestion-window
                # warning for the CI gate, not a failure.
                count = manifest.get("dataset_count")
                if isinstance(count, int) and count != len(rows):
                    errors.append(
                        f"build-manifest dataset_count={count} "
                        f"but snapshot.rows()={len(rows)}"
                    )
            else:
                manifest = {
                    "source": "live",
                    "dataset_count": len(rows),
                    "fetched_at": now.isoformat(),
                }
            try:  # best-effort disk cache write
                _disk_cache_path(database).write_text(
                    rows.to_json(orient="records"), encoding="utf-8"
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("snapshot disk cache write failed: %s", exc)
            return DatasetSnapshot(
                rows=rows,
                aggregations=aggregations,
                montages=montages,
                source="live",
                fetched_at=now,
                api_errors=list(errors),
                manifest=dict(manifest),
                metadata=_metadata_from_chart_data(datasets),
            )
        errors.append("chart-data returned 0 datasets")

    # 2. Disk cache (rows only; montage/metadata are not persisted, so those
    # accessors return None on a cached build). Skipped on force_refresh.
    if not force_refresh:
        path = _disk_cache_path(database)
        if path.exists() and path.stat().st_size > 0:
            try:
                cached_rows = pd.read_json(path, orient="records")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"disk cache read failed at {path}: {exc}")
                cached_rows = pd.DataFrame()
            if not cached_rows.empty:
                at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                return DatasetSnapshot(
                    rows=cached_rows,
                    aggregations={},
                    montages={},
                    source="cached",
                    fetched_at=at,
                    api_errors=list(errors),
                    manifest={
                        "source": "cached",
                        "dataset_count": len(cached_rows),
                        "fetched_at": at.isoformat(),
                    },
                )

    # 3. Package CSV — the floor. Always returns a snapshot; CI inspects
    # ``api_errors`` and refuses to publish on a non-live source.
    csv_rows = _read_package_csv(PACKAGE_CSV_PATH, errors=errors)
    if csv_rows is None:
        csv_rows = pd.DataFrame()
    at = (
        datetime.fromtimestamp(PACKAGE_CSV_PATH.stat().st_mtime, tz=timezone.utc)
        if PACKAGE_CSV_PATH.exists()
        else datetime.now(timezone.utc)
    )
    return DatasetSnapshot(
        rows=csv_rows,
        aggregations={},
        montages={},
        source="package-csv",
        fetched_at=at,
        api_errors=list(errors),
        manifest={
            "source": "package-csv",
            "dataset_count": len(csv_rows),
            "fetched_at": at.isoformat(),
        },
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

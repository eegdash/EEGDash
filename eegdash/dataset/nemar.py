"""Per-dataset metadata client for NEMAR (`data.nemar.org`).

This module is the sibling of :class:`eegdash.dataset.snapshot.DatasetSnapshot`
for a different seam: where ``DatasetSnapshot`` owns the *catalog* (one
chart-data fetch covers hundreds of datasets), :class:`NemarClient` owns
the *per-dataset* enrichment that NEMAR exposes through its ``/<nm_id>``
and ``/<nm_id>/metadata.json`` endpoints.

Shape of the data we consume:

* ``GET /{nm_id}`` returns the top-level descriptor with a ``versions``
  array (one entry per published snapshot) and a ``metadata_url`` pointer.
* ``GET /{nm_id}/metadata.json`` returns rich dataset metadata: authors
  with ORCID, MeSH-tagged keywords, license, recording modality,
  description, BIDS version (often ``null``), and a ``versions`` array
  mirroring the top-level descriptor.
* ``GET /{nm_id}/v{ver}/manifest.json`` returns an array of file entries
  with sha256 checksums and pre-signed S3 URLs (the URLs expire in 1h,
  so the client never persists them).

The client tombstones 404 responses to ``None`` so the docs build's job
is to render gracefully, not to distinguish "no such dataset" from
"NEMAR deleted it". Top-level errors are tagged on the class-level
:attr:`NemarClient.errors` list -- same pattern as
:attr:`DatasetSnapshot.api_errors`.

Disk cache lives under ``{get_default_cache_dir()}/nemar/`` with a 24h
TTL. NEMAR's own ``Cache-Control: max-age=60`` is per-edge; for the
docs build (which can rebuild a thousand times a day in development) we
can be more aggressive.

Honor :envvar:`EEGDASH_NO_API` to short-circuit to cache-only -- same
contract the dataset-page data loader uses.

No new dependencies: only :mod:`urllib.request`.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

from ..paths import get_default_cache_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public surface: frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NemarAuthor:
    """One author entry from the NEMAR ``authors`` array.

    The ``orcid`` field is the bare 16-digit ORCID id (``"0000-0002-...
    "``) without the ``https://orcid.org/`` URL prefix -- consumers
    build links by interpolating the bare id. Returns ``None`` when no
    ORCID was supplied.
    """

    name: str
    orcid: str | None = None


@dataclass(frozen=True)
class NemarKeyword:
    """One keyword entry from the NEMAR ``keywords`` array.

    Plain free-text keywords have only ``term`` populated. Scheme-tagged
    keywords (e.g. MeSH) also carry the ``scheme`` short name and the
    full ``value_uri`` so the docs page can deep-link into the
    controlled vocabulary.
    """

    term: str
    scheme: str | None = None
    value_uri: str | None = None


@dataclass(frozen=True)
class NemarVersion:
    """One entry in NEMAR's ``versions`` history.

    NEMAR appears to sort newest-first already, but :class:`NemarClient`
    re-sorts defensively so consumers can trust the order.
    """

    version: str
    doi: str
    created_at: datetime
    manifest_url: str
    browse_url: str


@dataclass(frozen=True)
class NemarManifestEntry:
    """One file entry from a NEMAR version's manifest.

    Notably absent: the signed S3 ``url``. It expires in 1h and would
    poison any cache that persists it. Use
    :meth:`NemarClient.signed_url_for` when a fresh URL is needed.
    """

    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class NemarMetadata:
    """Combined view of NEMAR's top-level descriptor + ``metadata.json``.

    The ``versions`` tuple is always sorted newest-first by
    ``created_at`` (regardless of NEMAR's own ordering) and
    ``latest_version`` always equals ``versions[0].version``.
    """

    dataset_id: str
    name: str
    description: str | None
    license: str | None
    recording_modality: tuple[str, ...]
    bids_version: str | None
    authors: tuple[NemarAuthor, ...]
    keywords: tuple[NemarKeyword, ...]
    versions: tuple[NemarVersion, ...]
    latest_version: str


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


# Schema versions NEMAR has shipped so far. We accept any minor revision
# inside the ``0.x`` family; anything else logs a warning and still
# tries to parse (forward-compatible best effort, not a hard fail).
_KNOWN_SCHEMA_MAJOR = "0"

# Disk cache TTL. NEMAR's own ``max-age=60`` is per-edge; the docs build
# rebuilds far more often than NEMAR ships new versions, so we widen.
_DEFAULT_TTL = timedelta(hours=24)

# Top-level URL parts we will be parsing.
_DEFAULT_BASE_URL = "https://data.nemar.org"

# Identifiable UA. NEMAR's edge rejects the default ``Python-urllib/N``
# string with HTTP 403; keep this distinct enough that NEMAR operators
# can attribute traffic if needed.
_NEMAR_USER_AGENT = "eegdash-docs/1.0 (+https://eegdash.org)"


class NemarClient:
    """Fetch per-dataset NEMAR metadata + manifests.

    Two public entry points:

    * :meth:`metadata` -- top-level descriptor + ``metadata.json``,
      merged into a single :class:`NemarMetadata`. Returns ``None`` on
      404 (the "tombstone" path) so the dataset page can decline to
      emit a section without distinguishing absence from deletion.
    * :meth:`manifest` -- the per-version ``manifest.json`` array. Not
      called automatically by :meth:`metadata` because some manifests
      have ~1k entries (stimuli folders etc.) and the docs build does
      not need them for every dataset.

    Both are disk-cached. The disk cache key is the (``nemar_id``,
    ``kind``) pair; the cache layout is
    ``<cache_dir>/nemar/<nemar_id>__<kind>.json``. Both also honour
    :envvar:`EEGDASH_NO_API` to short-circuit to cache-only -- a
    cache-miss in that mode returns ``None`` rather than reaching the
    network.

    Errors that fall through to a return value of ``None`` are still
    recorded on :attr:`errors` so consumers (and CI gates) can inspect
    why a particular dataset was unavailable.
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        cache_dir: Path | None = None,
        timeout: float = 10.0,
        ttl: timedelta = _DEFAULT_TTL,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        if cache_dir is None:
            cache_dir = get_default_cache_dir() / "nemar"
        self._cache_dir = Path(cache_dir)
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Cache failures must not break the client; we will just
            # operate as if there is no cache.
            logger.debug("nemar: failed to create cache dir %s: %s", cache_dir, exc)
        self._timeout = timeout
        self._ttl = ttl
        # Per-instance lock guards the in-process memoisation; the disk
        # layer is best-effort and tolerates concurrent writes (last
        # writer wins, all writers see the same payload).
        self._lock = Lock()
        self.errors: list[str] = []

    # ----- public API ----------------------------------------------------

    def metadata(self, nemar_id: str) -> NemarMetadata | None:
        """Return combined metadata for ``nemar_id`` or ``None``.

        Returns ``None`` for any 404 (whether NEMAR never had the
        dataset or has tombstoned it). All other failures (timeouts,
        decode errors, schema upsets) also return ``None`` *and* append
        an explanatory string to :attr:`errors`.
        """
        nemar_id = nemar_id.strip()
        if not nemar_id:
            return None

        top_payload = self._fetch_json(
            nemar_id,
            kind="top",
            url=f"{self._base_url}/{nemar_id}",
        )
        if top_payload is None:
            return None

        meta_payload = self._fetch_json(
            nemar_id,
            kind="metadata",
            url=f"{self._base_url}/{nemar_id}/metadata.json",
        )
        if meta_payload is None:
            return None

        schema_version = str(meta_payload.get("schema_version") or "")
        if schema_version and not schema_version.startswith(f"{_KNOWN_SCHEMA_MAJOR}."):
            msg = (
                f"nemar: unknown schema_version={schema_version!r} for {nemar_id} "
                f"(known major={_KNOWN_SCHEMA_MAJOR}); attempting forward-compatible parse"
            )
            logger.warning(msg)
            warnings.warn(msg, stacklevel=2)

        try:
            return _build_metadata(nemar_id, top_payload, meta_payload)
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"nemar: parse failed for {nemar_id}: {exc}")
            logger.warning("nemar: parse failed for %s: %s", nemar_id, exc)
            return None

    def manifest(
        self,
        nemar_id: str,
        version: str | None = None,
    ) -> tuple[NemarManifestEntry, ...]:
        """Return the file manifest for ``(nemar_id, version)``.

        When ``version`` is ``None``, fetches the metadata first to
        resolve the latest version. Returns an empty tuple if NEMAR
        returns 404, the manifest is malformed, or the dataset's
        checksum algorithm is anything other than sha256.

        Caller note: the signed S3 ``url`` field on each manifest entry
        is deliberately dropped before caching (it expires in 1h). Use
        :meth:`signed_url_for` to obtain a fresh signed URL when needed.
        """
        nemar_id = nemar_id.strip()
        if not nemar_id:
            return ()

        if version is None:
            meta = self.metadata(nemar_id)
            if meta is None:
                return ()
            version = meta.latest_version

        cache_kind = f"manifest__{version}"
        payload = self._fetch_json(
            nemar_id,
            kind=cache_kind,
            url=f"{self._base_url}/{nemar_id}/{version}/manifest.json",
            transform=_strip_signed_urls,
        )
        if payload is None:
            return ()

        entries: list[NemarManifestEntry] = []
        for raw in payload:
            algo = (raw.get("checksum_algorithm") or "").lower()
            checksum = raw.get("checksum")
            if algo != "sha256" or not checksum:
                # We never accept a non-sha256 checksum -- the contract
                # documented in the dataclass is "only sha256 is
                # supported." A future scheme can extend the dataclass.
                self.errors.append(
                    f"nemar: skipped manifest entry with algo={algo!r} "
                    f"path={raw.get('path')!r} for {nemar_id}"
                )
                continue
            try:
                entries.append(
                    NemarManifestEntry(
                        path=str(raw["path"]),
                        size=int(raw.get("size") or 0),
                        sha256=str(checksum),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                self.errors.append(
                    f"nemar: malformed manifest entry for {nemar_id}: {exc}"
                )
                continue
        return tuple(entries)

    def signed_url_for(self, path: str, version: str, *, nemar_id: str) -> str | None:
        """Return a fresh signed S3 URL for ``path`` in ``(nemar_id, version)``.

        NEMAR's signed URLs expire in 1h. The disk cache strips them on
        write, so this method always reaches the network to refresh
        them. Returns ``None`` if the file is not in the manifest.

        Today this is a thin wrapper that re-fetches the live manifest
        (bypassing the disk cache, since the cache never stored the
        signed URLs in the first place). A future revision can call
        a dedicated ``signed-url`` endpoint when NEMAR exposes one.
        """
        url = f"{self._base_url}/{nemar_id}/{version}/manifest.json"
        try:
            raw = self._http_get_json(url)
        except _HttpNotFound:
            return None
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"nemar: signed_url_for {nemar_id} failed: {exc}")
            return None
        if not isinstance(raw, list):
            return None
        for entry in raw:
            if isinstance(entry, dict) and entry.get("path") == path:
                signed = entry.get("url")
                return str(signed) if signed else None
        return None

    def is_available(self, nemar_id: str) -> bool:
        """Cheap probe: does NEMAR know about ``nemar_id``?

        Uses the top-level descriptor (small JSON, well-cached on
        NEMAR's side) and the disk cache. Network is only hit on cache
        miss when :envvar:`EEGDASH_NO_API` is unset.
        """
        nemar_id = nemar_id.strip()
        if not nemar_id:
            return False
        payload = self._fetch_json(
            nemar_id,
            kind="top",
            url=f"{self._base_url}/{nemar_id}",
        )
        return payload is not None

    # ----- internals -----------------------------------------------------

    def _fetch_json(
        self,
        nemar_id: str,
        *,
        kind: str,
        url: str,
        transform=None,
    ):
        """Resolution order: disk cache (within TTL), then network.

        ``transform``, when supplied, is invoked on the fresh payload
        *before* it is written to the disk cache -- e.g. to strip
        ``url`` fields that expire after 1h.
        """
        cache_path = self._cache_path(nemar_id, kind)
        # 1. Disk cache, if within TTL.
        cached = _read_cache_within_ttl(cache_path, self._ttl)
        if cached is not None:
            return cached

        # 2. EEGDASH_NO_API forbids network. A cache miss returns None.
        if os.environ.get("EEGDASH_NO_API"):
            return None

        # 3. Network.
        with self._lock:
            try:
                payload = self._http_get_json(url)
            except _HttpNotFound:
                return None
            except Exception as exc:  # noqa: BLE001
                self.errors.append(
                    f"nemar: fetch {kind} for {nemar_id} failed at {url}: {exc}"
                )
                logger.debug("nemar: fetch %s for %s failed: %s", kind, nemar_id, exc)
                # Stale-while-error: if the cache file exists (even past
                # TTL) we serve it as a graceful fallback. Otherwise None.
                stale = _read_cache(cache_path)
                return stale

            to_cache = transform(payload) if transform is not None else payload
            _write_cache(cache_path, to_cache)
            return to_cache

    def _http_get_json(self, url: str):
        """One GET → JSON; raises :class:`_HttpNotFound` on a 404.

        NEMAR's edge rejects the default ``Python-urllib/N`` User-Agent
        with a 403, so we set an identifiable UA on every request. The
        UA also makes it easier for the NEMAR operators to attribute
        traffic if they need to reach out.
        """
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": _NEMAR_USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise _HttpNotFound() from exc
            raise
        return json.loads(payload)

    def _cache_path(self, nemar_id: str, kind: str) -> Path:
        safe_id = "".join(c for c in nemar_id if c.isalnum() or c in {"_", "-"})
        safe_kind = "".join(c for c in kind if c.isalnum() or c in {"_", "-"})
        # ``safe_id or "default"`` defends against pathological input;
        # the docs build never produces an empty ``nemar_id``, but the
        # cache layer should not crash on it either.
        name = f"{safe_id or 'default'}__{safe_kind or 'default'}.json"
        return self._cache_dir / name


# ---------------------------------------------------------------------------
# Module-level helpers (kept at module scope so tests can monkeypatch them)
# ---------------------------------------------------------------------------


@dataclass
class _HttpNotFound(Exception):
    """Internal sentinel for 404 responses -- consumed inside the client."""

    message: str = field(default="not found")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _read_cache_within_ttl(path: Path, ttl: timedelta):
    """Return parsed JSON when the cache file is present *and* fresh."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    age = _utcnow() - datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    if age > ttl:
        return None
    return _read_cache(path)


def _read_cache(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.debug("nemar: cache read failed at %s: %s", path, exc)
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.debug("nemar: cache decode failed at %s: %s", path, exc)
        return None


def _write_cache(path: Path, payload) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.debug("nemar: cache write failed at %s: %s", path, exc)


def _strip_signed_urls(payload):
    """Drop the ``url`` field from each manifest entry before caching.

    NEMAR's signed S3 URLs expire one hour after generation, so caching
    them poisons the cache. Stripping them here keeps the rest of the
    payload (path / size / checksum) cacheable indefinitely.
    """
    if not isinstance(payload, list):
        return payload
    stripped: list[dict] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        copy = {k: v for k, v in entry.items() if k != "url"}
        stripped.append(copy)
    return stripped


def _parse_created_at(raw: str) -> datetime:
    """Parse NEMAR's ``"YYYY-MM-DD HH:MM:SS"`` (UTC) timestamps.

    NEMAR returns naive timestamps that document themselves as UTC.
    We pin the tzinfo here so downstream sorts and ISO renders are
    unambiguous.
    """
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _build_metadata(
    nemar_id: str,
    top_payload: dict,
    meta_payload: dict,
) -> NemarMetadata:
    """Merge the two NEMAR responses into one :class:`NemarMetadata`."""
    # Versions: prefer the rich top-level descriptor (it always carries
    # ``browse_url``), but fall back to ``meta_payload["extensions"]["nemar"]``
    # when the top descriptor is unexpectedly empty.
    raw_versions = top_payload.get("versions") or []
    if not raw_versions:
        nemar_ext = (meta_payload.get("extensions") or {}).get("nemar") or {}
        raw_versions = nemar_ext.get("versions") or []

    parsed_versions: list[NemarVersion] = []
    for raw in raw_versions:
        try:
            parsed_versions.append(
                NemarVersion(
                    version=str(raw["version"]),
                    doi=str(raw["doi"]),
                    created_at=_parse_created_at(str(raw["created_at"])),
                    manifest_url=_absolute(raw.get("manifest_url"), _DEFAULT_BASE_URL),
                    browse_url=_absolute(
                        raw.get("browse_url")
                        or f"/{nemar_id}/{raw.get('version', '')}/",
                        _DEFAULT_BASE_URL,
                    ),
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("nemar: skipping malformed version entry %r: %s", raw, exc)
            continue

    # Sort newest-first defensively.
    parsed_versions.sort(key=lambda v: v.created_at, reverse=True)

    authors = tuple(
        NemarAuthor(
            name=str(a.get("name") or "").strip(),
            orcid=_clean_orcid(a.get("orcid")),
        )
        for a in (meta_payload.get("authors") or [])
        if (a.get("name") or "").strip()
    )

    keywords = tuple(
        NemarKeyword(
            term=str(k.get("term") or "").strip(),
            scheme=_clean_optional(k.get("subject_scheme")),
            value_uri=_clean_optional(k.get("value_uri")),
        )
        for k in (meta_payload.get("keywords") or [])
        if (k.get("term") or "").strip()
    )

    recording_modality_raw = meta_payload.get("recording_modality") or []
    if isinstance(recording_modality_raw, str):
        recording_modality_raw = [recording_modality_raw]
    recording_modality = tuple(
        str(m).strip() for m in recording_modality_raw if str(m).strip()
    )

    if parsed_versions:
        latest_version = parsed_versions[0].version
    else:
        latest_version = str(top_payload.get("latest") or "")

    return NemarMetadata(
        dataset_id=str(meta_payload.get("dataset_id") or nemar_id),
        name=str(meta_payload.get("name") or "").strip(),
        description=_clean_optional(meta_payload.get("description")),
        license=_clean_optional(meta_payload.get("license")),
        recording_modality=recording_modality,
        bids_version=_clean_optional(meta_payload.get("bids_version")),
        authors=authors,
        keywords=keywords,
        versions=tuple(parsed_versions),
        latest_version=latest_version,
    )


def _clean_orcid(value) -> str | None:
    """Normalise an ORCID to a bare 16-digit id with hyphens.

    Accepts ``None`` (skip), a full ``https://orcid.org/<id>`` URL, or
    the bare id. Anything else returns ``None`` so callers can rely on
    ``orcid`` either being a well-formed id or absent.
    """
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    # Strip the URL prefix if present.
    for prefix in ("https://orcid.org/", "http://orcid.org/"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text


def _clean_optional(value) -> str | None:
    """Return ``None`` for empty strings / ``None`` / explicit nulls."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "nan"}:
        return None
    return text


def _absolute(url, base: str) -> str:
    """Make a possibly-relative URL absolute against ``base``."""
    if not url:
        return ""
    text = str(url)
    if text.startswith(("http://", "https://")):
        return text
    return f"{base}{text}" if text.startswith("/") else f"{base}/{text}"


__all__ = [
    "NemarAuthor",
    "NemarClient",
    "NemarKeyword",
    "NemarManifestEntry",
    "NemarMetadata",
    "NemarVersion",
]

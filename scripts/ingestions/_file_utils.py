"""File utilities for remote and local file access.

This module provides unified functions for:
- Listing files from various sources (HTTP, ZIP, WebDAV, Git)
- Peeking into ZIP files without downloading
- Rate-limited HTTP requests

Uses fsspec for unified filesystem access where possible.
"""

import json
import logging
import os
import re
import struct
import time
import xml.etree.ElementTree as ET
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import httpx

from _http import HTTPStatusError, RequestError, request_response

logger = logging.getLogger(__name__)


def _fetch_scidb_path(
    path: str,
    dataset_id: str,
    version: str,
    api_url: str,
    headers: dict,
    max_depth: int,
    depth: int = 0,
) -> list[dict]:
    """Recursively fetch files from SciDB path."""
    if depth > max_depth:
        return []

    body = {
        "dataSetId": dataset_id,
        "version": version,
        "path": path,
        "lastIndex": 0,
        "pageSize": 1000,
    }

    try:
        resp = request_response(
            "post", api_url, json_body=body, headers=headers, timeout=30
        )
        if not resp or resp.status_code != 200:
            return []
        data = resp.json()

        if data.get("code") != 20000:
            return []

        files = []
        for item in data.get("data", []):
            item_path = item.get("path", "")
            is_dir = item.get("dir", False)

            if is_dir:
                files.extend(
                    _fetch_scidb_path(
                        item_path,
                        dataset_id,
                        version,
                        api_url,
                        headers,
                        max_depth,
                        depth + 1,
                    )
                )
            else:
                # Strip version prefix from path
                rel_path = item_path.lstrip(f"/{version}/")
                files.append(
                    {
                        "name": rel_path,
                        "size": item.get("size", 0),
                        "md5": item.get("md5", ""),
                    }
                )

        return files

    except (
        httpx.RequestError,
        httpx.HTTPStatusError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
    ):
        # External-service listing failure: network blip, HTTP error,
        # or malformed JSON response. Treat the source as empty.
        return []


def _propfind_datarn(url: str, result: list, visited: set, depth: int = 0):
    """Recursively list files via WebDAV PROPFIND."""
    if depth > 10 or url in visited:
        return
    visited.add(url)

    try:
        resp = request_response(
            "PROPFIND",
            url,
            headers={"Depth": "1"},
            timeout=30,
        )
        if not resp or resp.status_code not in (200, 207):
            return

        root = ET.fromstring(resp.content)
        ns = {"d": "DAV:"}

        for response in root.findall(".//d:response", ns):
            href_elem = response.find("d:href", ns)
            if href_elem is None:
                continue
            href = unquote(href_elem.text or "")

            # Check if this is a collection (directory)
            is_collection = response.find(".//d:collection", ns) is not None

            if is_collection:
                # Recurse into subdirectory
                sub_url = urljoin(url, href)
                if sub_url != url:
                    _propfind_datarn(sub_url, result, visited, depth + 1)
            else:
                # Extract file info
                size_elem = response.find(".//d:getcontentlength", ns)
                size = int(size_elem.text or 0) if size_elem is not None else 0

                # Get relative path
                parsed = urlparse(href)
                path = parsed.path.lstrip("/")

                result.append(
                    {
                        "name": path,
                        "size": size,
                    }
                )

    except (
        httpx.RequestError,
        httpx.HTTPStatusError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
    ):
        # Recoverable external-service failure; we already have partial
        # `result`/`webdav_url` accumulated. Continue with what we have.
        pass


# BIDS file detection patterns
BIDS_ROOT_FILES = {
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "readme",
    "readme.md",
    "readme.txt",
    "changes",
}

BIDS_DATA_EXTENSIONS = {
    ".edf",
    ".bdf",
    ".vhdr",
    ".vmrk",
    ".eeg",
    ".set",
    ".fdt",
    ".cnt",
    ".mff",
    ".nwb",
    ".nii",
    ".nii.gz",
    ".json",
    ".tsv",
}


def is_bids_file(filename: str) -> bool:
    """Check if a filename looks like a BIDS file."""
    name_lower = filename.lower()
    base = Path(filename).name.lower()

    # Root BIDS files
    if base in BIDS_ROOT_FILES:
        return True

    # Subject/session patterns
    if "sub-" in name_lower:
        return True

    # BIDS data extensions
    for ext in BIDS_DATA_EXTENSIONS:
        if name_lower.endswith(ext):
            return True

    return False


def is_bids_root_file(filename: str) -> bool:
    """Check if file is a BIDS root indicator file."""
    return Path(filename).name.lower() in BIDS_ROOT_FILES


# =============================================================================
# Rate Limiting Decorator
# =============================================================================


def _is_rate_limited_retryable(exc: BaseException) -> bool:
    """Retry-predicate for ``rate_limited``: 429 + network errors only.

    Used as a tenacity ``retry_if_exception`` callback; defined at
    module level (rather than nested in ``rate_limited``) to satisfy
    the no-nested-functions lint rule. Programmer errors
    (AttributeError, TypeError, etc.) return False and propagate.
    """
    if isinstance(exc, HTTPStatusError):
        return exc.response is not None and exc.response.status_code == 429
    return isinstance(exc, RequestError)


def rate_limited(min_interval: float = 0.5, max_retries: int = 3):
    """Decorator for rate-limited HTTP requests with retry on 429/network errors.

    Wraps ``func`` so that:
    - Calls are spaced at least ``min_interval`` seconds apart.
    - HTTP 429 (Too Many Requests) and httpx network errors are retried
      with exponential backoff (factor of 2 between attempts), up to
      ``max_retries`` total attempts.
    - Non-429 HTTPStatusError surfaces immediately on the first attempt
      (404/5xx surface; tenacity-backed callers handle retry themselves
      via the ``_http`` helper).

    Parameters
    ----------
    min_interval : float
        Minimum seconds between consecutive calls (rate limit). Also
        used as the base for the exponential backoff between retries.
    max_retries : int
        Maximum total attempts (not delta — so ``max_retries=3`` makes
        up to 3 attempts).

    Notes
    -----
    Phase 9 audit-2 F1 consolidation: the previous version used a
    hand-rolled try/except retry loop. This rewrite uses ``tenacity``
    via ``stop_after_attempt`` / ``wait_exponential`` so it shares
    semantics with ``_http.request_json``. The behaviour change is
    documented in the test suite.
    """
    last_call = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from tenacity import (
                RetryError,
                Retrying,
                retry_if_exception,
                stop_after_attempt,
                wait_exponential,
            )

            # Enforce minimum interval between calls.
            elapsed = time.time() - last_call[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            retrying = Retrying(
                stop=stop_after_attempt(max_retries),
                wait=wait_exponential(
                    multiplier=min_interval, min=min_interval, max=60
                ),
                retry=retry_if_exception(_is_rate_limited_retryable),
            )
            result: Any = None
            try:
                for attempt in retrying:
                    with attempt:
                        result = func(*args, **kwargs)
            except RetryError:
                # Exhausted retries on a retryable exception path.
                # Preserve the legacy contract: return None rather than
                # propagating tenacity's RetryError wrapper.
                return None
            last_call[0] = time.time()
            return result

        return wrapper

    return decorator


# =============================================================================
# ZIP File Utilities
# =============================================================================


def peek_zip_contents(url: str, timeout: int = 30) -> list[dict] | None:
    """List ZIP contents using HTTP Range requests (no full download).

    Uses the ZIP End-of-Central-Directory structure to read file listing
    from the end of the file.

    Args:
        url: Direct download URL for the ZIP file
        timeout: Request timeout in seconds

    Returns
    -------
        List of {name, size, compressed_size} dicts, or None on error

    """
    try:
        # Get file size via HEAD
        head = request_response(
            "head",
            url,
            timeout=timeout,
            allow_redirects=True,
        )
        if not head or head.status_code != 200:
            return None

        file_size = int(head.headers.get("Content-Length", 0))
        if file_size < 22:  # Minimum ZIP size
            return None

        # Check if server supports Range requests
        if head.headers.get("Accept-Ranges", "").lower() != "bytes":
            return None

        # Read last 64KB to find EOCD (End of Central Directory)
        eocd_search_size = min(65536, file_size)
        range_start = file_size - eocd_search_size

        resp = request_response(
            "get",
            url,
            headers={"Range": f"bytes={range_start}-{file_size - 1}"},
            timeout=timeout,
        )
        if not resp or resp.status_code not in (200, 206):
            return None

        data = resp.content

        # Find EOCD signature (0x06054b50)
        eocd_sig = b"\x50\x4b\x05\x06"
        eocd_pos = data.rfind(eocd_sig)
        if eocd_pos == -1:
            return None

        # Parse EOCD
        eocd = data[eocd_pos : eocd_pos + 22]
        if len(eocd) < 22:
            return None

        cd_size = struct.unpack("<I", eocd[12:16])[0]
        cd_offset = struct.unpack("<I", eocd[16:20])[0]

        # Read Central Directory
        resp = request_response(
            "get",
            url,
            headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"},
            timeout=timeout,
        )
        if not resp or resp.status_code not in (200, 206):
            return None

        cd_data = resp.content
        files = []
        pos = 0
        cd_sig = b"\x50\x4b\x01\x02"

        while pos < len(cd_data) - 46:
            if cd_data[pos : pos + 4] != cd_sig:
                break

            compressed_size = struct.unpack("<I", cd_data[pos + 20 : pos + 24])[0]
            uncompressed_size = struct.unpack("<I", cd_data[pos + 24 : pos + 28])[0]
            name_len = struct.unpack("<H", cd_data[pos + 28 : pos + 30])[0]
            extra_len = struct.unpack("<H", cd_data[pos + 30 : pos + 32])[0]
            comment_len = struct.unpack("<H", cd_data[pos + 32 : pos + 34])[0]

            name_start = pos + 46
            name_end = name_start + name_len
            if name_end > len(cd_data):
                break

            filename = cd_data[name_start:name_end].decode("utf-8", errors="replace")

            # Skip directories
            if not filename.endswith("/"):
                files.append(
                    {
                        "name": filename,
                        "size": uncompressed_size,
                        "compressed_size": compressed_size,
                    }
                )

            pos = name_end + extra_len + comment_len

        return files if files else None

    except (struct.error, UnicodeDecodeError, ValueError, IndexError) as e:
        # ZIP central-directory parser. struct.error fires on truncated /
        # garbage bytes; UnicodeDecodeError on filenames in unexpected
        # encodings; IndexError on offsets past EOF. All recoverable —
        # caller treats "no peek possible" as "skip the optimisation".
        logger.debug("peek_zip_contents failed: %s", e)
        return None


# =============================================================================
# Source-Specific File Listing
# =============================================================================


def list_figshare_files(article_id: int | str, api_key: str = "") -> list[dict]:
    """List files from a Figshare article.

    .. warning::
        **Secondary Source.** CI exercises only OpenNeuro and NEMAR;
        this Adapter is best-effort and may silently drop fields
        not in the shared schema. Fix opportunistically when
        exercised; do not invest in depth until promoted in a
        future sprint. See
        ``ROBUSTNESS/ADRs/0001-secondary-source-deferral.md``.
    """
    headers = {"User-Agent": "EEGDash/1.0"}
    if api_key:
        headers["Authorization"] = f"token {api_key}"

    url = f"https://api.figshare.com/v2/articles/{article_id}/files"

    try:
        resp = request_response("get", url, headers=headers, timeout=30)
        if not resp or resp.status_code != 200:
            return []
        files = resp.json()

        result = []
        for f in files:
            file_info = {
                "name": f.get("name", ""),
                "size": f.get("size", 0),
                "download_url": f.get("download_url", ""),
            }

            # Try to peek into ZIP files
            if file_info["name"].lower().endswith(".zip") and file_info["download_url"]:
                zip_contents = peek_zip_contents(file_info["download_url"])
                if zip_contents:
                    file_info["zip_contents"] = zip_contents

            result.append(file_info)

        return result

    except (
        httpx.RequestError,
        httpx.HTTPStatusError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
    ):
        # External-service listing failure: network blip, HTTP error,
        # or malformed JSON response. Treat the source as empty.
        return []


def list_zenodo_files(record_id: int | str, api_key: str = "") -> list[dict]:
    """List files from a Zenodo record.

    .. warning::
        **Secondary Source.** Same caveats as
        :func:`list_figshare_files`. Zenodo's per-file ``checksum``
        field is now picked up by :func:`build_manifest` (was silently
        dropped before — see
        ``ROBUSTNESS/ADRs/0001-secondary-source-deferral.md``).
    """
    headers = {"User-Agent": "EEGDash/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"https://zenodo.org/api/records/{record_id}"

    try:
        resp = request_response("get", url, headers=headers, timeout=30)
        if not resp or resp.status_code != 200:
            return []
        data = resp.json()

        result = []
        for f in data.get("files", []):
            file_info = {
                "name": f.get("key", ""),
                "size": f.get("size", 0),
                "checksum": f.get("checksum", ""),
            }

            # Build download URL - try multiple sources
            # 1. New Zenodo API: links.self (content endpoint)
            # 2. Old Zenodo API: bucket + filename
            download_url = None

            # New API format
            file_links = f.get("links", {})
            if file_links.get("self"):
                download_url = file_links["self"]

            # Old API format fallback
            if not download_url:
                bucket = data.get("links", {}).get("bucket", "")
                if bucket:
                    download_url = f"{bucket}/{file_info['name']}"

            if download_url:
                file_info["download_url"] = download_url

                # Try to peek into ZIP files
                if file_info["name"].lower().endswith(".zip"):
                    zip_contents = peek_zip_contents(download_url)
                    if zip_contents:
                        file_info["zip_contents"] = zip_contents

            result.append(file_info)

        return result

    except (
        httpx.RequestError,
        httpx.HTTPStatusError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
    ):
        # External-service listing failure: network blip, HTTP error,
        # or malformed JSON response. Treat the source as empty.
        return []


def list_osf_files(node_id: str, path: str = "/") -> list[dict]:
    """Recursively list files from an OSF node.

    .. warning::
        **Secondary Source.** Same caveats as
        :func:`list_figshare_files`. Pagination is currently stubbed
        (line ~593): the helper recurses but does not chase ``next``
        page links, so OSF nodes with > 100 files may return a
        partial listing. See
        ``ROBUSTNESS/ADRs/0001-secondary-source-deferral.md``.
    """
    url = f"https://api.osf.io/v2/nodes/{node_id}/files/osfstorage{path}"
    headers = {"User-Agent": "EEGDash/1.0"}

    result = []

    try:
        resp = request_response("get", url, headers=headers, timeout=30)
        if not resp or resp.status_code != 200:
            return result

        data = resp.json()

        for item in data.get("data", []):
            attrs = item.get("attributes", {})
            kind = attrs.get("kind", "")
            name = attrs.get("name", "")
            item_path = attrs.get("materialized_path", f"{path}{name}")

            if kind == "folder":
                # Recurse into folder
                result.extend(list_osf_files(node_id, item_path))
            else:
                file_info = {
                    "name": item_path.lstrip("/"),
                    "size": attrs.get("size", 0),
                }

                # Get download URL
                links = item.get("links", {})
                if download_url := links.get("download"):
                    file_info["download_url"] = download_url

                    # Try to peek into ZIP files
                    if name.lower().endswith(".zip"):
                        zip_contents = peek_zip_contents(download_url)
                        if zip_contents:
                            file_info["zip_contents"] = zip_contents

                result.append(file_info)

        # Handle pagination
        next_url = data.get("links", {}).get("next")
        if next_url:
            next_resp = request_response("get", next_url, headers=headers, timeout=30)
            if next_resp and next_resp.status_code == 200:
                # Extract path from next URL and recurse
                pass  # Simplified - OSF pagination rarely needed for single datasets

    except (
        httpx.RequestError,
        httpx.HTTPStatusError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
    ):
        # Recoverable external-service failure; we already have partial
        # `result`/`webdav_url` accumulated. Continue with what we have.
        pass

    return result


def list_scidb_files(
    dataset_id: str, version: str = "V1", max_depth: int = 8
) -> list[dict]:
    """List files from SciDB using the public file tree API.

    .. warning::
        **Secondary Source.** Same caveats as
        :func:`list_figshare_files`. SciDB emits ``md5`` directly and
        is one of the few Adapters whose checksum survives the
        manifest pipeline. See
        ``ROBUSTNESS/ADRs/0001-secondary-source-deferral.md``.

    Args:
        dataset_id: The SciDB dataSetId (UUID format)
        version: Dataset version (default: V1)
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns
    -------
        List of file info dicts

    """
    api_url = (
        "https://www.scidb.cn/api/gin-sdb-filetree/public/file/childrenFileListByPath"
    )
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Accept": "application/json, text/plain, */*",
    }

    return _fetch_scidb_path(
        f"/{version}", dataset_id, version, api_url, headers, max_depth
    )


def list_datarn_files(source_url: str) -> list[dict]:
    """List files from data.ru.nl using WebDAV PROPFIND.

    .. warning::
        **Secondary Source.** Same caveats as
        :func:`list_figshare_files`. The WebDAV PROPFIND path emits
        only ``name`` and ``size`` — no checksum, no download URL
        (constructed by the consumer from the source URL). See
        ``ROBUSTNESS/ADRs/0001-secondary-source-deferral.md``.
    """
    # Try to get WebDAV URL from page JSON-LD
    webdav_url = None

    try:
        resp = request_response("get", source_url, timeout=30)
        if resp and resp.status_code == 200:
            ld_match = re.search(
                r'<script type="application/ld\+json">([^<]+)</script>', resp.text
            )
            if ld_match:
                ld_data = json.loads(ld_match.group(1))
                dist = ld_data.get("distribution", {})
                if isinstance(dist, dict):
                    webdav_url = dist.get("contentUrl")
    except (
        httpx.RequestError,
        httpx.HTTPStatusError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
    ):
        # Recoverable external-service failure; we already have partial
        # `result`/`webdav_url` accumulated. Continue with what we have.
        pass

    if not webdav_url:
        return []

    # List files via PROPFIND
    result = []
    visited = set()
    _propfind_datarn(webdav_url, result, visited)
    return result


_ANNEX_SIZE_RE = re.compile(r"-s(\d+)--")
# Anchored to the start of the basename. The trailing ``\.`` is enough to
# reject garbage like ``garbage-not-a-key.set`` because the regex demands
# the SHA-key shape *before* the extension dot.
_ANNEX_KEY_BASENAME_RE = re.compile(
    r"^(?:SHA256E|MD5E)-s\d+--[0-9a-f]+\.", flags=re.IGNORECASE
)
_ANNEX_POINTER_MAX_SIZE = 256


def parse_annex_size(text: str) -> int | None:
    """Extract the real file size from a git-annex key or pointer path.

    Git-annex encodes size in keys like ``MD5E-s{size}--{hash}.ext``.
    Returns ``None`` if *text* does not contain an annex size.
    """
    m = _ANNEX_SIZE_RE.search(text)
    return int(m.group(1)) if m else None


def _read_annex_pointer_text(path: Path) -> str | None:
    """Return the symlink target / smudged-pointer content for *path*.

    For both symlinks and small regular files (≤256B). Returns ``None``
    for missing paths, large files, OS errors, and non-decodable content.
    """
    if path.is_symlink():
        try:
            return str(path.readlink())
        except OSError:
            return None
    if path.is_file():
        try:
            if path.stat().st_size > _ANNEX_POINTER_MAX_SIZE:
                return None
            return path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            return None
    return None


def get_annex_file_key(path: Path) -> str | None:
    """Return the git-annex SHA key for *path*, or ``None``.

    Git-annex tracks binaries as either a symlink ending in
    ``.git/annex/objects/<X>/<Y>/<KEY>/<KEY>`` or a smudged pointer
    file whose content is a line like ``/annex/objects/<KEY>``. In both
    cases the SHA key is the final path segment. Returns ``None`` for
    regular git-tracked sidecars and real binaries.
    """
    text = _read_annex_pointer_text(path)
    if text is None or "/annex/objects/" not in text:
        return None
    candidate = text.strip().rsplit("/", 1)[-1]
    return candidate if _ANNEX_KEY_BASENAME_RE.match(candidate) else None


# Cap chosen to keep inlined sidecars well under MongoDB's 16 MB BSON
# document limit while still covering the long tail of legitimate
# events.tsv / participants.tsv files (typical: 1–10 KB; outlier: 1–2 MB).
_INLINE_SIDECAR_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_INLINE_SIDECAR_SUFFIXES = (".tsv", ".json", ".md", ".txt")
_INLINE_SIDECAR_BASENAMES = frozenset({"README", "CHANGES", "LICENSE", ".bidsignore"})


def read_inline_sidecar(path: Path) -> str | None:
    """Return UTF-8 contents of *path* if it's eligible for digest-time inlining.

    Intended **only** for the NEMAR digest pipeline's
    ``storage.sidecar_inline`` enrichment — not as a general text
    reader. Returns ``None`` for symlinks (annex pointers in this
    tree), oversized files, NUL-byte content, or anything outside the
    allowlist of small text sidecars. Empty files return ``""``.

    Files that don't decode as UTF-8 are intentionally normalized: we
    try ``latin-1`` then ``windows-1252`` and log a warning. BIDS spec
    mandates UTF-8 for sidecars, so non-UTF-8 inputs are pre-existing
    data quality issues that we silently fix on the way into Mongo.

    Note: callers in the digest pipeline are expected to filter out
    annex-managed files (via ``get_annex_file_key``) before invoking
    this; we don't re-check here. The basename allowlist below mirrors
    the apex-file enumeration ``NEMAR_ROOT_METADATA_FILES`` in
    ``eegdash/schemas.py`` — keep them aligned when adding entries.
    See also ``BIDS_ROOT_FILES``/``BIDS_REQUIRED_FILES`` in this module
    and ``_bids.py`` for the case-folded matching used at fetch time.
    """
    if not path.is_file() or path.is_symlink():
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > _INLINE_SIDECAR_MAX_BYTES:
        return None

    if (
        path.suffix.lower() not in _INLINE_SIDECAR_SUFFIXES
        and path.name not in _INLINE_SIDECAR_BASENAMES
    ):
        return None

    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        for enc in ("latin-1", "windows-1252"):
            try:
                decoded = data.decode(enc)
            except UnicodeDecodeError:
                continue
            logger.warning(
                "Inlining %s via %s fallback (file is not valid UTF-8); "
                "BIDS spec mandates UTF-8 — consider fixing upstream.",
                path,
                enc,
            )
            return decoded
        return None


def get_annex_file_size(path: Path) -> int:
    """Get file size, resolving git-annex pointers and symlinks.

    Returns the real data size for annex-managed files, the stat size
    for regular files, or 0 for broken symlinks without annex keys.
    """
    text = _read_annex_pointer_text(path)
    if text is not None and "/annex/" in text:
        annex_size = parse_annex_size(text)
        if annex_size is not None:
            return annex_size
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    return 0


def list_git_files(clone_dir: Path) -> list[dict]:
    """List files from a cloned git repository.

    Includes both regular files and symlinks (even broken ones like git-annex
    pointers).  Resolves git-annex pointer files and symlinks to their real
    data size.

    Implementation note (perf): walks via :func:`os.scandir` in a single
    iterative pass and classifies each entry from the cached dirent
    flags (``DT_REG`` / ``DT_LNK`` / ``DT_DIR``) instead of via
    ``Path.rglob`` + repeated ``is_file`` / ``is_symlink`` ``stat``
    syscalls.

    The walker's classification (is_dir / is_file / is_symlink) uses
    the cached dirent flags from scandir — zero stat syscalls. Per-
    emitted-file stat cost is unchanged from the prior code:
    ``get_annex_file_size`` (called once per emitted entry) still does
    its own Path-based stat probes to disambiguate git-annex pointer
    files from regular files. The Stage-2 speedup comes from cutting
    classification syscalls for the (much larger) population of
    non-emitted intermediate directory entries, not from changing the
    per-emitted-file cost.

    See ``tests/test_manifest_walk_perf.py`` for the regression guard.
    """
    result: list[dict] = []
    # Pre-resolve to absolute path once so ``os.path.relpath`` is purely
    # a string operation per entry, avoiding ``Path.__init__`` per file.
    clone_dir_str = os.fspath(clone_dir)
    stack: list[str] = [clone_dir_str]

    while stack:
        cur = stack.pop()
        try:
            scanner = os.scandir(cur)
        except (PermissionError, OSError, FileNotFoundError):
            continue

        with scanner:
            for entry in scanner:
                # Skip the top-level ".git" tree without ever stat-ing
                # its contents. Matches the prior ``".git" in path.parts``
                # guard for any path under the .git subtree, because we
                # never descend into it.
                name = entry.name
                if name == ".git":
                    continue

                try:
                    is_symlink = entry.is_symlink()
                    # ``follow_symlinks=False`` keeps broken symlinks
                    # classified as non-dir, matching the prior
                    # ``path.is_file()`` (which followed symlinks but
                    # would return False for broken pointers).
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    # Dirent went away between readdir and stat. Skip.
                    continue

                if is_dir and not is_symlink:
                    stack.append(entry.path)
                    continue

                # Mirror the prior classifier exactly: emit an entry
                # when the dirent is a regular file (``is_file`` after
                # following symlinks) OR when it's a symlink (broken or
                # otherwise — git-annex pointers).
                try:
                    is_file = entry.is_file()
                except OSError:
                    is_file = False

                if not (is_file or is_symlink):
                    continue

                # Size resolution preserves the exact semantics of
                # ``get_annex_file_size``: annex-keyed pointers → size
                # parsed from the key, broken symlinks → 0, regular
                # files → ``stat.st_size``.
                size = get_annex_file_size(Path(entry.path))
                rel = os.path.relpath(entry.path, clone_dir_str)
                result.append({"name": rel, "size": size})

    return result


# =============================================================================
# Manifest Building
# =============================================================================


def build_manifest(
    dataset_id: str,
    source: str,
    files: list[dict],
    metadata: dict | None = None,
) -> dict:
    """Build a standardized manifest from file listing.

    Args:
        dataset_id: Dataset identifier
        source: Source name (openneuro, figshare, etc.)
        files: List of file info dicts with at least 'name' and 'size'
        metadata: Optional additional metadata from consolidated files.
                  ALL metadata is preserved to ensure no information loss.

    Returns
    -------
        Manifest dict ready to be saved

    Note:
        Normalizes file info to use 'path' key (expected by digest script)
        and '_zip_contents' for ZIP contents.

        IMPORTANT: All metadata from the fetch step is preserved in the manifest
        to ensure downstream scripts (digest) have access to all collected data.

    """
    # Normalize file info format for digest script
    normalized_files = []
    for f in files:
        nf = {
            "path": f.get("path") or f.get("name", ""),
            "size": f.get("size", 0),
        }
        if download_url := f.get("download_url"):
            nf["download_url"] = download_url
        # Some Adapters (Zenodo) emit ``checksum`` instead of ``md5``;
        # accept either. Without this, Zenodo content hashes were
        # silently dropped at manifest time. See
        # ROBUSTNESS/ADRs/0001-secondary-source-deferral.md.
        if checksum := f.get("md5") or f.get("checksum"):
            nf["md5"] = checksum

        # Normalize zip_contents to _zip_contents (expected by digest)
        if zip_contents := f.get("zip_contents"):
            # Also normalize inner file format
            nf["_zip_contents"] = [
                {
                    "path": zf.get("path") or zf.get("name", ""),
                    "size": zf.get("size", 0),
                }
                for zf in zip_contents
            ]

        normalized_files.append(nf)

    # Identify BIDS root files
    bids_files = [f["path"] for f in normalized_files if is_bids_root_file(f["path"])]

    # Calculate totals
    total_size = sum(f.get("size", 0) for f in normalized_files)

    # Start with all metadata from the fetch step (preserve everything!)
    # This ensures no information is lost between fetch -> clone -> digest
    manifest = {}
    if metadata:
        # Copy all metadata except internal/computed fields
        exclude_keys = {"_files", "files", "total_files", "total_size", "bids_files"}
        for key, value in metadata.items():
            if key not in exclude_keys and value is not None:
                manifest[key] = value

    # Override/set core fields (these are authoritative from this step)
    manifest.update(
        {
            "dataset_id": dataset_id,
            "source": source,
            "total_files": len(normalized_files),
            "total_size": total_size,
            "bids_files": bids_files,
            "files": normalized_files,
        }
    )

    return manifest


def save_manifest(manifest: dict, output_dir: Path) -> Path:
    """Save manifest to disk."""
    manifest_dir = output_dir / manifest["dataset_id"]
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def list_local_bids_files(local_path: str | Path) -> list[dict]:
    """List files from a local BIDS directory.

    Args:
        local_path: Path to local BIDS dataset directory

    Returns
    -------
        List of file dicts with {name, size} for each file

    """
    local_path = Path(local_path)
    if not local_path.exists():
        return []

    files = []
    for root, dirs, filenames in os.walk(local_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for filename in filenames:
            # Skip hidden files
            if filename.startswith("."):
                continue

            filepath = Path(root) / filename
            rel_path = filepath.relative_to(local_path)

            try:
                size = filepath.stat().st_size
            except OSError:
                size = 0

            files.append(
                {
                    "name": str(rel_path),
                    "size": size,
                }
            )

    return files

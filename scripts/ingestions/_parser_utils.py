"""Common utilities for parser modules.

This module provides shared functionality for validating file paths,
reading files with encoding fallback, and fetching from S3 for
git-annex symlinks.

HTTP path note (2026-05-22)
---------------------------
The three S3 helpers (``fetch_bytes_from_s3``, ``head_content_length``,
``fetch_from_s3``) used to call ``urllib.request.urlopen`` directly.
The Stage-3 profile run of 2026-05-22 showed that the MEG montage
extractor calls ``head_content_length`` ~100x per dataset, and that
91% of the digest wall-clock went to TLS handshake / connect / read
for those serialised, unpooled requests.

This module now lazy-initialises a single shared :class:`httpx.Client`
per process (``_http_client()``), keeping connections alive across
calls to the same host. S3 is one host for every MEG montage URL, so
all subsequent HEAD requests reuse the first handshake. Expected
speedup on MEG-heavy datasets is ~5x (per the profile report).

The signature + return type + exception behaviour of the three public
functions are unchanged. Tests use ``respx`` to mock the httpx
transport (same pattern as ``_inject_config.py``).
"""

from __future__ import annotations

import atexit
import logging
import re
import threading
from pathlib import Path
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)


# ─── Pooled httpx client (shared across all S3 helpers) ───────────────────

# httpx.Client is thread-safe for synchronous use (per docs). The
# Stage-2 clone uses ThreadPoolExecutor → one client is fine. The
# Stage-3 digest uses multiprocessing spawn → each worker process
# gets its own client (re-import = new module-level state). No
# inter-process state to manage.

_HTTP_CLIENT: httpx.Client | None = None
_HTTP_CLIENT_LOCK = threading.Lock()


def _http_client() -> httpx.Client:
    """Return the lazily-initialised shared HTTP client.

    Pooled keep-alive connections (default httpx settings) eliminate the
    per-request TLS handshake overhead. Closed at interpreter exit via
    ``atexit`` so we don't leak sockets.
    """
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        with _HTTP_CLIENT_LOCK:
            if _HTTP_CLIENT is None:  # double-checked locking
                _HTTP_CLIENT = httpx.Client(
                    timeout=httpx.Timeout(30.0),
                    limits=httpx.Limits(
                        max_keepalive_connections=20,
                        max_connections=40,
                        keepalive_expiry=60.0,
                    ),
                    follow_redirects=True,
                )
                atexit.register(_close_http_client)
    return _HTTP_CLIENT


def _close_http_client() -> None:
    """Test helper + atexit hook to release the pooled client."""
    global _HTTP_CLIENT
    client = _HTTP_CLIENT
    _HTTP_CLIENT = None
    if client is not None:
        try:
            client.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Error closing httpx client at shutdown: %s", exc)


def reset_http_client_for_testing() -> None:
    """Drop the cached client so the next call rebuilds it.

    Tests that use respx may want to inspect call counts on a fresh
    transport; this resets the singleton so respx hooks land cleanly.
    """
    _close_http_client()


def is_broken_symlink(path: Path) -> bool:
    """Check if a path is a broken symlink (e.g., git-annex).

    Parameters
    ----------
    path : Path
        Path to check.

    Returns
    -------
    bool
        True if the path is a symlink but the target doesn't exist.

    """
    try:
        # is_symlink() returns True even for broken symlinks
        # exists() returns False for broken symlinks
        return path.is_symlink() and not path.exists()
    except (OSError, RuntimeError):
        return False


def extract_dataset_info(path: Path) -> tuple[str, str, str] | None:
    r"""Extract ``(source, dataset_id, relative_path)`` from a path inside a cloned dataset.

    Matches the two hosted source conventions currently supported:

    - ``ds\d+`` paths → ``source="openneuro"`` (e.g. ``ds001234/sub-01/eeg/x.vhdr``)
    - ``nm\d+`` paths → ``source="nemar"``   (e.g. ``nm000123/sub-01/meg/x.fif``)

    The returned ``relative_path`` is the path relative to the dataset
    root, normalised to forward slashes.

    Returns ``None`` when neither pattern matches.
    """
    path_str = str(path.resolve() if not is_broken_symlink(path) else path.absolute())
    for source, pat in (
        ("openneuro", r"[/\\](ds\d+)[/\\](.+)$"),
        ("nemar", r"[/\\](nm\d+)[/\\](.+)$"),
    ):
        m = re.search(pat, path_str)
        if m:
            ds_id = m.group(1)
            rel = m.group(2).replace("\\", "/")
            return source, ds_id, rel
    return None


def extract_openneuro_info(path: Path) -> tuple[str, str] | None:
    """Backwards-compatible OpenNeuro-only wrapper around ``extract_dataset_info``."""
    info = extract_dataset_info(path)
    if info is None or info[0] != "openneuro":
        return None
    _, dataset_id, relative_path = info
    return dataset_id, relative_path


def build_s3_url(dataset_id: str, relative_path: str, source: str = "openneuro") -> str:
    """Build S3 URL for a file in a dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g., "ds001234", "nm000123").
    relative_path : str
        Path within the dataset (e.g., "sub-01/eeg/file.vhdr").
    source : str
        Data source: ``"openneuro"`` or ``"nemar"``.

    Returns
    -------
    str
        S3 URL for the file.

    Raises
    ------
    ValueError
        If the source is not supported.

    """
    encoded_path = quote(relative_path, safe="/")
    if source == "openneuro":
        # OpenNeuro's git-annex remote uses ``exporttree=yes``, so files on
        # S3 are addressed by their BIDS path directly.
        return f"https://s3.amazonaws.com/openneuro.org/{dataset_id}/{encoded_path}"
    if source == "nemar":
        # NOTE: NEMAR's public README documents this same BIDS-path layout
        # (``aws s3 cp s3://nemar/<id>/path/to/file --no-sign-request``),
        # but in practice the bucket only serves files at
        # ``s3://nemar/<id>/objects/<git-annex-key>`` — anonymous GETs on
        # BIDS paths return 403, and ListObjectsV2 is denied. ``relative_path``
        # here must already be the ``objects/<annex-key>`` form; callers
        # cannot pass a BIDS path. See _resolve_nemar_annex_key (TODO).
        return f"https://s3.amazonaws.com/nemar/{dataset_id}/{encoded_path}"
    raise ValueError(f"Unsupported source for S3 URL: {source}")


def fetch_bytes_from_s3(
    url: str,
    *,
    start: int = 0,
    max_bytes: int = 262144,
    timeout: float = 30.0,
) -> bytes | None:
    """Range-fetch ``max_bytes`` starting at ``start`` from an S3 object.

    Used to pull MEG FIF / KIT SQD headers without downloading the full
    multi-GB recording. S3 always supports ``Range: bytes=start-end`` —
    the response is 206 Partial Content with the exact range requested.
    The MNE FIF reader only needs enough bytes to walk the ``FIFFB_ROOT``
    → ``FIFFB_MEAS_INFO`` → channel info blocks, usually well under
    256 KB even for 500-channel recordings.

    Uses the shared :func:`_http_client` (pooled keep-alive). Returns
    the raw byte slice on success, ``None`` on any network or protocol
    failure (caller decides whether to retry with a larger ``max_bytes``).
    """
    end = int(start) + int(max_bytes) - 1
    headers = {
        "Range": f"bytes={int(start)}-{end}",
        "Accept": "*/*",
    }
    logger.debug("Range-fetching bytes=%d-%d from %s", start, end, url)
    try:
        resp = _http_client().get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        # Servers that don't honour Range return 200 with the full
        # body; accept both but log when we got more than asked.
        data = resp.content
        if len(data) > max_bytes:
            logger.debug(
                "server ignored Range; got %d bytes (asked %d) from %s",
                len(data),
                max_bytes,
                url,
            )
        return data
    except httpx.HTTPError as e:
        logger.debug("HTTP/network error range-fetching %s: %s", url, e)
        return None


def head_content_length(url: str, *, timeout: float = 30.0) -> int | None:
    """Issue a HEAD request and return ``Content-Length`` as int.

    Uses the shared :func:`_http_client` (pooled keep-alive). The
    Stage-3 profile of 2026-05-22 showed this function is called ~100×
    per MEG dataset (one per FIF montage source); each call previously
    opened a fresh TLS connection. Sharing one client across calls
    yields the ~5x speedup the profile report quotes.

    Returns ``None`` on any network failure or when the header is absent.
    """
    try:
        resp = _http_client().head(url, timeout=timeout)
        resp.raise_for_status()
        raw = resp.headers.get("Content-Length")
        return int(raw) if raw is not None else None
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("HEAD %s failed: %s", url, exc)
        return None


def fetch_from_s3(
    url: str,
    timeout: float = 30.0,
    encodings: tuple[str, ...] = ("utf-8", "latin-1", "cp1252"),
) -> str | None:
    """Fetch text content from an S3 URL.

    Uses the shared :func:`_http_client` (pooled keep-alive). Tries each
    encoding in order; returns the first that decodes the body cleanly.

    Parameters
    ----------
    url : str
        S3 URL to fetch.
    timeout : float
        Request timeout in seconds.
    encodings : tuple[str, ...]
        Encodings to try when decoding the response.

    Returns
    -------
    str | None
        File content as string, or None if fetch or decode fails.
    """
    logger.debug("Fetching from S3: %s", url)
    try:
        resp = _http_client().get(url, timeout=timeout)
        resp.raise_for_status()
        content_bytes = resp.content
    except httpx.HTTPError as e:
        logger.debug("HTTP/network error fetching from S3: %s - %s", url, e)
        return None

    for encoding in encodings:
        try:
            content = content_bytes.decode(encoding)
            logger.debug("Successfully fetched %d bytes from S3", len(content_bytes))
            return content
        except UnicodeDecodeError:
            continue

    logger.warning("Failed to decode S3 content with any encoding: %s", url)
    return None


def path_is_within_root(path: Path | str, root: Path | str) -> bool:
    """Return True if ``path`` resolves inside ``root``.

    Defense-in-depth helper for path-traversal containment. Resolves
    both arguments to absolute paths and tests that the result of
    ``path`` is a sub-path of ``root``. Symlinks are followed, so a
    symlink-out-of-tree fails the check.

    Parameters
    ----------
    path : Path or str
        Candidate file or directory path.
    root : Path or str
        Trusted root directory that ``path`` must remain inside.

    Returns
    -------
    bool
        True if ``path`` is structurally contained in ``root`` AND both
        resolve without OS-level error. False otherwise — including
        when either side cannot be resolved (e.g. permission denied on
        an ancestor directory).

    Notes
    -----
    Phase 9 audit-3 F1+F2 fix. Internal trust model: paths from a
    manifest the pipeline itself built are already trusted; this helper
    exists so a future code path that accepts a user-supplied sidecar
    reference (BIDS ``IntendedFor``, ``.vhdr`` ``DataFile=``, etc.) can
    cheaply gate it.

    Examples
    --------
    >>> from pathlib import Path
    >>> root = Path("/data/ds002893").resolve()
    >>> path_is_within_root(root / "sub-01" / "eeg" / "x.set", root)
    True
    >>> path_is_within_root(root / ".." / "ds_other" / "x.set", root)
    False
    """
    try:
        p = Path(path).resolve()
        r = Path(root).resolve()
    except (OSError, RuntimeError):
        return False
    try:
        # is_relative_to landed in 3.9 — we target 3.10+ per pyproject.
        return p.is_relative_to(r)
    except (AttributeError, ValueError):
        # AttributeError: not reachable under py3.9+; defensive.
        # ValueError: shouldn't fire here but matches the docs contract.
        return False


def validate_file_path(path: Path) -> bool:
    """Check if a file path exists and is readable.

    Handles broken symlinks (e.g., git-annex) by resolving the path
    and checking if the target exists.

    Parameters
    ----------
    path : Path
        Path to validate.

    Returns
    -------
    bool
        True if the file exists and is readable, False otherwise.

    Notes
    -----
    For broken git-annex symlinks, this returns False. Use
    `read_with_encoding_fallback` which will attempt S3 fallback
    for OpenNeuro datasets.

    """
    if not path.exists():
        return False

    # Handle broken symlinks (git-annex)
    try:
        resolved = path.resolve()
        if not resolved.exists():
            return False
    except (OSError, RuntimeError):
        return False

    return True


def read_with_encoding_fallback(
    path: Path,
    encodings: tuple[str, ...] = ("utf-8", "latin-1", "cp1252"),
    s3_fallback: bool = True,
) -> str | None:
    """Read a file with encoding fallback and optional S3 fallback.

    Tries multiple encodings to read a text file. If the file is a broken
    git-annex symlink and belongs to an OpenNeuro dataset, attempts to
    fetch the content from S3.

    Parameters
    ----------
    path : Path
        Path to the file to read.
    encodings : tuple[str, ...]
        Tuple of encodings to try, in order.
    s3_fallback : bool
        If True, attempt to fetch from S3 for broken git-annex symlinks
        in OpenNeuro datasets.

    Returns
    -------
    str | None
        File content as string, or None if all methods fail.

    """
    # First, try to read the file directly
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, OSError):
            continue

    # If direct read failed and S3 fallback is enabled, check if it's a broken symlink
    if s3_fallback and is_broken_symlink(path):
        info = extract_openneuro_info(path)
        if info:
            dataset_id, relative_path = info
            s3_url = build_s3_url(dataset_id, relative_path)
            logger.info(
                "Using S3 fallback for git-annex symlink: %s -> %s",
                path.name,
                s3_url,
            )
            return fetch_from_s3(s3_url, encodings=encodings)
        else:
            logger.debug(
                "Broken symlink is not an OpenNeuro dataset, cannot use S3 fallback: %s",
                path,
            )

    return None

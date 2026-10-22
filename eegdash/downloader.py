# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""File downloading utilities for EEG data from cloud storage.

This module provides functions for downloading EEG data files and BIDS dependencies from
AWS S3 storage, with support for caching and progress tracking. It handles the communication
between the EEGDash metadata database and the actual EEG data stored in the cloud.

It talks to S3 through a plain synchronous ``boto3`` client rather than ``s3fs``.
``s3fs`` transitively pulls in ``aiobotocore``, which hard-pins ``botocore`` to a
narrow range and makes any environment that also needs ``boto3`` (moabb, awscli,
neuralbench, …) effectively unsolvable. All access here is anonymous, read-only,
and limited to a handful of operations (HEAD, GET, LIST), so ``boto3`` covers it
directly. See https://github.com/eegdash/EEGDash/issues/397.
"""

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Iterable, Sequence

import boto3
import rich.progress
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ReadTimeoutError,
    ResponseStreamingError,
)
from rich.console import Console
from tqdm.auto import tqdm

from .logging import logger


def _split_s3_uri(uri: str) -> tuple[str, str]:
    """Split an ``s3://bucket/key`` (or ``bucket/key``) string into (bucket, key)."""
    without_scheme = re.sub(r"^s3://", "", str(uri)).lstrip("/")
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


_STREAM_RETRIES = 3
_TRANSIENT_STREAM_ERRORS = (
    ResponseStreamingError,
    ConnectionClosedError,
    ReadTimeoutError,
)
# Still failing once boto3's own retries are spent; a bare 429 is not in
# botocore's throttling list at all.
_RETRY_STATUS = (429, 500, 503)
_RETRY_BACKOFF = 2.0  # seconds, doubled per attempt

# git-annex hash backends (``<BACKEND>[E]``) → hashlib algorithm name.
_ANNEX_HASHES = {
    "MD5": "md5",
    "SHA1": "sha1",
    "SHA224": "sha224",
    "SHA256": "sha256",
    "SHA384": "sha384",
    "SHA512": "sha512",
    "SHA3_224": "sha3_224",
    "SHA3_256": "sha3_256",
    "SHA3_384": "sha3_384",
    "SHA3_512": "sha3_512",
}
_ANNEX_KEY_PARTS_RE = re.compile(
    r"^(?P<backend>[A-Z][A-Z0-9_]*)(?P<fields>(?:-[smSC]\d+)*)--(?P<name>.+)$"
)


def annex_key_info(key: str) -> tuple[int | None, str | None, str | None]:
    """Return ``(size, algorithm, hexdigest)`` parsed from a git-annex key.

    Keys read ``<BACKEND>[E]-s<size>[-m<mtime>][-S<chunk>-C<n>]--<hex>[.<ext>]``.
    A value is ``None`` when the key does not carry it (``WORM``/``URL``
    backends have no hash) or the string is not a key. A chunk key names one
    piece of the content, so neither its size nor its hash describes the
    object and all three are ``None``.
    """
    m = _ANNEX_KEY_PARTS_RE.match(str(key or ""))
    if not m:
        return None, None, None
    fields = {f[0]: int(f[1:]) for f in m["fields"].split("-")[1:]}
    if "S" in fields or "C" in fields:
        return None, None, None
    size = fields.get("s")
    backend, name = m["backend"], m["name"]
    algorithm = _ANNEX_HASHES.get(backend)
    if algorithm is None and backend.endswith("E"):
        algorithm = _ANNEX_HASHES.get(backend[:-1])
        name = name.split(".", 1)[0]
    if algorithm is None:
        return size, None, None
    digest = name.lower()
    if not re.fullmatch(r"[0-9a-f]+", digest) or (
        len(digest) != 2 * hashlib.new(algorithm).digest_size
    ):
        return size, None, None
    return size, algorithm, digest


def _annex_key_of(uri: str) -> str | None:
    """The git-annex key a URI ends in (NEMAR ``…/objects/<key>``), else None."""
    name = str(uri).rstrip("/").rsplit("/", 1)[-1]
    return name if annex_key_info(name) != (None, None, None) else None


def _accept_stale_key(
    path: Path,
    got: int,
    annex_key: str | None,
    s3_uri: str,
    *,
    s3_size: int | None = None,
    filesystem: "S3Client | None" = None,
    version_id: str | None = None,
) -> bool:
    """Whether a file that disagrees with *annex_key* is still what S3 serves.

    A record's key comes from NEMAR's copy of an OpenNeuro dataset, while an
    ``openneuro.org`` URI serves OpenNeuro's latest snapshot: *got* bytes equal
    to S3's own size mean the record is stale, not the file bad. Never when
    the key names the object (NEMAR ``…/objects/<key>``), where a mismatch is
    a bad object (nm000289, 2026-10-16). Logs a warning when it accepts.
    """
    if not annex_key or _annex_key_of(s3_uri) == annex_key:
        return False
    if s3_size is None:
        s3_size = _remote_size(
            filesystem or get_s3_filesystem(),
            s3_uri,
            **({"version_id": version_id} if version_id else {}),
        )
    if s3_size is None or got != s3_size:
        return False
    logger.warning(
        "%s: record key %s says %s bytes, %s serves %s: the record is stale, "
        "file accepted",
        path,
        annex_key,
        annex_key_info(annex_key)[0],
        _split_s3_uri(s3_uri)[0],
        s3_size,
    )
    return True


def verify_file(
    path: Path, key_or_expected: str | int, *, verify_hash: bool = False
) -> Path:
    """Check *path* against an annex key (or a byte count); raise ``OSError``.

    The size is always checked when known, the key's hash only with
    *verify_hash* (it reads the whole file). A truncated NEMAR object
    (nm000289, 2026-10-16) passed every check before this one.
    """
    path = Path(path)
    if isinstance(key_or_expected, int):
        size, algorithm, digest = key_or_expected, None, None
    else:
        size, algorithm, digest = annex_key_info(key_or_expected)
    got = path.stat().st_size
    if size is not None and got != size:
        raise OSError(
            f"Incomplete download for {path}: expected {size} bytes, got {got}."
        )
    if verify_hash and algorithm is not None:
        h = hashlib.new(algorithm)
        with open(path, "rb") as fh:
            while chunk := fh.read(1024 * 1024):
                h.update(chunk)
        if h.hexdigest() != digest:
            raise OSError(
                f"Checksum mismatch for {path}: expected {algorithm} {digest}, "
                f"got {h.hexdigest()}."
            )
    return path


class S3Client:
    """Minimal, anonymous S3 accessor with an ``s3fs``-compatible surface.

    Exposes just the operations EEGDash needs — :meth:`info`, :meth:`get_file`,
    :meth:`ls`, :meth:`du` — backed by a synchronous ``boto3`` client configured
    for unsigned (anonymous) access to public buckets. Paths may be given as
    ``s3://bucket/key`` or ``bucket/key``.
    """

    def __init__(self, *, region: str = "us-east-2", max_concurrency: int = 20):
        self._client = boto3.client(
            "s3",
            region_name=region,
            config=Config(
                signature_version=UNSIGNED,
                max_pool_connections=max_concurrency,
                retries={"max_attempts": 5, "mode": "standard"},
            ),
        )

    def info(self, path: str, *, version_id: str | None = None) -> dict:
        """Return object metadata (``{"size": <bytes>}``) via a HEAD request."""
        bucket, key = _split_s3_uri(path)
        resp = self._client.head_object(
            Bucket=bucket, Key=key, **({"VersionId": version_id} if version_id else {})
        )
        return {"size": resp["ContentLength"]}

    def get_file(
        self,
        rpath: str,
        lpath: str,
        *,
        callback=None,
        version_id: str | None = None,
        annex_key: str | None = None,
        verify_hash: bool = False,
    ) -> None:
        """Download a single object with one unsigned GET (no HEAD, no LIST).

        Streams the body to a ``.part`` sibling and atomically renames it into
        place, so an interrupted transfer never leaves a truncated file at the
        destination (which ``skip_existing`` would later mistake for complete).
        Each chunk's byte count is fed to ``callback`` (a ``callback(nbytes)``
        callable) for progress.

        Error mapping matches the ``s3fs`` contract callers depend on: a missing
        key raises :class:`FileNotFoundError`, and a 403 — how S3 reports a
        missing object when anonymous ``ListBucket`` is denied, e.g. NEMAR —
        raises :class:`PermissionError`. A connection dropped mid-stream is
        retried (``_STREAM_RETRIES`` attempts) with a fresh GET, as is a
        429/500/503 that outlived boto3's own retries (with backoff).
        With ``annex_key`` the staged file must match the key's size (and hash,
        with ``verify_hash``) before it is published: S3's own length is that
        of the stored object, which can itself be short (nm000289, 2026-10-16).
        """
        for attempt in range(1, _STREAM_RETRIES + 1):
            try:
                return self._get_file_once(
                    rpath,
                    lpath,
                    callback=callback,
                    annex_key=annex_key,
                    verify_hash=verify_hash,
                    **({"version_id": version_id} if version_id else {}),
                )
            except _TRANSIENT_STREAM_ERRORS as e:
                if attempt == _STREAM_RETRIES:
                    raise
                # ponytail: progress may overshoot after a retry; reset the bar if it matters
                logger.warning(
                    "Retrying %s after %s (%d/%d)", rpath, e, attempt, _STREAM_RETRIES
                )
            except ClientError as e:
                status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if status not in _RETRY_STATUS or attempt == _STREAM_RETRIES:
                    raise
                logger.warning(
                    "Retrying %s after HTTP %s (%d/%d)",
                    rpath,
                    status,
                    attempt,
                    _STREAM_RETRIES,
                )
                time.sleep(_RETRY_BACKOFF * 2 ** (attempt - 1))

    def _get_file_once(
        self,
        rpath: str,
        lpath: str,
        *,
        callback=None,
        version_id: str | None = None,
        annex_key: str | None = None,
        verify_hash: bool = False,
    ) -> None:
        bucket, key = _split_s3_uri(rpath)
        try:
            resp = self._client.get_object(
                Bucket=bucket,
                Key=key,
                **({"VersionId": version_id} if version_id else {}),
            )
        except ClientError as e:
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = str(e.response.get("Error", {}).get("Code"))
            if status == 404 or code in ("NoSuchKey", "NoSuchBucket"):
                raise FileNotFoundError(rpath) from e
            if status == 403 or code in ("AccessDenied", "Forbidden"):
                raise PermissionError(rpath) from e
            raise

        body = resp["Body"]
        # Prefer the GET response's own size for the progress total, so the bar
        # is complete even if the caller's HEAD probe returned nothing.
        if callback is not None and hasattr(callback, "set_size"):
            length = resp.get("ContentLength")
            if length is not None:
                callback.set_size(length)

        lpath = Path(lpath)
        # Unique per caller: records of one dataset share sidecars (events.tsv,
        # electrodes.tsv, coordsystem.json) and download_all fetches them on
        # threads; a fixed ".part" let one GET truncate another's staging file.
        tmp = lpath.with_name(
            f"{lpath.name}.{os.getpid()}.{threading.get_ident()}.part"
        )
        try:
            with open(tmp, "wb") as fh:
                for chunk in body.iter_chunks(chunk_size=1024 * 1024):
                    fh.write(chunk)
                    if callback is not None:
                        callback(len(chunk))
            length = resp.get("ContentLength")
            if length is not None and tmp.stat().st_size != length:
                raise OSError(f"Incomplete download for {rpath} -> {lpath}.")
            if annex_key:
                try:
                    verify_file(tmp, annex_key, verify_hash=verify_hash)
                except OSError as e:
                    # length is S3's own size of the object (checked above)
                    if not _accept_stale_key(
                        lpath, tmp.stat().st_size, annex_key, rpath, s3_size=length
                    ):
                        raise OSError(f"{e} ({rpath} -> {lpath})") from e
            tmp.replace(lpath)
        except BaseException:
            # Interruption / mid-stream error: drop the partial so the next run
            # re-downloads instead of trusting a truncated file.
            tmp.unlink(missing_ok=True)
            raise

    def _iter_objects(self, path: str, *, delimiter: str | None = None):
        """Yield each object dict under a prefix via paginated ListObjectsV2.

        Pass ``delimiter="/"`` to list a single directory level (like
        ``s3fs.ls``); omit it to walk the prefix recursively (for ``du``).
        """
        bucket, key = _split_s3_uri(path)
        paginator = self._client.get_paginator("list_objects_v2")
        kwargs = {"Bucket": bucket, "Prefix": key}
        if delimiter:
            kwargs["Delimiter"] = delimiter
        for page in paginator.paginate(**kwargs):
            yield from page.get("Contents", [])

    def ls(self, path: str, detail: bool = False):
        """List the immediate objects under a prefix as ``bucket/key`` strings.

        Lists one directory level (``Delimiter="/"``), like ``s3fs.ls`` — a
        recursive walk would let a nested subdirectory's keys be flattened into
        the parent by callers that take ``Path(name)`` (e.g. CTF ``.ds`` dirs
        with a nested ``hz.ds``).
        """
        bucket, _ = _split_s3_uri(path)
        prefix = path if path.endswith("/") else path + "/"
        names = [
            f"{bucket}/{obj['Key']}"
            for obj in self._iter_objects(prefix, delimiter="/")
        ]
        if detail:
            return [{"name": n, "type": "file"} for n in names]
        return names

    def du(self, path: str) -> int:
        """Sum the size in bytes of every object under a prefix (recursive)."""
        return sum(obj["Size"] for obj in self._iter_objects(path))


def get_s3_filesystem(
    *,
    max_concurrency: int = 20,
    region: str = "us-east-2",
) -> S3Client:
    """Get an anonymous S3 accessor for public buckets.

    Parameters
    ----------
    max_concurrency : int
        Size of the underlying connection pool (default 20).
    region : str
        AWS region for the S3 endpoint (default ``"us-east-2"``).

    Returns
    -------
    S3Client
        An anonymous S3 accessor with an ``s3fs``-compatible surface.

    """
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
    return S3Client(region=region, max_concurrency=max_concurrency)


def get_s3path(s3_bucket: str, filepath: str) -> str:
    """Construct an S3 URI from a bucket and file path.

    Parameters
    ----------
    s3_bucket : str
        The S3 bucket name (e.g., "s3://my-bucket").
    filepath : str
        The path to the file within the bucket.

    Returns
    -------
    str
        The full S3 URI (e.g., "s3://my-bucket/path/to/file").

    """
    s3_bucket = str(s3_bucket).rstrip("/")
    filepath = str(filepath).lstrip("/")
    return f"{s3_bucket}/{filepath}" if filepath else s3_bucket


def download_s3_file(
    s3_path: str,
    local_path: Path,
    *,
    filesystem: S3Client | None = None,
    version_id: str | None = None,
    annex_key: str | None = None,
    verify_hash: bool = False,
) -> Path:
    """Download a single file from S3 to a local path.

    Handles the download of a raw EEG data file from an S3 bucket, caching it
    at the specified local path. Creates parent directories if they do not exist.
    The size in a git-annex key beats the HEAD, which NEMAR denies: a truncated
    object (nm000289, 2026-10-16) was otherwise accepted as complete.

    Parameters
    ----------
    s3_path : str
        The full S3 URI of the file to download.
    local_path : pathlib.Path
        The local file path where the downloaded file will be saved.
    filesystem : S3Client | None
        Optional pre-created accessor to reuse across multiple downloads.
    version_id : str | None
        Select this exact S3 object version for HEAD and GET. A sibling
        ``.s3-version.json`` records the cache identity; unmarked caches are
        refreshed even when the byte count matches.
    annex_key : str | None
        git-annex key of the object, giving its size and hash. Defaults to the
        key the URI ends in (NEMAR ``…/objects/<key>``), if any.
    verify_hash : bool
        Also check the key's hash, of a cached file and of a fresh download.
        Off by default: it reads the whole file.

    Returns
    -------
    pathlib.Path
        The local path to the downloaded file.

    """
    filesystem = filesystem or get_s3_filesystem()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    annex_key = annex_key or _annex_key_of(s3_path)
    key_size = annex_key_info(annex_key)[0] if annex_key else None
    # The key does not name the object (an OpenNeuro twin): S3's size may win.
    stale_ok = bool(annex_key) and _annex_key_of(s3_path) != annex_key

    # A size match alone cannot identify a version (HEAD may have equal size).
    marker = local_path.with_name(local_path.name + ".s3-version.json")
    identity = {"uri": s3_path, "version_id": version_id}
    cached_identity = None
    if marker.exists():
        try:
            cached_identity = json.loads(marker.read_text())
        except (OSError, ValueError):
            pass
    remote_size = (
        key_size
        if key_size is not None
        else _remote_size(filesystem, s3_path, version_id=version_id)
    )
    cache_matches = cached_identity == identity if version_id else not marker.exists()
    if cache_matches:
        try:
            local_size = local_path.stat().st_size
        except FileNotFoundError:
            pass  # Another caller may be refreshing this shared sidecar.
        else:
            if remote_size is None or local_size == remote_size:
                if not (verify_hash and annex_key):
                    return local_path
                try:
                    return verify_file(local_path, annex_key, verify_hash=True)
                except OSError as e:
                    logger.warning("Refetching %s: %s", s3_path, e)
            elif stale_ok and _accept_stale_key(
                local_path,
                local_size,
                annex_key,
                s3_path,
                filesystem=filesystem,
                version_id=version_id,
            ):
                return local_path

    # Invalidate before replacing the data, so a failed transfer cannot certify
    # an object downloaded for another pin. get_file replaces data atomically.
    marker.unlink(missing_ok=True)
    _filesystem_get(
        filesystem=filesystem,
        s3path=s3_path,
        filepath=local_path,
        size=remote_size,
        **({"version_id": version_id} if version_id else {}),
        **({"annex_key": annex_key, "verify_hash": verify_hash} if annex_key else {}),
    )
    got = local_path.stat().st_size
    # get_file has checked a stale-key download against S3's own size.
    if remote_size is not None and got != remote_size and not stale_ok:
        raise OSError(
            f"Incomplete download for {s3_path} -> {local_path} "
            f"(expected {remote_size} bytes, got {got})."
        )

    if version_id:
        marker.write_text(json.dumps(identity))
    return local_path


def download_s3_directory(
    s3_path: str, local_dir: Path, *, filesystem: S3Client | None = None
) -> Path:
    """Download every object under an S3 prefix, keeping relative paths.

    For recordings stored as directories (CTF ``.ds``, MEF3 ``.mefd``, whose
    files sit in nested ``.timd/.segd`` folders). Needs ListBucket, which
    OpenNeuro allows and NEMAR's bucket does not.
    """
    filesystem = filesystem or get_s3_filesystem()
    bucket, key = _split_s3_uri(s3_path)
    prefix = key.rstrip("/") + "/"
    pairs = [
        (f"s3://{bucket}/{obj['Key']}", local_dir / obj["Key"][len(prefix) :])
        for obj in filesystem._iter_objects(f"s3://{bucket}/{prefix}")
    ]
    if not pairs:
        raise FileNotFoundError(s3_path)
    download_files(pairs, filesystem=filesystem, skip_existing=True)
    return local_dir


def download_files(
    files: Sequence[tuple[str, Path]] | Iterable[tuple[str, Path]],
    *,
    filesystem: S3Client | None = None,
    skip_existing: bool = True,
    skip_missing: bool = False,
    annex_keys: dict[Path, str] | None = None,
) -> list[Path]:
    """Download multiple S3 URIs to local destinations.

    Sizes come from the git-annex key a URI ends in, else from a HEAD, so
    NEMAR objects (no anonymous HEAD) are size-checked too.

    Parameters
    ----------
    files : iterable of (str, Path)
        Pairs of (S3 URI, local destination path).
    filesystem : S3Client | None
        Optional pre-created accessor to reuse across multiple downloads.
    skip_existing : bool
        If True, do not download files that already exist locally.
    skip_missing : bool
        If True, skip files that do not exist on S3 instead of raising.
    annex_keys : dict of Path to str | None
        git-annex key per destination, for URIs that do not end in one (an
        OpenNeuro twin's BIDS-named objects); it beats the HEAD size.

    """
    filesystem = filesystem or get_s3_filesystem()
    downloaded: list[Path] = []
    for uri, dest in files:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        key = (annex_keys or {}).get(dest) or _annex_key_of(uri)
        remote_size = annex_key_info(key)[0] if key else None
        if remote_size is None:
            remote_size = _remote_size(filesystem, uri)

        if skip_existing:
            try:
                local_size = dest.stat().st_size
            except FileNotFoundError:
                pass
            else:
                if remote_size is None or local_size == remote_size:
                    continue
                if _accept_stale_key(dest, local_size, key, uri, filesystem=filesystem):
                    continue

        # Never unlink a shared destination: get_file validates its own staging
        # file and replaces atomically, keeping concurrent readers safe.

        try:
            _filesystem_get(
                filesystem=filesystem,
                s3path=uri,
                filepath=dest,
                size=remote_size,
                **({"annex_key": key} if key else {}),
            )
        except FileNotFoundError:
            if skip_missing:
                logger.warning("File not found on S3, skipping: %s", uri)
                continue
            raise

        stale_ok = bool(key) and _annex_key_of(uri) != key
        if (
            remote_size is not None
            and dest.stat().st_size != remote_size
            and not stale_ok
        ):
            got = dest.stat().st_size
            raise OSError(
                f"Incomplete download for {uri} -> {dest} "
                f"(expected {remote_size} bytes, got {got})."
            )

        downloaded.append(dest)
    return downloaded


def _remote_size(
    filesystem: S3Client, s3path: str, *, version_id: str | None = None
) -> int | None:
    try:
        info = filesystem.info(
            s3path, **({"version_id": version_id} if version_id else {})
        )
    except Exception:
        return None
    # Not `size or Size`: a legitimate 0-byte object is falsy and must not be
    # treated as "unknown" (that would skip size verification for empty keys).
    size = info.get("size")
    if size is None:
        size = info.get("Size")
    if size is None:
        return None
    try:
        return int(size)
    except Exception:
        return None


class RichCallback:
    """Progress callback rendering with Rich; callable for ``boto3`` transfers."""

    def __init__(self, size: int | None = None, description: str = ""):
        self.progress = rich.progress.Progress(
            rich.progress.TextColumn("[bold blue]{task.description}"),
            rich.progress.BarColumn(bar_width=None),
            rich.progress.TaskProgressColumn(),
            "•",
            rich.progress.DownloadColumn(),
            "•",
            rich.progress.TransferSpeedColumn(),
            "•",
            rich.progress.TimeRemainingColumn(),
        )
        self.task_id = self.progress.add_task(description, total=size)
        self.progress.start()

    def set_size(self, size):
        self.progress.update(self.task_id, total=size)

    def relative_update(self, inc=1):
        self.progress.update(self.task_id, advance=inc)

    def __call__(self, inc):
        # boto3 invokes the callback with the number of bytes just transferred.
        self.relative_update(inc)

    def close(self):
        self.progress.stop()


class TqdmCallback:
    """Progress callback rendering with tqdm; callable for ``boto3`` transfers."""

    def __init__(self, size: int | None = None, description: str = ""):
        self.bar = tqdm(
            total=size,
            desc=description,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
            leave=True,
            mininterval=0.2,
            smoothing=0.1,
            miniters=1,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]",
        )

    def set_size(self, size):
        self.bar.total = size
        self.bar.refresh()

    def __call__(self, inc):
        self.bar.update(inc)

    def close(self):
        self.bar.close()


def _filesystem_get(
    filesystem: S3Client,
    s3path: str,
    filepath: Path,
    *,
    size: int | None = None,
    version_id: str | None = None,
    annex_key: str | None = None,
    verify_hash: bool = False,
) -> Path:
    """Perform the file download with a progress bar (Rich if a TTY, else tqdm).

    Parameters
    ----------
    filesystem : S3Client
        The accessor to use for the download.
    s3path : str
        The full S3 URI of the source file.
    filepath : pathlib.Path
        The local destination path.

    Returns
    -------
    pathlib.Path
        The local path to the downloaded file.

    """
    # Show the BIDS-named local destination, not the SHA-keyed S3 object,
    # so the progress bar matches what the user expects on disk.
    filename = Path(filepath).name
    description = f"Downloading {filename}"

    # Check if we should use Rich
    use_rich = False
    try:
        # Check if console is available and interactive-ish
        console = Console()
        if console.is_terminal:  # or some other heuristic if needed
            use_rich = True
    except Exception:
        pass

    if use_rich:
        callback = RichCallback(size=size, description=description)
    else:
        callback = TqdmCallback(size=size, description=description)

    try:
        # NEMAR denies anonymous ListBucket but allows GetObject. ``get_file``
        # issues a single GetObject — no HeadObject, no ListObjectsV2 — so it
        # works on those buckets. All callers here pass single object URIs; the
        # CTF directory case in base.py pre-expands via ``S3Client.ls`` into
        # per-file pairs.
        filesystem.get_file(
            s3path,
            str(filepath),
            callback=callback,
            **({"version_id": version_id} if version_id else {}),
            **(
                {"annex_key": annex_key, "verify_hash": verify_hash}
                if annex_key
                else {}
            ),
        )
    finally:
        # Ensure callback is closed properly (important for Rich to clean up display)
        if hasattr(callback, "close"):
            callback.close()

    return filepath


__all__ = [
    "annex_key_info",
    "download_s3_file",
    "download_files",
    "get_s3path",
    "get_s3_filesystem",
    "S3Client",
    "verify_file",
]

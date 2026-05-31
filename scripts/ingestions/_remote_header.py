"""Remote-header substrate: a budget-capped, block-cached, Range-backed reader
plus a per-source locate layer (RH2).

This is the shared transport for the "few bytes / zero bytes" header tiers
described in
``docs/superpowers/specs/2026-05-31-efficient-remote-header-reader-design.md``:

* :class:`RangeReader` — a seekable, file-like object backed by HTTP Range GETs.
  It caches whole blocks, counts every byte fetched against a hard budget, and
  refuses to buffer a full-object body when a server ignores ``Range`` and
  answers ``200``. That last guard (``RangeUnsupportedError``) is what keeps a
  multi-MB signal file from ever being streamed in full.
* :func:`locate` — resolves the free annex-key *size* (T1) for any source and
  the remote *URL* (T2) per source. OpenNeuro objects are addressable on S3 by
  their BIDS path; NEMAR's S3 is closed, so its URL is ``None`` (the caller
  drops to T3) while the size is still returned.

Design contract: nothing here raises on a *recoverable* failure. The fetcher
returning ``None`` (network/protocol error) is a miss, not a crash. The two
explicit exceptions — :class:`RangeUnsupportedError` and :class:`ByteBudgetExceededError`
— are control-flow signals the cascade catches to fall back to a cheaper tier.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable

from _file_utils import parse_annex_size
from _parser_utils import build_s3_url, fetch_bytes_from_s3, head_content_length

logger = logging.getLogger(__name__)

# A fetcher pulls a byte range: ``fetcher(url, start=, max_bytes=, timeout=)``.
Fetcher = Callable[..., "bytes | None"]

_DEFAULT_BUDGET = 128 * 1024
_DEFAULT_BLOCK = 64 * 1024


class RangeUnsupportedError(Exception):
    """The endpoint ignored ``Range`` and answered with the full object.

    Raised *before* a large body is buffered so a signal file is never
    streamed in full. The cascade catches this and drops the record to T3.
    """


class ByteBudgetExceededError(Exception):
    """A read would push total bytes fetched past the per-file budget.

    Raised instead of degrading into an unbounded read. The cascade catches
    this and drops the record to T3 rather than paying for a full download.
    """


class RangeReader:
    """A seekable, file-like reader backed by budgeted HTTP Range GETs.

    Parameters
    ----------
    url:
        The remote object URL (an OpenNeuro S3 URL in practice).
    budget:
        Hard cap on total bytes fetched across the lifetime of this reader.
        A fetch that would exceed it raises :class:`ByteBudgetExceededError`.
    block:
        Block granularity. Reads are served from whole-block fetches that are
        cached, so repeated small reads within one block cost one round-trip
        (this bounds h5py / MAT B-tree walks).
    fetcher:
        Injectable transport with the signature of
        :func:`_parser_utils.fetch_bytes_from_s3`. Defaults to that function;
        tests inject a fake that slices a known buffer.

    The object is usable two ways:

    * ``read(offset, length)`` — explicit ranged read of *length* bytes at
      *offset* (the tier readers use this).
    * file-like ``read(n)`` / ``seek`` / ``tell`` — so ``h5py.File(reader)``
      and a MAT reader can consume it. ``read()`` with no argument reads to
      EOF (bounded by the resolved object size and the budget).
    """

    def __init__(
        self,
        url: str,
        *,
        budget: int = _DEFAULT_BUDGET,
        block: int = _DEFAULT_BLOCK,
        fetcher: Fetcher | None = None,
    ) -> None:
        if block <= 0:
            raise ValueError("block must be positive")
        self.url = url
        self.budget = int(budget)
        self.block = int(block)
        self._fetcher: Fetcher = fetcher or fetch_bytes_from_s3
        self.bytes_fetched = 0
        self._cache: dict[int, bytes] = {}
        self._pos = 0
        self._size: int | None = None
        self._size_resolved = False

    # ── block transport ────────────────────────────────────────────────────

    def _fetch_block(self, block_index: int) -> bytes:
        """Fetch (and cache) one whole block; enforce budget + Range guard."""
        cached = self._cache.get(block_index)
        if cached is not None:
            return cached

        if self.bytes_fetched + self.block > self.budget:
            raise ByteBudgetExceededError(
                f"fetching block {block_index} ({self.block} B) would exceed "
                f"budget {self.budget} B (already fetched {self.bytes_fetched} B)"
            )

        start = block_index * self.block
        data = self._fetcher(self.url, start=start, max_bytes=self.block, timeout=30.0)
        if data is None:
            # Recoverable transport miss — empty block, no budget charge.
            return b""

        # Range-ignored detection: a correct partial response is at most
        # ``block`` bytes. Materially more means the server streamed the whole
        # object (HTTP 200) — abort before this large body is used anywhere.
        if len(data) > self.block:
            raise RangeUnsupportedError(
                f"server returned {len(data)} B for a {self.block} B Range "
                f"request on {self.url} (Range ignored)"
            )

        # Charge a full block even on a short (EOF) read: the round-trip is the
        # cost we budget against, and a short read also reveals the size.
        self.bytes_fetched += self.block
        if len(data) < self.block:
            # Short read ⇒ this block is the last; the object ends here.
            self._size = start + len(data)
            self._size_resolved = True
        self._cache[block_index] = data
        return data

    # ── explicit ranged read ────────────────────────────────────────────────

    def read(self, offset: int | None = None, length: int | None = None) -> bytes:
        """Read bytes.

        Two call shapes:

        * ``read(offset, length)`` — return *length* bytes starting at
          *offset* (explicit ranged read).
        * ``read(n)`` / ``read()`` — file-like: read *n* bytes (or to EOF when
          ``n`` is ``None``) from the current position, advancing it.
        """
        if length is None:
            return self._read_filelike(offset)
        return self._read_range(int(offset or 0), int(length))

    def _read_range(self, offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        end = offset + length
        first_block = offset // self.block
        last_block = (end - 1) // self.block
        chunks: list[bytes] = []
        for bi in range(first_block, last_block + 1):
            block = self._fetch_block(bi)
            block_start = bi * self.block
            lo = max(offset, block_start) - block_start
            hi = min(end, block_start + len(block)) - block_start
            if hi > lo:
                chunks.append(block[lo:hi])
            if len(block) < self.block:
                break  # hit EOF inside this block
        return b"".join(chunks)

    # ── file-like adapter (h5py.File(reader) / MAT reader) ──────────────────

    def _resolve_size(self) -> int | None:
        """Best-effort total object size (for SEEK_END / read-to-EOF).

        Prefers a ``content_length`` the fetcher exposes (test fakes do), then
        a HEAD ``Content-Length``. A short-read during normal reads also fills
        it. Returns ``None`` if undiscoverable (callers then bound by budget).
        """
        if self._size_resolved:
            return self._size
        probe = getattr(self._fetcher, "content_length", None)
        if isinstance(probe, int) and probe >= 0:
            self._size = probe
            self._size_resolved = True
            return self._size
        cl = head_content_length(self.url)
        if cl is not None:
            self._size = int(cl)
        self._size_resolved = True
        return self._size

    def _read_filelike(self, n: int | None) -> bytes:
        if n is not None and int(n) < 0:
            n = None
        if n is None:
            size = self._resolve_size()
            if size is None:
                # Unknown size: read forward block-by-block until a short read,
                # bounded by the budget (raises ByteBudgetExceededError past it).
                return self._read_to_eof_unbounded()
            length = max(0, size - self._pos)
        else:
            length = int(n)
        data = self._read_range(self._pos, length)
        self._pos += len(data)
        return data

    def _read_to_eof_unbounded(self) -> bytes:
        chunks: list[bytes] = []
        while True:
            block = self._fetch_block(self._pos // self.block)
            block_start = (self._pos // self.block) * self.block
            lo = self._pos - block_start
            piece = block[lo:]
            chunks.append(piece)
            self._pos += len(piece)
            if len(block) < self.block:
                break
        return b"".join(chunks)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = int(offset)
        elif whence == io.SEEK_CUR:
            self._pos += int(offset)
        elif whence == io.SEEK_END:
            size = self._resolve_size()
            if size is None:
                raise OSError("cannot SEEK_END: object size unknown")
            self._pos = int(size) + int(offset)
        else:  # pragma: no cover — defensive
            raise ValueError(f"invalid whence: {whence}")
        if self._pos < 0:
            self._pos = 0
        return self._pos

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False


def _size_from_record(record: dict) -> int | None:
    """Recover the free annex-key size from whatever field carries the key.

    Tries, in order: ``annex_keys[bids_relpath]``, then any annex_keys value,
    then ``storage`` / ``storage_base`` / ``bids_relpath`` text. Returns
    ``None`` when no field encodes a ``-s<size>--`` annex size.
    """
    relpath = record.get("bids_relpath")
    annex_keys = record.get("annex_keys")
    if isinstance(annex_keys, dict):
        candidates: list[str] = []
        if relpath and relpath in annex_keys:
            candidates.append(annex_keys[relpath])
        candidates.extend(v for k, v in annex_keys.items() if v not in candidates)
        for cand in candidates:
            if isinstance(cand, str):
                size = parse_annex_size(cand)
                if size is not None:
                    return size
    for field in ("storage", "storage_base", "bids_relpath"):
        value = record.get(field)
        if isinstance(value, str):
            size = parse_annex_size(value)
            if size is not None:
                return size
    return None


def _url_from_record(record: dict) -> str | None:
    """Derive the remote object URL for a record, per source.

    OpenNeuro (``ds…``) objects are addressable on S3 by their BIDS path →
    return a built S3 URL. NEMAR (``nm…``) S3 is closed → ``None`` (caller
    drops to T3). Any other / unknown dataset prefix → ``None``.
    """
    dataset = record.get("dataset")
    relpath = record.get("bids_relpath")
    if not isinstance(dataset, str) or not isinstance(relpath, str) or not relpath:
        return None
    if dataset.startswith("ds"):
        try:
            return build_s3_url(dataset, relpath, source="openneuro")
        except ValueError:  # pragma: no cover — defensive
            return None
    # NEMAR (nm…) S3 is closed; everything else is an unknown source.
    return None


def locate(record: dict) -> tuple[int | None, str | None]:
    """Resolve ``(size_from_annex_key, remote_url_or_None)`` for a record.

    * **size** — free, from the git-annex key (``-s<size>--``); works for any
      source (OpenNeuro and NEMAR alike).
    * **url** — OpenNeuro → an S3 URL built from ``dataset`` + ``bids_relpath``;
      NEMAR / unknown → ``None`` (S3 closed → the caller falls to T3).
    """
    return _size_from_record(record), _url_from_record(record)


__all__ = [
    "ByteBudgetExceededError",
    "RangeReader",
    "RangeUnsupportedError",
    "locate",
]

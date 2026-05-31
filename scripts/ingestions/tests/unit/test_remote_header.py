"""Unit tests for ``_remote_header`` — RangeReader + locate substrate (RH2).

These tests NEVER touch the network. ``RangeReader`` accepts an injected
``fetcher`` callable (same signature as
``_parser_utils.fetch_bytes_from_s3``) so the transport is fully faked. The
fakes are MODULE-LEVEL callables (a pre-commit hook forbids functions/classes
nested inside a ``def``), each asserting the Range it was asked for and
returning slices of a known bytes buffer.
"""

from __future__ import annotations

import pytest

from _remote_header import (
    ByteBudgetExceeded,
    RangeReader,
    RangeUnsupported,
    locate,
)

# ─── A known buffer the fakes slice (256 KiB of a repeating byte pattern) ───
_BUF = bytes((i * 7 + 3) & 0xFF for i in range(256 * 1024))


# ─── Module-level fake fetchers (NO nested defs — pre-commit forbids them) ──


class SliceFetcher:
    """Honest ranged fetcher: returns exactly ``_BUF[start:start+max_bytes]``.

    Records every ``(start, max_bytes)`` call so tests can assert the cache
    collapses repeated reads within one block into a single fetch.
    """

    def __init__(self, buf: bytes = _BUF) -> None:
        self.buf = buf
        self.calls: list[tuple[int, int]] = []
        # Lets RangeReader resolve total size for SEEK_END without a HEAD.
        self.content_length = len(buf)

    def __call__(
        self,
        url: str,
        *,
        start: int = 0,
        max_bytes: int = 262144,
        timeout: float = 30.0,
    ) -> bytes | None:
        self.calls.append((int(start), int(max_bytes)))
        return self.buf[int(start) : int(start) + int(max_bytes)]


class WholeBodyFetcher:
    """Dishonest fetcher: server ignored Range, streamed the WHOLE buffer.

    Returns far more than ``max_bytes`` so ``RangeReader`` must detect the
    non-ranged response and raise ``RangeUnsupported`` before buffering it.
    """

    def __init__(self, buf: bytes = _BUF) -> None:
        self.buf = buf
        self.calls: list[tuple[int, int]] = []

    def __call__(
        self,
        url: str,
        *,
        start: int = 0,
        max_bytes: int = 262144,
        timeout: float = 30.0,
    ) -> bytes | None:
        self.calls.append((int(start), int(max_bytes)))
        # Ignore Range entirely — hand back the full object.
        return self.buf


class NoneFetcher:
    """Fetcher that always fails (returns None), e.g. network/protocol error."""

    def __call__(
        self,
        url: str,
        *,
        start: int = 0,
        max_bytes: int = 262144,
        timeout: float = 30.0,
    ) -> bytes | None:
        return None


_URL = "https://s3.amazonaws.com/openneuro.org/ds000001/sub-01/eeg/x.edf"


# ─── read(offset, length) ──────────────────────────────────────────────────


def test_ranged_read_returns_requested_slice():
    fetcher = SliceFetcher()
    reader = RangeReader(_URL, fetcher=fetcher)
    data = reader.read(1000, 256)
    assert data == _BUF[1000:1256]


def test_ranged_read_accounts_bytes_fetched():
    fetcher = SliceFetcher()
    reader = RangeReader(_URL, block=64 * 1024, fetcher=fetcher)
    reader.read(0, 256)
    # The first read pulls exactly one block (64 KiB), not 256 bytes.
    assert reader.bytes_fetched == 64 * 1024
    assert fetcher.calls == [(0, 64 * 1024)]


def test_block_cache_collapses_two_reads_in_one_block():
    fetcher = SliceFetcher()
    reader = RangeReader(_URL, block=64 * 1024, fetcher=fetcher)
    a = reader.read(10, 8)
    b = reader.read(100, 8)
    assert a == _BUF[10:18]
    assert b == _BUF[100:108]
    # Both reads live inside block 0 -> exactly ONE fetcher call.
    assert len(fetcher.calls) == 1
    assert reader.bytes_fetched == 64 * 1024


def test_read_spanning_two_blocks_fetches_both():
    fetcher = SliceFetcher()
    reader = RangeReader(_URL, block=64 * 1024, fetcher=fetcher)
    # Straddle the 64 KiB boundary.
    start = 64 * 1024 - 4
    data = reader.read(start, 8)
    assert data == _BUF[start : start + 8]
    assert len(fetcher.calls) == 2
    assert reader.bytes_fetched == 2 * 64 * 1024


def test_repeated_block_read_does_not_refetch():
    fetcher = SliceFetcher()
    reader = RangeReader(_URL, block=64 * 1024, fetcher=fetcher)
    reader.read(0, 8)
    reader.read(0, 8)
    reader.read(20, 8)
    assert len(fetcher.calls) == 1


# ─── Range-ignored detection (200 on a large object) ───────────────────────


def test_whole_body_response_raises_range_unsupported():
    fetcher = WholeBodyFetcher()
    reader = RangeReader(_URL, block=64 * 1024, fetcher=fetcher)
    with pytest.raises(RangeUnsupported):
        reader.read(0, 8)


# ─── Byte-budget enforcement ───────────────────────────────────────────────


def test_over_budget_raises_byte_budget_exceeded():
    fetcher = SliceFetcher()
    # Budget smaller than two blocks -> the second block trips the budget.
    reader = RangeReader(_URL, budget=64 * 1024, block=64 * 1024, fetcher=fetcher)
    reader.read(0, 8)  # block 0: ok (64 KiB == budget)
    with pytest.raises(ByteBudgetExceeded):
        reader.read(64 * 1024 + 1, 8)  # block 1 would exceed budget


def test_budget_allows_reads_up_to_limit():
    fetcher = SliceFetcher()
    reader = RangeReader(_URL, budget=128 * 1024, block=64 * 1024, fetcher=fetcher)
    reader.read(0, 8)
    reader.read(64 * 1024, 8)  # second block, exactly at budget
    assert reader.bytes_fetched == 128 * 1024


# ─── File-like adapter (.read/.seek/.tell) for h5py.File(reader) ────────────


def test_file_like_read_seek_tell():
    fetcher = SliceFetcher()
    reader = RangeReader(_URL, block=64 * 1024, fetcher=fetcher)
    assert reader.tell() == 0
    first = reader.read(16)  # file-like: positional read of n bytes
    assert first == _BUF[0:16]
    assert reader.tell() == 16
    reader.seek(1000)
    assert reader.tell() == 1000
    assert reader.read(4) == _BUF[1000:1004]


def test_file_like_seek_whence_modes():
    fetcher = SliceFetcher()
    reader = RangeReader(_URL, block=64 * 1024, fetcher=fetcher)
    reader.seek(100)
    reader.seek(10, 1)  # SEEK_CUR
    assert reader.tell() == 110
    reader.seek(-5, 2)  # SEEK_END relative to known size (buffer length)
    assert reader.tell() == len(_BUF) - 5


def test_file_like_read_none_returns_all_remaining():
    fetcher = SliceFetcher()
    reader = RangeReader(_URL, block=64 * 1024, fetcher=fetcher)
    reader.seek(len(_BUF) - 10)
    rest = reader.read()  # read() with no arg -> to EOF
    assert rest == _BUF[-10:]


def test_failed_fetch_returns_empty_without_raising_budget():
    fetcher = NoneFetcher()
    reader = RangeReader(_URL, fetcher=fetcher)
    # A None from the transport is a recoverable miss -> empty bytes, no crash.
    assert reader.read(0, 8) == b""


# ─── locate(record) ────────────────────────────────────────────────────────


def _openneuro_record() -> dict:
    relpath = "sub-01/eeg/sub-01_task-rest_eeg.edf"
    return {
        "dataset": "ds004577",
        "bids_relpath": relpath,
        "storage_base": "s3://openneuro.org/ds004577",
        "annex_keys": {
            relpath: "MD5E-s2254720--0123456789abcdef0123456789abcdef.edf",
        },
    }


def _nemar_record() -> dict:
    relpath = "sub-01/eeg/sub-01_task-rest_eeg.edf"
    return {
        "dataset": "nm000123",
        "bids_relpath": relpath,
        "storage_base": "s3://nemar/nm000123",
        "annex_keys": {
            "sub-01/eeg/sub-01_task-rest_eeg.edf": (
                "SHA256E-s987654--"
                "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789.edf"
            ),
        },
    }


def test_locate_openneuro_returns_size_and_s3_url():
    size, url = locate(_openneuro_record())
    assert size == 2254720
    assert url is not None
    assert "openneuro.org/ds004577" in url
    assert url.endswith("sub-01/eeg/sub-01_task-rest_eeg.edf")


def test_locate_nemar_returns_size_but_no_url():
    size, url = locate(_nemar_record())
    assert size == 987654  # S3 closed for NEMAR, but the annex key still gives size
    assert url is None


def test_locate_size_from_relpath_when_annex_keys_missing():
    # Some records carry the size encoded in the relpath / pointer text and no
    # annex_keys dict — locate() must still recover the size.
    rec = {
        "dataset": "ds000001",
        "bids_relpath": "sub-01/eeg/x.edf",
        "annex_keys": None,
        # The annex key sits in a storage/relpath-ish field instead.
        "storage": "/annex/objects/.../MD5E-s4096--deadbeef.edf",
    }
    size, url = locate(rec)
    assert size == 4096
    assert url is not None  # OpenNeuro dataset -> URL still derivable


def test_locate_unknown_source_returns_none_url():
    rec = {
        "dataset": "zz999",  # neither ds nor nm
        "bids_relpath": "sub-01/eeg/x.edf",
        "annex_keys": {"sub-01/eeg/x.edf": "MD5E-s10--abc.edf"},
    }
    size, url = locate(rec)
    assert size == 10
    assert url is None


def test_locate_no_size_anywhere_returns_none_size():
    rec = {
        "dataset": "ds000001",
        "bids_relpath": "sub-01/eeg/x.edf",
        "annex_keys": {"sub-01/eeg/x.edf": "no-size-here.edf"},
    }
    size, url = locate(rec)
    assert size is None
    assert url is not None  # URL is still derivable even with no size

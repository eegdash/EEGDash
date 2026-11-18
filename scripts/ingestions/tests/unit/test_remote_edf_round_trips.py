"""The last-resort EDF header read must not cost one HTTP request per signal.

Nightly digests of nm000108, nm000237, nm000238, nm000281, nm000335 and on007358
exceeded the 1800 s per-dataset limit (22-23 Oct 2026). cProfile on Voyager
(prof1023): >99% of the time in RemoteHeaderStep._fill_edf, 12,625 range
requests for 100 header reads, because the RangeReader block was the 256-byte
main header, so 256 x n_signals of signal headers took n_signals round trips.
The main header and all signal headers now come in two requests.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import _metadata_cascade as mc
from _metadata_cascade import CascadeContext, CascadeResult, RemoteHeaderStep

N_SIGNALS = 125
_KEY = "SHA256E-s999999--" + "a" * 64 + ".edf"


def _edf_bytes(n_signals=N_SIGNALS, n_records=3600, spr=250):
    main = bytearray(b" " * 256)
    main[236:244] = str(n_records).ljust(8).encode()
    main[244:252] = b"1       "
    main[252:256] = str(n_signals).ljust(4).encode()
    sig = bytearray(b" " * (256 * n_signals))
    for i in range(n_signals):
        sig[16 * i : 16 * i + 16] = f"EEG {i:03d}".ljust(16).encode()
        off = 216 * n_signals + 8 * i
        sig[off : off + 8] = str(spr).ljust(8).encode()
    return bytes(main + sig)


class _CountingFetcher:
    def __init__(self, data):
        self.data, self.calls = data, 0

    def __call__(self, url, *, start=0, max_bytes=262144, **_kw):
        self.calls += 1
        return self.data[start : start + max_bytes]


def _ctx(tmp_path):
    root = tmp_path / "nm000108"
    rel = "sub-01/eeg/sub-01_task-x_eeg.edf"
    (root / rel).parent.mkdir(parents=True)
    os.symlink(f"../../.git/annex/objects/x/y/{_KEY}/{_KEY}", root / rel)
    return CascadeContext(SimpleNamespace(bidsdir=str(root)), str(root / rel))


def test_edf_header_read_takes_two_requests(monkeypatch, tmp_path):
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    monkeypatch.setattr(mc._remote_header, "locate", lambda _rec: (None, None))
    fetch = _CountingFetcher(_edf_bytes())
    monkeypatch.setattr(mc._remote_header, "fetch_bytes_from_s3", fetch)
    result = CascadeResult()
    RemoteHeaderStep(last_resort=True).fill(_ctx(tmp_path), result)
    assert result.ntimes == 3600 * 250
    assert result.sampling_frequency == 250.0
    assert result.nchans == N_SIGNALS
    assert fetch.calls <= 2

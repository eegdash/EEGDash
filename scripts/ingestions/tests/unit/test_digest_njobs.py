"""n_jobs thread-pool parallelism in ``_enumerate_via_bids``.

The parallel path overlaps the I/O-bound ``extract_record`` across threads but
finishes (montage/telemetry/collection) sequentially in file order — so its output
must be byte-identical to the serial path. These tests pin that invariant.
"""

from __future__ import annotations

from pathlib import Path

import _bids_digest as bd


class _FakeBids:
    def __init__(self, files):
        self._files = files

    def get_files(self):
        return self._files


def _fake_extract_record(
    bids_dataset, bids_file, dataset_id, source, digested_at, source_adapter=None
):
    # Deterministic per-file record; the value encodes the file so order is checkable.
    name = str(bids_file)
    return {"bids_relpath": name, "datatype": "eeg", "nchans": 1, "ntimes": len(name)}


_FILES = [f"sub-{i:02d}/eeg/sub-{i:02d}_task-x_eeg.edf" for i in range(20)]


def _patch(monkeypatch):
    monkeypatch.setattr(bd, "extract_record", _fake_extract_record)
    monkeypatch.setattr(bd, "is_neuro_data_file", lambda _p: True)
    monkeypatch.setattr(
        bd,
        "extract_dataset_metadata",
        lambda *a, **k: {"dataset_id": "ds", "source": "openneuro"},
    )
    monkeypatch.setattr(bd, "fingerprint_from_files", lambda *a, **k: "f" * 16)
    monkeypatch.setattr(bd, "_attach_montage_to_record", lambda *a, **k: [])


def _relpaths(n_jobs, monkeypatch):
    _patch(monkeypatch)
    res = bd._enumerate_via_bids(
        Path("/tmp/ds"),
        "ds",
        "openneuro",
        None,
        "2026-05-31T00:00:00Z",
        _FakeBids(list(_FILES)),
        n_jobs=n_jobs,
    )
    return [r["bids_relpath"] for r in res.records]


def test_njobs_parallel_output_identical_to_serial(monkeypatch):
    serial = _relpaths(1, monkeypatch)
    parallel = _relpaths(8, monkeypatch)
    assert serial == parallel
    assert serial == _FILES  # order preserved, none dropped


def test_njobs_one_is_serial_path(monkeypatch):
    # n_jobs=1 must produce the full set in order (the existing behaviour).
    assert _relpaths(1, monkeypatch) == _FILES

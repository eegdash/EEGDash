"""Unit tests for the public ``EEGDashDataset`` helpers.

These tests cover ``summary``, ``preview``, ``filter``,
``estimate_download_size``, and ``ensure_downloaded`` using synthetic /
mocked records and a tiny BIDS scratch dataset constructed via
``mne_bids``. No network access is performed.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, write_raw_bids

from eegdash.dataset.dataset import (
    EEGDashDataset,
    RecordingPreview,
)
from eegdash.dataset.exceptions import PreviewError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_records() -> list[dict[str, Any]]:
    """Three minimal records exercising filter/summary semantics."""
    return [
        {
            "dataset": "ds_demo",
            "bids_relpath": "sub-01/eeg/sub-01_ses-1_task-rest_run-1_eeg.set",
            "bidspath": "ds_demo/sub-01/eeg/sub-01_ses-1_task-rest_run-1_eeg.set",
            "storage": {"backend": "local", "base": "/tmp/ds_demo"},
            "subject": "01",
            "session": "1",
            "run": "1",
            "task": "rest",
            "modality": "eeg",
            "datatype": "eeg",
            "nchans": 64,
            "sampling_frequency": 500.0,
            "ntimes": 5000,
            "duration": 10.0,
            "size_bytes": 1_000_000,
            "entities_mne": {"subject": "01", "task": "rest"},
        },
        {
            "dataset": "ds_demo",
            "bids_relpath": "sub-02/eeg/sub-02_ses-1_task-rest_run-1_eeg.set",
            "bidspath": "ds_demo/sub-02/eeg/sub-02_ses-1_task-rest_run-1_eeg.set",
            "storage": {"backend": "local", "base": "/tmp/ds_demo"},
            "subject": "02",
            "session": "1",
            "run": "1",
            "task": "rest",
            "modality": "eeg",
            "datatype": "eeg",
            "nchans": 64,
            "sampling_frequency": 500.0,
            "ntimes": 7500,
            "duration": 15.0,
            "size_bytes": 2_500_000,
            "entities_mne": {"subject": "02", "task": "rest"},
        },
        {
            "dataset": "ds_demo",
            "bids_relpath": "sub-02/eeg/sub-02_ses-2_task-oddball_run-1_eeg.set",
            "bidspath": "ds_demo/sub-02/eeg/sub-02_ses-2_task-oddball_run-1_eeg.set",
            "storage": {"backend": "local", "base": "/tmp/ds_demo"},
            "subject": "02",
            "session": "2",
            "run": "1",
            "task": "oddball",
            "modality": "eeg",
            "datatype": "eeg",
            "nchans": 32,
            "sampling_frequency": 250.0,
            "ntimes": 5000,
            "duration": 20.0,
            "size_bytes": 1_500_000,
            "entities_mne": {"subject": "02", "task": "oddball"},
        },
    ]


@pytest.fixture
def synthetic_dataset(tmp_path: Path, synthetic_records) -> EEGDashDataset:
    """Build an EEGDashDataset on top of synthetic records.

    The ``records`` constructor path skips DB queries and only attempts
    minimal validation.
    """
    # Update storage base to a real tmp path so validate_record passes.
    for record in synthetic_records:
        record["storage"]["base"] = str(tmp_path / "ds_demo")
    (tmp_path / "ds_demo").mkdir(parents=True, exist_ok=True)
    return EEGDashDataset(
        cache_dir=tmp_path,
        dataset="ds_demo",
        records=synthetic_records,
        download=False,
    )


@pytest.fixture
def toy_bids_dataset(tmp_path: Path) -> Path:
    """Create a tiny but real BIDS dataset on disk to exercise preview()."""
    bids_root = tmp_path / "ds_preview"
    bids_root.mkdir()
    sfreq = 100
    info = mne.create_info(ch_names=["O1", "O2", "Cz"], sfreq=sfreq, ch_types="eeg")
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1e-6, size=(3, sfreq * 12))  # 12 seconds
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(
        mne.Annotations(onset=[1.0, 4.0], duration=[0.5, 0.25], description=["a", "b"])
    )
    bids_path = BIDSPath(
        subject="01",
        task="rest",
        datatype="eeg",
        root=bids_root,
        extension=".set",
    )
    write_raw_bids(
        raw,
        bids_path,
        verbose=False,
        overwrite=True,
        allow_preload=True,
        format="EEGLAB",
    )
    return bids_root


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


def test_summary_returns_dict_with_expected_keys(synthetic_dataset):
    report = synthetic_dataset.summary()
    expected_keys = {
        "n_records",
        "n_subjects",
        "n_tasks",
        "n_sessions",
        "n_runs",
        "modalities",
        "channel_counts",
        "sampling_rates",
        "total_duration_seconds",
        "cache_path",
        "estimated_size_bytes",
    }
    assert expected_keys == set(report.keys())


def test_summary_aggregates_counts(synthetic_dataset):
    report = synthetic_dataset.summary()
    assert report["n_records"] == 3
    assert report["n_subjects"] == 2  # subjects "01", "02"
    assert report["n_tasks"] == 2  # "rest", "oddball"
    assert report["n_sessions"] == 2  # "1", "2"
    assert report["n_runs"] == 1  # all are run "1"
    assert report["modalities"] == {"eeg"}
    assert isinstance(report["channel_counts"], Counter)
    assert report["channel_counts"][64] == 2
    assert report["channel_counts"][32] == 1
    assert isinstance(report["sampling_rates"], Counter)
    assert report["sampling_rates"][500.0] == 2
    assert report["sampling_rates"][250.0] == 1
    assert report["total_duration_seconds"] == pytest.approx(10.0 + 15.0 + 20.0)
    assert report["estimated_size_bytes"] == 1_000_000 + 2_500_000 + 1_500_000


def test_summary_cache_path_reflects_constructor(synthetic_dataset, tmp_path):
    report = synthetic_dataset.summary()
    assert report["cache_path"] == str(tmp_path)


def test_summary_verbose_prints(synthetic_dataset, capsys):
    report = synthetic_dataset.summary(verbose=True)
    out = capsys.readouterr().out
    assert "EEGDashDataset summary" in out
    assert "records:       3" in out
    # Returned report still matches non-verbose version.
    assert report["n_records"] == 3


def test_summary_handles_missing_size(tmp_path):
    records = [
        {
            "dataset": "ds_e",
            "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
            "bidspath": "ds_e/sub-01/eeg/sub-01_task-rest_eeg.set",
            "storage": {"backend": "local", "base": str(tmp_path / "ds_e")},
            "subject": "01",
            "task": "rest",
            "modality": "eeg",
            "nchans": 8,
            "sampling_frequency": 100.0,
            "ntimes": 1000,  # duration derivable from sfreq + ntimes
        }
    ]
    (tmp_path / "ds_e").mkdir()
    ds = EEGDashDataset(
        cache_dir=tmp_path, dataset="ds_e", records=records, download=False
    )
    report = ds.summary()
    assert report["estimated_size_bytes"] is None
    # duration recomputed from ntimes/sfreq when not provided.
    assert report["total_duration_seconds"] == pytest.approx(10.0)


def test_summary_empty_dataset(tmp_path, monkeypatch):
    # Build an empty EEGDashDataset by bypassing constructor checks.
    ds = EEGDashDataset.__new__(EEGDashDataset)
    ds.cache_dir = tmp_path
    ds.records = []
    ds.datasets = []
    ds.cumulative_sizes_cache = []
    report = ds.summary()
    assert report["n_records"] == 0
    assert report["n_subjects"] == 0
    assert report["modalities"] == set()
    assert report["channel_counts"] == Counter()
    assert report["sampling_rates"] == Counter()
    assert report["total_duration_seconds"] == 0.0
    assert report["estimated_size_bytes"] is None
    assert report["cache_path"] == str(tmp_path)


# ---------------------------------------------------------------------------
# filter()
# ---------------------------------------------------------------------------


def test_filter_single_kwarg(synthetic_dataset):
    sub = synthetic_dataset.filter(subject="02")
    assert isinstance(sub, EEGDashDataset)
    assert sub is not synthetic_dataset
    assert len(sub.datasets) == 2
    assert len(sub.records) == 2
    assert {r["subject"] for r in sub.records} == {"02"}
    # Original dataset is untouched.
    assert len(synthetic_dataset.datasets) == 3
    assert len(synthetic_dataset.records) == 3


def test_filter_list_value_or_semantics(synthetic_dataset):
    sub = synthetic_dataset.filter(task=["rest", "oddball"])
    assert len(sub.records) == 3


def test_filter_combines_kwargs_with_and(synthetic_dataset):
    sub = synthetic_dataset.filter(subject="02", task="rest")
    assert len(sub.records) == 1
    assert sub.records[0]["subject"] == "02"
    assert sub.records[0]["task"] == "rest"


def test_filter_no_match_returns_empty_dataset(synthetic_dataset):
    sub = synthetic_dataset.filter(subject="99")
    assert isinstance(sub, EEGDashDataset)
    assert len(sub.datasets) == 0
    assert len(sub.records) == 0
    # Empty dataset still summarizes cleanly.
    report = sub.summary()
    assert report["n_records"] == 0


def test_filter_unknown_field_raises(synthetic_dataset):
    with pytest.raises(ValueError, match="Unknown filter field"):
        synthetic_dataset.filter(banana="yes")


def test_filter_no_kwargs_returns_clone(synthetic_dataset):
    sub = synthetic_dataset.filter()
    assert sub is not synthetic_dataset
    assert len(sub.records) == len(synthetic_dataset.records)
    # Mutating the clone's records does not change the original.
    sub.records.pop()
    assert len(synthetic_dataset.records) == 3


def test_filter_uses_entities_fallback(tmp_path):
    records = [
        {
            "dataset": "ds_x",
            "bids_relpath": "sub-01/eeg/sub-01_task-A_eeg.set",
            "bidspath": "ds_x/sub-01/eeg/sub-01_task-A_eeg.set",
            "storage": {"backend": "local", "base": str(tmp_path / "ds_x")},
            "entities": {"subject": "01", "task": "A"},
            "modality": "eeg",
        },
        {
            "dataset": "ds_x",
            "bids_relpath": "sub-02/eeg/sub-02_task-B_eeg.set",
            "bidspath": "ds_x/sub-02/eeg/sub-02_task-B_eeg.set",
            "storage": {"backend": "local", "base": str(tmp_path / "ds_x")},
            "entities": {"subject": "02", "task": "B"},
            "modality": "eeg",
        },
    ]
    (tmp_path / "ds_x").mkdir()
    ds = EEGDashDataset(
        cache_dir=tmp_path, dataset="ds_x", records=records, download=False
    )
    sub = ds.filter(subject="02")
    assert len(sub.records) == 1
    assert sub.records[0]["entities"]["subject"] == "02"


# ---------------------------------------------------------------------------
# preview()
# ---------------------------------------------------------------------------


def test_preview_returns_recording_preview(toy_bids_dataset, tmp_path):
    ds = EEGDashDataset(
        cache_dir=toy_bids_dataset.parent,
        dataset="ds_preview",
        download=False,
    )
    preview = ds.preview(index=0)

    assert isinstance(preview, RecordingPreview)
    # raw is materialized.
    assert preview.raw is not None
    assert preview.raw.info["sfreq"] == 100
    # snippet covers first 5 seconds of all channels.
    assert preview.snippet.shape == (3, 5 * 100)
    # annotations come through as plain dicts.
    assert isinstance(preview.annotations, list)
    assert len(preview.annotations) == 2
    assert preview.annotations[0] == {
        "onset": pytest.approx(1.0),
        "duration": pytest.approx(0.5),
        "description": "a",
    }
    # metadata is at minimum a dict.
    assert isinstance(preview.metadata, dict)
    # plot helper is callable.
    assert callable(preview.plot)


def test_preview_negative_index_supported(toy_bids_dataset):
    ds = EEGDashDataset(
        cache_dir=toy_bids_dataset.parent,
        dataset="ds_preview",
        download=False,
    )
    preview = ds.preview(index=-1)
    assert preview.index == len(ds.datasets) - 1


def test_preview_index_out_of_range_raises(toy_bids_dataset):
    ds = EEGDashDataset(
        cache_dir=toy_bids_dataset.parent,
        dataset="ds_preview",
        download=False,
    )
    with pytest.raises(PreviewError, match="out of range"):
        ds.preview(index=99)


def test_preview_empty_dataset_raises(tmp_path):
    ds = EEGDashDataset.__new__(EEGDashDataset)
    ds.cache_dir = tmp_path
    ds.records = []
    ds.datasets = []
    ds.cumulative_sizes_cache = []
    with pytest.raises(PreviewError, match="empty"):
        ds.preview()


def test_preview_load_failure_wrapped(toy_bids_dataset):
    ds = EEGDashDataset(
        cache_dir=toy_bids_dataset.parent,
        dataset="ds_preview",
        download=False,
    )

    class _Boom(Exception):
        pass

    class _BadDataset:
        record = {"dataset": "ds_preview", "subject": "01"}

        @property
        def raw(self):
            raise _Boom("synthetic load failure")

    ds.datasets = [_BadDataset()]
    ds.records = [_BadDataset.record]
    with pytest.raises(PreviewError) as exc_info:
        ds.preview(index=0)
    assert isinstance(exc_info.value.__cause__, _Boom)
    assert exc_info.value.index == 0


# ---------------------------------------------------------------------------
# estimate_download_size()
# ---------------------------------------------------------------------------


def test_estimate_download_size_uses_record_sizes(synthetic_dataset):
    estimate = synthetic_dataset.estimate_download_size()
    assert estimate == {
        "bytes": 1_000_000 + 2_500_000 + 1_500_000,
        "n_records": 3,
        "source": "records",
    }


def test_estimate_download_size_falls_back_to_summary(tmp_path):
    records = [
        {
            "dataset": "ds_no_size",
            "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
            "bidspath": "ds_no_size/sub-01/eeg/sub-01_task-rest_eeg.set",
            "storage": {"backend": "local", "base": str(tmp_path / "ds_no_size")},
            "subject": "01",
            "task": "rest",
            "modality": "eeg",
        },
        {
            "dataset": "ds_no_size",
            "bids_relpath": "sub-02/eeg/sub-02_task-rest_eeg.set",
            "bidspath": "ds_no_size/sub-02/eeg/sub-02_task-rest_eeg.set",
            "storage": {"backend": "local", "base": str(tmp_path / "ds_no_size")},
            "subject": "02",
            "task": "rest",
            "modality": "eeg",
        },
    ]
    (tmp_path / "ds_no_size").mkdir()
    ds = EEGDashDataset(
        cache_dir=tmp_path,
        dataset="ds_no_size",
        records=records,
        download=False,
    )
    # No record carries a byte count -> fall back to dataset_doc.
    ds.dataset_doc = {"size_bytes": 12_345_678, "total_files": 2}
    estimate = ds.estimate_download_size()
    assert estimate == {
        "bytes": 12_345_678,
        "n_records": 2,
        "source": "summary",
    }


def test_estimate_download_size_unknown_when_no_hints(tmp_path):
    records = [
        {
            "dataset": "ds_unknown",
            "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
            "bidspath": "ds_unknown/sub-01/eeg/sub-01_task-rest_eeg.set",
            "storage": {"backend": "local", "base": str(tmp_path / "ds_unknown")},
            "subject": "01",
            "task": "rest",
            "modality": "eeg",
        }
    ]
    (tmp_path / "ds_unknown").mkdir()
    ds = EEGDashDataset(
        cache_dir=tmp_path,
        dataset="ds_unknown",
        records=records,
        download=False,
    )
    ds.dataset_doc = None
    estimate = ds.estimate_download_size()
    assert estimate == {"bytes": None, "n_records": 1, "source": "unknown"}


def test_estimate_download_size_picks_storage_size(tmp_path):
    records = [
        {
            "dataset": "ds_storage",
            "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
            "bidspath": "ds_storage/sub-01/eeg/sub-01_task-rest_eeg.set",
            "storage": {
                "backend": "local",
                "base": str(tmp_path / "ds_storage"),
                "size_bytes": 4096,
            },
            "subject": "01",
            "task": "rest",
            "modality": "eeg",
        }
    ]
    (tmp_path / "ds_storage").mkdir()
    ds = EEGDashDataset(
        cache_dir=tmp_path,
        dataset="ds_storage",
        records=records,
        download=False,
    )
    estimate = ds.estimate_download_size()
    assert estimate["bytes"] == 4096
    assert estimate["source"] == "records"


def test_estimate_download_size_empty_dataset(tmp_path):
    ds = EEGDashDataset.__new__(EEGDashDataset)
    ds.cache_dir = tmp_path
    ds.records = []
    ds.datasets = []
    ds.cumulative_sizes_cache = []
    ds.dataset_doc = None
    estimate = ds.estimate_download_size()
    assert estimate == {"bytes": None, "n_records": 0, "source": "unknown"}


# ---------------------------------------------------------------------------
# ensure_downloaded()
# ---------------------------------------------------------------------------


class _StubRecording:
    """Minimal stand-in for an ``EEGDashRaw`` instance."""

    def __init__(self, filecache, dep_paths=None):
        self.filecache = filecache
        self._dep_paths = list(dep_paths or [])


def _empty_eegdash_dataset(tmp_path: Path) -> EEGDashDataset:
    """Build an EEGDashDataset that bypasses the constructor's discovery."""
    ds = EEGDashDataset.__new__(EEGDashDataset)
    ds.cache_dir = tmp_path
    ds.records = []
    ds.datasets = []
    ds.cumulative_sizes_cache = []
    ds.dataset_doc = None
    ds.n_jobs = 1
    return ds


def test_ensure_downloaded_returns_summary(synthetic_dataset, monkeypatch):
    """download_all is invoked once and the summary reports counts/bytes."""
    calls: list[dict[str, Any]] = []

    def _fake_download_all(self, n_jobs=None):
        calls.append({"n_jobs": n_jobs})

    monkeypatch.setattr(EEGDashDataset, "download_all", _fake_download_all)

    summary = synthetic_dataset.ensure_downloaded(n_jobs=2)

    assert summary["n_records"] == 3
    assert summary["n_downloaded"] + summary["n_skipped"] == 3
    # Every record has size_bytes -> bytes_total is summed from records.
    assert summary["bytes_total"] == 1_000_000 + 2_500_000 + 1_500_000
    assert calls == [{"n_jobs": 2}]


def test_ensure_downloaded_counts_skipped_vs_downloaded(tmp_path, monkeypatch):
    ds = _empty_eegdash_dataset(tmp_path)
    monkeypatch.setattr(EEGDashDataset, "download_all", lambda self, n_jobs=None: None)

    cached_path = tmp_path / "cached.set"
    cached_path.write_bytes(b"x")
    missing_path = tmp_path / "missing.set"
    ds.datasets = [
        _StubRecording(filecache=cached_path),
        _StubRecording(filecache=missing_path),
        _StubRecording(filecache=None),  # also counted as missing
    ]

    summary = ds.ensure_downloaded()
    assert summary["n_records"] == 3
    assert summary["n_skipped"] == 1
    assert summary["n_downloaded"] == 2
    # No size hints anywhere -> bytes_total falls back to None.
    assert summary["bytes_total"] is None


def test_ensure_downloaded_empty_dataset_skips_download_all(tmp_path, monkeypatch):
    ds = _empty_eegdash_dataset(tmp_path)
    called = {"count": 0}

    def _fake_download_all(self, n_jobs=None):
        called["count"] += 1

    monkeypatch.setattr(EEGDashDataset, "download_all", _fake_download_all)

    summary = ds.ensure_downloaded()
    assert summary == {
        "n_records": 0,
        "n_downloaded": 0,
        "n_skipped": 0,
        "bytes_total": None,
    }
    # Skip the underlying download for empty datasets.
    assert called["count"] == 0


def test_ensure_downloaded_progress_falls_back_when_tqdm_missing(tmp_path, monkeypatch):
    ds = _empty_eegdash_dataset(tmp_path)
    cached_path = tmp_path / "cached.set"
    cached_path.write_bytes(b"x")
    ds.datasets = [_StubRecording(filecache=cached_path)]
    monkeypatch.setattr(EEGDashDataset, "download_all", lambda self, n_jobs=None: None)

    # Force the tqdm import to fail and ensure we still get a clean summary.
    import builtins

    real_import = builtins.__import__

    def _missing_tqdm(name, *args, **kwargs):
        if name.startswith("tqdm"):
            raise ImportError("synthetic missing tqdm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _missing_tqdm)
    summary = ds.ensure_downloaded(progress=True)
    assert summary["n_records"] == 1
    assert summary["n_skipped"] == 1
    assert summary["n_downloaded"] == 0


def test_ensure_downloaded_raises_when_download_all_missing(tmp_path):
    ds = _empty_eegdash_dataset(tmp_path)
    # Hide download_all on this instance only; the class attribute remains.
    ds.download_all = None  # type: ignore[assignment]
    with pytest.raises(NotImplementedError, match="download_all"):
        ds.ensure_downloaded()

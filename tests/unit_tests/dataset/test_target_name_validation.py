# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Validation of ``target_name`` at construction time (#21)."""

import pytest

from eegdash import EEGDashDataset


def _record(tmp_path, *, task="rest", age=25, dataset="dsTest"):
    record = {
        "dataset": dataset,
        "bids_relpath": f"sub-01/eeg/sub-01_task-{task}_eeg.set",
        "bidspath": f"{dataset}/sub-01/eeg/sub-01_task-{task}_eeg.set",
        "storage": {"backend": "local", "base": str(tmp_path)},
        "entities_mne": {"subject": "01", "task": task},
        "ntimes": 100,
        "sampling_frequency": 100,
    }
    if age is not None:
        record["age"] = age
    return record


def test_misspelled_target_name_raises(tmp_path):
    records = [_record(tmp_path, age=25)]
    with pytest.raises(ValueError, match="target_name='p-factor' has no usable"):
        EEGDashDataset(
            cache_dir=tmp_path,
            dataset="dsTest",
            records=records,
            download=False,
            target_name="p-factor",
        )


def test_all_missing_target_raises(tmp_path):
    records = [_record(tmp_path, age=None), _record(tmp_path, age=None)]
    with pytest.raises(ValueError, match="has no usable"):
        EEGDashDataset(
            cache_dir=tmp_path,
            dataset="dsTest",
            records=records,
            download=False,
            target_name="age",
        )


def test_valid_target_passes_and_populates_column(tmp_path):
    records = [_record(tmp_path, age=25), _record(tmp_path, age=30)]
    ds = EEGDashDataset(
        cache_dir=tmp_path,
        dataset="dsTest",
        records=records,
        download=False,
        target_name="age",
    )
    assert len(ds.datasets) == 2
    # target column is surfaced in each recording description
    assert all("age" in d.description.index for d in ds.datasets)


def test_no_target_name_is_noop(tmp_path):
    records = [_record(tmp_path, age=None)]
    ds = EEGDashDataset(
        cache_dir=tmp_path,
        dataset="dsTest",
        records=records,
        download=False,
    )
    assert len(ds.datasets) == 1

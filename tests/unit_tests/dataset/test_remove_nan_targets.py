# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""``remove_nan_targets`` filtering of NaN-target recordings (#22)."""

import logging

from eegdash import EEGDashDataset


def _record(tmp_path, *, sub="01", task="rest", age=25, dataset="dsTest"):
    record = {
        "dataset": dataset,
        "bids_relpath": f"sub-{sub}/eeg/sub-{sub}_task-{task}_eeg.set",
        "bidspath": f"{dataset}/sub-{sub}/eeg/sub-{sub}_task-{task}_eeg.set",
        "storage": {"backend": "local", "base": str(tmp_path)},
        "entities_mne": {"subject": sub, "task": task},
        "ntimes": 100,
        "sampling_frequency": 100,
    }
    if age is not None:
        record["age"] = age
    return record


def test_partial_nan_kept_by_default(tmp_path):
    records = [
        _record(tmp_path, sub="01", age=25),
        _record(tmp_path, sub="02", age=None),
    ]
    ds = EEGDashDataset(
        cache_dir=tmp_path,
        dataset="dsTest",
        records=records,
        download=False,
        target_name="age",
    )
    assert len(ds.datasets) == 2


def test_partial_nan_dropped_when_enabled(tmp_path, caplog):
    records = [
        _record(tmp_path, sub="01", age=25),
        _record(tmp_path, sub="02", age=None),
    ]
    with caplog.at_level(logging.WARNING):
        ds = EEGDashDataset(
            cache_dir=tmp_path,
            dataset="dsTest",
            records=records,
            download=False,
            target_name="age",
            remove_nan_targets=True,
        )
    assert len(ds.datasets) == 1
    assert any("Dropped 1/2" in r.message for r in caplog.records)


def test_no_drop_without_target(tmp_path):
    records = [_record(tmp_path, sub="01", age=None)]
    ds = EEGDashDataset(
        cache_dir=tmp_path,
        dataset="dsTest",
        records=records,
        download=False,
        remove_nan_targets=True,
    )
    assert len(ds.datasets) == 1

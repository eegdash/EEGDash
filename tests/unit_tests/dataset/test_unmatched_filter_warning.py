# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Warn when explicit filter values match no records (#141)."""

import logging

from eegdash import EEGDashDataset


def _record(tmp_path, *, sub="01", task="rest", dataset="dsTest"):
    return {
        "dataset": dataset,
        "bids_relpath": f"sub-{sub}/eeg/sub-{sub}_task-{task}_eeg.set",
        "bidspath": f"{dataset}/sub-{sub}/eeg/sub-{sub}_task-{task}_eeg.set",
        "storage": {"backend": "local", "base": str(tmp_path)},
        "entities_mne": {"subject": sub, "task": task},
        "ntimes": 100,
        "sampling_frequency": 100,
    }


def test_unmatched_value_in_list_warns(tmp_path, caplog):
    records = [_record(tmp_path, task="rest")]
    with caplog.at_level(logging.WARNING):
        EEGDashDataset(
            cache_dir=tmp_path,
            dataset="dsTest",
            records=records,
            download=False,
            task=["rest", "wrongtask"],
        )
    msgs = " ".join(r.message for r in caplog.records)
    assert "wrongtask" in msgs and "matched no records" in msgs


def test_fully_matched_filter_does_not_warn(tmp_path, caplog):
    records = [_record(tmp_path, task="rest")]
    with caplog.at_level(logging.WARNING):
        EEGDashDataset(
            cache_dir=tmp_path,
            dataset="dsTest",
            records=records,
            download=False,
            task="rest",
        )
    assert not any("matched no records" in r.message for r in caplog.records)


def test_all_matched_list_does_not_warn(tmp_path, caplog):
    records = [
        _record(tmp_path, sub="01", task="rest"),
        _record(tmp_path, sub="02", task="oddball"),
    ]
    with caplog.at_level(logging.WARNING):
        EEGDashDataset(
            cache_dir=tmp_path,
            dataset="dsTest",
            records=records,
            download=False,
            task=["rest", "oddball"],
        )
    assert not any("matched no records" in r.message for r in caplog.records)

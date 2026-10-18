"""Recordings whose sidecar says ``RecordingType: "epoched"`` (EEGDash/EEGDash#452 follow-up).

mne-bids 0.20 refuses them in ``read_raw_bids``. EEGDashRaw reads them with
``mne_bids.read_epochs_bids``, so the BIDS sidecars (channels.tsv types and bads,
events.tsv epochs) are kept, and concatenates the epochs into ``raw``.
"""

import json
import logging
import warnings
from unittest.mock import patch

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, read_epochs_bids, write_raw_bids

from eegdash.dataset.base import EEGDashRaw

SFREQ = 100.0
CH_NAMES = ["LA1", "LA2", "LA3", "G1", "G2"]
CH_TYPES = ["seeg", "seeg", "seeg", "ecog", "ecog"]
# Trial starts (s) and names; with EpochLength 1 s they cover 4 of the 10 s.
ONSETS = [1.0, 3.0, 5.0, 7.5]
TRIALS = ["go", "stop", "go", "stop"]


def _write_ieeg(root, *, epoched=True, epoch_length=1.0):
    """Write sub-01/ieeg/sub-01_task-trials_ieeg.vhdr; flag its sidecar as epoched."""
    info = mne.create_info(CH_NAMES, SFREQ, CH_TYPES)
    rng = np.random.default_rng(0)
    raw = mne.io.RawArray(rng.standard_normal((5, int(10 * SFREQ))) * 1e-5, info)
    raw.info["bads"] = ["LA2"]
    raw.set_annotations(mne.Annotations(ONSETS, [1.0] * 4, TRIALS))
    bids_path = BIDSPath(
        subject="01", task="trials", datatype="ieeg", suffix="ieeg", root=root
    )
    with warnings.catch_warnings():  # no electrodes.tsv / iEEGReference
        warnings.simplefilter("ignore")
        write_raw_bids(
            raw,
            bids_path,
            format="BrainVision",
            allow_preload=True,
            overwrite=True,
            verbose="ERROR",
        )
    sidecar = root / "sub-01" / "ieeg" / "sub-01_task-trials_ieeg.json"
    meta = json.loads(sidecar.read_text())
    if epoched:
        meta["RecordingType"] = "epoched"
        if epoch_length is not None:
            meta["EpochLength"] = epoch_length
    sidecar.write_text(json.dumps(meta))
    return bids_path.update(extension=".vhdr")


def _eegdashraw(root, dataset_id="ds-epoched"):
    relpath = "sub-01/ieeg/sub-01_task-trials_ieeg.vhdr"
    record = {
        "dataset": dataset_id,
        "bids_relpath": relpath,
        "datatype": "ieeg",
        "suffix": "ieeg",
        "extension": ".vhdr",
        "storage": {"backend": "local", "base": str(root), "raw_key": relpath},
        "entities": {"subject": "01", "task": "trials"},
        "entities_mne": {"subject": "01", "task": "trials"},
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        return EEGDashRaw(record, cache_dir=str(root.parent))


@pytest.fixture
def epoched_root(tmp_path):
    root = tmp_path / "ds-epoched"
    _write_ieeg(root)
    return root


def test_epoched_recording_keeps_bids_sidecars(epoched_root):
    """channels.tsv types and bads survive; one epoch per events.tsv row."""
    ds = _eegdashraw(epoched_root)
    raw = ds.raw

    assert isinstance(raw, mne.io.BaseRaw)
    assert raw.get_channel_types() == CH_TYPES  # the direct reader says "eeg"
    assert raw.info["bads"] == ["LA2"]
    assert raw.n_times == len(ONSETS) * int(SFREQ)  # not the 10 s file


def test_epoched_raw_matches_read_epochs_bids(epoched_root):
    """The raw is the read_epochs_bids epochs end to end, same info."""
    bids_path = BIDSPath(
        subject="01",
        task="trials",
        datatype="ieeg",
        suffix="ieeg",
        extension=".vhdr",
        root=epoched_root,
    )
    reference = read_epochs_bids(bids_path, verbose="ERROR")
    raw = _eegdashraw(epoched_root).raw

    data = raw.get_data()
    for i, epoch in enumerate(reference.get_data()):
        np.testing.assert_array_equal(data[:, i * 100 : (i + 1) * 100], epoch)
    assert [c["unit"] for c in raw.info["chs"]] == [
        c["unit"] for c in reference.info["chs"]
    ]


def test_epoched_raw_annotates_epochs_and_joins(epoched_root):
    raw = _eegdashraw(epoched_root).raw
    ann = raw.annotations

    trials = [a for a in ann if not a["description"].endswith("boundary")]
    assert [a["description"] for a in trials] == TRIALS
    np.testing.assert_allclose([a["onset"] for a in trials], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose([a["duration"] for a in trials], 1.0)
    for marker in ("BAD boundary", "EDGE boundary"):
        joins = [a["onset"] for a in ann if a["description"] == marker]
        np.testing.assert_allclose(joins, [1.0, 2.0, 3.0])


def test_epochs_property_returns_the_bids_epochs(epoched_root):
    ds = _eegdashraw(epoched_root)
    epochs = ds.epochs

    assert isinstance(epochs, mne.BaseEpochs)
    assert len(epochs) == len(ONSETS)
    assert set(epochs.event_id) == {"go", "stop"}
    assert epochs.get_channel_types() == CH_TYPES
    np.testing.assert_array_equal(epochs.events[:, 0], [100, 300, 500, 750])
    assert ds.epochs is epochs  # read once, then kept


def test_continuous_recording_is_unchanged(tmp_path):
    root = tmp_path / "ds-epoched"
    _write_ieeg(root, epoched=False)
    ds = _eegdashraw(root)

    with patch("mne_bids.read_epochs_bids") as read_epochs:
        raw = ds.raw
    read_epochs.assert_not_called()
    assert raw.n_times == int(10 * SFREQ)
    assert raw.get_channel_types() == CH_TYPES
    assert not any("boundary" in d for d in raw.annotations.description)
    assert ds.epochs is None


def test_read_epochs_bids_failure_falls_back_to_direct_reader(tmp_path, caplog):
    """No EpochLength: read_epochs_bids cannot slice the file; say why, read it raw."""
    root = tmp_path / "ds-epoched"
    _write_ieeg(root, epoch_length=None)
    ds = _eegdashraw(root)

    with caplog.at_level(logging.WARNING, logger="eegdash"):
        raw = ds.raw

    assert raw.n_times == int(10 * SFREQ)
    assert ds.epochs is None
    assert any(
        "read_epochs_bids failed" in r.getMessage() and "EpochLength" in r.getMessage()
        for r in caplog.records
    )


def test_epochs_to_raw_event_offset_and_layout():
    """With tmin < 0 the epoch annotation starts at its event, -tmin into the epoch."""
    from eegdash.dataset.epochs import epochs_to_raw

    info = mne.create_info(["a", "b"], 10.0, "eeg")
    data = np.arange(2 * 2 * 5, dtype=float).reshape(2, 2, 5)  # 2 epochs, 0.5 s
    events = np.array([[10, 0, 1], [40, 0, 2]])
    epochs = mne.EpochsArray(
        data, info, events, tmin=-0.2, event_id={"x": 1, "y": 2}, verbose="ERROR"
    )

    raw = epochs_to_raw(epochs)

    np.testing.assert_array_equal(raw.get_data(), np.hstack(list(data)))
    trials = [a for a in raw.annotations if a["description"] in ("x", "y")]
    np.testing.assert_allclose([a["onset"] for a in trials], [0.2, 0.7])
    np.testing.assert_allclose([a["duration"] for a in trials], 0.3)


def test_epochs_to_raw_rejects_empty_epochs():
    from eegdash.dataset.epochs import epochs_to_raw

    info = mne.create_info(["a"], 10.0, "eeg")
    epochs = mne.EpochsArray(np.zeros((1, 1, 5)), info, verbose="ERROR")
    epochs.drop([0], verbose="ERROR")
    with pytest.raises(ValueError, match="no epoch"):
        epochs_to_raw(epochs)


def test_epoched_recording_windows_one_per_epoch(epoched_root):
    """Braindecode trial windows come from the epoch annotations, not the joins."""
    from braindecode.datasets import BaseConcatDataset
    from braindecode.preprocessing import create_windows_from_events

    windows = create_windows_from_events(
        BaseConcatDataset([_eegdashraw(epoched_root)]),
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
    )

    assert len(windows) == len(ONSETS)
    targets = [windows[i][1] for i in range(len(windows))]
    assert targets[0] == targets[2] != targets[1] == targets[3]  # go, stop, go, stop

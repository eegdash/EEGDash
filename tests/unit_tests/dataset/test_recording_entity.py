"""Keep EMG2Pose recording-side entities when constructing MNE-BIDS paths."""

import pytest

from eegdash.dataset.base import EEGDashRaw


@pytest.mark.parametrize("indexed", [False, True])
def test_recording_side_is_preserved(tmp_path, indexed):
    relative = "sub-06/ses-01/emg/sub-06_ses-01_task-emg_run-09_recording-right_emg.bdf"
    entities = {"subject": "06", "session": "01", "task": "emg", "run": "09"}
    if indexed:
        entities["recording"] = "right"
    recording = EEGDashRaw(
        record={
            "dataset": "nm000281",
            "bids_relpath": relative,
            "bidspath": f"nm000281/{relative}",
            "datatype": "emg",
            "suffix": "emg",
            "extension": ".bdf",
            "entities_mne": entities,
            "storage": {"backend": "local", "base": str(tmp_path)},
        },
        cache_dir=str(tmp_path),
    )
    assert recording.bidspath.recording == "right"
    assert recording.bidspath.fpath.name == relative.rsplit("/", 1)[-1]

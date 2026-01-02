from datetime import datetime, timezone

import mne
import numpy as np
import pandas as pd
import pytest

from eegdash.hbn.preprocessing import hbn_ec_ec_reannotation
from eegdash.hbn.windows import build_trial_table


@pytest.fixture
def mock_raw():
    # Create valid MNE Raw object
    sfreq = 100
    info = mne.create_info(ch_names=["Fz"], sfreq=sfreq, ch_types=["eeg"])
    data = np.zeros((1, 100 * 60))  # 60 seconds
    raw = mne.io.RawArray(data, info)

    # Add annotations
    # instructed_toCloseEyes at 10s
    # instructed_toOpenEyes at 40s
    raw.set_meas_date(datetime(2020, 1, 1, tzinfo=timezone.utc))

    my_annot = mne.Annotations(
        onset=[10, 40],
        duration=[0, 0],
        description=["instructed_toCloseEyes", "instructed_toOpenEyes"],
    )
    raw.set_annotations(my_annot)
    return raw


def test_hbn_ec_ec_reannotation(mock_raw):
    # Apply reannotation
    preproc = hbn_ec_ec_reannotation()
    raw_new = preproc.transform(mock_raw)

    events, event_id = mne.events_from_annotations(raw_new)

    # Expect:
    # eyes_closed (id 1): 15s to 29s after 10s -> 25s, 27s ...
    # Wait, code says: start_times = event[0] + np.arange(15, 29, 2) * sfreq
    # original event sample: 10 * 100 = 1000.
    # offsets: 1500, 1700, 1900, 2100, 2300, 2500, 2700. (numpy arange exclude end 29)
    # 7 events.

    # eyes_open (id 2): 5s to 19s after 40s -> 45s, 47s...
    # offsets: 500, ... 1700 (relative).
    # 4000 + 500 = 4500.
    # range(5, 19, 2) -> 5, 7, 9, 11, 13, 15, 17. (7 events)

    # Check counts
    assert "eyes_closed" in event_id
    assert "eyes_open" in event_id

    closed_id = event_id["eyes_closed"]
    open_id = event_id["eyes_open"]

    n_closed = np.sum(events[:, 2] == closed_id)
    n_open = np.sum(events[:, 2] == open_id)

    assert n_closed == 7  # arange(15, 29, 2) -> 7 items
    assert n_open == 7  # arange(5, 19, 2) -> 7 items


def test_build_trial_table():
    data = [
        {"onset": 5, "duration": 0, "value": "contrastTrial_start"},
        {"onset": 6, "duration": 0, "value": "right_target"},
        {"onset": 0, "duration": 0, "value": "contrastTrial_start"},
        {"onset": 1, "duration": 0, "value": "left_target"},
        {
            "onset": 1.5,
            "duration": 0,
            "value": "left_buttonPress",
            "feedback": "smiley_face",
        },
        {"onset": 10, "duration": 0, "value": "end_experiment"},
    ]
    df = pd.DataFrame(data)

    table = build_trial_table(df)

    assert len(table) == 2
    first = table.iloc[0]
    assert first["trial_start_onset"] == 0
    assert first["trial_stop_onset"] == 5
    assert first["stimulus_onset"] == 1
    assert first["response_onset"] == 1.5
    assert first["rt_from_stimulus"] == pytest.approx(0.5)
    assert first["rt_from_trialstart"] == pytest.approx(1.5)
    assert first["response_type"] == "left_buttonPress"
    assert bool(first["correct"]) is True

    second = table.iloc[1]
    assert second["trial_start_onset"] == 5
    assert second["trial_stop_onset"] == 10
    assert second["stimulus_onset"] == 6
    assert pd.isna(second["response_onset"])
    assert pd.isna(second["rt_from_stimulus"])
    assert pd.isna(second["rt_from_trialstart"])
    assert second["response_type"] is None
    assert second["correct"] is None


def test_build_trial_table_without_feedback_column():
    data = [
        {"onset": 0, "duration": 0, "value": "contrastTrial_start"},
        {"onset": 1, "duration": 0, "value": "right_target"},
        {"onset": 1.5, "duration": 0, "value": "right_buttonPress"},
        {"onset": 4, "duration": 0, "value": "end_experiment"},
    ]
    df = pd.DataFrame(data)

    table = build_trial_table(df)

    assert len(table) == 1
    row = table.iloc[0]
    assert row["trial_start_onset"] == 0
    assert row["trial_stop_onset"] == 4
    assert row["stimulus_onset"] == 1
    assert row["response_onset"] == 1.5
    assert row["rt_from_stimulus"] == pytest.approx(0.5)
    assert row["rt_from_trialstart"] == pytest.approx(1.5)
    assert row["response_type"] == "right_buttonPress"
    assert row["correct"] is None


def test_annotate_trials_with_target(tmp_path):
    # Setup dummy BIDS structure
    sub = "01"
    ses = "01"
    task = "test"
    run = "01"

    bids_root = tmp_path / "bids_dataset"
    bids_root.mkdir()

    # Create events file
    events_dir = bids_root / f"sub-{sub}" / f"ses-{ses}" / "eeg"
    events_dir.mkdir(parents=True)
    events_fname = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_events.tsv"

    # Minimal events for a valid trial
    events_data = {
        "onset": [1.0, 2.0, 3.0, 4.0],
        "duration": [0, 0, 0, 0],
        "value": [
            "contrastTrial_start",
            "left_target",
            "left_buttonPress",
            "contrastTrial_start",  # Next trial to end previous
        ],
    }
    pd.DataFrame(events_data).to_csv(events_dir / events_fname, sep="\t", index=False)

    # Mock Raw
    raw_fname = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_eeg.vhdr"
    raw_path = events_dir / raw_fname
    # We don't need real EEG file, just the path matching for get_bids_path

    # Create a raw object with matching filename
    info = mne.create_info(["ch1"], 1000.0, ["eeg"])
    raw = mne.io.RawArray(np.zeros((1, 5000)), info)

    # Mock filenames attribute (hacky but works for the test)
    raw._filenames = [str(raw_path)]

    # Run annotation
    # We need to compute what rt_from_stimulus would be:
    # stim at 2.0, resp at 3.0 -> rt = 1.0
    from eegdash.hbn.windows import annotate_trials_with_target

    raw = annotate_trials_with_target(raw, target_field="rt_from_stimulus")

    assert len(raw.annotations) == 1
    assert raw.annotations.description[0] == "contrast_trial_start"
    assert raw.annotations.onset[0] == 1.0

    # Check extras
    extras = raw.annotations.extras[0]
    assert extras["target"] == 1.0
    assert extras["stimulus_onset"] == 2.0
    assert extras["response_onset"] == 3.0


def test_add_aux_anchors():
    from eegdash.hbn.windows import add_aux_anchors

    info = mne.create_info(["ch1"], 1000.0, ["eeg"])
    raw = mne.io.RawArray(np.zeros((1, 5000)), info)

    # Create an annotation with extras
    extras = {"stimulus_onset": 2.0, "response_onset": 3.0, "rt_from_stimulus": 1.0}

    ann = mne.Annotations(
        onset=[1.0],
        duration=[2.0],
        description=["contrast_trial_start"],
        extras=[extras],
    )
    raw.set_annotations(ann)

    raw = add_aux_anchors(raw)

    # Should have 3 annotations now: start, stim_anchor, resp_anchor
    assert len(raw.annotations) == 3

    descs = raw.annotations.description
    onsets = raw.annotations.onset

    assert "contrast_trial_start" in descs
    assert "stimulus_anchor" in descs
    assert "response_anchor" in descs

    # Check times
    stim_idx = np.where(descs == "stimulus_anchor")[0][0]
    assert onsets[stim_idx] == 2.0

    resp_idx = np.where(descs == "response_anchor")[0][0]
    assert onsets[resp_idx] == 3.0


def test_add_extras_columns():
    from braindecode.datasets import BaseConcatDataset
    from braindecode.datasets.base import BaseDataset
    from eegdash.hbn.windows import add_extras_columns

    # 1. Create BaseConcatDataset with one recording having annotations+extras
    info = mne.create_info(["ch1"], 100.0, ["eeg"])
    raw = mne.io.RawArray(np.zeros((1, 1000)), info)

    extras = {"target": 0.5, "correct": 1}
    ann = mne.Annotations(
        onset=[1.0],
        duration=[1.0],
        description=["contrast_trial_start"],
        extras=[extras],
    )
    raw.set_annotations(ann)

    base_ds = BaseDataset(raw, description=None)
    original_concat_ds = BaseConcatDataset([base_ds])

    # 2. Create WindowsDataset (mocked)
    # We need a metadata df
    metadata = pd.DataFrame(
        {
            "i_window_in_trial": [0, 1],
            "i_start_in_trial": [0, 100],
            "i_stop_in_trial": [100, 200],
            "target": [-1, -1],  # placeholder
        }
    )

    # We can mock WindowsDataset by just attaching metadata to a BaseDataset (duck typing)
    win_ds = BaseDataset(raw, description=None)
    win_ds.metadata = metadata

    windows_concat_ds = BaseConcatDataset([win_ds])

    # 3. Run add_extras_columns
    windows_concat_ds = add_extras_columns(windows_concat_ds, original_concat_ds)

    # 4. Verify metadata
    md = windows_concat_ds.datasets[0].metadata

    assert "target" in md.columns
    # Both windows belong to the 0-th trial (cumulatively)
    # The function maps windows to trials based on i_window_in_trial==0

    assert md.iloc[0]["target"] == 0.5
    assert md.iloc[1]["target"] == 0.5
    assert md.iloc[0]["correct"] == 1

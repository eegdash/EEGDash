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

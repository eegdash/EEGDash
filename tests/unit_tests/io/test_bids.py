"""eegdash.io.bids: sidecars MNE-BIDS 0.20's read_raw_bids refuses, read under tolerant().

One BrainVision iEEG file per case; only ieeg.json / events.tsv are broken the way
NEMAR deposits break them. ``mne_bids.read_raw_bids`` fails on each, ours reads it.
"""

import json
import warnings
from datetime import UTC, datetime
from unittest.mock import patch

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids

from eegdash.dataset.base import EEGDashRaw
from eegdash.io import tolerant

SFREQ = 100.0
CH_TYPES = ["seeg", "seeg", "ecog"]
RELPATH = "sub-01/ieeg/sub-01_task-trials_ieeg.vhdr"


def _write(root, *, sidecar=None, events=None, meas_date=None):
    info = mne.create_info(["LA1", "LA2", "G1"], SFREQ, CH_TYPES)
    data = np.random.default_rng(0).standard_normal((3, int(10 * SFREQ))) * 1e-5
    raw = mne.io.RawArray(data, info)
    raw.set_meas_date(meas_date)
    raw.info["bads"] = ["LA2"]
    bp = BIDSPath(
        subject="01", task="trials", datatype="ieeg", suffix="ieeg", root=root
    )
    with warnings.catch_warnings():  # no electrodes.tsv / iEEGReference
        warnings.simplefilter("ignore")
        write_raw_bids(
            raw, bp, format="BrainVision", allow_preload=True, verbose="ERROR"
        )
    ieeg = root / "sub-01" / "ieeg"
    meta_path = ieeg / "sub-01_task-trials_ieeg.json"
    meta_path.write_text(
        json.dumps({**json.loads(meta_path.read_text()), **(sidecar or {})})
    )
    if events:
        (ieeg / "sub-01_task-trials_events.tsv").write_text(events)
    return bp.update(extension=".vhdr")


# nm000403/485/487-style: a stimulus row at every trial start, no EpochLength,
# and one trial past the end. read_epochs_bids fails on each of these.
EPOCHED_EVENTS = (
    "onset\tduration\ttrial_type\tvalue\tsample\n"
    "1.0\t2.0\ttrial\t1\t100\n"
    "1.0\t0.0\tstim\tS  3\t100\n"
    "4.0\t2.0\ttrial\t1\t400\n"
    "9.5\t2.0\ttrial\t1\t950\n"
)


def test_epoched_continuous_container_reads_as_raw(tmp_path):
    bp = _write(
        tmp_path / "ds", sidecar={"RecordingType": "epoched"}, events=EPOCHED_EVENTS
    )
    with pytest.raises(RuntimeError, match='RecordingType is "epoched"'):
        read_raw_bids(bp, verbose="ERROR")

    with tolerant():
        raw = read_raw_bids(bp, verbose="ERROR")

    assert raw.n_times == int(10 * SFREQ)  # the whole file, not epochs
    assert raw.get_channel_types() == CH_TYPES  # channels.tsv applied
    assert raw.info["bads"] == ["LA2"]
    assert sorted(set(raw.annotations.description)) == ["stim", "trial"]


def test_reserved_events_column_is_renamed(tmp_path):
    events = "onset\tduration\ttrial_type\tdescription\n1.0\t0.5\tgo\tleft hand\n"
    bp = _write(tmp_path / "ds", events=events)  # nm000536
    with pytest.raises(ValueError, match="reserved"):
        read_raw_bids(bp, verbose="ERROR")

    with tolerant():
        raw = read_raw_bids(bp, verbose="ERROR")

    assert raw.annotations.extras[0]["events_description"] == "left hand"


def test_negative_duration_becomes_zero(tmp_path):
    events = "onset\tduration\ttrial_type\n2.0\t0.5\tgo\n9.0\t-1.0\tphoneme\n"
    bp = _write(tmp_path / "ds", events=events)  # on004703
    with pytest.raises(AssertionError):
        read_raw_bids(bp, verbose="ERROR")

    with tolerant():
        raw = read_raw_bids(bp, verbose="ERROR")

    assert list(raw.annotations.duration) == [0.5, 0.0]


def test_age_that_is_a_year_leaves_birthday_unset(tmp_path):
    meas_date = datetime(2019, 1, 1, tzinfo=UTC)
    bp = _write(tmp_path / "ds", meas_date=meas_date)
    participants = tmp_path / "ds" / "participants.tsv"
    participants.write_text(  # nm000149: the age column holds the year 2018
        "participant_id\tage\tsex\thand\nsub-01\t2018\tF\tR\n"
    )
    with pytest.raises(OverflowError, match="date value out of range"):
        read_raw_bids(bp, verbose="ERROR")

    with tolerant():
        raw = read_raw_bids(bp, verbose="ERROR")

    assert raw.info["meas_date"] == meas_date
    assert "birthday" not in raw.info["subject_info"]  # never a made-up date
    assert raw.info["subject_info"]["sex"] == 2  # the rest of the row is kept
    assert raw.info["subject_info"]["hand"] == 1


def test_eegdashraw_loads_epoched_recording(tmp_path):
    root = tmp_path / "ds-epoched"
    _write(root, sidecar={"RecordingType": "epoched"}, events=EPOCHED_EVENTS)
    record = {
        "dataset": "ds-epoched",
        "bids_relpath": RELPATH,
        "datatype": "ieeg",
        "suffix": "ieeg",
        "extension": ".vhdr",
        "storage": {"backend": "local", "base": str(root), "raw_key": RELPATH},
        "entities": {"subject": "01", "task": "trials"},
        "entities_mne": {"subject": "01", "task": "trials"},
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        raw = EEGDashRaw(record, cache_dir=str(tmp_path)).raw

    assert raw.n_times == int(10 * SFREQ)
    assert raw.get_channel_types() == CH_TYPES
    assert "trial" in raw.annotations.description

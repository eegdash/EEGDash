"""eegdash.io.mef: MEF3 sessions that would crash pymef are refused first.

on004624 sub-c04 Longterm Passive run-10: channel CH20's segment has only its
``.tmet``; pymef 1.4.8 segfaults on it (SIGSEGV killed the read-check job twice).
pymef is replaced by a stub there, so the test can never crash the test process.
Channels of different lengths (on004624, 15 sessions) are written with pymef.
"""

import logging
import warnings

import mne
import numpy as np
import pymef.mef_session
import pytest
from mne_bids import BIDSPath, read_raw_bids

from eegdash.io import get_reader, tolerant


class _Reached(Exception):
    pass


@pytest.fixture
def pymef_stub(monkeypatch):
    def stub(*_args, **_kwargs):
        raise _Reached("pymef opened the session")

    monkeypatch.setattr(pymef.mef_session, "MefSession", stub)


def _session(tmp_path, ch20_files=(".tmet", ".tidx", ".tdat")):
    root = tmp_path / "sub-c04_ses-Longterm_task-Passive_acq-rec_run-10_ieeg.mefd"
    for ch, files in (("CH0", (".tmet", ".tidx", ".tdat")), ("CH20", ch20_files)):
        seg = root / f"{ch}.timd" / f"{ch}-000000.segd"
        seg.mkdir(parents=True)
        for ext in files:
            (seg / f"{ch}-000000{ext}").write_bytes(b"x")
    return root


def test_segment_without_data_is_refused_before_pymef(tmp_path, pymef_stub):
    root = _session(tmp_path, ch20_files=(".tmet",))
    with (
        tolerant(),
        pytest.raises(ValueError, match="CH20-000000.segd lacks .tidx, .tdat"),
    ):
        mne.io.read_raw_mef(root)
    with pytest.raises(ValueError, match="Incomplete MEF3 session"):
        get_reader(".mefd")(root)


def test_complete_session_reaches_pymef(tmp_path, pymef_stub):
    with tolerant(), pytest.raises(_Reached):
        mne.io.read_raw_mef(_session(tmp_path))


SFREQ, START = 100.0, 1_600_000_000_000_000  # Hz, uUTC


def _write_session(path, lengths, starts, gap=None, gap_us=1_000_000):
    """Unencrypted channels written with pymef (no stub); ``gap=(ch, sample)`` splits
    that channel there into a second segment that starts ``gap_us`` late.
    """
    ms = pymef.mef_session.MefSession(str(path), None, new_session=True)
    data = np.arange(max(lengths) * len(lengths), dtype="int32").reshape(
        len(lengths), -1
    )
    for k, (n, start) in enumerate(zip(lengths, starts)):
        cut = gap[1] if gap and gap[0] == k else n
        for seg, (i0, i1, t0) in enumerate([(0, cut, start), (cut, n, start + gap_us)]):
            if i1 > i0:
                t0 += int(i0 / SFREQ * 1e6)
                end = t0 + int((i1 - i0) / SFREQ * 1e6)
                section_2 = {
                    "sampling_frequency": SFREQ,
                    "units_conversion_factor": 1.0,
                }
                section_2.update(start_sample=i0, recording_duration=end - t0)
                ms.write_mef_ts_segment_metadata(
                    f"CH{k}", seg, None, None, t0, end, section_2, {}
                )
                ms.write_mef_ts_segment_data(
                    f"CH{k}", seg, None, None, 100, data[k, i0:i1]
                )
    return path, data


def test_trailing_channel_tail_is_cropped_to_the_common_length(tmp_path, caplog):
    # Same start, CH1 200 samples longer.
    root, data = _write_session(
        tmp_path / "run-6_ieeg.mefd", (1000, 1200, 1000), [START] * 3
    )
    with pytest.raises(ValueError, match="inconsistent n_samples"):
        mne.io.read_raw_mef(root)

    with tolerant(), caplog.at_level(logging.WARNING, logger="eegdash"):
        raw = mne.io.read_raw_mef(root, preload=True, verbose="ERROR")
    assert "dropped at the start: none; at the end: CH1 200." in caplog.text
    assert raw.n_times == 1000  # never padded
    cals = np.array([ch["cal"] for ch in raw.info["chs"]])
    np.testing.assert_allclose(raw.get_data() / cals[:, None], data[:, :1000])


def test_channels_that_start_apart_are_read_on_their_common_window(tmp_path, caplog):
    # on004624 sub-c02 Functional Sensorimotor run-3: 32 channels start 0.2 s
    # (200 samples) after the other 7 and all end together. Here CH1 starts 2 s later and
    # CH2 ends 1 s earlier: the window all channels have is [2 s, 11 s].
    starts = [START, START + 2_000_000, START]
    root, data = _write_session(
        tmp_path / "run-7_ieeg.mefd", (1200, 1000, 1100), starts
    )
    with tolerant(), caplog.at_level(logging.WARNING, logger="eegdash"):
        raw = mne.io.read_raw_mef(root, preload=True, verbose="ERROR")
    assert "dropped at the start: CH0 200, CH2 200" in caplog.text
    assert "at the end: CH0 100, CH1 100" in caplog.text
    assert raw.n_times == 900  # never padded
    assert raw.info["meas_date"].timestamp() == (START + 2_000_000) / 1e6
    cals = np.array([ch["cal"] for ch in raw.info["chs"]])
    want = [data[0, 200:1100], data[1, :900], data[2, 200:1100]]  # same times
    np.testing.assert_allclose(raw.get_data() / cals[:, None], want)


def _bids_session(tmp_path, *args, **kwargs):
    """A BIDS tree with one session (``_write_session`` args), events.tsv onset 5 s."""
    root = tmp_path / "bids"
    ieeg = root / "sub-01" / "ieeg"
    ieeg.mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        '{"Name": "t", "BIDSVersion": "1.9"}'
    )
    _, data = _write_session(ieeg / "sub-01_task-x_ieeg.mefd", *args, **kwargs)
    (ieeg / "sub-01_task-x_events.tsv").write_text(
        "onset\tduration\ttrial_type\n5.0\t1.0\tgo\n"
    )
    bp = BIDSPath(
        subject="01",
        task="x",
        datatype="ieeg",
        suffix="ieeg",
        extension=".mefd",
        root=root,
    )
    return bp, data


def test_events_tsv_stay_on_their_samples_after_the_lead_is_dropped(tmp_path):
    # on004624: events.tsv onsets count from the aux channels' start (the BCI2000
    # StimulusCode changes at onset x 1000 samples), 0.1-0.2 s before the neural ones.
    bp, _ = _bids_session(tmp_path, (1200, 1000), [START, START + 2_000_000])
    with tolerant():
        raw = read_raw_bids(bp, verbose="ERROR")
    assert list(raw.annotations.onset) == [3.0]  # 5 s after CH0's first sample


def test_a_leading_gap_on_some_channels_is_dropped_with_the_lead(tmp_path, caplog):
    # on004624 sub-c02 Sensorimotor run-6 / sub-c04 run-11: the neural channels start
    # 0.1 s after the aux ones, then skip 0.1 s; from 0.3 s on all are continuous and
    # end together. Here CH1 has samples at 0.1-0.19 s, then from 0.3 s to 40 s.
    bp, data = _bids_session(
        tmp_path, (4000, 3980), [START, START + 100_000], (1, 10), 100_000
    )
    with pytest.raises(ValueError, match="inconsistent n_samples"):
        read_raw_bids(bp, verbose="ERROR")

    with tolerant(), caplog.at_level(logging.WARNING, logger="eegdash"):
        raw = read_raw_bids(bp, verbose="ERROR")
    assert "dropped at the start: CH0 30, CH1 10; at the end: none." in caplog.text
    assert raw.n_times == 3970  # never padded
    assert raw.info["meas_date"].timestamp() == (START + 300_000) / 1e6
    cals = np.array([ch["cal"] for ch in raw.info["chs"]])
    want = [data[0, 30:4000], data[1, 10:3980]]  # both from 0.3 s on
    np.testing.assert_allclose(raw.get_data() / cals[:, None], want)
    np.testing.assert_allclose(raw.annotations.onset, [4.7])  # still at 5 s of CH0


@pytest.mark.parametrize(
    "lengths, starts, gap, gap_us, match",
    [
        # a 1 s gap in the middle of CH1 only (the Stimulation runs)
        (
            (1200, 1000, 1200),
            [START, START + 2_000_000, START],
            (1, 500),
            1_000_000,
            "discontinuities",
        ),
        # the leading gap above, but 30 of 2000 samples (> 1 %)
        ((2000, 1980), [START, START + 100_000], (1, 10), 100_000, "discontinuities"),
        # a leading gap ending 1.8 s in (> 1 s), though < 1 % of 20000 samples
        (
            (20000, 19830),
            [START, START + 100_000],
            (1, 10),
            1_600_000,
            "discontinuities",
        ),
        # a start half a sample off
        (
            (1200, 1000, 1200),
            [START, START + 2_005_000, START],
            None,
            0,
            "whole number of samples",
        ),
    ],
)
def test_channels_not_on_one_time_base_are_refused(
    tmp_path, lengths, starts, gap, gap_us, match
):
    # Sample i would not be the same time on every channel without dropping more
    # than a leading part of the recording.
    root, _ = _write_session(tmp_path / "run-9_ieeg.mefd", lengths, starts, gap, gap_us)
    with tolerant(), pytest.raises(ValueError, match=match):
        mne.io.read_raw_mef(root)


def test_equal_lengths_read_as_before(tmp_path, caplog):
    root, data = _write_session(tmp_path / "run-8_ieeg.mefd", (1000, 1000), [START] * 2)
    ref = mne.io.read_raw_mef(root, preload=True)
    with tolerant(), warnings.catch_warnings(), caplog.at_level(logging.WARNING):
        warnings.simplefilter("error")
        raw = mne.io.read_raw_mef(root, preload=True)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    np.testing.assert_array_equal(raw.get_data(), ref.get_data())

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


def _write_session(path, lengths, starts, gap=None):
    """Unencrypted channels written with pymef (no stub); ``gap=(ch, sample)`` splits
    that channel there into a second segment that starts 1 s late.
    """
    ms = pymef.mef_session.MefSession(str(path), None, new_session=True)
    data = np.arange(max(lengths) * len(lengths), dtype="int32").reshape(
        len(lengths), -1
    )
    for k, (n, start) in enumerate(zip(lengths, starts)):
        cut = gap[1] if gap and gap[0] == k else n
        for seg, (i0, i1, t0) in enumerate(
            [(0, cut, start), (cut, n, start + 1_000_000)]
        ):
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
    # on004624 sub-c02 Functional Sensorimotor run-6: {206800, 207000} samples.
    root, data = _write_session(
        tmp_path / "run-6_ieeg.mefd", (1000, 1200, 1000), [START] * 3
    )
    with pytest.raises(ValueError, match="inconsistent n_samples"):
        mne.io.read_raw_mef(root)

    with tolerant(), caplog.at_level(logging.WARNING, logger="eegdash"):
        raw = mne.io.read_raw_mef(root, preload=True, verbose="ERROR")
    assert "Samples dropped from the end: CH1 200." in caplog.text
    assert raw.n_times == 1000  # never padded
    cals = np.array([ch["cal"] for ch in raw.info["chs"]])
    np.testing.assert_allclose(raw.get_data() / cals[:, None], data[:, :1000])


def test_channels_that_start_apart_are_refused(tmp_path):
    starts = [START, START + 2_000_000, START]  # CH1 starts 2 s later
    root, _ = _write_session(tmp_path / "run-7_ieeg.mefd", (1000, 1200, 1000), starts)
    with tolerant(), pytest.raises(ValueError, match="do not start at the same time"):
        mne.io.read_raw_mef(root)


def test_channels_with_different_gaps_are_refused(tmp_path):
    root, _ = _write_session(
        tmp_path / "run-9_ieeg.mefd", (1000, 1200, 1000), [START] * 3, gap=(1, 500)
    )
    with tolerant(), pytest.raises(ValueError, match="discontinuities"):
        mne.io.read_raw_mef(root)


def test_equal_lengths_read_as_before(tmp_path, caplog):
    root, data = _write_session(tmp_path / "run-8_ieeg.mefd", (1000, 1000), [START] * 2)
    ref = mne.io.read_raw_mef(root, preload=True)
    with tolerant(), warnings.catch_warnings(), caplog.at_level(logging.WARNING):
        warnings.simplefilter("error")
        raw = mne.io.read_raw_mef(root, preload=True)
    assert caplog.text == ""
    np.testing.assert_array_equal(raw.get_data(), ref.get_data())

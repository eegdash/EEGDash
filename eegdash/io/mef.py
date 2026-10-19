"""MNE's MEF3 reader, refusing a session that would crash pymef.

pymef 1.4.8 (meflib) does not check that a segment has its files: a ``.segd``
holding only its ``.tmet`` (on004624 sub-c04 Longterm Passive run-10, channel
CH20; the OpenNeuro deposit has no ``.tdat``/``.tidx`` there either) ends the
process with SIGSEGV in ``read_mef_session_metadata``, which killed the
on004624 read-check job twice. The session is checked first and refused with a
Python error, so one bad recording no longer takes the whole job down.

MNE refuses a session whose channels have different lengths ("MEF channels have
inconsistent n_samples"). In on004624 (15 sessions) some channels run 100-200
samples longer than the others, with the same start time and rate: a trailing
tail. The session's metadata says so, so all channels are read for the length
they have in common, with a warning naming the samples dropped per channel.
Nothing is padded, and channels that do not start at the same time are refused.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import mne
import pymef.mef_session
from mne.io.mef import mef as _mef
from mne.io.mef._utils import _mef_get
from mne.utils import warn

_SEGMENT_FILES = (".tmet", ".tidx", ".tdat")  # metadata, indices, data
_orig_init = _mef.RawMEF.__init__


def check_session(fname):
    """Raise ``ValueError`` if a time-series segment lacks one of its files."""
    incomplete = [
        f"{seg.parent.name}/{seg.name} lacks {', '.join(missing)}"
        for seg in sorted(Path(fname).glob("*.timd/*.segd"))
        if (missing := [ext for ext in _SEGMENT_FILES if not any(seg.glob(f"*{ext}"))])
    ]
    if incomplete:
        raise ValueError(
            f"Incomplete MEF3 session {Path(fname).name}: {'; '.join(incomplete)}"
        )


def _init(self, fname, *args, **kwargs):
    check_session(fname)
    _orig_init(self, fname, *args, **kwargs)


def _gaps(session, ch, n):
    """(sample, uUTC) of each discontinuity before sample ``n``, the start included."""
    flags, _, samples, times = session.get_channel_toc(ch)
    keep = (flags == 1) & (samples < n)
    return tuple(zip(samples[keep].tolist(), times[keep].tolist()))


def crop_to_common_length(session):
    """Give every channel the length they all have, in the metadata MNE reads.

    Only for channels that start at the same time and have their discontinuities
    at the same samples and times: pymef reads samples by index, so the longer
    channels just lose their tail.
    """
    if session.session_md is None:  # a new session, opened to write
        return session
    channels = session.session_md["time_series_channels"]
    n_samples = {
        ch: _mef_get(md["section_2"], "number_of_samples", kind="int")
        for ch, md in channels.items()
    }
    if len(set(n_samples.values())) < 2:
        return session
    starts = {
        ch: _mef_get(md["channel_specific_metadata"], "earliest_start_time", kind="int")
        for ch, md in channels.items()
    }
    if len(set(starts.values())) > 1:
        raise ValueError(
            f"MEF channels have inconsistent n_samples and do not start at the same "
            f"time (earliest_start_time {sorted(set(starts.values()))}), so they "
            f"cannot be cropped to a common length."
        )
    common = min(n_samples.values())
    gaps = {ch: _gaps(session, ch, common) for ch in channels}
    if len(set(gaps.values())) > 1:
        raise ValueError(
            "MEF channels have inconsistent n_samples and their discontinuities "
            "(sample, uUTC) differ before the common length, so the same sample "
            "index is not the same time on every channel."
        )
    for md in channels.values():
        md["section_2"]["number_of_samples"] = common
    dropped = ", ".join(
        f"{ch} {n - common}" for ch, n in n_samples.items() if n > common
    )
    warn(
        f"MEF channels have inconsistent n_samples; reading the {common} samples all "
        f"channels have. Samples dropped from the end: {dropped}."
    )
    return session


def _cropping(open_session):
    def session(*args, **kwargs):
        return crop_to_common_length(open_session(*args, **kwargs))

    return session


@contextmanager
def fixes():
    """Check MEF3 sessions before pymef opens them; crop trailing channel tails."""
    session = _cropping(pymef.mef_session.MefSession)
    with (
        mock.patch.object(_mef.RawMEF, "__init__", _init),
        mock.patch.object(pymef.mef_session, "MefSession", session),
    ):
        yield


def read_raw_mef(fname, *args, **kwargs):
    """Read a MEF3 session with :func:`mne.io.read_raw_mef` under :func:`fixes`."""
    with fixes():
        return mne.io.read_raw_mef(fname, *args, **kwargs)

"""MNE's MEF3 reader, refusing a session that would crash pymef.

pymef 1.4.8 (meflib) does not check that a segment has its files: a ``.segd``
holding only its ``.tmet`` (on004624 sub-c04 Longterm Passive run-10, channel
CH20; the OpenNeuro deposit has no ``.tdat``/``.tidx`` there either) ends the
process with SIGSEGV in ``read_mef_session_metadata``, which killed the
on004624 read-check job twice. The session is checked first and refused with a
Python error, so one bad recording no longer takes the whole job down.

MNE refuses a session whose channels have different lengths ("MEF channels have
inconsistent n_samples"). In on004624 the 32 neural channels start 0.1-0.2 s
after the aux channels (accelerometers, BCI2000 states) and they all end
together; in some sessions a channel ends a few samples early. With one rate,
starts a whole number of samples apart, and the same discontinuities, every
channel is read on the time window they all have, so sample ``i`` is the same
time on every channel: leading samples of the earlier channels and the tail of
the longer ones are dropped, with a warning that names them per channel.
events.tsv onsets, counted from the earliest start, are moved back by the
dropped lead (``bids.py``). Nothing is padded; anything else is refused.
"""

import logging
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from unittest import mock

import mne
import pymef.mef_session
from mne.io.mef import mef as _mef
from mne.io.mef._utils import _mef_get

logger = logging.getLogger("eegdash")
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
    # read_raw_bids counts events.tsv onsets from this many seconds before sample 0
    self._eegdash_lead_s = getattr(self._raw_extras[0]["session"], "lead_s", 0.0)


def _gaps(session, ch, first, n):
    """(sample, uUTC) of each discontinuity in samples ``first + 1 .. first + n - 1``,
    counted from ``first``.
    """
    flags, _, samples, times = session.get_channel_toc(ch)
    keep = (flags == 1) & (samples > first) & (samples < first + n)
    return tuple(zip((samples[keep] - first).tolist(), times[keep].tolist()))


def _read_shifted(read, first, channel_map, sample_map, process_n=None):
    """Read MNE's ``[start, stop]`` from ``first[channel]`` samples on, per channel."""
    chs = [channel_map] if isinstance(channel_map, str) else channel_map
    i0, i1 = sample_map
    return read(channel_map, [[i0 + first[c], i1 + first[c]] for c in chs], process_n)


def _dropped(n_of):
    return ", ".join(f"{ch} {n}" for ch, n in n_of.items() if n) or "none"


def crop_to_common_window(session):
    """Read every channel on the time window they all have, in the metadata MNE reads.

    A channel that starts ``k`` samples later than another (on004624: 100-200
    samples, the channels ending together) is read from the later start: the
    earlier channels lose ``k`` leading samples, the longer ones their tail.
    Only for one rate, starts a whole number of samples apart, and
    discontinuities at the same times inside the window, so sample ``i`` is the
    same time on every channel. pymef reads samples by index per channel, so
    MNE's read is shifted per channel.
    """
    if session.session_md is None:  # a new session, opened to write
        return session
    channels = session.session_md["time_series_channels"]
    md2 = {ch: md["section_2"] for ch, md in channels.items()}
    n_samples = {
        ch: _mef_get(md, "number_of_samples", kind="int") for ch, md in md2.items()
    }
    rates = {_mef_get(md, "sampling_frequency", kind="float") for md in md2.values()}
    if len(set(n_samples.values())) < 2 or len(rates) > 1:  # MNE refuses other rates
        return session
    (rate,) = rates
    starts = {
        ch: _mef_get(md["channel_specific_metadata"], "earliest_start_time", kind="int")
        for ch, md in channels.items()
    }
    t0 = max(starts.values())
    first = {ch: (t0 - t) * rate / 1e6 for ch, t in starts.items()}
    if any(k != round(k) for k in first.values()):
        raise ValueError(
            "MEF channels have inconsistent n_samples and start times "
            f"{sorted(set(starts.values()))} that are not a whole number of samples "
            f"apart at {rate} Hz, so no sample is the same time on every channel."
        )
    first = {ch: round(k) for ch, k in first.items()}
    common = min(n_samples[ch] - k for ch, k in first.items())
    if common <= 0:
        raise ValueError("MEF channels have time ranges that do not overlap.")
    if len({_gaps(session, ch, first[ch], common) for ch in channels}) > 1:
        raise ValueError(
            "MEF channels have inconsistent n_samples and their discontinuities "
            "(sample, uUTC) differ inside the window all channels have, so the "
            "same sample index is not the same time on every channel."
        )
    # pymef's metadata views its C structures (it clips reads to them): copy, then edit.
    for md in channels.values():
        md["section_2"] = md["section_2"].copy()
        md["section_2"]["number_of_samples"] = common
    smd = session.session_md
    smd["session_specific_metadata"] = smd["session_specific_metadata"].copy()
    smd["session_specific_metadata"]["earliest_start_time"] = t0
    session.read_ts_channels_sample = partial(
        _read_shifted, session.read_ts_channels_sample, first
    )
    session.lead_s = (t0 - min(starts.values())) / 1e6
    logger.warning(  # an eegdash warning: read_raw_bids reads with MNE at verbose=ERROR
        "%s: MEF channels start or end apart; reading the %d samples all channels "
        "have, from uUTC %d (%g s after the earliest start; events.tsv onsets are "
        "moved back by as much). Samples dropped at the start: %s; at the end: %s.",
        Path(session.path).name,
        common,
        t0,
        session.lead_s,
        _dropped(first),
        _dropped({ch: n_samples[ch] - first[ch] - common for ch in channels}),
    )
    return session


def _cropping(open_session):
    def session(*args, **kwargs):
        return crop_to_common_window(open_session(*args, **kwargs))

    return session


@contextmanager
def fixes():
    """Check MEF3 sessions before pymef opens them; read channels on a common window."""
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

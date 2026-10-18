"""MNE's MEF3 reader, refusing a session that would crash pymef.

pymef 1.4.8 (meflib) does not check that a segment has its files: a ``.segd``
holding only its ``.tmet`` (on004624 sub-c04 Longterm Passive run-10, channel
CH20; the OpenNeuro deposit has no ``.tdat``/``.tidx`` there either) ends the
process with SIGSEGV in ``read_mef_session_metadata``, which killed the
on004624 read-check job twice. The session is checked first and refused with a
Python error, so one bad recording no longer takes the whole job down.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import mne
from mne.io.mef import mef as _mef

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


@contextmanager
def fixes():
    """Check MEF3 sessions before pymef opens them."""
    with mock.patch.object(_mef.RawMEF, "__init__", _init):
        yield


def read_raw_mef(fname, *args, **kwargs):
    """Read a MEF3 session with :func:`mne.io.read_raw_mef` under :func:`fixes`."""
    with fixes():
        return mne.io.read_raw_mef(fname, *args, **kwargs)

"""MNE's EDF/BDF reader, tolerant of an invalid start time, explicit about a bad header.

The fixes wrap ``_read_edf_header`` of ``mne/io/edf/edf.py`` (MNE 1.13, unchanged
on main as of 2026-10-18); the upstream PR is still to be written:

- A start time with 60 seconds (``10.47.60``: on003555, on003775, on004381,
  on004752) makes ``datetime.replace`` raise ``ValueError: second must be in
  0..59`` after the date was read, so the whole file is refused. MNE already
  reads a file whose start date is invalid, with ``meas_date`` unset and the
  warning "Invalid measurement date encountered in the header."; an invalid
  start time now goes the same way. No date or time is made up.
- A header whose header-size field disagrees with its number of signals
  (on003343: 20 signals, which take 5376 bytes, in a 4608-byte header) fails a
  bare ``assert``. The file is still refused, now with a ``ValueError`` that
  says why ("Bad EDF/BDF header: ..."), which EEGDash reports as a data
  integrity error instead of retrying without the events files.
"""

import io
import re
from contextlib import contextmanager
from unittest import mock

import mne
from mne.io.edf import edf as _edf
from mne.utils import warn

_orig_read_edf_header = _edf._read_edf_header
# What datetime.replace raises for an hour, minute or second out of range.
_BAD_TIME = re.compile(r"^(hour|minute|second) must be in")


class _WithoutStartDate(io.RawIOBase):
    """The file as is, except that MNE cannot parse its start date.

    The header start date (bytes 168-176) is blanked and, in an EDF+ recording
    id (``Startdate 02-MAR-2002 ...``), the date is crossed out, so that
    ``_read_edf_header`` takes its own invalid-date branch.
    """

    def __init__(self, fname):
        self._fid = open(fname, "rb")
        head = bytearray(self._fid.read(184))
        recording_id = head[88:168].decode("latin-1").rstrip().split(" ")
        if len(recording_id) == 5:  # the layout MNE parses the date from
            start = 88 + len(recording_id[0]) + 1
            head[start : start + len(recording_id[1])] = b"X" * len(recording_id[1])
        head[168:176] = b" " * 8
        self._head = bytes(head)
        self._fid.seek(0)

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        return self._fid.seek(offset, whence)

    def tell(self):
        return self._fid.tell()

    def readinto(self, buffer):
        pos = self._fid.tell()
        n = self._fid.readinto(buffer)
        if pos < len(self._head):
            end = min(n, len(self._head) - pos)
            buffer[:end] = self._head[pos : pos + end]
        return n

    def close(self):
        self._fid.close()
        super().close()


def _header_size_mismatch(fname):
    with open(fname, "rb") as fid:
        head = fid.read(256)
    try:
        n_signals, n_bytes = int(head[252:256]), int(head[184:192])
    except ValueError:
        return None
    if n_bytes == 256 * (n_signals + 1):
        return None
    return (
        f"Bad EDF/BDF header: {n_signals} signals take {256 * (n_signals + 1)} "
        f"bytes, but the header-size field says {n_bytes}."
    )


def _read_edf_header(fname, *args, **kwargs):
    try:
        return _orig_read_edf_header(fname, *args, **kwargs)
    except AssertionError:
        message = None if _edf._file_like(fname) else _header_size_mismatch(fname)
        if message is None:
            raise
        raise ValueError(message) from None
    except ValueError as err:
        if not _BAD_TIME.match(str(err)) or _edf._file_like(fname):
            raise
        warn(f"Invalid measurement time in the header ({err}); meas_date is unset.")
    with _WithoutStartDate(fname) as fid:
        return _orig_read_edf_header(fid, *args, **kwargs)


@contextmanager
def fixes():
    """Patch MNE's EDF/BDF header reader while reading."""
    with mock.patch.object(_edf, "_read_edf_header", _read_edf_header):
        yield


def read_raw_edf(input_fname, **kwargs):
    """Read an EDF file with :func:`mne.io.read_raw_edf` under :func:`fixes`."""
    with fixes():
        return mne.io.read_raw_edf(input_fname, **kwargs)


def read_raw_bdf(input_fname, **kwargs):
    """Read a BDF file with :func:`mne.io.read_raw_bdf` under :func:`fixes`."""
    with fixes():
        return mne.io.read_raw_bdf(input_fname, **kwargs)

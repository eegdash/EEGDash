"""MNE's EDF/BDF reader, tolerant of an invalid start time, explicit about a bad header.

The fixes wrap ``_read_edf_header`` of ``mne/io/edf/edf.py`` (MNE 1.13, unchanged
on main as of 2026-10-18); the upstream PR is still to be written:

- A start time with 60 seconds (``10.47.60``: on003555, on003775, on004381,
  on004752) makes ``datetime.replace`` raise ``ValueError: second must be in
  0..59`` after the date was read, so the whole file is refused. MNE already
  reads a file whose start date is invalid, with ``meas_date`` unset and the
  warning "Invalid measurement date encountered in the header."; an invalid
  start time now goes the same way. No date or time is made up.
- A header whose header-size field disagrees with its number of signals fails
  a bare ``assert``. on003343 (59 BDF): 20 signals take 5376 bytes, the field
  says 4608 (or 4352), and the file size is exactly field + n_records x record
  size: the writer put the data at the field's offset, then wrote the full
  header over its first 768 (1024) bytes. MNE already reads the data from the
  field's offset; it gets the header with the field corrected, in memory, and
  the samples of the first channel under the header are annotated
  ``BAD_header_overlap`` (kept by ``read_raw_bids``), with an eegdash warning. Any other
  mismatch is refused with a ``ValueError`` that says why ("Bad EDF/BDF
  header: ..."). Nothing is written; once the deposit has a correct header
  (nemarDatasets/on003343#3) this path is not taken.
"""

import io
import logging
import os
import re
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest import mock

import mne
import mne_bids.read
from mne.io.edf import edf as _edf
from mne.utils import warn

logger = logging.getLogger("eegdash")
_orig_read_edf_header = _edf._read_edf_header
# What datetime.replace raises for an hour, minute or second out of range.
_BAD_TIME = re.compile(r"^(hour|minute|second) must be in")


class _View(io.RawIOBase):
    """``head``, then the file's bytes from ``skip`` on: a header fixed in memory."""

    def __init__(self, fname, head, skip):
        self._fid = open(fname, "rb")
        self._head, self._shift, self._pos = head, skip - len(head), 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        end = os.fstat(self._fid.fileno()).st_size - self._shift
        self._pos = (offset, self._pos + offset, end + offset)[whence]
        return self._pos

    def tell(self):
        return self._pos

    def readinto(self, buffer):
        n = max(0, min(len(buffer), len(self._head) - self._pos))
        buffer[:n] = self._head[self._pos : self._pos + n]
        if n < len(buffer):
            self._fid.seek(self._pos + n + self._shift)
            n += self._fid.readinto(memoryview(buffer)[n:])
        self._pos += n
        return n

    def close(self):
        self._fid.close()
        super().close()


def _without_start_date(fname):
    """The file as is, except that MNE cannot parse its start date.

    The header start date (bytes 168-176) is blanked and, in an EDF+ recording
    id (``Startdate 02-MAR-2002 ...``), the date is crossed out, so that
    ``_read_edf_header`` takes its own invalid-date branch.
    """
    with open(fname, "rb") as fid:
        head = bytearray(fid.read(184))
    recording_id = head[88:168].decode("latin-1").rstrip().split(" ")
    if len(recording_id) == 5:  # the layout MNE parses the date from
        start = 88 + len(recording_id[0]) + 1
        head[start : start + len(recording_id[1])] = b"X" * len(recording_id[1])
    head[168:176] = b" " * 8
    return _View(fname, bytes(head), len(head))


def _header_over_data(fname, n_bytes):
    """on003343: data at the header-size field, the full header written over it.

    Returns the header with its size field corrected and the field's value (the
    data offset), or ``None`` unless the file size says exactly that and only
    the first channel's samples of record 1 lie under the header.
    """
    with open(fname, "rb") as fid:
        head = fid.read(256)
        try:
            n_signals, offset = int(head[252:256]), int(head[184:192])
            n_records = int(head[236:244])
            size = 256 * (n_signals + 1)
            head += fid.read(size - 256)
            at = 256 + 216 * n_signals  # samples per record, 8 bytes per signal
            n_samps = [int(head[at + 8 * k : at + 8 * k + 8]) for k in range(n_signals)]
        except ValueError:
            return None
        # ponytail: only the first channel under the header (on003343); a longer
        # overlap would need one BAD annotation per channel.
        if not 0 < size - offset <= n_samps[0] * n_bytes:
            return None
        if (
            os.fstat(fid.fileno()).st_size
            != offset + n_records * sum(n_samps) * n_bytes
        ):
            return None
    return head[:184] + str(size).ljust(8).encode() + head[192:], offset


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


def _read_edf_header(fname, exclude, infer_types, file_type, *args, **kwargs):
    args = (exclude, infer_types, file_type, *args)
    try:
        return _orig_read_edf_header(fname, *args, **kwargs)
    except AssertionError:
        n_bytes = 3 if file_type == _edf.FileType.BDF else 2
        fixed = None if _edf._file_like(fname) else _header_over_data(fname, n_bytes)
        if fixed is None:
            message = None if _edf._file_like(fname) else _header_size_mismatch(fname)
            if message is None:
                raise
            raise ValueError(message) from None
    except ValueError as err:
        if not _BAD_TIME.match(str(err)) or _edf._file_like(fname):
            raise
        warn(f"Invalid measurement time in the header ({err}); meas_date is unset.")
        with _without_start_date(fname) as fid:
            return _orig_read_edf_header(fid, *args, **kwargs)

    head, offset = fixed
    with _View(fname, head, offset) as fid:
        edf_info, orig_units = _orig_read_edf_header(fid, *args, **kwargs)
    edf_info["data_offset"] = offset  # what MNE reads the data from, as the field says
    n_bad = -(-(len(head) - offset) // n_bytes)
    sfreq = edf_info["n_samps"][0] / edf_info["record_length"][0]
    sel = list(edf_info["sel"])
    name = edf_info["ch_names"][sel.index(0)] if 0 in sel else None
    edf_info["header_overlap"] = (name, n_bad / sfreq) if name else None
    logger.warning(  # an eegdash warning: read_raw_bids reads with MNE at verbose=ERROR
        "%s: the header-size field says %d bytes, but its %d signals take %d, and "
        "the file size puts the data at %d: the header covers the first %d samples "
        "of the first channel. Read with the field corrected in memory; those "
        "samples are annotated BAD_header_overlap.",
        Path(fname).name,
        offset,
        edf_info["nchan"],
        len(head),
        offset,
        n_bad,
    )
    return edf_info, orig_units


def _annotate_header_overlap(init):
    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if overlap := self._raw_extras[0].get("header_overlap"):
            name, duration = overlap
            self.annotations.append(0, duration, "BAD_header_overlap", [[name]])

    return __init__


@contextmanager
def fixes():
    """Patch MNE's EDF/BDF header reader while reading."""
    keep = (*mne_bids.read.ANNOTATIONS_TO_KEEP, "BAD_header_overlap")
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(_edf, "_read_edf_header", _read_edf_header)
        )
        stack.enter_context(
            mock.patch.object(mne_bids.read, "ANNOTATIONS_TO_KEEP", keep)
        )
        for cls in (_edf.RawEDF, _edf.RawBDF):
            init = _annotate_header_overlap(cls.__init__)
            stack.enter_context(mock.patch.object(cls, "__init__", init))
        yield


def read_raw_edf(input_fname, **kwargs):
    """Read an EDF file with :func:`mne.io.read_raw_edf` under :func:`fixes`."""
    with fixes():
        return mne.io.read_raw_edf(input_fname, **kwargs)


def read_raw_bdf(input_fname, **kwargs):
    """Read a BDF file with :func:`mne.io.read_raw_bdf` under :func:`fixes`."""
    with fixes():
        return mne.io.read_raw_bdf(input_fname, **kwargs)

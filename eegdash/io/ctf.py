"""MNE's CTF reader, tolerant of an EEG position file that only says ``n/a``.

Wraps ``_read_eeg`` of ``mne/io/ctf/eeg.py`` (MNE 1.13):

- ``<name>.eeg`` inside the ``.ds`` holds just ``n/a`` (on004278: every run; the
  OpenNeuro deposit file is those 3 bytes). It is BIDS's "not available" where
  CTF expects ``<n> <label> <x> <y> <z>`` lines, and ``_read_eeg`` raises
  "Illegal data in EEG position file". It is read as the absent file it stands
  for, which MNE already supports (no EEG positions); nothing is made up.
"""

import logging
import os
from contextlib import contextmanager
from unittest import mock

import mne
from mne.io.ctf import ctf as _ctf
from mne.io.ctf import eeg as _eeg

logger = logging.getLogger("eegdash")
_orig_read_eeg = _eeg._read_eeg


def _read_eeg(directory):
    fname, found = _eeg._make_ctf_name(directory, "eeg", raise_error=False)
    if found and os.path.getsize(fname) < 16:
        with open(fname, "rb") as fid:
            if fid.read().strip().lower() == b"n/a":
                logger.warning("%s holds only 'n/a': no EEG positions.", fname)
                return None
    return _orig_read_eeg(directory)


@contextmanager
def fixes():
    """Patch MNE's CTF EEG-position reader while reading."""
    with mock.patch.object(_ctf, "_read_eeg", _read_eeg):
        yield


def read_raw_ctf(directory, *args, **kwargs):
    """Read a CTF ``.ds`` with :func:`mne.io.read_raw_ctf` under :func:`fixes`."""
    with fixes():
        return mne.io.read_raw_ctf(directory, *args, **kwargs)

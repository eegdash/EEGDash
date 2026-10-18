"""MNE-BIDS's ``read_raw_bids``, tolerant of two sidecar layouts real deposits have.

Each fix patches one global of ``mne_bids/read.py`` (MNE-BIDS 0.20, unchanged on
main 085e309 as of 2026-10-18); the upstream PRs are still to be written:

- ``RecordingType: "epoched"`` on a continuous container (``.vhdr``, ``.edf``,
  ``.bdf``): ``read_raw_bids`` refuses the file and points to
  ``read_epochs_bids``, which slices it with ``EpochLength`` and fails on most
  real events.tsv files (no ``EpochLength``, stimulus rows, empty or
  non-integer events). The file on disk is continuous, so it is read as Raw;
  events.tsv still becomes annotations. Only ``.set`` with trials is refused.
- An events.tsv column named after a reserved ``mne.Annotations`` key
  (``description`` in nm000536) makes the extras raise ``ValueError``; the
  column is kept as ``events_<name>``.
- A negative events.tsv ``duration`` (on004703: -16.35 s) trips an assertion in
  ``mne.Annotations.crop``; it becomes 0 and a warning is logged.
- A participants.tsv ``age`` that is not an age in years (nm000149: the year
  2018) puts the birthday, ``meas_date - age``, before year 1:
  ``OverflowError``. The birthday is left unset, as MNE-BIDS does when
  ``meas_date`` is unknown; the other participant fields are kept.
"""

import logging
from contextlib import ExitStack, contextmanager
from unittest import mock

import numpy as np
from mne_bids import config as _config
from mne_bids import read as _read

logger = logging.getLogger("eegdash")
_RESERVED = ("onset", "duration", "description", "ch_names", "hed_string")
_orig_make_annotation_kwargs = _read._make_annotation_kwargs
_orig_handle_participants_reading = _read._handle_participants_reading


def _make_annotation_kwargs(*args, **kwargs):
    info, rows = _orig_make_annotation_kwargs(*args, **kwargs)
    negative = info["duration"] < 0
    if negative.any():
        logger.warning("events.tsv: %d negative duration(s) set to 0.", negative.sum())
        info["duration"] = np.where(negative, 0.0, info["duration"])
    if info.get("extras"):
        info["extras"] = [
            {f"events_{k}" if k in _RESERVED else k: v for k, v in extra.items()}
            for extra in info["extras"]
        ]
    return info, rows


def _handle_participants_reading(participants_fname, raw, subject):
    try:
        return _orig_handle_participants_reading(participants_fname, raw, subject)
    except OverflowError:
        logger.warning(
            "participants.tsv: the age of %s is not an age in years; birthday unset.",
            subject,
        )
    meas_date = raw.info["meas_date"]
    with raw.info._unlock():
        raw.info["meas_date"] = None  # MNE-BIDS then skips the birthday
    try:
        return _orig_handle_participants_reading(participants_fname, raw, subject)
    finally:
        with raw.info._unlock():
            raw.info["meas_date"] = meas_date


@contextmanager
def fixes():
    """Patch MNE-BIDS's reader globals while reading."""
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                _read, "_EPOCHED_EXTS", frozenset(_config._EPOCH_READER_NAMES)
            )
        )
        stack.enter_context(
            mock.patch.object(_read, "_make_annotation_kwargs", _make_annotation_kwargs)
        )
        stack.enter_context(
            mock.patch.object(
                _read, "_handle_participants_reading", _handle_participants_reading
            )
        )
        yield

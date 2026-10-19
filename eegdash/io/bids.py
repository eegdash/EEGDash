"""MNE-BIDS's ``read_raw_bids``, tolerant of two sidecar layouts real deposits have.

Each fix patches one global of MNE-BIDS 0.20 (``read.py``, ``tsv_handler.py``; unchanged on
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
- A TSV that is not UTF-8 (on004588 sub-S37 events.tsv): MNE-BIDS checks only
  the first 64 KiB for UTF-8 and otherwise reads latin-1, so a later non-UTF-8
  byte raises ``UnicodeDecodeError``, and cp1252 characters (``–``, ``€``, ``’``)
  come back as control characters. The whole file is checked, and cp1252 is
  tried before latin-1. This replaces EEGDash's rewrite of the cached TSV to
  UTF-8, which made every new session download it again.
- A reader that drops a recording's leading samples (``mef.py``: on004624's
  neural channels start 0.1-0.2 s after its aux channels) sets
  ``raw._eegdash_lead_s``; events.tsv onsets, counted from the earliest start,
  are moved back by as much so they stay on the same samples.

Several ``space-`` electrodes files matching a recording equally well (nm000485
``space-Other`` and ``space-Talairach``) make MNE-BIDS read no positions, and
``BIDSPath(space=)`` would rename the data file too. The electrodes and
coordsystem lookups then take one space: a named coordinate system before
``Other``, then alphabetical. Positions are never merged across spaces.

:func:`without_events` reads a recording as if it had no events.tsv, for an
events file MNE-BIDS cannot turn into annotations. Only reads in the calling
thread are affected; nothing in the dataset folder is renamed, so other runs
read in parallel keep their own events.
"""

import codecs
import gzip
import logging
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from unittest import mock

import numpy as np
from mne_bids import config as _config
from mne_bids import get_entities_from_fname
from mne_bids import read as _read
from mne_bids import tsv_handler as _tsv_handler

logger = logging.getLogger("eegdash")
_RESERVED = ("onset", "duration", "description", "ch_names", "hed_string")
_orig_make_annotation_kwargs = _read._make_annotation_kwargs
_orig_handle_participants_reading = _read._handle_participants_reading
_orig_find_matching_sidecar = _read._find_matching_sidecar
_orig_handle_events_reading = _read._handle_events_reading
_skip_events = ContextVar("eegdash_skip_events_tsv", default=False)
_lead_s = ContextVar("eegdash_events_lead_s", default=0.0)


def _make_annotation_kwargs(*args, **kwargs):
    info, rows = _orig_make_annotation_kwargs(*args, **kwargs)
    info["onset"] = info["onset"] - _lead_s.get()
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


def _handle_events_reading(events_fname, raw, **kwargs):
    """events.tsv onsets count from the recording's first sample; a reader that
    dropped leading samples (``mef.crop_to_common_window``) sets ``_eegdash_lead_s``.
    """
    token = _lead_s.set(getattr(raw, "_eegdash_lead_s", 0.0))
    try:
        return _orig_handle_events_reading(events_fname, raw, **kwargs)
    finally:
        _lead_s.reset(token)


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


# ponytail: wraps a private MNE-BIDS helper; switch to a read_raw_bids option to
# skip events.tsv once MNE-BIDS has one (none in 0.20 or on main 085e309).
def _find_matching_sidecar(bids_path, suffix=None, extension=None, on_error="raise"):
    if suffix == "events" and _skip_events.get():
        return None  # read_raw_bids then reads no events.tsv / events.json
    if suffix in ("electrodes", "coordsystem") and bids_path.space is None:
        bids_path = _pick_space(bids_path, log=suffix == "electrodes")
    return _orig_find_matching_sidecar(
        bids_path, suffix=suffix, extension=extension, on_error=on_error
    )


def _electrodes(bids_path):
    return _orig_find_matching_sidecar(bids_path, "electrodes", ".tsv", "ignore")


def _pick_space(bids_path, log=False):
    """Give ``bids_path`` one ``space`` when its electrodes.tsv is a tie of spaces."""
    if bids_path.root is None or _electrodes(bids_path) is not None:
        return bids_path
    subject_dir = Path(bids_path.root) / f"sub-{bids_path.subject}"
    spaces = {
        get_entities_from_fname(p.name)["space"]
        for p in subject_dir.rglob("*_space-*_electrodes.tsv")
    }
    for space in sorted(spaces, key=lambda s: (s == "Other", s)):
        candidate = bids_path.copy().update(space=space, check=False)
        if _electrodes(candidate) is not None:
            if log:
                logger.warning(
                    "%s: electrodes.tsv in several spaces (%s); reading space-%s.",
                    bids_path.basename,
                    ", ".join(sorted(spaces)),
                    space,
                )
            return candidate
    return bids_path


@contextmanager
def without_events():
    """Make ``read_raw_bids`` in this thread (or task) skip the events sidecars."""
    token = _skip_events.set(True)
    try:
        yield
    finally:
        _skip_events.reset(token)


def tsv_encoding(fname):
    """Return the encoding to read the TSV ``fname`` with (UTF-8, else cp1252, else latin-1).

    Replaces ``mne_bids.tsv_handler._detect_file_encoding`` while reading.
    """
    fname = Path(fname)
    opener = gzip.open if fname.suffix == ".gz" else open
    with opener(fname, "rb") as f:
        # ponytail: whole file in memory, fine for sidecars; chunked if a GB TSV shows up
        content = f.read()
    if content.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if content.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    for encoding in ("utf-8", "cp1252"):
        try:
            content.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"  # decodes any byte


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
            mock.patch.object(_read, "_handle_events_reading", _handle_events_reading)
        )
        stack.enter_context(
            mock.patch.object(
                _read, "_handle_participants_reading", _handle_participants_reading
            )
        )
        stack.enter_context(
            mock.patch.object(_read, "_find_matching_sidecar", _find_matching_sidecar)
        )
        stack.enter_context(
            mock.patch.object(_tsv_handler, "_detect_file_encoding", tsv_encoding)
        )
        yield

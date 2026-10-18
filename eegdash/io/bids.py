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
"""

from contextlib import ExitStack, contextmanager
from unittest import mock

from mne_bids import config as _config
from mne_bids import read as _read

_RESERVED = ("onset", "duration", "description", "ch_names", "hed_string")
_orig_make_annotation_kwargs = _read._make_annotation_kwargs


def _make_annotation_kwargs(*args, **kwargs):
    info, rows = _orig_make_annotation_kwargs(*args, **kwargs)
    if info.get("extras"):
        info["extras"] = [
            {f"events_{k}" if k in _RESERVED else k: v for k, v in extra.items()}
            for extra in info["extras"]
        ]
    return info, rows


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
        yield

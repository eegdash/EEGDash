"""MNE's EEGLAB reader, tolerant of two struct layouts real deposits have.

Each fix wraps one helper of ``mne/io/eeglab/eeglab.py`` (MNE 1.13, unchanged on
main as of 2026-10-15) and is the diff proposed upstream (``upstream/eeglab.md``):

- A ``chanlocs`` field empty on every channel comes back from pymatreader as one
  empty array, not one per channel (on004166: ``chanlocs.type`` and ``.ref``).
  ``_dol_to_lod`` then raises ``IndexError`` or, when the empty field happens to
  come first, returns no channels at all (default names ``EEG 000`` ...).
- Events without a ``type`` field (on008496: one event, ``latency`` only)
  break ``_read_annotations_eeglab``; they get the description ``"n/a"``.
- Events with an empty latency (on004350: a trailing event with every field
  empty) make ``np.array(EEG.event.latency)`` ragged, and events before the
  first sample (on004851: latency -1.3e9) make ``_check_latencies`` raise
  although ``set_annotations`` has already cropped them away. Both are
  dropped with a warning, as EEGLAB's ``eeg_checkset`` drops out-of-range
  events.

One more fix picks the data file at read time (``_check_load_mat``). It replaces
EEGDash's old rewrite of the cached ``.set`` (``savemat``), which made every new
session download the ``.set`` again and let a second process read it half
written. When ``EEG.data`` names another file than ``<set stem>.fdt`` while that
file exists, the own ``.fdt`` is read. MNE trusts ``EEG.data`` whenever the file
it names exists, and on004348 has 7 ``*_task-wake_eeg.set`` that name the same
subject's ``*_task-sleep_eeg.fdt``. (MNE already falls back to the own ``.fdt``
when the named one is missing.)
"""

import inspect
import logging
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest import mock

import mne
import numpy as np
from mne.io.eeglab import eeglab as _eeglab
from mne.io.eeglab import read_raw_eeglab as _mne_read_raw_eeglab
from mne.utils import warn

from .eeglab_salvage import read_raw_truncated_eeglab

logger = logging.getLogger("eegdash")

_orig = {
    name: getattr(_eeglab, name)
    for name in (
        "_dol_to_lod",
        "_bunchify",
        "_read_annotations_eeglab",
        "_check_load_mat",
    )
}


def own_data_file(set_fname, dataname):
    """Return the ``.fdt`` to read for ``set_fname`` whose ``EEG.data`` is ``dataname``.

    ``<set stem>.fdt`` when it exists, else the file ``EEG.data`` names (which
    may not exist). A ``dataname`` that is the ``.set`` itself (samples stored
    in the ``.set``) is kept.
    """
    set_fname = Path(set_fname)
    named = set_fname.parent / str(dataname)
    own = set_fname.with_suffix(".fdt")
    # ponytail: the BIDS name wins over EEG.data; wrong only if a deposit stores
    # the right samples under another name next to a wrong <stem>.fdt.
    if named.suffix.lower() == ".set" or not own.is_file():
        return named
    return own


def _len(value):
    if isinstance(value, np.ndarray):
        return len(value) if value.ndim else 0
    return len(value) if isinstance(value, list) else 0


def _is_empty(value):
    if isinstance(value, np.ndarray):
        return value.size == 0
    return isinstance(value, list) and not value


def _dol_to_lod(dol):
    # MNE takes the length of the first field, and pymatreader orders the fields
    # of a v7.3 struct by set() iteration (string hash), so the first can be the
    # empty one: then MNE drops every channel name instead of raising.
    n = max(map(_len, dol.values()), default=0)
    empty = [np.array([])] * n
    return _orig["_dol_to_lod"](
        {k: empty if _is_empty(v) else v for k, v in dol.items()}
    )


def _bunchify(items):
    items = _orig["_bunchify"](items)
    for item in items:
        if isinstance(item, dict) and "latency" in item:  # an event, not an epoch
            item.setdefault("type", "n/a")
    return items


def _no_sample(latency):
    # NaN is left to MNE, which drops it with its own warning.
    return _is_empty(latency) or (np.ndim(latency) == 0 and latency < 0)


def _read_annotations_eeglab(eeg, uint16_codec=None):
    event = getattr(eeg, "event", None)
    latency = event.get("latency") if isinstance(event, dict) else None
    if isinstance(latency, list):
        keep = [i for i, x in enumerate(latency) if not _no_sample(x)]
        if len(keep) < len(latency):
            warn(
                f"{len(latency) - len(keep)} EEGLAB event(s) have an empty or "
                "negative latency (no sample in the recording) and are dropped."
            )
            eeg.event = {
                k: [v[i] for i in keep]
                if isinstance(v, list) and len(v) == len(latency)
                else v
                for k, v in event.items()
            }
    return _orig["_read_annotations_eeglab"](eeg, uint16_codec=uint16_codec)


def _check_load_mat(fname, uint16_codec, *, preload=False):
    eeg = _orig["_check_load_mat"](fname, uint16_codec, preload=preload)
    if not isinstance(eeg.get("data"), str) or not eeg.data:
        return eeg
    fname = Path(fname)
    data_file = own_data_file(fname, eeg.data)
    named = fname.parent / eeg.data
    if data_file != named:
        if named.is_file():  # MNE would read it; it logs nothing at verbose=ERROR
            logger.warning(
                "%s: EEG.data names %r, the data file of another recording; "
                "reading %s instead.",
                fname.name,
                eeg.data,
                data_file.name,
            )
        eeg.data = data_file.name  # MNE resolves it next to the .set
    return eeg


@contextmanager
def fixes():
    """Patch MNE's EEGLAB helpers while reading."""
    with ExitStack() as stack:
        for name in _orig:
            stack.enter_context(mock.patch.object(_eeglab, name, globals()[name]))
        stack.enter_context(
            mock.patch.object(mne.io, "read_raw_eeglab", _read_or_salvage)
        )
        yield


def _read_or_salvage(input_fname, **kwargs):
    inspect.signature(_mne_read_raw_eeglab).bind(input_fname, **kwargs)
    try:
        return _mne_read_raw_eeglab(input_fname, **kwargs)
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as original:
        if kwargs.get("eog", ()) not in ((), None):
            raise original from None  # salvage uses only deposited channel types
        try:
            return read_raw_truncated_eeglab(
                input_fname, preload=kwargs.get("preload", False)
            )
        except (OSError, ValueError, KeyError, UnicodeError) as refusal:
            logger.debug("Truncated EEGLAB salvage refused: %s", refusal)
            raise original from None


def read_raw_eeglab(input_fname, **kwargs):
    """Read an EEGLAB file with :func:`mne.io.read_raw_eeglab` under :func:`fixes`."""
    with fixes():
        return _read_or_salvage(input_fname, **kwargs)

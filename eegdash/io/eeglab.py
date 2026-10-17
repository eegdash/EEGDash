"""MNE's EEGLAB reader, tolerant of two struct layouts real deposits have.

Each fix wraps one helper of ``mne/io/eeglab/eeglab.py`` (MNE 1.13, unchanged on
main as of 2026-10-15) and is the diff proposed upstream (``upstream/eeglab.md``):

- A ``chanlocs`` field empty on every channel comes back from pymatreader as one
  empty array, not one per channel (on004166: ``chanlocs.type`` and ``.ref``).
  ``_dol_to_lod`` then raises ``IndexError`` or, when the empty field happens to
  come first, returns no channels at all (default names ``EEG 000`` ...).
- Events without a ``type`` field (on008496: one event, ``latency`` only)
  break ``_read_annotations_eeglab``; they get the description ``"n/a"``.
"""

from contextlib import ExitStack, contextmanager
from unittest import mock

import numpy as np
from mne.io.eeglab import eeglab as _eeglab
from mne.io.eeglab import read_raw_eeglab as _mne_read_raw_eeglab

_orig = {name: getattr(_eeglab, name) for name in ("_dol_to_lod", "_bunchify")}


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


@contextmanager
def fixes():
    """Patch MNE's EEGLAB helpers while reading."""
    with ExitStack() as stack:
        for name in _orig:
            stack.enter_context(mock.patch.object(_eeglab, name, globals()[name]))
        yield


def read_raw_eeglab(input_fname, **kwargs):
    """Read an EEGLAB file with :func:`mne.io.read_raw_eeglab` under :func:`fixes`."""
    with fixes():
        return _mne_read_raw_eeglab(input_fname, **kwargs)

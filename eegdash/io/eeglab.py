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

:func:`crop_truncated_fdt` reads a ``.fdt`` that is shorter than its ``.set``
declares (ds006554 sub-11: 198,072,000 B = 773,718.75 samples x 64 channels,
where ``pnts`` is 970,060; the stored object and the annex key have that size).
MNE takes ``pnts`` from the header and fails on the first read past the end of
the file. The complete samples are kept with ``Raw.crop`` and a warning that
gives the loss; nothing is padded.
"""

import os
from contextlib import ExitStack, contextmanager
from unittest import mock

import numpy as np
from mne.io.eeglab import eeglab as _eeglab
from mne.io.eeglab import read_raw_eeglab as _mne_read_raw_eeglab
from mne.utils import warn

_orig = {
    name: getattr(_eeglab, name)
    for name in ("_dol_to_lod", "_bunchify", "_read_annotations_eeglab")
}


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


def crop_truncated_fdt(raw):
    """Crop an EEGLAB ``raw`` (``preload=False``) to the samples its ``.fdt`` holds.

    The ``.fdt`` stores float32 samples, all channels of one sample together, so
    a file cut short loses the end of the recording. Returns the cropped raw, or
    ``None`` when ``raw`` does not read a single ``.fdt`` or the file holds every
    declared sample (or none).
    """
    fnames = [str(f) for f in (getattr(raw, "filenames", None) or []) if f]
    if len(fnames) != 1 or not fnames[0].lower().endswith(".fdt"):
        return None
    size = os.path.getsize(fnames[0])
    sample_bytes = 4 * raw.info["nchan"]
    n, keep = raw.n_times, size // sample_bytes
    if not 0 < keep < n:
        return None
    warn(
        f"{os.path.basename(fnames[0])}: truncated EEGLAB data file. It holds "
        f"{size} of the {n * sample_bytes} bytes the .set declares"
        f"{' (cut inside a sample)' if size % sample_bytes else ''}. Reading the "
        f"first {keep} of {n} samples; the last {(n - keep) / raw.info['sfreq']:.3f} s "
        "are not in the file."
    )
    return raw.crop(tmax=raw.times[keep - 1])

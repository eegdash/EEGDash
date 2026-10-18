"""What EEGDash can read: one table, extension -> reader, tagged with its upstream status.

- ``"mne"``: MNE's own reader, as is.
- ``"tolerant"``: MNE's reader run under the fixes in ``eegdash/io/<fmt>.py``, which
  are not in a MNE release yet (PR drafts: ``eegdash-io-2026-10-17/upstream/``).
- ``"vendored"``: MNE has no reader; ours is written to move to ``mne/io/<fmt>/``.
  MNE's is used as soon as ``mne.io`` has it.

Delete an eegdash module once the MNE release with its fix is the floor in
``pyproject.toml``.
"""

import threading
from contextlib import ExitStack, contextmanager

import mne

from . import bids, ctf, edf, eeglab, mef, nwb, snirf

# extension: (name in mne.io, eegdash reader, upstream status). The extensions are
# MNE-BIDS 0.20's ``config.reader`` (lower-cased) plus ``.nwb``; the client keeps
# only records with one of them.
READERS = {
    ".con": ("read_raw_kit", None, "mne"),
    ".sqd": ("read_raw_kit", None, "mne"),
    ".fif": ("read_raw_fif", None, "mne"),
    ".pdf": ("read_raw_bti", None, "mne"),
    ".ds": ("read_raw_ctf", ctf.read_raw_ctf, "tolerant"),  # ctf.py
    ".vhdr": ("read_raw_brainvision", None, "mne"),
    ".edf": ("read_raw_edf", edf.read_raw_edf, "tolerant"),  # edf.py
    ".bdf": ("read_raw_bdf", edf.read_raw_bdf, "tolerant"),  # edf.py
    ".lay": ("read_raw_persyst", None, "mne"),
    ".eeg": ("read_raw_nihon", None, "mne"),  # Nihon Kohden ".EEG"
    ".cnt": ("read_raw_cnt", None, "mne"),
    ".bin": ("read_raw_egi", None, "mne"),
    ".cdt": ("read_raw_curry", None, "mne"),
    ".mefd": ("read_raw_mef", mef.read_raw_mef, "tolerant"),  # mef.py
    ".set": ("read_raw_eeglab", eeglab.read_raw_eeglab, "tolerant"),  # eeglab.md
    ".snirf": ("read_raw_snirf", snirf.read_raw_snirf, "tolerant"),  # snirf.md
    ".nwb": ("read_raw_nwb", nwb.read_raw_nwb, "vendored"),  # mne-python#10427
}


def get_reader(ext):
    """Return the reader for ``ext`` (lower-case, with the dot)."""
    name, ours, status = READERS[ext]
    return ours if status == "tolerant" else getattr(mne.io, name, ours)


_lock = threading.Lock()
_active = 0
_stack: ExitStack | None = None


@contextmanager
def tolerant():
    """Apply every pending MNE and MNE-BIDS fix while reading (wrap ``read_raw_bids`` in it).

    The fixes patch module globals of MNE. Reads run on threads
    (``download_all``, ``n_jobs``), and interleaved ``mock.patch`` enter/exit
    pairs restore the wrong function, so the patches are applied once by the
    first reader and removed by the last one.
    """
    global _active, _stack
    with _lock:
        if _active == 0:
            _stack = ExitStack()
            for module in (bids, ctf, edf, eeglab, mef, snirf):
                _stack.enter_context(module.fixes())
        _active += 1
    try:
        yield
    finally:
        with _lock:
            _active -= 1
            if _active == 0:
                _stack.close()
                _stack = None

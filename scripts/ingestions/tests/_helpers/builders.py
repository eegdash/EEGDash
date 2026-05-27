"""Synthetic-fixture builders shared by parser, digest, and pipeline tests.

Each builder constructs a minimal but valid binary fixture on disk so a
test can exercise the parser/digest path without committing the file
into ``eegdash-testing-data``. Anything that can be regenerated cheaply
lives here; anything that needs real signal data lives in the corpus.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


def build_synthetic_set_v5(
    path: Path,
    *,
    srate: float = 250.0,
    nbchan: int = 32,
    pnts: int = 5000,
    ch_names: list[str] | None = None,
) -> Path:
    """Construct a minimal EEGLAB .set file in MAT v5 format.

    EEGLAB's struct shape (from the .set parser's reading code):
        EEG.srate         — sampling rate in Hz
        EEG.nbchan        — channel count
        EEG.pnts          — number of samples
        EEG.chanlocs      — struct array with .labels (channel names)
    """
    import scipy.io as scipy_io

    if ch_names is None:
        ch_names = [f"Ch{i + 1}" for i in range(nbchan)]

    chanlocs_dtype = np.dtype([("labels", "O")])
    chanlocs = np.zeros(nbchan, dtype=chanlocs_dtype)
    for i, name in enumerate(ch_names):
        chanlocs[i] = (name,)

    eeg_struct = {
        "srate": np.array([[srate]]),
        "nbchan": np.array([[nbchan]]),
        "pnts": np.array([[pnts]]),
        "chanlocs": chanlocs,
    }
    scipy_io.savemat(str(path), {"EEG": eeg_struct}, do_compression=False)
    return path


def build_mefd_around_real_tmet(
    tmp_path: Path,
    real_tmet: Path,
    *,
    channels: list[str] | None = None,
) -> Path:
    """Construct a .mefd directory using a real .tmet fixture.

    Per the MEF 3.0 directory layout, each channel needs its own
    ``<channel>.timd/<channel>-000000.segd/<channel>-000000.tmet`` path.
    Copies the same real .tmet into each channel's location so every
    channel loads the canonical fixture. Returns the .mefd path.
    """
    if channels is None:
        channels = ["EKG", "LAD1", "LAD2"]

    mefd = tmp_path / "sub-test_ieeg.mefd"
    mefd.mkdir()
    for ch in channels:
        segd = mefd / f"{ch}.timd" / f"{ch}-000000.segd"
        segd.mkdir(parents=True)
        shutil.copy(real_tmet, segd / f"{ch}-000000.tmet")
    return mefd


__all__ = [
    "build_synthetic_set_v5",
    "build_mefd_around_real_tmet",
]

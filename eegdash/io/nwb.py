"""Read the ElectricalSeries of an NWB file as a lazy MNE Raw (h5py only).

MNE has no NWB reader (mne-tools/mne-python#10427) and MNE-BIDS raises for
``.nwb``. Written to move to ``mne/io/nwb/`` (draft: ``upstream/nwb.md``);
``eegdash.io.READERS`` switches to MNE's as soon as ``mne.io.read_raw_nwb`` exists.
"""

from __future__ import annotations

import h5py
import numpy as np
from mne import create_info

# private, but where MNE's own readers import it from (1.11-1.13)
from mne._fiff.utils import _mult_cal_one
from mne.io import BaseRaw


class RawNWB(BaseRaw):
    """Raw object for the single ElectricalSeries of an NWB file (channels: sEEG)."""

    def __init__(self, fname, *, preload=False, verbose=None):
        with h5py.File(fname, "r") as f:
            series = [
                g
                for g in f.get("acquisition", {}).values()
                if g.attrs.get("neurodata_type") == "ElectricalSeries"
            ]
            if len(series) != 1:
                raise ValueError(f"Expected one ElectricalSeries, found {len(series)}")
            # ponytail: one series, starting_time.rate, volts, no offset; this is
            # every NWB file in the catalogue. Timestamps/multi-series when one shows up.
            es = series[0]
            data = es["data"]
            unit = str(data.attrs.get("unit", "volts")).lower()
            if unit not in ("volt", "volts", "v") or data.attrs.get("offset", 0):
                raise ValueError(f"Only volts without offset are read, got {unit!r}")
            ids = f[es["electrodes"].attrs["table"]]["id"][()][es["electrodes"][()]]
            if len(ids) != data.shape[1]:
                raise ValueError(f"{len(ids)} electrodes for {data.shape[1]} columns")
            info = create_info(
                [str(i) for i in ids], float(es["starting_time"].attrs["rate"]), "seeg"
            )
            for ch in info["chs"]:
                ch["cal"] = float(data.attrs.get("conversion", 1.0))
            n_times, path = data.shape[0], data.name
        super().__init__(
            info=info,
            last_samps=[n_times - 1],
            filenames=[str(fname)],
            preload=preload,
            raw_extras=[{"path": path}],
            orig_format="double",
            verbose=verbose,
        )

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data (stored as n_times x n_channels)."""
        with h5py.File(self.filenames[fi], "r") as f:
            block = f[self._raw_extras[fi]["path"]][start:stop].T.astype(np.float64)
        _mult_cal_one(data, block, idx, cals, mult)


def read_raw_nwb(fname, *, preload=False, verbose=None):
    """Read the ElectricalSeries of an NWB file; see :class:`RawNWB`."""
    return RawNWB(fname, preload=preload, verbose=verbose)

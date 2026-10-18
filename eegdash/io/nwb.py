"""Read the voltage series of an NWB file as a lazy MNE Raw (h5py only).

MNE has no NWB reader (mne-tools/mne-python#10427) and MNE-BIDS raises for
``.nwb``. Written to move to ``mne/io/nwb/`` (draft: ``upstream/nwb.md``);
``eegdash.io.READERS`` switches to MNE's as soon as ``mne.io.read_raw_nwb`` exists.

The series is the file's one ``ElectricalSeries``; without one, its one generic
``TimeSeries`` with 2-D data in a voltage unit (nm000369 "microvolts" at a
``starting_time`` rate, nm000371 "µV" with per-sample ``timestamps``).
"""

from __future__ import annotations

import logging

import h5py
import numpy as np
from mne import create_info

# private, but where MNE's own readers import it from (1.11-1.13)
from mne._fiff.utils import _mult_cal_one
from mne.io import BaseRaw

logger = logging.getLogger("eegdash")
_VOLTS = {
    **dict.fromkeys(("v", "volt", "volts"), 1.0),
    **dict.fromkeys(("mv", "millivolt", "millivolts"), 1e-3),
    **dict.fromkeys(("µv", "μv", "uv", "microvolt", "microvolts"), 1e-6),
}


def _unit(series):
    return str(series["data"].attrs.get("unit", "volts")).strip().lower()


def _find_series(f):
    acq = [g for g in f.get("acquisition", {}).values() if isinstance(g, h5py.Group)]
    series = [g for g in acq if g.attrs.get("neurodata_type") == "ElectricalSeries"]
    if not series:
        series = [
            g
            for g in acq
            if g.attrs.get("neurodata_type") == "TimeSeries"
            and "data" in g
            and g["data"].ndim == 2
            and _unit(g) in _VOLTS
        ]
    if len(series) != 1:
        raise ValueError(
            f"Expected one ElectricalSeries (or one 2-D voltage TimeSeries), "
            f"found {len(series)}"
        )
    return series[0]


def _sfreq(series):
    if "starting_time" in series:
        return float(series["starting_time"].attrs["rate"])
    ts = series["timestamps"][()]
    # The nominal rate: the median step, which must agree with the mean rate
    # over the whole recording (nm000371: LSL timestamps jitter, even backwards).
    median = 1.0 / float(np.median(np.diff(ts)))
    mean = (len(ts) - 1) / float(ts[-1] - ts[0])
    if not np.isclose(median, mean, rtol=1e-3):
        raise ValueError(
            f"Irregular timestamps: median rate {median} Hz, mean rate {mean} Hz"
        )
    if not np.all(np.diff(ts) > 0):
        logger.warning(
            "NWB timestamps are not monotonic; using the nominal rate %s Hz.", median
        )
    return median


class RawNWB(BaseRaw):
    """Raw object for the voltage series of an NWB file (channels: sEEG)."""

    def __init__(self, fname, *, preload=False, verbose=None):
        with h5py.File(fname, "r") as f:
            # ponytail: one series, volts without offset; multi-series when one shows up.
            es = _find_series(f)
            data = es["data"]
            unit = _unit(es)
            if unit not in _VOLTS or data.attrs.get("offset", 0):
                raise ValueError(f"Only volts without offset are read, got {unit!r}")
            if "electrodes" in es:
                ids = f[es["electrodes"].attrs["table"]]["id"][()][es["electrodes"][()]]
            else:  # a TimeSeries has no electrode table; channels.tsv names them
                ids = np.arange(data.shape[1])
            if len(ids) != data.shape[1]:
                raise ValueError(f"{len(ids)} electrodes for {data.shape[1]} columns")
            info = create_info([str(i) for i in ids], _sfreq(es), "seeg")
            for ch in info["chs"]:
                ch["cal"] = float(data.attrs.get("conversion", 1.0)) * _VOLTS[unit]
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
    """Read the voltage series of an NWB file; see :class:`RawNWB`."""
    return RawNWB(fname, preload=preload, verbose=verbose)

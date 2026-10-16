"""Read the ElectricalSeries of an NWB file as an MNE Raw (lazy, h5py only).

MNE has no NWB reader (mne-tools/mne-python#10427) and MNE-BIDS raises for
``.nwb`` in ``read_raw_bids``. This module is written to be moved to
``mne/io/nwb/`` as-is; delete it here once the installed MNE ships
``mne.io.read_raw_nwb``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from mne import create_info

# private, but where MNE's own readers import it from (1.11-1.13)
from mne._fiff.utils import _mult_cal_one
from mne.io import BaseRaw

_VOLT_UNITS = {"volt", "volts", "v"}


def _electrical_series(f: h5py.File, series: str | None) -> h5py.Group:
    found = {
        name: obj
        for name, obj in f.get("acquisition", {}).items()
        if isinstance(obj, h5py.Group)
        and obj.attrs.get("neurodata_type") == "ElectricalSeries"
    }
    if series is not None:
        if series not in found:
            raise ValueError(f"No ElectricalSeries {series!r}; found {sorted(found)}")
        return found[series]
    if len(found) != 1:
        raise ValueError(
            f"Expected one ElectricalSeries in /acquisition, found {sorted(found)}; "
            "pass series=<name>."
        )
    return next(iter(found.values()))


def _sfreq(es: h5py.Group) -> float:
    if "starting_time" in es and "rate" in es["starting_time"].attrs:
        return float(es["starting_time"].attrs["rate"])
    if "timestamps" in es:
        steps = np.diff(es["timestamps"][: min(len(es["timestamps"]), 100_001)])
        rate = 1.0 / np.median(steps)
        # ponytail: checks the first 100k steps only; a gap later on goes unnoticed
        if not np.allclose(steps, 1.0 / rate, rtol=1e-3):
            raise ValueError("Timestamps are not uniformly sampled.")
        return float(rate)
    raise ValueError("ElectricalSeries has neither starting_time.rate nor timestamps.")


def _ch_names(f: h5py.File, es: h5py.Group) -> list[str]:
    rows = es["electrodes"][()]
    table = f[es["electrodes"].attrs["table"]]
    col = table["label"] if "label" in table else table["id"]
    names = [v.decode() if isinstance(v, bytes) else str(v) for v in col[()][rows]]
    if len(set(names)) != len(names):
        names = [str(i) for i in table["id"][()][rows]]
    return names


class RawNWB(BaseRaw):
    """Raw object for the ElectricalSeries of an NWB file.

    Parameters
    ----------
    fname : path-like
        Path to the ``.nwb`` (HDF5) file.
    series : str | None
        Name of the ElectricalSeries under ``/acquisition``. ``None`` takes the
        only one and raises if there are several.
    ch_type : str
        Channel type for all channels. Defaults to ``"seeg"`` like
        :func:`mne.io.read_raw_mef`; use :meth:`mne.io.Raw.set_channel_types`
        (or a BIDS ``channels.tsv``) to refine.
    preload : bool | str
        See :class:`mne.io.Raw`.
    verbose : bool | str | int | None
        See :func:`mne.verbose`.

    """

    def __init__(
        self, fname, *, series=None, ch_type="seeg", preload=False, verbose=None
    ):
        fname = Path(fname)
        with h5py.File(fname, "r") as f:
            es = _electrical_series(f, series)
            data = es["data"]
            if data.ndim != 2:
                raise ValueError(
                    f"Expected (n_times, n_channels) data, got {data.shape}"
                )
            unit = str(data.attrs.get("unit", "volts")).lower()
            if unit not in _VOLT_UNITS:
                raise ValueError(f"Unsupported unit {unit!r}; only volts are read.")
            if float(data.attrs.get("offset", 0.0)) != 0.0:
                raise ValueError("Non-zero ElectricalSeries offset is not supported.")
            conversion = float(data.attrs.get("conversion", 1.0))
            n_times, n_chan = data.shape
            names = _ch_names(f, es)
            if len(names) != n_chan:
                raise ValueError(f"{len(names)} electrodes for {n_chan} data columns")
            info = create_info(names, _sfreq(es), ch_type)
            path = es.name
        for ch in info["chs"]:
            ch["cal"] = conversion
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
        """Read a chunk of raw data."""
        with h5py.File(self.filenames[fi], "r") as f:
            # stored (n_times, n_channels); h5py reads the contiguous time slice
            block = f[self._raw_extras[fi]["path"]]["data"][start:stop].T
        _mult_cal_one(data, block.astype(np.float64), idx, cals, mult)


def read_raw_nwb(fname, *, series=None, ch_type="seeg", preload=False, verbose=None):
    """Read the ElectricalSeries of an NWB file.

    Parameters
    ----------
    fname : path-like
        Path to the ``.nwb`` file.
    series : str | None
        ElectricalSeries name under ``/acquisition`` (``None``: the only one).
    ch_type : str
        Channel type for all channels (default ``"seeg"``).
    preload : bool | str
        See :class:`mne.io.Raw`.
    verbose : bool | str | int | None
        See :func:`mne.verbose`.

    Returns
    -------
    raw : RawNWB

    """
    return RawNWB(
        fname, series=series, ch_type=ch_type, preload=preload, verbose=verbose
    )

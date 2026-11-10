"""MNE's SNIRF reader, tolerant of the metadata real deposits leave out.

Each fix wraps one helper of ``mne/io/snirf/_snirf.py`` (MNE 1.13, unchanged on
main as of 2026-10-15) and is the diff proposed upstream (``upstream/snirf.md``):

- ``TimeUnit`` / ``SubjectID`` missing (on007714, on007719). Both are required
  by the spec, which gives no default; seconds is assumed (with a warning) and
  matches the sidecar ``SamplingFrequency`` on those files; the id is left empty.
- ``LengthUnit`` / ``dataUnit`` spelled out ("millimeter", "dimensionless").
- String fields stored as 2-D arrays (ds007463, ds007473).
- Time-domain channels have unit ``FIFF_UNIT_UNITLESS``, which
  ``Info.set_channel_types`` (called by MNE-BIDS) does not know (ds006545).
- Single-chromophore processed data (e.g. HbO-only without HbR): MNE's
  internal validator enforces HbO/HbR pairs during Raw construction,
  which is a processing requirement, not a container format constraint.
- Processed HbO/HbR files where ``/nirs/probe/wavelengths`` is missing
  or empty: MNE requires at least two wavelengths before inspecting
  the data type, even though processed Hb channels never use them.
"""

from contextlib import ExitStack, contextmanager
from unittest import mock

import h5py
import mne.preprocessing.nirs as _nirs
import numpy as np
from mne import pick_types
from mne._fiff import meas_info
from mne.io.constants import FIFF
from mne.io.snirf import _snirf
from mne.io.snirf import read_raw_snirf as _mne_read_raw_snirf
from mne.utils import warn

_orig = {
    name: getattr(_snirf, name)
    for name in (
        "_correct_shape",
        "_get_timeunit_scaling",
        "_get_lengthunit_scaling",
        "_get_dataunit_scaling",
    )
}
_orig_validate = _nirs._validate_nirs_info
_LENGTH_UNITS = {"meter": "m", "centimeter": "cm", "millimeter": "mm"}
_HB_LABELS = {"HbO", "HbR"}


def _validate_nirs_info(info, *args, **kwargs):
    try:
        return _orig_validate(info, *args, **kwargs)
    except ValueError as exc:
        if "At least two chromophore" not in str(exc):
            raise
        picks = pick_types(info, fnirs=True, exclude=[])
        types = info.get_channel_types()
        kinds = {types[p] for p in picks}
        names = [info["ch_names"][p] for p in picks]
        if len(picks) and kinds in ({"hbo"}, {"hbr"}) and len(set(names)) == len(names):
            warn(f"fNIRS data holds a single chromophore ({kinds.pop()}); read without HbO/HbR pairs.")
            return picks
        raise


def _label(ds):
    v = np.asarray(ds[()]).reshape(-1)[0]
    return v.decode() if isinstance(v, bytes) else str(v)


def _processed_hb_only(f):
    try:
        group = f["nirs"]["data1"]
        lists = [k for k in group.keys() if k.startswith("measurementList")]
        if not lists:
            return False
        for k in lists:
            ml = group[k]
            if int(np.asarray(ml["dataType"][()]).reshape(-1)[0]) != 99999:
                return False
            if "dataTypeLabel" not in ml or _label(ml["dataTypeLabel"]) not in _HB_LABELS:
                return False
        return True
    except Exception:
        return False


class _File(h5py.File):
    def get(self, name, default=None, *args, **kwargs):
        out = super().get(name, default, *args, **kwargs)
        if str(name).strip("/") == "nirs/probe/wavelengths":
            empty = out is None or (isinstance(out, h5py.Dataset) and out.size == 0)
            if empty and _processed_hb_only(self):
                warn("SNIRF file has no probe wavelengths (a required field); its HbO/HbR data does not use them.")
                return np.array([0, 0])
        return out


def _correct_shape(arr):
    if arr.dtype == object and arr.shape == () and arr.item() is None:
        return np.array([b""])  # absent field: h5py .get() gave None
    return _orig["_correct_shape"](arr).reshape(-1)


def _get_timeunit_scaling(time_unit):
    if time_unit is None:
        warn("SNIRF file has no TimeUnit (a required tag); assuming seconds.")
        time_unit = "s"
    return _orig["_get_timeunit_scaling"](time_unit)


def _get_lengthunit_scaling(length_unit):
    return _orig["_get_lengthunit_scaling"](_LENGTH_UNITS.get(length_unit, length_unit))


def _get_dataunit_scaling(unit):
    return _orig["_get_dataunit_scaling"]("" if unit == "dimensionless" else unit)


@contextmanager
def fixes():
    """Patch MNE's SNIRF helpers (and the unit table) while reading."""
    with ExitStack() as stack:
        for name in _orig:
            stack.enter_context(mock.patch.object(_snirf, name, globals()[name]))
        stack.enter_context(mock.patch.object(_nirs, "_validate_nirs_info", _validate_nirs_info))
        stack.enter_context(mock.patch.object(h5py, "File", _File))
        stack.enter_context(
            mock.patch.dict(
                meas_info._unit2human,
                {FIFF.FIFF_UNIT_UNITLESS: "unitless", FIFF.FIFF_UNIT_SEC2: "s^2"},
            )
        )
        yield


def read_raw_snirf(fname, optode_frame="unknown", **kwargs):
    """Read a SNIRF file with :func:`mne.io.read_raw_snirf` under :func:`fixes`."""
    with fixes():
        return _mne_read_raw_snirf(fname, optode_frame, **kwargs)

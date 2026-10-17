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
"""

from contextlib import ExitStack, contextmanager
from unittest import mock

import numpy as np
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
_LENGTH_UNITS = {"meter": "m", "centimeter": "cm", "millimeter": "mm"}


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

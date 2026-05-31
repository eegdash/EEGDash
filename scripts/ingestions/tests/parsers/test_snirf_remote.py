"""Tests for the remote-ready SNIRF ``n_times`` reader (file-like input).

``snirf_n_times_from_fileobj`` takes any seekable read-only file-like
(e.g. ``_remote_header.RangeReader`` or ``io.BytesIO``) and returns the
time-axis length, reading only HDF5 metadata (``.shape``) — never the
signal. These tests exercise it against small in-memory HDF5 files.
"""

from __future__ import annotations

import io

import pytest

# h5py is required to BUILD the in-memory fixtures and is the engine the
# reader uses; missing → skip the whole module.
h5py = pytest.importorskip("h5py")

import numpy as np

from _snirf_parser import snirf_n_times_from_fileobj

# Time-axis length encoded in every well-formed fixture below.
_N_TIMES = 1234
_N_CHANS = 8


def _build_snirf_with_datatimeseries() -> io.BytesIO:
    """Well-formed SNIRF: nirs/data1/dataTimeSeries of shape (n_times, nchans)."""
    bio = io.BytesIO()
    f = h5py.File(bio, "w")
    nirs = f.create_group("nirs")
    d = nirs.create_group("data1")
    d.create_dataset("dataTimeSeries", data=np.zeros((_N_TIMES, _N_CHANS)))
    f.close()
    bio.seek(0)
    return bio


def _build_snirf_time_only() -> io.BytesIO:
    """SNIRF variant with only a /time dataset (no dataTimeSeries)."""
    bio = io.BytesIO()
    f = h5py.File(bio, "w")
    nirs = f.create_group("nirs")
    d = nirs.create_group("data1")
    d.create_dataset("time", data=np.zeros((_N_TIMES,)))
    f.close()
    bio.seek(0)
    return bio


def test_snirf_n_times_from_datatimeseries():
    """dataTimeSeries.shape[0] is the preferred time-axis length."""
    bio = _build_snirf_with_datatimeseries()
    assert snirf_n_times_from_fileobj(bio) == _N_TIMES


def test_snirf_n_times_from_time_only():
    """With no dataTimeSeries, fall back to the /time dataset length."""
    bio = _build_snirf_time_only()
    assert snirf_n_times_from_fileobj(bio) == _N_TIMES


def test_snirf_n_times_junk_returns_none():
    """A non-HDF5 BytesIO returns None (never raises)."""
    junk = io.BytesIO(b"NOT HDF5 AT ALL" * 16)
    assert snirf_n_times_from_fileobj(junk) is None

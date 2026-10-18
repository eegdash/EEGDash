"""eegdash.io.nwb, laid out like the catalogue's NWB files (nm000350, on008610)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from eegdash.dataset.base import _read_nwb_with_channels_tsv
from eegdash.dataset.dataset import _VALID_DATA_EXTENSIONS
from eegdash.io.nwb import read_raw_nwb


def _write(path, data, *, conversion=1.0, unit="volts", n_series=1):
    with h5py.File(path, "w") as f:
        table = f.create_group("general/extracellular_ephys/electrodes")
        table.create_dataset("id", data=np.arange(data.shape[1]))
        for i in range(n_series):
            es = f.create_group(f"acquisition/es{i}")
            es.attrs["neurodata_type"] = "ElectricalSeries"
            d = es.create_dataset("data", data=data)
            d.attrs["conversion"] = conversion
            d.attrs["unit"] = unit
            es.create_dataset("electrodes", data=np.arange(data.shape[1])).attrs[
                "table"
            ] = table.ref
            es.create_dataset("starting_time", data=0.0).attrs["rate"] = 1000.0
    return path


def test_scaled_lazy_read(tmp_path):
    data = np.arange(150, dtype=np.float32).reshape(50, 3)  # (n_times, n_channels)
    raw = read_raw_nwb(_write(tmp_path / "a.nwb", data, conversion=1e-3))
    assert (raw.info["sfreq"], raw.ch_names, raw.preload) == (
        1000.0,
        ["0", "1", "2"],
        False,
    )
    np.testing.assert_allclose(raw.get_data(), data.T * 1e-3, rtol=1e-6)
    np.testing.assert_allclose(
        raw.get_data(picks=[2], start=10, stop=13), data[10:13, 2:3].T * 1e-3, rtol=1e-6
    )
    assert ".nwb" in _VALID_DATA_EXTENSIONS


@pytest.mark.parametrize("kwargs", [{"unit": "degrees"}, {"n_series": 2}])
def test_rejects_what_it_cannot_read(tmp_path, kwargs):
    with pytest.raises(ValueError):
        read_raw_nwb(_write(tmp_path / "b.nwb", np.zeros((10, 2)), **kwargs))


def test_channels_tsv_names_types_bads(tmp_path):
    path = _write(tmp_path / "sub-01_task-x_ieeg.nwb", np.zeros((10, 2)))
    (tmp_path / "sub-01_task-x_channels.tsv").write_text(
        "name\ttype\tunits\tstatus\nch01\tECOG\tV\tgood\nch02\tECOG\tV\tbad\n"
    )
    raw = _read_nwb_with_channels_tsv(path)
    assert raw.ch_names == ["ch01", "ch02"]
    assert raw.get_channel_types() == ["ecog", "ecog"]
    assert raw.info["bads"] == ["ch02"]


def _write_timeseries(path, data, *, unit, rate=None, timestamps=None, extra=()):
    """A generic TimeSeries in acquisition, no electrode table (nm000369, nm000371)."""
    with h5py.File(path, "w") as f:
        for name in extra:  # 1-D, unitless companions (nm000371: Audio, Stimulus)
            ts = f.create_group(f"acquisition/{name}")
            ts.attrs["neurodata_type"] = "TimeSeries"
            ts.create_dataset("data", data=np.zeros(data.shape[0])).attrs["unit"] = ""
        ts = f.create_group("acquisition/iEEG")
        ts.attrs["neurodata_type"] = "TimeSeries"
        d = ts.create_dataset("data", data=data)
        d.attrs["unit"] = unit
        d.attrs["conversion"] = 1.0
        if rate is not None:
            ts.create_dataset("starting_time", data=0.0).attrs["rate"] = rate
        else:
            ts.create_dataset("timestamps", data=timestamps)
    return path


def test_timeseries_microvolts_at_rate(tmp_path):
    """nm000369: ``acquisition/ECoG`` is a TimeSeries in microvolts at 512 Hz."""
    data = np.arange(40, dtype=float).reshape(20, 2)
    raw = read_raw_nwb(
        _write_timeseries(tmp_path / "a.nwb", data, unit="microvolts", rate=512.0)
    )
    assert (raw.info["sfreq"], raw.ch_names) == (512.0, ["0", "1"])
    np.testing.assert_allclose(raw.get_data(), data.T * 1e-6)


def test_timeseries_jittered_timestamps_nominal_rate(tmp_path):
    """nm000371: "µV", 1024 Hz LSL timestamps that jitter (even backwards)."""
    n = 2048
    ts = 100.0 + np.arange(n) / 1024.0
    ts[[10, 500]] += [0.013, -0.012]  # out of order, as in the deposit
    data = np.ones((n, 3))
    raw = read_raw_nwb(
        _write_timeseries(
            tmp_path / "b.nwb", data, unit="µV", timestamps=ts, extra=("Audio",)
        )
    )
    assert raw.info["sfreq"] == 1024.0
    np.testing.assert_allclose(raw.get_data(start=n - 5), 1e-6)


def test_timeseries_irregular_timestamps_refused(tmp_path):
    ts = np.cumsum(np.r_[np.full(50, 0.001), np.full(50, 0.004)])
    with pytest.raises(ValueError, match="Irregular timestamps"):
        read_raw_nwb(
            _write_timeseries(
                tmp_path / "c.nwb", np.ones((100, 2)), unit="V", timestamps=ts
            )
        )


def test_timeseries_not_voltage_is_not_picked(tmp_path):
    with pytest.raises(ValueError, match="found 0"):
        read_raw_nwb(
            _write_timeseries(tmp_path / "d.nwb", np.ones((10, 2)), unit="", rate=10.0)
        )

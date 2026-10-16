"""eegdash.dataset.nwb: the two NWB layouts in the catalogue (nm000350 NWB 2.0, on008610 NWB 2.10)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from eegdash.dataset.base import _read_nwb_with_channels_tsv
from eegdash.dataset.dataset import _VALID_DATA_EXTENSIONS
from eegdash.dataset.io import _load_raw_direct
from eegdash.dataset.nwb import RawNWB, read_raw_nwb


def _write(
    path,
    data,
    *,
    conversion,
    unit="volts",
    rate=1000.0,
    labels=None,
    timestamps=None,
    extra_series=False,
):
    with h5py.File(path, "w") as f:
        f.attrs["nwb_version"] = "2.10.0"
        table = f.create_group("general/extracellular_ephys/electrodes")
        n = data.shape[1]
        table.create_dataset("id", data=np.arange(n))
        if labels is not None:
            table.create_dataset(
                "label", data=np.array(labels, dtype=h5py.string_dtype())
            )
        acq = f.create_group("acquisition")
        names = ["ElectricalSeries"] + (["lfp2"] if extra_series else [])
        for name in names:
            es = acq.create_group(name)
            es.attrs["neurodata_type"] = "ElectricalSeries"
            d = es.create_dataset("data", data=data)
            d.attrs["conversion"] = conversion
            d.attrs["unit"] = unit
            el = es.create_dataset("electrodes", data=np.arange(n))
            el.attrs["table"] = table.ref
            if timestamps is None:
                st = es.create_dataset("starting_time", data=0.0)
                st.attrs["rate"] = rate
            else:
                es.create_dataset("timestamps", data=timestamps)
    return path


def test_values_scaled_to_volts_and_read_lazily(tmp_path):
    data = np.arange(50 * 3, dtype=np.float32).reshape(50, 3)  # (n_times, n_channels)
    raw = read_raw_nwb(
        _write(tmp_path / "a.nwb", data, conversion=1e-3)
    )  # nm000350: mV
    assert isinstance(raw, RawNWB)
    assert raw.info["sfreq"] == 1000.0
    assert raw.ch_names == ["0", "1", "2"]
    assert raw.get_channel_types() == ["seeg"] * 3
    assert not raw.preload
    np.testing.assert_allclose(raw.get_data(), data.T * 1e-3, rtol=1e-6)
    np.testing.assert_allclose(
        raw.get_data(picks=[2], start=10, stop=13), data[10:13, 2:3].T * 1e-3, rtol=1e-6
    )


def test_labels_and_timestamps(tmp_path):
    data = np.ones((20, 2))
    ts = np.arange(20) / 500.0
    path = _write(
        tmp_path / "b.nwb", data, conversion=1e-6, labels=["A1", "A2"], timestamps=ts
    )
    raw = read_raw_nwb(path)
    assert raw.ch_names == ["A1", "A2"]
    assert raw.info["sfreq"] == pytest.approx(500.0)


def test_nwb_is_loadable_through_eegdash(tmp_path):
    assert ".nwb" in _VALID_DATA_EXTENSIONS
    raw = _load_raw_direct(
        _write(tmp_path / "c.nwb", np.zeros((10, 2)), conversion=1.0)
    )
    assert isinstance(raw, RawNWB)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"unit": "degrees"}, "only volts"),
        ({"timestamps": np.array([0.0, 0.001, 0.5, 0.501])}, "not uniformly"),
        ({"extra_series": True}, "pass series"),
    ],
)
def test_rejects_what_it_cannot_read_correctly(tmp_path, kwargs, match):
    n = 4 if "timestamps" in kwargs else 10
    path = _write(tmp_path / "d.nwb", np.zeros((n, 2)), conversion=1.0, **kwargs)
    with pytest.raises(ValueError, match=match):
        read_raw_nwb(path)


def test_series_selects_one_of_several(tmp_path):
    path = _write(
        tmp_path / "e.nwb", np.zeros((10, 2)), conversion=1.0, extra_series=True
    )
    assert read_raw_nwb(path, series="lfp2").n_times == 10


def test_channels_tsv_sets_types_and_bads(tmp_path):
    path = _write(
        tmp_path / "sub-01_task-x_ieeg.nwb", np.zeros((10, 2)), conversion=1.0
    )
    (tmp_path / "sub-01_task-x_channels.tsv").write_text(
        "name\ttype\tunits\tstatus\n0\tECOG\tV\tgood\n1\tECOG\tV\tbad\n"
    )
    raw = _read_nwb_with_channels_tsv(path)
    assert raw.get_channel_types() == ["ecog", "ecog"]
    assert raw.info["bads"] == ["1"]

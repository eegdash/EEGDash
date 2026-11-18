"""Only complete, identified MAT-v5 sample columns may survive a cut."""

import json
from unittest.mock import patch

import mne
import numpy as np
import pytest
from scipy.io import savemat

from eegdash.dataset.base import EEGDashRaw
from eegdash.io import get_reader, tolerant
from eegdash.io.eeglab_salvage import read_raw_truncated_eeglab


def _cut(tmp_path, layout="flat", dtype="<f4", keep=7):
    path = tmp_path / "sub-01_task-x_eeg.set"
    data = (np.arange(60).reshape(3, 20) + 0.25).astype(dtype)
    fields = dict(nbchan=3, pnts=20, trials=1, srate=100, data=data)
    if layout == "first":
        fields = dict(data=data)
    savemat(path, {"EEG": fields} if layout == "struct" else fields)
    blob = path.read_bytes()
    start = blob.index(data.tobytes(order="F"))
    path.write_bytes(blob[: start + keep * 3 * data.itemsize + 1])
    path.with_name("sub-01_task-x_channels.tsv").write_text(
        "\ufeffname\ttype\tunits\nFz\tEEG\tµV\nEOG\tEOG\tuV\nTrig\tTRIG\tn/a\n"
    )
    path.with_suffix(".json").write_text(json.dumps({"SamplingFrequency": 100}))
    return path, data[:, :keep]


@pytest.mark.parametrize("layout", ["flat", "struct", "first"])
@pytest.mark.parametrize("dtype", ["<f4", "<f8"])
def test_salvage_exact_samples(tmp_path, layout, dtype):
    path, expected = _cut(tmp_path, layout, dtype)
    before = path.read_bytes()
    with pytest.raises(Exception):
        mne.io.read_raw_eeglab(path, verbose="error")
    with pytest.warns(RuntimeWarning, match="DataIntegrity"):
        read_raw_truncated_eeglab(path)
    with pytest.warns(RuntimeWarning, match="DataIntegrity.*lost_fraction=0.65"):
        raw = get_reader(".set")(path, preload=False, verbose="error")
    assert raw.n_times == 7
    assert raw.ch_names == ["Fz", "EOG", "Trig"]
    assert raw.get_channel_types() == ["eeg", "eog", "stim"]
    expected = expected.astype(float)
    expected[:2] *= 1e-6
    np.testing.assert_allclose(raw.get_data(), expected)
    assert "lost_fraction=0.65" in raw.info["description"]
    assert "metadata" in raw.info["description"]
    assert path.read_bytes() == before
    # The same reader must be reached through MNE-BIDS's module-global call.
    with tolerant(), pytest.warns(RuntimeWarning, match="DataIntegrity"):
        out = mne.io.read_raw_eeglab(path, verbose="error")
    np.testing.assert_array_equal(out.get_data(), raw.get_data())


@pytest.mark.parametrize(
    "damage", ["no_channels", "wrong_count", "units", "rate", "no_samples"]
)
def test_refuse_unknown_metadata_or_no_samples(tmp_path, damage):
    path, _ = _cut(tmp_path, keep=0 if damage == "no_samples" else 7)
    channels = path.with_name("sub-01_task-x_channels.tsv")
    if damage == "no_channels":
        channels.rename(tmp_path / "sub-02_task-other_channels.tsv")
    elif damage == "wrong_count":
        channels.write_text("name\ttype\tunits\nFz\tEEG\tuV\n")
    elif damage == "units":
        channels.write_text(channels.read_text().replace("µV", "n/a"))
    elif damage == "rate":
        path.with_suffix(".json").write_text('{"SamplingFrequency": 200}')
    with pytest.raises(Exception):
        get_reader(".set")(path, verbose="error")


def test_dataset_path(tmp_path):
    root = tmp_path / "ds1"
    folder = root / "sub-01" / "eeg"
    folder.mkdir(parents=True)
    path, _ = _cut(folder)
    (root / "dataset_description.json").write_text(
        '{"Name":"test","BIDSVersion":"1.9.0"}'
    )
    record = dict(
        dataset="ds1",
        bids_relpath=str(path.relative_to(root)),
        storage={"backend": "local", "base": str(root)},
    )
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))
    with pytest.warns(RuntimeWarning, match="DataIntegrity"):
        raw = ds.raw
    assert raw.n_times == 7
    assert "lost_fraction=0.65" in raw.info["description"]


@pytest.mark.parametrize(
    "damage", ["complex", "compressed", "epochs", "dimensions", "before_data"]
)
def test_refuse_unsupported_mat(tmp_path, damage):
    path, _ = _cut(tmp_path)
    if damage == "before_data":
        path.write_bytes(path.read_bytes()[:140])
    elif damage == "compressed":
        savemat(path, {"data": np.ones((3, 20))}, do_compression=True)
        path.write_bytes(path.read_bytes()[:-2])
    else:
        data = np.arange(60, dtype=np.float32).reshape(3, 20)
        if damage == "complex":
            data = data.astype(complex) + 1j
        fields = dict(
            nbchan=4 if damage == "dimensions" else 3,
            trials=2 if damage == "epochs" else 1,
            pnts=20,
            srate=100,
            data=data,
        )
        savemat(path, fields)
        blob = path.read_bytes()
        payload = data.real.tobytes(order="F")
        offset = blob.index(payload)
        path.write_bytes(blob[: offset + 3 * 7 * data.real.dtype.itemsize])
    with pytest.raises(ValueError):
        read_raw_truncated_eeglab(path)


def test_lazy_selection_preload_and_crop(tmp_path):
    path, _ = _cut(tmp_path)
    with pytest.warns(RuntimeWarning, match="DataIntegrity"):
        raw = read_raw_truncated_eeglab(path)
    data = raw.get_data()
    selected = raw.copy().pick(["Trig", "Fz"]).crop(tmin=0.02)
    np.testing.assert_array_equal(selected.get_data(), data[[2, 0], 2:])
    with pytest.warns(RuntimeWarning, match="DataIntegrity"):
        preloaded = read_raw_truncated_eeglab(path, preload=True)
    np.testing.assert_array_equal(preloaded.get_data(), data)


@pytest.mark.parametrize("kwargs", [{"typo": True}, {"eog": ["Fz"]}])
def test_do_not_silently_discard_reader_options(tmp_path, kwargs):
    path, _ = _cut(tmp_path)
    with pytest.raises((TypeError, AttributeError, OSError, ValueError)):
        get_reader(".set")(path, **kwargs)


def test_reject_units_conflicting_with_eeglab_calibration(tmp_path):
    path, _ = _cut(tmp_path)
    channels = path.with_name("sub-01_task-x_channels.tsv")
    channels.write_text(channels.read_text().replace("µV", "V"))
    with pytest.raises(ValueError, match="microvolt storage convention"):
        read_raw_truncated_eeglab(path)


@pytest.mark.parametrize("unit", ["uV", "V"])
def test_intact_calibration_parity_or_explicit_refusal(tmp_path, unit):
    path, _ = _cut(tmp_path)
    channels = path.with_name("sub-01_task-x_channels.tsv")
    channels.write_text(
        "name\ttype\tunits\nFz\tEEG\t"
        + unit
        + "\nCz\tEEG\t"
        + unit
        + "\nPz\tEEG\t"
        + unit
        + "\n"
    )
    data = np.arange(60, dtype=np.float32).reshape(3, 20)
    chanlocs = np.array(
        [(name,) for name in ["Fz", "Cz", "Pz"]], dtype=[("labels", "O")]
    )
    fields = dict(
        nbchan=3,
        pnts=20,
        trials=1,
        srate=100.0,
        xmin=0.0,
        xmax=0.19,
        data=data,
        chanlocs=chanlocs,
        event=[],
        epoch=[],
    )
    savemat(path, {"EEG": fields})
    intact = mne.io.read_raw_eeglab(path, preload=True, verbose="error").get_data()
    blob = path.read_bytes()
    offset = blob.index(data.tobytes(order="F"))
    path.write_bytes(blob[: offset + 7 * 3 * 4 + 1])
    if unit == "V":
        with pytest.raises(ValueError, match="microvolt storage convention"):
            read_raw_truncated_eeglab(path)
    else:
        with pytest.warns(RuntimeWarning, match="DataIntegrity"):
            raw = read_raw_truncated_eeglab(path)
        np.testing.assert_allclose(raw.get_data(), intact[:, :7])

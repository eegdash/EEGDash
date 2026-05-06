from types import SimpleNamespace

import pytest

from eegdash.local_bids import (
    _get_file_metadata,
    _normalize_modalities,
    discover_local_bids_records,
)


@pytest.mark.parametrize(
    "modality_filter,expected",
    [
        (None, ["eeg", "meg", "ieeg", "nirs", "fnirs", "emg"]),
        ("fnirs", ["nirs"]),
        ([" eeg ", "fnirs"], ["eeg", "nirs"]),
        ([], ["eeg", "meg", "ieeg", "nirs", "fnirs", "emg"]),
    ],
)
def test_normalize_modalities(modality_filter, expected):
    assert _normalize_modalities(modality_filter) == expected


class _Helper:
    def __init__(self, *, fail_attr=False, fail_channels=False):
        self.fail_attr = fail_attr
        self.fail_channels = fail_channels

    def get_bids_file_attribute(self, name, _path):
        if self.fail_attr:
            raise RuntimeError("attr fail")
        return {"sfreq": 128.0, "nchans": 64, "ntimes": 1024}[name]

    def channel_labels(self, _path):
        if self.fail_channels:
            raise RuntimeError("ch fail")
        return ["C3", "C4"]


@pytest.mark.parametrize(
    "helper,expected",
    [
        (
            None,
            {
                "sampling_frequency": None,
                "nchans": None,
                "ntimes": None,
                "ch_names": None,
            },
        ),
        (
            _Helper(),
            {
                "sampling_frequency": 128.0,
                "nchans": 64,
                "ntimes": 1024,
                "ch_names": ["C3", "C4"],
            },
        ),
        (
            _Helper(fail_channels=True),
            {
                "sampling_frequency": 128.0,
                "nchans": 64,
                "ntimes": 1024,
                "ch_names": None,
            },
        ),
        (
            _Helper(fail_attr=True),
            {
                "sampling_frequency": None,
                "nchans": None,
                "ntimes": None,
                "ch_names": None,
            },
        ),
    ],
)
def test_get_file_metadata(helper, expected):
    assert _get_file_metadata(helper, "dummy-file") == expected


def test_discover_builds_matching_args(monkeypatch, tmp_path):
    captured = {}

    def fake_find_matching_paths(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "eegdash.local_bids.find_matching_paths", fake_find_matching_paths
    )
    records = discover_local_bids_records(
        tmp_path,
        {
            "dataset": "ds1",
            "modality": "eeg",
            "subject": ["01", "02"],
            "session": "01",
            "run": [],
        },
    )

    assert records == []
    assert captured["subjects"] == ["01", "02"]
    assert captured["sessions"] == ["01"]
    assert "runs" not in captured


def test_discover_filters_missing_derivative_and_invalid_extensions(
    monkeypatch, tmp_path
):
    root = tmp_path / "ds1"
    valid = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.vhdr"
    derivative = root / "derivatives" / "sub-01" / "eeg" / "sub-01_task-rest_eeg.vhdr"
    invalid = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.json"

    valid.parent.mkdir(parents=True)
    derivative.parent.mkdir(parents=True)
    valid.touch()
    derivative.touch()
    invalid.touch()

    outside = tmp_path / "outside_task-rest_eeg.vhdr"
    outside.touch()

    paths = [
        SimpleNamespace(
            fpath=str(valid),
            datatype="eeg",
            suffix="eeg",
            subject="01",
            session=None,
            task="rest",
            run=None,
            acquisition=None,
        ),
        SimpleNamespace(
            fpath=str(derivative),
            datatype="eeg",
            suffix="eeg",
            subject="01",
            session=None,
            task="rest",
            run=None,
            acquisition=None,
        ),
        SimpleNamespace(
            fpath=str(invalid),
            datatype="eeg",
            suffix="eeg",
            subject="01",
            session=None,
            task="rest",
            run=None,
            acquisition=None,
        ),
        SimpleNamespace(
            fpath=str(outside),
            datatype="eeg",
            suffix="eeg",
            subject="09",
            session=None,
            task="rest",
            run=None,
            acquisition=None,
        ),
    ]

    monkeypatch.setattr("eegdash.local_bids.find_matching_paths", lambda **_: paths)
    monkeypatch.setattr(
        "eegdash.local_bids.EEGBIDSDataset", lambda **_: None, raising=False
    )

    records = discover_local_bids_records(root, {"dataset": "ds1", "modality": "eeg"})

    assert len(records) == 2
    assert {record["entities"]["subject"] for record in records} == {"01", "09"}
    # outside path cannot be made relative, so fallback is basename-only
    outside_record = [r for r in records if r["entities"]["subject"] == "09"][0]
    assert outside_record["bids_relpath"] == "outside_task-rest_eeg.vhdr"


def test_discover_fallback_glob_after_valueerror(monkeypatch, tmp_path):
    root = tmp_path / "ds1"
    eeg_dir = root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-01_task-rest_run-01_eeg.vhdr").touch()
    (eeg_dir / "sub-01_task-rest_eeg.json").touch()

    def fail_find_matching_paths(**_kwargs):
        raise ValueError("bad entity")

    monkeypatch.setattr(
        "eegdash.local_bids.find_matching_paths", fail_find_matching_paths
    )

    records = discover_local_bids_records(root, {"dataset": "ds1", "modality": "eeg"})

    assert len(records) == 1
    assert records[0]["entities"]["subject"] == "01"
    assert records[0]["entities"]["run"] == "01"
    assert records[0]["storage"]["backend"] == "local"


def test_discover_directory_format_fallback(monkeypatch, tmp_path):
    root = tmp_path / "ds1"
    ds_dir = root / "sub-01" / "meg" / "sub-01_task-rest_run-01_meg.ds"
    ds_dir.mkdir(parents=True)

    monkeypatch.setattr("eegdash.local_bids.find_matching_paths", lambda **_: [])

    records = discover_local_bids_records(root, {"dataset": "ds1", "modality": "meg"})

    assert len(records) == 1
    assert records[0]["entities"]["subject"] == "01"
    assert records[0]["datatype"] == "meg"
    assert records[0]["suffix"] == "meg"

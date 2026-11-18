"""Offline local directory recordings (MEF3 ``.mefd``, CTF ``.ds``) stay local.

mne-bids returns files and ``.ds`` folders only, so a ``.mefd`` session is
found by the directory scan in ``local_bids``. That scan built its records
without ``storage_backend="local"``, so they took the ``"s3"`` default: offline
loading then sent an anonymous S3 LIST for a prefix built from the local path.
"""

import json
import urllib.request

import boto3
import numpy as np
import pytest
import requests

from eegdash import downloader
from eegdash.dataset.dataset import EEGDashDataset

_SFREQ = 256
_N_SAMPLES = 2560


@pytest.fixture
def no_network(monkeypatch):
    """Fail every S3 or HTTP access, and remember it if a caller swallows it."""
    calls = []

    def refuse(name):
        def _refuse(*args, **kwargs):
            calls.append((name, args))
            raise AssertionError(f"network access in offline mode: {name}{args}")

        return _refuse

    monkeypatch.setattr(downloader, "get_s3_filesystem", refuse("get_s3_filesystem"))
    monkeypatch.setattr(boto3, "client", refuse("boto3.client"))
    monkeypatch.setattr(requests.Session, "request", refuse("requests"))
    monkeypatch.setattr(urllib.request, "urlopen", refuse("urlopen"))
    return calls


def _write_mef_session(path):
    """Write a 2-channel, 10 s, unencrypted MEF3 session with pymef."""
    from pymef.mef_session import MefSession

    path.parent.mkdir(parents=True, exist_ok=True)
    session = MefSession(str(path), "", read_metadata=False, new_session=True)
    start = 946684800000000
    duration = int(_N_SAMPLES / _SFREQ * 1e6)
    section_2 = {
        "recording_duration": duration,
        "sampling_frequency": _SFREQ,
        "units_conversion_factor": 1.0,
        "units_description": "uV",
        "start_sample": 0,
        "number_of_discontinuities": 1,
    }
    section_3 = {"recording_time_offset": start - 1_000_000, "GMT_offset": 0}
    for i, ch in enumerate(("EEG1", "EEG2")):
        session.write_mef_ts_segment_metadata(
            ch, 0, "", "", start, start + duration, section_2, section_3
        )
        data = (np.arange(_N_SAMPLES) % 100 + i).astype(np.int32)
        session.write_mef_ts_segment_data(ch, 0, "", "", 256, data)


def _bids_root(cache_dir, dataset):
    root = cache_dir / dataset
    root.mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": dataset, "BIDSVersion": "1.9.0"})
    )
    return root


def test_offline_mefd_record_is_local_and_reads_without_network(tmp_path, no_network):
    pytest.importorskip("pymef")
    dataset = "localmef"
    root = _bids_root(tmp_path, dataset)
    stem = "sub-01_ses-01_task-rest_run-01_ieeg"
    ieeg_dir = root / "sub-01" / "ses-01" / "ieeg"
    _write_mef_session(ieeg_dir / f"{stem}.mefd")
    (ieeg_dir / f"{stem}.json").write_text(
        json.dumps({"TaskName": "rest", "SamplingFrequency": _SFREQ})
    )

    ds = EEGDashDataset(
        cache_dir=tmp_path, dataset=dataset, modality="ieeg", download=False
    )

    assert len(ds.datasets) == 1
    rec = ds.datasets[0]
    assert rec.record["storage"]["backend"] == "local"
    assert rec.record["storage"]["base"] == str(root)
    assert rec.filecache == ieeg_dir / f"{stem}.mefd"

    data = rec.raw.get_data()
    assert data.shape == (2, _N_SAMPLES)
    assert no_network == []

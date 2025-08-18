import sys
import types
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_dataset_loads_without_eegdash(monkeypatch, tmp_path):
    """Dataset should load from records without contacting network resources."""

    # Stub external dependencies to avoid heavy imports
    for name in [
        "numpy",
        "xarray",
        "dotenv",
        "joblib",
        "pymongo",
        "s3fs",
        "mne_bids",
        "pandas",
        "bids",
        "fsspec",
        "braindecode",
        "braindecode.datasets",
    ]:
        sys.modules.setdefault(name, types.ModuleType(name))

    # Create minimal mne package with required submodules
    mne_pkg = types.ModuleType("mne")
    mne_pkg.__path__ = []
    sys.modules["mne"] = mne_pkg
    sys.modules["mne._fiff"] = types.ModuleType("mne._fiff")
    sys.modules["mne._fiff.utils"] = types.ModuleType("mne._fiff.utils")
    sys.modules["mne._fiff.utils"]._read_segments_file = lambda *a, **k: None
    sys.modules["mne.io"] = types.ModuleType("mne.io")
    sys.modules["mne.io"].BaseRaw = object

    sys.modules["numpy"].ndarray = object
    sys.modules["xarray"].DataArray = object
    sys.modules["dotenv"].load_dotenv = lambda *a, **k: None
    sys.modules["joblib"].Parallel = lambda *a, **k: []
    sys.modules["joblib"].delayed = lambda f: f
    sys.modules["pymongo"].InsertOne = object
    sys.modules["pymongo"].UpdateOne = object
    sys.modules["pymongo"].MongoClient = object
    sys.modules["s3fs"].S3FileSystem = object
    sys.modules["bids"].BIDSLayout = object
    sys.modules["fsspec.callbacks"] = types.ModuleType("callbacks")
    sys.modules["fsspec.callbacks"].TqdmCallback = object
    sys.modules["mne_bids"].BIDSPath = object

    dummy_concat = type("BaseConcatDataset", (), {"__init__": lambda self, datasets=None: setattr(self, "datasets", datasets)})
    sys.modules["braindecode.datasets"].BaseConcatDataset = dummy_concat
    sys.modules["braindecode.datasets"].BaseDataset = object

    # Create minimal package placeholder to load submodules without executing __init__
    pkg = types.ModuleType("eegdash")
    pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "eegdash")]
    sys.modules.setdefault("eegdash", pkg)

    # Dynamically import eegdash.api without running package __init__
    api_spec = importlib.util.spec_from_file_location(
        "eegdash.api", Path(pkg.__path__[0]) / "api.py"
    )
    api_module = importlib.util.module_from_spec(api_spec)
    sys.modules["eegdash.api"] = api_module
    api_spec.loader.exec_module(api_module)
    EEGDashDataset = api_module.EEGDashDataset

    # Fake base dataset to avoid heavy dependencies and network calls
    class FakeBaseDataset:
        def __init__(self, record, cache_dir, s3_bucket=None, **kwargs):
            self.record = record
            self.raw = "DATA"

    monkeypatch.setattr(api_module, "EEGDashBaseDataset", FakeBaseDataset)

    # Patch EEGDash to raise if instantiated (simulating offline/no DB access)
    def raise_eegdash(*args, **kwargs):
        raise RuntimeError(
            "EEGDash should not be instantiated when records are provided"
        )

    monkeypatch.setattr(api_module, "EEGDash", raise_eegdash)

    record = {
        "dataset": "ds",
        "bidspath": "ds/file",
        "bidsdependencies": [],
        "subject": "01",
        "session": "",
        "run": "",
        "task": "rest",
        "modality": "eeg",
        "sampling_frequency": 1,
        "nchans": 1,
        "ntimes": 1,
    }

    dummy = SimpleNamespace(close=lambda: None)

    dataset = EEGDashDataset(records=[record], cache_dir=str(tmp_path), eeg_dash_instance=dummy)

    assert dataset.datasets[0].raw == "DATA"
    assert dataset.datasets[0].record == record

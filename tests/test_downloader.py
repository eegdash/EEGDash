import os
from pathlib import Path

os.environ.setdefault("MNE_USE_USER_CONFIG", "false")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import pytest
import xarray as xr

import eegdash.downloader as downloader

OPENNEURO_EEG_FILE = (
    "s3://openneuro.org/ds005505/sub-NDARAC904DMU/eeg/"
    "sub-NDARAC904DMU_task-RestingState_eeg.set"
)
OPENNEURO_SMALL_FILES = [
    "ds005505/dataset_description.json",
    "ds005505/participants.tsv",
]

CHALLENGE_EEG_FILE = (
    "s3://nmdatasets/NeurIPS25/R5_mini_L100_bdf/ds005509/sub-NDARAH793FBF/eeg/"
    "sub-NDARAH793FBF_task-DespicableMe_eeg.set"
)
CHALLENGE_SMALL_FILES = [
    "dataset_description.json",
    "participants.tsv",
]
CHALLENGE_SMALL_FILES_ORIGINAL = [
    "ds005509/dataset_description.json",
    "ds005509/participants.tsv",
]


@pytest.fixture(scope="module")
def download_dir():
    """Provide a shared cache directory for tests that need to cache datasets."""
    from pathlib import Path

    from eegdash.paths import get_default_cache_dir

    download_dir = Path(get_default_cache_dir())
    download_dir.mkdir(parents=True, exist_ok=True)
    return download_dir


def _require_s3(uri: str) -> None:
    """Skip the test if the requested S3 object cannot be reached."""
    try:
        filesystem = downloader.get_s3_filesystem()
        filesystem.info(uri)
    except Exception as exc:  # pragma: no cover - defensive skip
        pytest.skip(f"S3 resource {uri} not reachable: {exc}")


@pytest.fixture(scope="module")
def openneuro_local_file(download_dir: Path) -> Path:
    _require_s3("s3://openneuro.org/ds005505/dataset_description.json")
    destination = download_dir / Path(OPENNEURO_EEG_FILE).name
    if not destination.exists():
        downloader.download_s3_file(
            OPENNEURO_EEG_FILE,
            destination,
            s3_open_neuro=True,
        )
    return destination


@pytest.fixture(scope="module")
def challenge_local_file(download_dir: Path) -> Path:
    _require_s3(
        "s3://nmdatasets/NeurIPS25/R5_mini_L100_bdf/ds005509/dataset_description.json"
    )
    destination = download_dir / Path(CHALLENGE_EEG_FILE).with_suffix(".bdf").name
    if not destination.exists():
        downloader.download_s3_file(
            CHALLENGE_EEG_FILE,
            download_dir / Path(CHALLENGE_EEG_FILE).name,
            s3_open_neuro=False,
        )
    return destination


def test_download_s3_file_openneuro_writes_to_destination(openneuro_local_file: Path):
    assert openneuro_local_file.exists()
    assert openneuro_local_file.suffix == ".set"
    assert openneuro_local_file.stat().st_size > 0


def test_download_s3_file_competition_dataset_converts_to_bdf(
    challenge_local_file: Path,
):
    assert challenge_local_file.exists()
    assert challenge_local_file.suffix == ".bdf"
    assert challenge_local_file.stat().st_size > 0


def test_download_dependencies_fetches_sidecar_files(download_dir: Path):
    _require_s3("s3://openneuro.org/ds005505/dataset_description.json")
    downloader.download_dependencies(
        s3_bucket="s3://openneuro.org",
        bids_dependencies=OPENNEURO_SMALL_FILES,
        bids_dependencies_original=OPENNEURO_SMALL_FILES,
        cache_dir=download_dir,
        dataset_folder="ds005505",
        record={"dataset": "ds005505"},
        s3_open_neuro=True,
    )

    for rel_path in OPENNEURO_SMALL_FILES:
        local_path = download_dir / rel_path
        assert local_path.exists()
        assert local_path.stat().st_size > 0


def test_download_dependencies_handles_competition_paths(download_dir: Path):
    _require_s3(
        "s3://nmdatasets/NeurIPS25/R5_mini_L100_bdf/ds005509/dataset_description.json"
    )
    downloader.download_dependencies(
        s3_bucket="s3://nmdatasets/NeurIPS25/R5_mini_L100_bdf",
        bids_dependencies=CHALLENGE_SMALL_FILES,
        bids_dependencies_original=CHALLENGE_SMALL_FILES_ORIGINAL,
        cache_dir=download_dir,
        dataset_folder="ds005509-bdf-mini",
        record={"dataset": "ds005509"},
        s3_open_neuro=False,
    )

    for rel in CHALLENGE_SMALL_FILES:
        local_path = download_dir / "ds005509-bdf-mini" / rel
        assert local_path.exists()
        assert local_path.stat().st_size > 0


def test_load_eeg_from_s3_returns_dataarray():
    _require_s3(OPENNEURO_EEG_FILE)
    result = downloader.load_eeg_from_s3(OPENNEURO_EEG_FILE)

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("channel", "time")
    assert result.sizes["channel"] > 0
    assert result.sizes["time"] > 0
    assert result.dtype.kind == "f"

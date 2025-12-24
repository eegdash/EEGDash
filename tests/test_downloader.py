import os
from pathlib import Path

os.environ.setdefault("MNE_USE_USER_CONFIG", "false")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import pytest

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
    "s3://nmdatasets/NeurIPS25/R5_mini_L100_bdf/sub-NDARAH793FBF/eeg/"
    "sub-NDARAH793FBF_task-DespicableMe_eeg.bdf"
)
CHALLENGE_SMALL_FILES = [
    "dataset_description.json",
    "participants.tsv",
]


def _require_s3(uri: str) -> None:
    """Skip the test if the requested S3 object cannot be reached."""
    try:
        filesystem = downloader.get_s3_filesystem()
        filesystem.info(uri)
    except Exception as exc:  # pragma: no cover - defensive skip
        pytest.skip(f"S3 resource {uri} not reachable: {exc}")


@pytest.fixture(scope="module")
def openneuro_local_file(cache_dir: Path) -> Path:
    _require_s3("s3://openneuro.org/ds005505/dataset_description.json")
    destination = cache_dir / Path(OPENNEURO_EEG_FILE).name
    if not destination.exists():
        downloader.download_s3_file(OPENNEURO_EEG_FILE, destination)
    return destination


@pytest.fixture(scope="module")
def challenge_local_file(cache_dir: Path) -> Path:
    _require_s3(
        "s3://nmdatasets/NeurIPS25/R5_mini_L100_bdf/dataset_description.json"
    )
    destination = cache_dir / Path(CHALLENGE_EEG_FILE).name
    if not destination.exists():
        downloader.download_s3_file(CHALLENGE_EEG_FILE, destination)
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


def test_download_dependencies_fetches_sidecar_files(cache_dir: Path):
    _require_s3("s3://openneuro.org/ds005505/dataset_description.json")
    pairs = [
        (downloader.get_s3path("s3://openneuro.org", rel_path), cache_dir / rel_path)
        for rel_path in OPENNEURO_SMALL_FILES
    ]
    downloader.download_files(pairs)

    for rel_path in OPENNEURO_SMALL_FILES:
        local_path = cache_dir / rel_path
        assert local_path.exists()
        assert local_path.stat().st_size > 0


def test_download_dependencies_handles_competition_paths(cache_dir: Path):
    _require_s3(
        "s3://nmdatasets/NeurIPS25/R5_mini_L100_bdf/dataset_description.json"
    )
    dataset_subdir = cache_dir / "ds005509-bdf-mini"
    dataset_subdir.mkdir(parents=True, exist_ok=True)
    pairs = [
        (
            downloader.get_s3path(
                "s3://nmdatasets/NeurIPS25/R5_mini_L100_bdf", rel
            ),
            dataset_subdir / rel,
        )
        for rel in CHALLENGE_SMALL_FILES
    ]
    downloader.download_files(pairs)

    for rel in CHALLENGE_SMALL_FILES:
        local_path = dataset_subdir / rel
        assert local_path.exists()
        assert local_path.stat().st_size > 0

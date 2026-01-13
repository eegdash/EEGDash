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
    _require_s3("s3://nmdatasets/NeurIPS25/R5_mini_L100_bdf/dataset_description.json")
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
    _require_s3("s3://nmdatasets/NeurIPS25/R5_mini_L100_bdf/dataset_description.json")
    dataset_subdir = cache_dir / "ds005509-bdf-mini"
    dataset_subdir.mkdir(parents=True, exist_ok=True)
    pairs = [
        (
            downloader.get_s3path("s3://nmdatasets/NeurIPS25/R5_mini_L100_bdf", rel),
            dataset_subdir / rel,
        )
        for rel in CHALLENGE_SMALL_FILES
    ]
    downloader.download_files(pairs)

    for rel in CHALLENGE_SMALL_FILES:
        local_path = dataset_subdir / rel
        assert local_path.exists()
        assert local_path.stat().st_size > 0


def test_downloader_gap(tmp_path):
    # Trigger downloader.py 99 (return local_path)
    p = tmp_path / "dummy.txt"
    p.write_text("hello")
    # download_s3_file will return local_path if it exists and remote_size is None
    from unittest.mock import patch

    with patch("eegdash.downloader._remote_size", return_value=None):
        res = downloader.download_s3_file("s3://bucket/dummy.txt", p)
        assert res == p


from unittest.mock import MagicMock, patch

import pytest


def test_download_s3_file_incomplete_raises_oserror(tmp_path):
    """Test that incomplete download raises OSError (line 92-96)."""
    from eegdash.downloader import download_s3_file

    local_file = tmp_path / "test.txt"

    mock_fs = MagicMock()
    # Remote says file is 100 bytes
    mock_fs.info.return_value = {"size": 100}

    # Mock _filesystem_get to write fewer bytes than expected
    def mock_get(filesystem, s3path, filepath, size):
        filepath.write_bytes(b"short")  # Only 5 bytes, not 100

    with patch("eegdash.downloader._filesystem_get", mock_get):
        with pytest.raises(OSError, match="Incomplete download"):
            download_s3_file(
                s3_path="s3://bucket/test.txt",
                local_path=local_file,
                filesystem=mock_fs,
            )

    pass


def test_download_file_incomplete_error(tmp_path):
    """Test that incomplete download raises OSError."""
    from unittest.mock import MagicMock, patch

    import pytest

    from eegdash.downloader import download_s3_file

    local_path = tmp_path / "incomplete_file.txt"

    mock_fs = MagicMock()
    mock_fs.info.return_value = {"size": 1000}  # Expected 1000 bytes

    # Mock _filesystem_get to create a file with wrong size (simulates incomplete download)
    def mock_get(*args, **kwargs):
        local_path.write_bytes(b"short")  # Only 5 bytes, not 1000

    with patch("eegdash.downloader._filesystem_get", side_effect=mock_get):
        with pytest.raises(OSError, match="Incomplete download"):
            download_s3_file("s3://bucket/file.txt", local_path, filesystem=mock_fs)


def test_download_file_existing_with_unknown_remote_size(tmp_path):
    """Test download when remote size is None and file exists."""
    from unittest.mock import MagicMock

    from eegdash.downloader import download_s3_file

    local_file = tmp_path / "test.txt"
    local_file.write_text("existing content")

    mock_fs = MagicMock()
    # Return None for remote size (line 84)
    mock_fs.info.side_effect = Exception("No size info")

    result = download_s3_file(
        s3_path="s3://bucket/test.txt", local_path=local_file, filesystem=mock_fs
    )
    # Should return existing file since we can't verify size
    assert result == local_file


def test_download_files_skip_existing_with_matching_size(tmp_path):
    """Test download_files skips files with matching size."""
    from unittest.mock import MagicMock

    from eegdash.downloader import download_files

    local_file = tmp_path / "test.txt"
    local_file.write_bytes(b"12345")  # 5 bytes

    mock_fs = MagicMock()
    mock_fs.info.return_value = {"size": 5}  # Remote size matches

    # Line 130: skip existing file with matching size
    result = download_files(
        [("s3://bucket/test.txt", local_file)],
        filesystem=mock_fs,
        skip_existing=True,
    )
    # Should return empty since file was skipped
    assert local_file not in result


def test_download_files_skip_existing_false(tmp_path):
    """Test download_files with skip_existing=False removes existing files."""
    from unittest.mock import MagicMock, patch

    import pytest

    from eegdash.downloader import download_files

    local_file = tmp_path / "test.txt"
    local_file.write_bytes(b"old content")

    mock_fs = MagicMock()
    mock_fs.info.return_value = {"size": 10}

    # Line 130: dest exists, skip_existing=False, unlinks file
    with patch("eegdash.downloader._filesystem_get") as mock_get:
        # Simulate new file being written
        def write_new_content(*args, **kwargs):
            local_file.write_bytes(b"new content!")  # 12 bytes != 10

        mock_get.side_effect = write_new_content

        # Lines 136-137: size mismatch raises OSError
        with pytest.raises(OSError, match="Incomplete download"):
            download_files(
                [("s3://bucket/test.txt", local_file)],
                filesystem=mock_fs,
                skip_existing=False,
            )


def test_download_file_remote_size_none(tmp_path):
    """Test download when remote size is None (line 84)."""
    from unittest.mock import MagicMock

    from eegdash.downloader import download_s3_file

    local_path = tmp_path / "test.txt"
    local_path.write_text("existing content")

    mock_fs = MagicMock()
    mock_fs.info.side_effect = Exception("No size info")

    # File exists, remote size is None - should return early
    result = download_s3_file("s3://bucket/key", local_path, filesystem=mock_fs)
    assert result == local_path


def test_download_file_incomplete(tmp_path):
    """Test download raises on incomplete download (line 99)."""
    from pathlib import Path
    from unittest.mock import MagicMock

    import pytest

    from eegdash.downloader import download_s3_file

    local_path = tmp_path / "test.txt"

    mock_fs = MagicMock()
    mock_fs.info.return_value = {"size": 1000}  # Expected size

    def mock_get(rpath, lpath, **kwargs):
        Path(lpath).write_text("short")  # Write less than expected

    mock_fs.get = mock_get

    with pytest.raises(OSError, match="Incomplete download"):
        download_s3_file("s3://bucket/key", local_path, filesystem=mock_fs)


def test_download_files_skip_existing(tmp_path):
    """Test download_files with skip_existing (line 130)."""
    from unittest.mock import MagicMock

    from eegdash.downloader import download_files

    existing_file = tmp_path / "existing.txt"
    existing_file.write_text("existing content")

    mock_fs = MagicMock()
    mock_fs.info.return_value = {"size": len("existing content")}

    files = [("s3://bucket/existing.txt", existing_file)]

    result = download_files(files, filesystem=mock_fs, skip_existing=True)
    # Should skip and return empty list
    assert result == []


def test_remote_size_returns_none_on_exception():
    """Test _remote_size returns None on exception (lines 136-137)."""
    from unittest.mock import MagicMock

    from eegdash.downloader import _remote_size

    mock_fs = MagicMock()
    mock_fs.info.side_effect = Exception("Error")

    result = _remote_size(mock_fs, "s3://bucket/key")
    assert result is None


def test_remote_size_invalid_size_type():
    """Test _remote_size with non-integer size."""
    from unittest.mock import MagicMock

    from eegdash.downloader import _remote_size

    mock_fs = MagicMock()
    mock_fs.info.return_value = {"size": "not_a_number"}

    result = _remote_size(mock_fs, "s3://bucket/key")
    assert result is None


def test_downloader_get_s3_util():
    import s3fs

    from eegdash import downloader

    fs = downloader.get_s3_filesystem()
    assert isinstance(fs, s3fs.S3FileSystem)
    assert downloader.get_s3path("s3://bucket", "file") == "s3://bucket/file"
    assert downloader.get_s3path("s3://bucket", "/file") == "s3://bucket/file"
    assert downloader.get_s3path("s3://bucket", "") == "s3://bucket"


def test_downloader_remote_size_errors():
    from unittest.mock import MagicMock

    from eegdash import downloader

    mock_fs = MagicMock()
    mock_fs.info.side_effect = Exception("S3 Error")
    assert downloader._remote_size(mock_fs, "s3://b/f") is None

    mock_fs.info.side_effect = None
    mock_fs.info.return_value = {}  # No size key
    assert downloader._remote_size(mock_fs, "s3://b/f") is None

    mock_fs.info.return_value = {"Size": "not_int"}
    assert downloader._remote_size(mock_fs, "s3://b/f") is None


def test_downloader_download_s3_file_exists_match(tmp_path):
    from unittest.mock import MagicMock

    from eegdash import downloader

    f = tmp_path / "test.txt"
    f.write_text("content")

    mock_fs = MagicMock()
    # Remote size matches local (7 bytes)
    mock_fs.info.return_value = {"size": 7}

    res = downloader.download_s3_file("s3://b/test.txt", f, filesystem=mock_fs)
    assert res == f
    mock_fs.get.assert_not_called()


def test_downloader_download_s3_file_incomplete(tmp_path):
    from pathlib import Path
    from unittest.mock import MagicMock

    import pytest

    from eegdash import downloader

    f = tmp_path / "incomplete.txt"
    f.touch()  # size 0

    mock_fs = MagicMock()
    mock_fs.info.return_value = {"size": 100}

    # Mock get to recreate file but with wrong size
    def mock_get(s3, local, **kwargs):
        Path(local).write_text("b" * 50)  # only 50 bytes, expected 100

    mock_fs.get.side_effect = mock_get

    with pytest.raises(OSError, match="Incomplete download"):
        downloader.download_s3_file("s3://b/f", f, filesystem=mock_fs)

    assert not f.exists()  # Should remain unlinked


def test_downloader_batch_skip_existing(tmp_path):
    from pathlib import Path
    from unittest.mock import MagicMock

    from eegdash import downloader

    f1 = tmp_path / "f1"
    f1.write_text("a")
    f2 = tmp_path / "f2"

    mock_fs = MagicMock()
    # f1 size 1 (matches "a"), f2 size 10 (needs DL)
    mock_fs.info.side_effect = [{"size": 1}, {"size": 10}]

    # For f2, assume 'get' writes the file
    def mock_get(s3, local, **kwargs):
        Path(local).write_text("b" * 10)

    mock_fs.get.side_effect = mock_get

    files = [("s3://b/f1", f1), ("s3://b/f2", f2)]
    downloaded = downloader.download_files(
        files, filesystem=mock_fs, skip_existing=True
    )

    assert len(downloaded) == 1
    assert downloaded == [f2]


def test_filesystem_get_rich_fallback(tmp_path):
    """Test _filesystem_get uses TQDM or Rich."""
    from unittest.mock import MagicMock, patch

    import eegdash.downloader as downloader

    mock_fs = MagicMock()
    dest = tmp_path / "test.txt"

    # Force NO Rich
    with patch("eegdash.downloader.Console") as mock_console:
        mock_console.return_value.is_terminal = False
        with patch("eegdash.downloader.TqdmCallback") as mock_tqdm:
            downloader._filesystem_get(mock_fs, "s3://b/f", dest)
            mock_tqdm.assert_called()


def test_download_files_skip_existing_check_explicit(tmp_path):
    """Test download_files checks remote size for skipping."""
    from unittest.mock import MagicMock, patch

    import eegdash.downloader as downloader

    mock_fs = MagicMock()
    dest = tmp_path / "existing.txt"
    dest.write_text("content")

    # Remote size = local size -> skip
    with patch("eegdash.downloader._remote_size", return_value=7):
        downloader.download_files(
            [("s3://b/f", dest)], filesystem=mock_fs, skip_existing=True
        )
        # Should NOT call get
        mock_fs.get.assert_not_called()

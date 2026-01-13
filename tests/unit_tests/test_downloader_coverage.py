from unittest.mock import MagicMock, patch

import pytest

from eegdash.downloader import RichCallback, download_s3_file


def test_download_s3_file_size_mismatch(tmp_path):
    local_file = tmp_path / "test.txt"
    mock_fs = MagicMock()
    # Mock remote size check returning 100
    mock_fs.info.return_value = {"size": 100}

    # Mock get creating a file of size 50 (mismatch)
    def side_effect_get(s3path, filepath, callback=None):
        with open(filepath, "wb") as f:
            f.write(b"x" * 50)

    mock_fs.get.side_effect = side_effect_get

    with pytest.raises(OSError, match="Incomplete download"):
        download_s3_file("s3://bucket/test.txt", local_file, filesystem=mock_fs)


def test_rich_callback():
    # Smoke test for RichCallback logic
    # We patch rich.progress to avoid actual console output during tests
    with patch("rich.progress.Progress") as mock_progress:
        cb = RichCallback(size=100, description="Test")
        cb.set_size(200)
        cb.relative_update(10)
        cb.close()

        mock_progress.return_value.add_task.assert_called()
        mock_progress.return_value.start.assert_called()
        mock_progress.return_value.stop.assert_called()


def test_remote_size_helper():
    from eegdash.downloader import _remote_size

    mock_fs = MagicMock()
    mock_fs.info.return_value = {"size": 123}
    assert _remote_size(mock_fs, "path") == 123

    mock_fs.info.side_effect = Exception("error")
    assert _remote_size(mock_fs, "path") is None

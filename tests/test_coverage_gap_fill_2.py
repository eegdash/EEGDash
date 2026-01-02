import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import s3fs

from eegdash import bids_metadata, downloader, paths, schemas
from eegdash.dataset.base import EEGDashRaw
from eegdash.http_api_client import get_client

# --- Downloader Tests ---


def test_downloader_get_s3_util():
    fs = downloader.get_s3_filesystem()
    assert isinstance(fs, s3fs.S3FileSystem)
    assert downloader.get_s3path("s3://bucket", "file") == "s3://bucket/file"
    assert downloader.get_s3path("s3://bucket", "/file") == "s3://bucket/file"
    assert downloader.get_s3path("s3://bucket", "") == "s3://bucket"


def test_downloader_remote_size_errors():
    mock_fs = MagicMock()
    mock_fs.info.side_effect = Exception("S3 Error")
    assert downloader._remote_size(mock_fs, "s3://b/f") is None

    mock_fs.info.side_effect = None
    mock_fs.info.return_value = {}  # No size key
    assert downloader._remote_size(mock_fs, "s3://b/f") is None

    mock_fs.info.return_value = {"Size": "not_int"}
    assert downloader._remote_size(mock_fs, "s3://b/f") is None


def test_downloader_download_s3_file_exists_match(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("content")

    mock_fs = MagicMock()
    # Remote size matches local (7 bytes)
    mock_fs.info.return_value = {"size": 7}

    res = downloader.download_s3_file("s3://b/test.txt", f, filesystem=mock_fs)
    assert res == f
    mock_fs.get.assert_not_called()


def test_downloader_download_s3_file_incomplete(tmp_path):
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


# --- BIDS Metadata Tests ---


def test_bids_meta_get_entity():
    rec_v1 = {"subject": "01"}
    assert bids_metadata.get_entity_from_record(rec_v1, "subject") == "01"

    rec_v2 = {"entities": {"subject": "02"}}
    assert bids_metadata.get_entity_from_record(rec_v2, "subject") == "02"

    # Priority
    rec_mix = {"subject": "01", "entities": {"subject": "02"}}
    assert bids_metadata.get_entity_from_record(rec_mix, "subject") == "02"


def test_bids_meta_build_query_errors():
    with pytest.raises(ValueError, match="Unsupported query"):
        bids_metadata.build_query_from_kwargs(invalid_field="x")

    with pytest.raises(ValueError, match="Received None"):
        bids_metadata.build_query_from_kwargs(subject=None)

    with pytest.raises(ValueError, match="empty list"):
        bids_metadata.build_query_from_kwargs(subject=[])

    with pytest.raises(ValueError, match="empty string"):
        bids_metadata.build_query_from_kwargs(subject="")


def test_bids_meta_merge_query_conflict():
    q1 = {"subject": "01"}
    # build_query_from_kwargs converts scalar to scalar, not $in for single value
    # But check_constraint logic handles scalar vs $in

    with pytest.raises(ValueError, match="Conflicting"):
        bids_metadata.merge_query(q1, subject="02")


def test_bids_meta_participants_tsv(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()

    # Missing file
    assert bids_metadata.participants_row_for_subject(d, "01") is None

    # Empty file
    tsv = d / "participants.tsv"
    tsv.touch()
    assert (
        bids_metadata.participants_row_for_subject(d, "01") is None
    )  # read_csv might fail or return empty df

    # Valid file
    tsv.write_text("participant_id\tage\nsub-01\t20\nsub-02\t30")
    row = bids_metadata.participants_row_for_subject(d, "01")
    assert row is not None
    assert row["age"] == "20"

    row2 = bids_metadata.participants_row_for_subject(d, "99")
    assert row2 is None


def test_bids_meta_attach_exceptions():
    # Pass garbage to trigger exceptions
    bids_metadata.attach_participants_extras("not_raw", {}, {"a": 1})
    # Should not raise


# --- Base (EEGDashRaw) Tests ---


def test_base_ensure_raw_failure(tmp_path):
    record = {
        "dataset": "ds",
        "bids_relpath": "f.set",
        "storage": {"base": str(tmp_path), "backend": "local"},
        # Missing ntimes/sfreq to force load
    }
    # Create file so exists check passes
    d = tmp_path / "ds"
    d.mkdir()
    (d / "f.set").touch()

    with patch("eegdash.dataset.base.validate_record", return_value=[]):
        ds = EEGDashRaw(record, str(tmp_path))

        # Mock download success
        with patch.object(ds, "_download_required_files"):
            # Mock load failure
            with patch("mne_bids.read_raw_bids", side_effect=ValueError("Bad file")):
                with pytest.raises(ValueError):
                    ds._ensure_raw()

                # Length check
                # Should return 0 and log warning
                assert len(ds) == 0


def test_base_len_from_metadata(tmp_path):
    record = {
        "dataset": "ds",
        "bids_relpath": "f.set",
        "storage": {"base": str(tmp_path), "backend": "local"},
        "ntimes": 100,
        "sampling_frequency": 10,
    }
    with patch("eegdash.dataset.base.validate_record", return_value=[]):
        ds = EEGDashRaw(record, str(tmp_path))
        assert len(ds) == 1000


# --- HTTP API Client Tests ---


def test_api_client_session_auth():
    with patch.dict(os.environ, {"EEGDASH_ADMIN_TOKEN": "adm"}):
        client = get_client(auth_token="usr")
        assert client._session.headers["Authorization"] == "Bearer usr"
        assert client._session.headers["X-EEGDASH-TOKEN"] == "adm"


def test_api_pagination():
    client = get_client()
    mock_resp_1 = MagicMock()
    mock_resp_1.json.return_value = {"data": [{"id": i} for i in range(1000)]}
    mock_resp_2 = MagicMock()
    mock_resp_2.json.return_value = {"data": [{"id": 1001}]}

    client._session.get = MagicMock(side_effect=[mock_resp_1, mock_resp_2])

    res = client.find(query={})
    assert len(res) == 1001
    assert client._session.get.call_count == 2


def test_api_methods():
    client = get_client()
    client._session.post = MagicMock()
    client._session.patch = MagicMock()

    client.insert_one({"a": 1})
    client.insert_many([{"a": 1}])
    client.update_many({"a": 1}, {"b": 2})

    assert client._session.post.call_count == 2
    assert client._session.patch.call_count == 1


# --- Schemas Tests ---


def test_sanitize_run():
    assert schemas._sanitize_run_for_mne(1) == "1"
    assert schemas._sanitize_run_for_mne("01") == "01"
    assert schemas._sanitize_run_for_mne("run") is None  # Not digit
    assert schemas._sanitize_run_for_mne(None) is None


def test_validate_record():
    assert "missing: dataset" in schemas.validate_record({})
    assert "missing: storage" in schemas.validate_record(
        {"dataset": "d", "bids_relpath": "p", "bidspath": "p"}
    )
    # Need non-empty storage to bypass "missing: storage" check
    assert "missing: storage.base" in schemas.validate_record(
        {
            "dataset": "d",
            "bids_relpath": "p",
            "bidspath": "p",
            "storage": {"backend": "local"},
        }
    )


def test_create_record_validation():
    with pytest.raises(ValueError):
        schemas.create_record(dataset="", storage_base="b", bids_relpath="p")

    rec = schemas.create_record(dataset="d", storage_base="b", bids_relpath="p", run=1)
    assert rec["entities_mne"]["run"] == "1"


# --- Paths Tests ---


def test_paths_resolution(tmp_path):
    # 1. Env var
    with patch.dict(os.environ, {"EEGDASH_CACHE_DIR": str(tmp_path / "env")}):
        assert paths.get_default_cache_dir() == tmp_path / "env"

    # 2. Local fallback (assuming cwd mock hard, but we can verify created logical path)
    # We can mock cwd
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        with patch.dict(os.environ, {}, clear=True):
            # Ensure hidden dir fallback
            assert paths.get_default_cache_dir() == tmp_path / ".eegdash_cache"

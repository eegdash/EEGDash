"""Tests for the source-listing adapters (HTTP-based + filesystem)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
import tenacity

from _file_utils import (
    list_datarn_files,
    list_figshare_files,
    list_git_files,
    list_local_bids_files,
    list_osf_files,
    list_scidb_files,
    list_zenodo_files,
)

# ─── Figshare ─────────────────────────────────────────────────────────────


@respx.mock
def test_figshare_happy_path():
    """Standard Figshare response → files with name/size/url."""
    respx.get("https://api.figshare.com/v2/articles/12345/files").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "name": "sub-01_eeg.edf",
                    "size": 1048576,
                    "download_url": "https://figshare.com/ndownloader/files/1",
                },
                {
                    "name": "sub-02_eeg.edf",
                    "size": 2097152,
                    "download_url": "https://figshare.com/ndownloader/files/2",
                },
            ],
        )
    )
    files = list_figshare_files(12345)
    assert len(files) == 2
    assert files[0]["name"] == "sub-01_eeg.edf"
    assert files[0]["size"] == 1048576
    assert files[0]["download_url"].startswith("https://figshare.com/")


@respx.mock
def test_figshare_empty_response():
    """Zero files → empty list."""
    respx.get("https://api.figshare.com/v2/articles/99999/files").mock(
        return_value=httpx.Response(200, json=[])
    )
    assert list_figshare_files(99999) == []


@respx.mock
def test_figshare_http_error_returns_empty():
    """HTTP 4xx → empty list, no exception."""
    respx.get("https://api.figshare.com/v2/articles/404/files").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    assert list_figshare_files(404) == []


@pytest.mark.slow
@respx.mock
def test_figshare_5xx_triggers_retries_then_propagates():
    """5xx exhausts tenacity retries and raises RetryError (unlike 4xx, which returns empty)."""
    respx.get("https://api.figshare.com/v2/articles/500/files").mock(
        return_value=httpx.Response(500, text="server error")
    )
    with pytest.raises(tenacity.RetryError):
        list_figshare_files(500)


@respx.mock
def test_figshare_malformed_json_returns_empty():
    respx.get("https://api.figshare.com/v2/articles/666/files").mock(
        return_value=httpx.Response(200, text="not json at all")
    )
    assert list_figshare_files(666) == []


@respx.mock
def test_figshare_uses_api_key_header():
    """api_key is sent as an Authorization header."""
    route = respx.get("https://api.figshare.com/v2/articles/777/files").mock(
        return_value=httpx.Response(200, json=[])
    )
    list_figshare_files(777, api_key="secret")
    assert route.calls.last is not None
    headers = route.calls.last.request.headers
    assert "Authorization" in headers
    assert "secret" in headers["Authorization"]


# ─── Zenodo ───────────────────────────────────────────────────────────────


@respx.mock
def test_zenodo_happy_path_uses_checksum_field():
    """Zenodo `checksum` field (md5:...) is surfaced by the adapter (regression)."""
    respx.get("https://zenodo.org/api/records/12345").mock(
        return_value=httpx.Response(
            200,
            json={
                "files": [
                    {
                        "key": "sub-01_eeg.edf",
                        "size": 1024,
                        "checksum": "md5:abcdef123",
                        "links": {
                            "self": "https://zenodo.org/api/records/12345/files/sub-01_eeg.edf"
                        },
                    }
                ],
                "links": {"bucket": "https://zenodo.org/api/files/bucket-12345"},
            },
        )
    )
    files = list_zenodo_files(12345)
    assert len(files) == 1
    assert files[0]["name"] == "sub-01_eeg.edf"
    assert files[0]["size"] == 1024
    assert files[0].get("checksum") == "md5:abcdef123"


@respx.mock
def test_zenodo_falls_back_to_bucket_url_when_no_links_self():
    """Old Zenodo API shape (no file-level links.self) → URL built from record-level links.bucket."""
    respx.get("https://zenodo.org/api/records/12345").mock(
        return_value=httpx.Response(
            200,
            json={
                "files": [{"key": "old_file.edf", "size": 512, "checksum": "md5:xyz"}],
                "links": {"bucket": "https://zenodo.org/api/files/bucket-12345"},
            },
        )
    )
    files = list_zenodo_files(12345)
    assert len(files) == 1
    assert (
        files[0]
        .get("download_url", "")
        .startswith("https://zenodo.org/api/files/bucket-12345")
    )


@respx.mock
def test_zenodo_404_returns_empty():
    respx.get("https://zenodo.org/api/records/99999").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    assert list_zenodo_files(99999) == []


@respx.mock
def test_zenodo_no_files_field_returns_empty():
    """Zenodo record with metadata but no files key → empty list."""
    respx.get("https://zenodo.org/api/records/55555").mock(
        return_value=httpx.Response(200, json={"links": {}})
    )
    assert list_zenodo_files(55555) == []


# ─── OSF ──────────────────────────────────────────────────────────────────


@respx.mock
def test_osf_walks_files_recursively():
    """OSF adapter walks folders recursively (single file at root)."""
    respx.get("https://api.osf.io/v2/nodes/abc123/files/osfstorage/").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "attributes": {
                            "kind": "file",
                            "name": "sub-01_eeg.edf",
                            "size": 4096,
                            "materialized_path": "/sub-01_eeg.edf",
                        },
                        "links": {"download": "https://files.osf.io/v1/abc/download"},
                    }
                ],
                "links": {},  # no pagination
            },
        )
    )
    files = list_osf_files("abc123")
    assert len(files) == 1
    assert files[0]["name"] == "sub-01_eeg.edf"
    assert "download_url" in files[0]


@respx.mock
def test_osf_404_returns_empty():
    respx.get("https://api.osf.io/v2/nodes/missing/files/osfstorage/").mock(
        return_value=httpx.Response(404, json={"errors": []})
    )
    assert list_osf_files("missing") == []


# ─── Cross-source invariants ──────────────────────────────────────────────


@respx.mock
def test_all_adapters_emit_name_and_size_for_basic_responses():
    """ADR 0001: every adapter emits at least name + size on a happy-path response."""
    # Figshare
    respx.get("https://api.figshare.com/v2/articles/1/files").mock(
        return_value=httpx.Response(200, json=[{"name": "f.edf", "size": 1}])
    )
    # Zenodo
    respx.get("https://zenodo.org/api/records/2").mock(
        return_value=httpx.Response(
            200,
            json={
                "files": [{"key": "z.edf", "size": 2}],
                "links": {"bucket": "https://zenodo.org/api/files/b"},
            },
        )
    )
    # OSF
    respx.get("https://api.osf.io/v2/nodes/x/files/osfstorage/").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "attributes": {
                            "kind": "file",
                            "name": "o.edf",
                            "size": 3,
                            "materialized_path": "/o.edf",
                        },
                        "links": {"download": "https://files.osf.io/v1/x/dl"},
                    }
                ]
            },
        )
    )

    figshare = list_figshare_files(1)
    zenodo = list_zenodo_files(2)
    osf = list_osf_files("x")

    for files, expected_size in (
        (figshare, 1),
        (zenodo, 2),
        (osf, 3),
    ):
        assert files
        assert "name" in files[0]
        assert "size" in files[0]
        assert files[0]["size"] == expected_size


@respx.mock
def test_all_adapters_return_empty_list_on_http_404():
    """ADR 0001: HTTP errors yield empty list, not exception."""
    routes = [
        ("https://api.figshare.com/v2/articles/404/files", 404),
        ("https://zenodo.org/api/records/404", 404),
        ("https://api.osf.io/v2/nodes/404/files/osfstorage/", 404),
    ]
    for url, status in routes:
        respx.get(url).mock(return_value=httpx.Response(status))
    assert list_figshare_files(404) == []
    assert list_zenodo_files(404) == []
    assert list_osf_files("404") == []


@pytest.mark.slow
@respx.mock
def test_all_adapters_tolerate_network_failure():
    """httpx.ConnectError → empty list for all adapters.

    Slow: production backoff_factor=1.0 means 3 retries × 3 adapters ≈ 9 s.
    """
    routes = [
        "https://api.figshare.com/v2/articles/dead/files",
        "https://zenodo.org/api/records/dead",
        "https://api.osf.io/v2/nodes/dead/files/osfstorage/",
    ]
    for url in routes:
        respx.get(url).mock(side_effect=httpx.ConnectError("simulated network down"))

    assert list_figshare_files("dead") == []
    assert list_zenodo_files("dead") == []
    assert list_osf_files("dead") == []


# ─── list_local_bids_files ────────────────────────────────────────────────


def test_local_bids_missing_path_returns_empty():
    """Non-existent directory → empty list, no exception."""
    assert list_local_bids_files("/totally/no/such/path") == []


def test_local_bids_empty_directory_returns_empty(tmp_path: Path):
    assert list_local_bids_files(tmp_path) == []


def test_local_bids_walks_subdirectories(tmp_path: Path):
    """Nested subdirectory files are included."""
    sub_dir = tmp_path / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)
    (sub_dir / "sub-01_eeg.edf").write_bytes(b"x" * 100)
    (sub_dir / "sub-01_channels.tsv").write_text("name\nCz\n")
    (tmp_path / "dataset_description.json").write_text("{}")

    files = list_local_bids_files(tmp_path)
    names = {f["name"] for f in files}
    assert "dataset_description.json" in names
    assert "sub-01/eeg/sub-01_eeg.edf" in names
    assert "sub-01/eeg/sub-01_channels.tsv" in names


def test_local_bids_skips_hidden_files_and_dirs(tmp_path: Path):
    """Files/dirs starting with ``.`` (e.g. git-annex objects) are skipped."""
    (tmp_path / "real.txt").write_text("ok")
    (tmp_path / ".hidden_file").write_text("skipped")
    hidden_dir = tmp_path / ".git"
    hidden_dir.mkdir()
    (hidden_dir / "config").write_text("skipped")

    files = list_local_bids_files(tmp_path)
    names = {f["name"] for f in files}
    assert "real.txt" in names
    assert ".hidden_file" not in names
    assert not any(n.startswith(".git/") for n in names)


def test_local_bids_reports_file_sizes(tmp_path: Path):
    """Each file entry includes real size in bytes."""
    (tmp_path / "small.txt").write_bytes(b"abc")
    (tmp_path / "larger.bin").write_bytes(b"x" * 1024)

    files = list_local_bids_files(tmp_path)
    by_name = {f["name"]: f["size"] for f in files}
    assert by_name["small.txt"] == 3
    assert by_name["larger.bin"] == 1024


def test_local_bids_accepts_string_path(tmp_path: Path):
    """Path and str arguments produce identical results."""
    (tmp_path / "file.txt").touch()
    out_str = list_local_bids_files(str(tmp_path))
    out_path = list_local_bids_files(tmp_path)
    assert out_str == out_path


# ─── list_scidb_files (HTTP tree-recursing API) ───────────────────────────


_SCIDB_URL = (
    "https://www.scidb.cn/api/gin-sdb-filetree/public/file/childrenFileListByPath"
)


@respx.mock
def test_scidb_happy_path_flat_directory():
    """SciDB success marker is code=20000; type flag is `dir` (not isDirectory)."""
    respx.post(_SCIDB_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 20000,
                "data": [
                    {
                        "path": "/V1/sub-01_eeg.edf",
                        "size": 1024,
                        "dir": False,
                        "md5": "abc123",
                    },
                    {
                        "path": "/V1/README",
                        "size": 100,
                        "dir": False,
                    },
                ],
            },
        )
    )
    files = list_scidb_files("test-dataset-uuid")
    assert len(files) == 2
    for f in files:
        assert "name" in f
        assert "size" in f
        assert "md5" in f


@respx.mock
def test_scidb_404_returns_empty():
    respx.post(_SCIDB_URL).mock(return_value=httpx.Response(404))
    assert list_scidb_files("missing-uuid") == []


@respx.mock
def test_scidb_empty_response_returns_empty():
    respx.post(_SCIDB_URL).mock(
        return_value=httpx.Response(200, json={"code": 20000, "data": []})
    )
    assert list_scidb_files("empty-dataset") == []


@respx.mock
def test_scidb_non_success_code_returns_empty():
    """Non-20000 response code (even with HTTP 200) → empty list."""
    respx.post(_SCIDB_URL).mock(
        return_value=httpx.Response(
            200,
            json={"code": 50000, "message": "internal error", "data": []},
        )
    )
    assert list_scidb_files("err-dataset") == []


# ─── list_datarn_files (WebDAV PROPFIND) ──────────────────────────────────


@pytest.mark.parametrize(
    ("url", "mock_response"),
    [
        pytest.param(
            "https://data.ru.nl/study/12345",
            httpx.Response(200, text="<html><body>no json-ld here</body></html>"),
            id="no_ld_json",
        ),
        pytest.param(
            "https://data.ru.nl/study/missing",
            httpx.Response(404),
            id="source_url_404",
        ),
        pytest.param(
            "https://data.ru.nl/study/55555",
            httpx.Response(
                200,
                text=(
                    '<html><script type="application/ld+json">'
                    '{"distribution": {"@type": "DataDownload"}}'
                    "</script></html>"
                ),
            ),
            id="no_webdav_url_in_distribution",
        ),
    ],
)
@respx.mock
def test_datarn_returns_empty(url: str, mock_response: httpx.Response):
    """Unreachable source or no usable distribution → empty list."""
    respx.get(url).mock(return_value=mock_response)
    assert list_datarn_files(url) == []


# ─── list_git_files (filesystem walk) ─────────────────────────────────────


def test_list_git_files_empty_directory(tmp_path: Path):
    assert list_git_files(tmp_path) == []


def test_list_git_files_walks_bids_tree(tmp_path: Path):
    """BIDS tree is enumerated as a flat list with relative paths."""
    sub_dir = tmp_path / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)
    (sub_dir / "sub-01_eeg.edf").write_bytes(b"x" * 50)
    (sub_dir / "sub-01_channels.tsv").write_text("name\nCz\n")
    (tmp_path / "dataset_description.json").write_text("{}")

    files = list_git_files(tmp_path)
    assert len(files) >= 3
    for f in files:
        assert "path" in f or "name" in f


def test_list_git_files_handles_nonexistent_directory():
    """Missing directory → [] or no crash."""
    result = list_git_files(Path("/totally/no/such/path"))
    assert isinstance(result, list)


def test_list_git_files_emits_broken_symlinks(tmp_path: Path):
    """Dangling git-annex symlinks are emitted (size=0), not silently dropped."""
    real = tmp_path / "real.edf"
    real.write_bytes(b"x" * 10)
    (tmp_path / "alive.edf").symlink_to(real)
    (tmp_path / "dead.edf").symlink_to(tmp_path / "no_such_target")

    names = {f["name"] for f in list_git_files(tmp_path)}
    assert "real.edf" in names
    assert "alive.edf" in names
    assert "dead.edf" in names


def test_list_git_files_skips_dot_git_without_descending(tmp_path: Path):
    """.git is pruned at dirent level — descending into it inflates manifests (regression guard)."""
    git_dir = tmp_path / ".git" / "objects" / "ab"
    git_dir.mkdir(parents=True)
    (git_dir / "deadbeef").write_text("blob")
    (tmp_path / "real.tsv").write_text("x")

    names = {f["name"] for f in list_git_files(tmp_path)}
    assert names == {"real.tsv"}

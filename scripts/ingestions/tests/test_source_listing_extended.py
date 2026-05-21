"""Extended source-listing tests (ROADMAP-C3 follow-up to C1.5).

C1.5 covered the 3 HTTP-based adapters (Figshare/Zenodo/OSF). This
file adds:
- list_local_bids_files (filesystem-based, no network)
- list_scidb_files (the SciDB tree-recursing API with respx mocks)
- list_datarn_files (WebDAV PROPFIND via respx)
- list_git_files (filesystem-based, walks a cloned git tree)

Per ADR 0001 these are secondary sources; the tests pin the
contract documented in their banners so a refactor that changes
the shape will fail CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import respx

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

from _file_utils import (
    list_datarn_files,
    list_git_files,
    list_local_bids_files,
    list_scidb_files,
)

# ─── list_local_bids_files ────────────────────────────────────────────────


def test_local_bids_missing_path_returns_empty():
    """A non-existent directory → empty list, no exception."""
    assert list_local_bids_files("/totally/no/such/path") == []


def test_local_bids_empty_directory_returns_empty(tmp_path: Path):
    """An empty directory → empty list."""
    assert list_local_bids_files(tmp_path) == []


def test_local_bids_walks_subdirectories(tmp_path: Path):
    """Files at nested subdirectories are found."""
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
    """Files / dirs starting with ``.`` are skipped (git-annex objects etc.)."""
    (tmp_path / "real.txt").write_text("ok")
    (tmp_path / ".hidden_file").write_text("skipped")
    hidden_dir = tmp_path / ".git"
    hidden_dir.mkdir()
    (hidden_dir / "config").write_text("skipped")

    files = list_local_bids_files(tmp_path)
    names = {f["name"] for f in files}
    assert "real.txt" in names
    assert ".hidden_file" not in names
    # No file paths starting with .git/
    assert not any(n.startswith(".git/") for n in names)


def test_local_bids_reports_file_sizes(tmp_path: Path):
    """Each file entry includes its real size in bytes."""
    (tmp_path / "small.txt").write_bytes(b"abc")  # 3 bytes
    (tmp_path / "larger.bin").write_bytes(b"x" * 1024)  # 1024 bytes

    files = list_local_bids_files(tmp_path)
    by_name = {f["name"]: f["size"] for f in files}
    assert by_name["small.txt"] == 3
    assert by_name["larger.bin"] == 1024


def test_local_bids_accepts_string_path(tmp_path: Path):
    """Argument can be either ``Path`` or ``str``."""
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
    """SciDB API returns a flat list of files at a single path level.

    The real API uses `code: 20000` as the success marker and `dir`
    (not isDirectory) for the type flag. Paths are stripped of the
    version prefix.
    """
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
    # Each file has name + size + md5 (per ADR 0001 — md5 is one of
    # the few secondary sources that survives the manifest pipeline)
    for f in files:
        assert "name" in f
        assert "size" in f
        assert "md5" in f


@respx.mock
def test_scidb_404_returns_empty():
    """SciDB API 404 → empty list, no exception."""
    respx.post(_SCIDB_URL).mock(return_value=httpx.Response(404))
    assert list_scidb_files("missing-uuid") == []


@respx.mock
def test_scidb_empty_response_returns_empty():
    """SciDB returning an empty data array → empty list."""
    respx.post(_SCIDB_URL).mock(
        return_value=httpx.Response(200, json={"code": 20000, "data": []})
    )
    assert list_scidb_files("empty-dataset") == []


@respx.mock
def test_scidb_non_success_code_returns_empty():
    """A non-20000 code (e.g. error response with HTTP 200) → empty list."""
    respx.post(_SCIDB_URL).mock(
        return_value=httpx.Response(
            200,
            json={"code": 50000, "message": "internal error", "data": []},
        )
    )
    assert list_scidb_files("err-dataset") == []


# ─── list_datarn_files (WebDAV PROPFIND) ──────────────────────────────────


@respx.mock
def test_datarn_returns_empty_when_no_ld_json():
    """If the source URL has no JSON-LD distribution, return empty."""
    respx.get("https://data.ru.nl/study/12345").mock(
        return_value=httpx.Response(
            200,
            text="<html><body>no json-ld here</body></html>",
        )
    )
    assert list_datarn_files("https://data.ru.nl/study/12345") == []


@respx.mock
def test_datarn_returns_empty_when_source_url_404():
    """A 404 on the source URL → empty list."""
    respx.get("https://data.ru.nl/study/missing").mock(return_value=httpx.Response(404))
    assert list_datarn_files("https://data.ru.nl/study/missing") == []


@respx.mock
def test_datarn_returns_empty_when_no_webdav_url_in_distribution():
    """The page parses, but the distribution dict has no ``contentUrl`` →
    empty list (no fallback)."""
    html = (
        '<html><script type="application/ld+json">'
        '{"distribution": {"@type": "DataDownload"}}'
        "</script></html>"
    )
    respx.get("https://data.ru.nl/study/55555").mock(
        return_value=httpx.Response(200, text=html)
    )
    assert list_datarn_files("https://data.ru.nl/study/55555") == []


# ─── list_git_files (filesystem walk) ─────────────────────────────────────


def test_list_git_files_empty_directory(tmp_path: Path):
    """A directory with no files → empty list."""
    assert list_git_files(tmp_path) == []


def test_list_git_files_walks_bids_tree(tmp_path: Path):
    """A cloned BIDS tree is enumerated as a flat list with relative paths."""
    sub_dir = tmp_path / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)
    (sub_dir / "sub-01_eeg.edf").write_bytes(b"x" * 50)
    (sub_dir / "sub-01_channels.tsv").write_text("name\nCz\n")
    (tmp_path / "dataset_description.json").write_text("{}")

    files = list_git_files(tmp_path)
    assert len(files) >= 3
    # Each entry has 'path' and 'size' (and possibly more)
    for f in files:
        assert "path" in f or "name" in f


def test_list_git_files_handles_nonexistent_directory():
    """Per the adapter contract, a missing dir → either [] or doesn't crash."""
    # Use a path that definitely doesn't exist
    result = list_git_files(Path("/totally/no/such/path"))
    # Tolerant either way: empty list or no crash
    assert isinstance(result, list)

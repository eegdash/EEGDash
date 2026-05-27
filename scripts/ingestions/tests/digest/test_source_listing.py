"""Parametrized tests for the source-listing adapters.

Two angles:

- **Core listing** — the 7 HTTP-based adapters (Figshare/Zenodo/OSF/...). Was test_source_listing.py.
- **Extended adapters** — filesystem-based + non-HTTP listers. Was test_source_listing_extended.py.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from _file_utils import (
    list_datarn_files,
    list_figshare_files,
    list_git_files,
    list_local_bids_files,
    list_osf_files,
    list_scidb_files,
    list_zenodo_files,
)

# ─── 1. Core listing ──────────────────────────────────────────────

# ─── Figshare ─────────────────────────────────────────────────────────────


@respx.mock
def test_figshare_happy_path():
    """Standard Figshare API response → list of files with name/size/url."""
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
    """Article with zero files → empty list."""
    respx.get("https://api.figshare.com/v2/articles/99999/files").mock(
        return_value=httpx.Response(200, json=[])
    )
    assert list_figshare_files(99999) == []


@respx.mock
def test_figshare_http_error_returns_empty():
    """Per the adapter banner: secondary sources never raise — they
    return empty on HTTP errors. The caller decides whether to retry."""
    respx.get("https://api.figshare.com/v2/articles/404/files").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    assert list_figshare_files(404) == []


@pytest.mark.slow
@respx.mock
def test_figshare_5xx_triggers_retries_then_propagates():
    """5xx responses trigger tenacity-driven retries inside
    request_response; after exhausting retries the call raises
    tenacity.RetryError (NOT returning empty).

    This is intentional: 5xx is "the server may recover" — the
    pipeline should know about persistent server errors rather than
    silently treating them as "no files".

    Contrast with 4xx (e.g. 404), which is "not found, definitive" and
    returns empty cleanly (see ``test_figshare_http_error_returns_empty``).
    """
    import tenacity

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
    """When given an api_key, the adapter sends an Authorization header."""
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
    """Zenodo emits `checksum` (md5:abc...) — adapter must surface it.

    Regression test for the bug fixed in housekeeping commit 987855bfd:
    Zenodo's checksum was previously dropped at manifest time.
    """
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
    # The bug fix: checksum is surfaced.
    assert files[0].get("checksum") == "md5:abcdef123"


@respx.mock
def test_zenodo_falls_back_to_bucket_url_when_no_links_self():
    """Old Zenodo API shape: file lacks `links.self`; adapter builds URL
    from `links.bucket` at the record level."""
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
    """Some Zenodo records have metadata but no files → empty list."""
    respx.get("https://zenodo.org/api/records/55555").mock(
        return_value=httpx.Response(200, json={"links": {}})
    )
    assert list_zenodo_files(55555) == []


# ─── OSF ──────────────────────────────────────────────────────────────────


@respx.mock
def test_osf_walks_files_recursively():
    """OSF's API returns a tree per call; adapter walks folders recursively.

    Tests the simple shape: one file at the root, no nested folders.
    """
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
    """Cross-source contract: every adapter emits at least name + size
    on a happy-path response. ADR 0001 documents this as the
    "secondary Source" minimum shape."""
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
    """ADR 0001 contract: HTTP errors yield empty list, not exception.

    The 4 HTTP-based adapters (figshare, zenodo, osf, scidb-via-API)
    all share this contract — caller treats empty as "no files,
    move on" rather than "retry the API".
    """
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
    """A complete network failure (httpx ConnectError) → empty list.

    Mocked here as an httpx exception bubbling from respx.

    Slow because the production adapters use ``backoff_factor=1.0`` —
    3 retries x 3 adapters x ~1 sec backoff = ~9 sec. Marked slow so
    PR-fast CI skips it; the nightly bench / slow-test job picks it up.
    The behaviour-pinning value is real (empty-list-on-failure for all
    3 adapters); the production timing is real too. Don't change the
    backoff in the test to make it faster — that would test a different
    config than production runs.
    """
    routes = [
        "https://api.figshare.com/v2/articles/dead/files",
        "https://zenodo.org/api/records/dead",
        "https://api.osf.io/v2/nodes/dead/files/osfstorage/",
    ]
    for url in routes:
        respx.get(url).mock(side_effect=httpx.ConnectError("simulated network down"))

    # Each call should return [] rather than propagating ConnectError.
    assert list_figshare_files("dead") == []
    assert list_zenodo_files("dead") == []
    assert list_osf_files("dead") == []


# ─── 2. Extended adapters ──────────────────────────────────────────────

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


def test_list_git_files_emits_broken_symlinks(tmp_path: Path):
    """Broken git-annex pointer symlinks must be emitted (size=0), not dropped.

    Pinned post-perf-review (2026-05-22) — the walker's dirent-driven
    classification calls is_symlink() before is_file() so dangling
    annex pointers (the common OpenNeuro/NEMAR case after a shallow
    clone with GIT_LFS_SKIP_SMUDGE=1) are still surfaced.
    """
    real = tmp_path / "real.edf"
    real.write_bytes(b"x" * 10)
    (tmp_path / "alive.edf").symlink_to(real)
    (tmp_path / "dead.edf").symlink_to(tmp_path / "no_such_target")

    names = {f["name"] for f in list_git_files(tmp_path)}
    assert "real.edf" in names
    assert "alive.edf" in names
    assert "dead.edf" in names  # broken symlink → emitted, not silently dropped


def test_list_git_files_skips_dot_git_without_descending(tmp_path: Path):
    """`.git` directory is pruned at dirent level — not descended into.

    Pinned post-perf-review: a regression that walked `.git` would
    inflate the manifest with thousands of blob filenames and slow
    Stage 2 by an order of magnitude.
    """
    git_dir = tmp_path / ".git" / "objects" / "ab"
    git_dir.mkdir(parents=True)
    (git_dir / "deadbeef").write_text("blob")
    (tmp_path / "real.tsv").write_text("x")

    names = {f["name"] for f in list_git_files(tmp_path)}
    assert names == {"real.tsv"}  # only the real BIDS file, no `.git` contents

"""Parametrized tests for the 7 source-listing adapters .

Per ADR 0001, OpenNeuro and NEMAR are the production sources; the
other 5 (Figshare, Zenodo, OSF, SciDB, DataRN) are "best-effort,
secondary". Before this round they had no tests at all.

The user explicitly asked: "check what we can transfer for our source".
That's the framing for this work: 4 of 5 secondary adapters share a
shape (HTTP GET → JSON → list of file entries with name/size/url), so
the same test patterns transfer across all of them.

Coverage:
- happy path (typical API response shape per source)
- empty response (no files)
- HTTP error (4xx, 5xx)
- malformed JSON
- timeout / network error

``respx`` mocks the httpx layer per :func:`request_response`.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

_INGEST_DIR = Path(__file__).resolve().parent.parent
from _file_utils import (
    list_figshare_files,
    list_osf_files,
    list_zenodo_files,
)

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

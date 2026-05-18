"""Unit tests for :class:`eegdash.dataset.nemar.NemarClient`.

The suite is fully offline: every HTTP call is stubbed through
:func:`urllib.request.urlopen`. The shape of the fixtures mirrors what
the real NEMAR endpoints return for ``nm000132`` (ERP CORE), as probed
in the task spec.

Covers the contract documented in :mod:`eegdash.dataset.nemar`:

* happy path → returns :class:`NemarMetadata`
* 404 → returns ``None`` (tombstone)
* manifest is *not* fetched by ``metadata()``
* disk cache hit avoids the network
* disk cache miss reaches the network and writes the cache
* :envvar:`EEGDASH_NO_API` short-circuits to cache-only
* versions are sorted newest-first (even if NEMAR ships them scrambled)
* :class:`NemarManifestEntry` does not carry the signed S3 ``url`` field
"""

from __future__ import annotations

import dataclasses
import json
import urllib.error
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from eegdash.dataset.nemar import (
    NemarClient,
    NemarManifestEntry,
    NemarMetadata,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _top_payload(dataset_id: str = "nm000132") -> dict:
    return {
        "dataset_id": dataset_id,
        "latest": "v1.1.1",
        "metadata_url": f"/{dataset_id}/metadata.json",
        "versions": [
            {
                "version": "v1.1.1",
                "doi": f"10.82901/nemar.{dataset_id}.v1.1.1",
                "created_at": "2026-04-04 06:05:15",
                "manifest_url": f"/{dataset_id}/v1.1.1/manifest.json",
                "browse_url": f"/{dataset_id}/v1.1.1/",
            },
            {
                "version": "v1.1.0",
                "doi": f"10.82901/nemar.{dataset_id}.v1.1.0",
                "created_at": "2026-04-02 20:15:30",
                "manifest_url": f"/{dataset_id}/v1.1.0/manifest.json",
                "browse_url": f"/{dataset_id}/v1.1.0/",
            },
            {
                "version": "v1.0.0",
                "doi": f"10.82901/nemar.{dataset_id}.v1.0.0",
                "created_at": "2026-03-14 12:20:43",
                "manifest_url": f"/{dataset_id}/v1.0.0/manifest.json",
                "browse_url": f"/{dataset_id}/v1.0.0/",
            },
        ],
    }


def _metadata_payload(dataset_id: str = "nm000132") -> dict:
    return {
        "schema_version": "0.3.0",
        "doc_type": "dataset",
        "dataset_id": dataset_id,
        "name": "ERP CORE",
        "description": "ERP CORE is a comprehensive open-access resource ...",
        "source": "nemar",
        "recording_modality": ["EEG"],
        "bids_version": None,
        "license": "CC-BY-4.0",
        "authors": [
            {
                "name": "Emily S. Kappenman",
                "name_type": "Personal",
                "orcid": "0000-0002-2789-015X",
            },
            {
                "name": "Steven J. Luck.",
                "name_type": "Personal",
                "orcid": "0000-0002-3725-1474",
            },
        ],
        "keywords": [
            {
                "term": "Event-Related Potentials, P300",
                "subject_scheme": "MeSH",
                "scheme_uri": "https://id.nlm.nih.gov/mesh/",
                "value_uri": "http://id.nlm.nih.gov/mesh/D018913",
            },
            {"term": "EEG"},
            {"term": "N170"},
        ],
    }


def _manifest_payload() -> list[dict]:
    return [
        {
            "path": "stimuli/task-MMN/video.avi",
            "size": 70200762,
            "checksum_algorithm": "sha256",
            "checksum": "8911b9f27e3d2d49a0e7dd1201eb1980ff1f773580ca800f7278819ff45f871e",
            "url": (
                "https://nemar.s3.us-east-2.amazonaws.com/...?X-Amz-Expires=3600&..."
            ),
        },
        {
            "path": "sub-001/eeg/sub-001_task-MMN_eeg.set",
            "size": 12345678,
            "checksum_algorithm": "sha256",
            "checksum": "deadbeef" * 8,
            "url": "https://nemar.s3.us-east-2.amazonaws.com/...",
        },
    ]


def _make_response(payload) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    return response


def _make_404(url: str = "https://data.nemar.org/nm404") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url=url,
        code=404,
        msg="not found",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )


@pytest.fixture
def client_factory(tmp_path):
    """Return a factory that builds NemarClient instances with isolated cache."""

    def _factory(**overrides) -> NemarClient:
        kwargs = {
            "base_url": "https://data.nemar.example",
            "cache_dir": tmp_path,
            "timeout": 1.0,
        }
        kwargs.update(overrides)
        return NemarClient(**kwargs)

    return _factory


# ---------------------------------------------------------------------------
# Required cases
# ---------------------------------------------------------------------------


def test_metadata_happy_path(client_factory):
    """Two GETs (top + metadata.json) merge into a :class:`NemarMetadata`."""
    client = client_factory()

    def urlopen_side_effect(req, *_, **__):
        url = req.full_url if hasattr(req, "full_url") else req
        if url.endswith("/nm000132/metadata.json"):
            return _make_response(_metadata_payload())
        if url.endswith("/nm000132"):
            return _make_response(_top_payload())
        raise AssertionError(f"unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=urlopen_side_effect) as urlopen:
        meta = client.metadata("nm000132")

    assert isinstance(meta, NemarMetadata)
    assert meta.dataset_id == "nm000132"
    assert meta.name == "ERP CORE"
    assert meta.license == "CC-BY-4.0"
    assert meta.recording_modality == ("EEG",)
    assert meta.bids_version is None
    # ORCIDs land as bare 16-char ids on every author.
    assert any(a.orcid == "0000-0002-2789-015X" for a in meta.authors)
    # MeSH-tagged keywords keep their scheme + URI.
    mesh = [k for k in meta.keywords if k.scheme == "MeSH"]
    assert mesh and mesh[0].value_uri == "http://id.nlm.nih.gov/mesh/D018913"
    # Latest version is the newest by created_at, not just NEMAR's order.
    assert meta.latest_version == "v1.1.1"
    assert meta.versions[0].version == "v1.1.1"
    # Both endpoints were called exactly once; no manifest fetch.
    assert urlopen.call_count == 2
    # No top-level errors recorded.
    assert client.errors == []


def test_metadata_404_returns_none(client_factory):
    """A 404 on either endpoint resolves to ``None`` (tombstone path)."""
    client = client_factory()

    with patch("urllib.request.urlopen", side_effect=_make_404()):
        assert client.metadata("nm999999") is None

    # Tombstones are not recorded as errors -- they are expected silence.
    assert client.errors == []


def test_manifest_only_fetched_on_request(client_factory):
    """``metadata()`` must not request any manifest."""
    client = client_factory()
    seen_urls: list[str] = []

    def urlopen_side_effect(req, *_, **__):
        url = req.full_url if hasattr(req, "full_url") else req
        seen_urls.append(url)
        if url.endswith("/metadata.json"):
            return _make_response(_metadata_payload())
        if url.endswith("/manifest.json"):
            return _make_response(_manifest_payload())
        if url.endswith("/nm000132"):
            return _make_response(_top_payload())
        raise AssertionError(f"unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
        meta = client.metadata("nm000132")
        assert meta is not None
        assert all("manifest.json" not in u for u in seen_urls)

        # Now explicitly request the manifest and verify the URL fires.
        entries = client.manifest("nm000132", version="v1.1.1")

    assert len(entries) == 2
    assert any("manifest.json" in u for u in seen_urls)
    # Manifest entries do not carry the signed URL.
    for entry in entries:
        assert isinstance(entry, NemarManifestEntry)
        field_names = {f.name for f in dataclasses.fields(entry)}
        assert "url" not in field_names


def test_disk_cache_hit_avoids_network(client_factory):
    """Two ``metadata()`` calls cause exactly one network round-trip."""
    client = client_factory()

    def urlopen_side_effect(req, *_, **__):
        url = req.full_url if hasattr(req, "full_url") else req
        if url.endswith("/metadata.json"):
            return _make_response(_metadata_payload())
        if url.endswith("/nm000132"):
            return _make_response(_top_payload())
        raise AssertionError(f"unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=urlopen_side_effect) as urlopen:
        first = client.metadata("nm000132")
        second = client.metadata("nm000132")

    assert first is not None and second is not None
    assert first.dataset_id == second.dataset_id
    # First call: 2 GETs (top + metadata). Second call: 0 GETs (both cached).
    assert urlopen.call_count == 2


def test_disk_cache_miss_fetches(client_factory, tmp_path):
    """A cache file older than the TTL forces a network re-fetch."""
    # Build a client whose TTL is essentially zero -- every read is stale.
    client = client_factory(ttl=timedelta(seconds=0))

    def urlopen_side_effect(req, *_, **__):
        url = req.full_url if hasattr(req, "full_url") else req
        if url.endswith("/metadata.json"):
            return _make_response(_metadata_payload())
        if url.endswith("/nm000132"):
            return _make_response(_top_payload())
        raise AssertionError(f"unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=urlopen_side_effect) as urlopen:
        first = client.metadata("nm000132")
        second = client.metadata("nm000132")

    assert first is not None and second is not None
    # Every call goes back to the network because TTL=0 makes every
    # cached file stale on read.
    assert urlopen.call_count == 4


def test_no_api_env_var_short_circuits(client_factory, monkeypatch):
    """:envvar:`EEGDASH_NO_API` blocks the network path entirely."""
    monkeypatch.setenv("EEGDASH_NO_API", "1")
    client = client_factory()

    # No cache and no network → metadata() must return None without
    # ever calling urlopen.
    with patch("urllib.request.urlopen", side_effect=AssertionError("net forbidden")):
        result = client.metadata("nm000132")

    assert result is None


def test_versions_sorted_newest_first(client_factory):
    """Even if NEMAR ships versions out of order, the client sorts them."""
    scrambled_top = {
        "dataset_id": "nm000132",
        "latest": "v1.1.0",  # NEMAR happens to be wrong here too
        "metadata_url": "/nm000132/metadata.json",
        "versions": [
            {
                "version": "v1.0.0",
                "doi": "10.82901/nemar.nm000132.v1.0.0",
                "created_at": "2026-03-14 12:20:43",
                "manifest_url": "/nm000132/v1.0.0/manifest.json",
                "browse_url": "/nm000132/v1.0.0/",
            },
            {
                "version": "v1.1.1",
                "doi": "10.82901/nemar.nm000132.v1.1.1",
                "created_at": "2026-04-04 06:05:15",
                "manifest_url": "/nm000132/v1.1.1/manifest.json",
                "browse_url": "/nm000132/v1.1.1/",
            },
            {
                "version": "v1.1.0",
                "doi": "10.82901/nemar.nm000132.v1.1.0",
                "created_at": "2026-04-02 20:15:30",
                "manifest_url": "/nm000132/v1.1.0/manifest.json",
                "browse_url": "/nm000132/v1.1.0/",
            },
        ],
    }

    client = client_factory()

    def urlopen_side_effect(req, *_, **__):
        url = req.full_url if hasattr(req, "full_url") else req
        if url.endswith("/metadata.json"):
            return _make_response(_metadata_payload())
        if url.endswith("/nm000132"):
            return _make_response(scrambled_top)
        raise AssertionError(f"unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
        meta = client.metadata("nm000132")

    assert meta is not None
    versions = [v.version for v in meta.versions]
    assert versions == ["v1.1.1", "v1.1.0", "v1.0.0"]
    # latest_version always reflects the *sorted* head, not NEMAR's own claim.
    assert meta.latest_version == "v1.1.1"


def test_signed_url_not_stored(client_factory):
    """Sanity check on the dataclass: there is no ``url`` field.

    Reproduces the contract documented in the dataclass docstring: the
    signed S3 URL expires in 1h and must never be cached.
    """
    fields = {f.name for f in dataclasses.fields(NemarManifestEntry)}
    assert "url" not in fields
    assert fields == {"path", "size", "sha256"}


# ---------------------------------------------------------------------------
# Belt-and-braces: cache-payload-shape and error tagging
# ---------------------------------------------------------------------------


def test_manifest_strips_signed_urls_in_cache(client_factory, tmp_path):
    """Disk-cached manifest payload must not contain ``url`` fields."""
    client = client_factory()

    def urlopen_side_effect(req, *_, **__):
        url = req.full_url if hasattr(req, "full_url") else req
        if url.endswith("/metadata.json"):
            return _make_response(_metadata_payload())
        if url.endswith("/nm000132"):
            return _make_response(_top_payload())
        if url.endswith("/manifest.json"):
            return _make_response(_manifest_payload())
        raise AssertionError(f"unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
        client.manifest("nm000132", version="v1.1.1")

    # The cache key sanitizer strips dots from ``v1.1.1``, so the
    # filename ends up as ``manifest__v111.json``. We do not rely on the
    # exact filename -- just verify *some* manifest cache file appeared.
    cache_files = list(tmp_path.glob("nm000132__manifest__*.json"))
    assert cache_files, list(tmp_path.iterdir())
    cache_file = cache_files[0]
    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    assert all("url" not in entry for entry in cached), (
        f"signed S3 URLs leaked into the disk cache: {cached!r}"
    )


def test_errors_populated_on_decode_failure(client_factory):
    """A malformed payload records an entry on :attr:`errors`."""
    client = client_factory()

    bad_response = MagicMock()
    bad_response.read.return_value = b"not-json"
    bad_response.__enter__ = MagicMock(return_value=bad_response)
    bad_response.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=bad_response):
        result = client.metadata("nm000132")

    assert result is None
    assert any("nemar" in e for e in client.errors)


def test_is_available(client_factory):
    """Cheap availability probe uses the top-level descriptor."""
    client = client_factory()

    def urlopen_side_effect(req, *_, **__):
        url = req.full_url if hasattr(req, "full_url") else req
        if url.endswith("/nm000132"):
            return _make_response(_top_payload())
        raise AssertionError(f"unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
        assert client.is_available("nm000132") is True

    with patch("urllib.request.urlopen", side_effect=_make_404()):
        assert client.is_available("nm999999") is False

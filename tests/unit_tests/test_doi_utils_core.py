import socket
import urllib.error
from pathlib import Path

import pytest

from scripts.citations import doi_utils


@pytest.mark.parametrize(
    "raw,expected,valid",
    [
        ("10.1000/xyz", "10.1000/xyz", True),
        (" doi:10.1000/xyz ", "10.1000/xyz", True),
        ("https://doi.org/10.1000/xyz", "10.1000/xyz", True),
        ("http://dx.doi.org/10.1000/xyz", "10.1000/xyz", True),
        ("", None, False),
        (None, None, False),
        ("not-a-doi", "not-a-doi", False),
    ],
)
def test_normalize_and_validate_doi(raw, expected, valid):
    assert doi_utils.normalize_doi(raw) == expected
    assert doi_utils.is_valid_doi(raw) is valid


def test_resolve_doi_success_and_failure(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"title":"T","author":[{"family":"Doe"}],"issued":{"date-parts":[[2024]]}}'

    monkeypatch.setattr(doi_utils.urllib.request, "urlopen", lambda *_a, **_k: _Resp())
    ok = doi_utils.resolve_doi("10.1000/xyz")
    assert ok is not None
    assert ok["title"] == "T"

    def _raise(*_a, **_k):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(doi_utils.urllib.request, "urlopen", _raise)
    assert doi_utils.resolve_doi("10.1000/xyz") is None


def test_resolve_doi_handles_socket_timeout(monkeypatch):
    def _raise_timeout(*_a, **_k):
        raise socket.timeout("timed out")

    monkeypatch.setattr(doi_utils.urllib.request, "urlopen", _raise_timeout)
    assert doi_utils.resolve_doi("10.1000/xyz") is None


def test_resolve_doi_cached_cache_hit_and_miss(monkeypatch):
    cache = {"10.1000/xyz": {"title": "cached"}}
    assert doi_utils.resolve_doi_cached("10.1000/xyz", cache, delay=0) == {
        "title": "cached"
    }

    monkeypatch.setattr(
        doi_utils,
        "resolve_doi",
        lambda *_a, **_k: {"title": "T", "author": [], "issued": {}},
    )
    monkeypatch.setattr(doi_utils.time, "sleep", lambda *_a, **_k: None)
    out = doi_utils.resolve_doi_cached("10.1000/new", cache, delay=0)
    assert out is not None
    assert "10.1000/new" in cache


def test_compact_metadata_extracts_expected_fields():
    compact = doi_utils._compact_metadata(
        {
            "title": "Paper",
            "type": "article-journal",
            "author": [{"family": "Doe", "given": "Jane"}, {"literal": "NoName"}],
            "issued": {"date-parts": [[2023, 5, 1]]},
        }
    )
    assert compact == {
        "title": "Paper",
        "authors": [{"family": "Doe", "given": "Jane"}],
        "year": 2023,
        "type": "article-journal",
    }


def test_cache_load_and_save_roundtrip(tmp_path: Path):
    cache_path = tmp_path / "cache.json"
    data = {"10.1000/a": {"title": "A"}}
    doi_utils.save_cache(data, cache_path)
    assert doi_utils.load_cache(cache_path) == data
    assert doi_utils.load_cache(tmp_path / "missing.json") == {}


def test_extract_surnames_and_overlap():
    a = [{"family": "Doe"}, {"family": "Smith", "given": "A"}, {"given": "NoFamily"}]
    b = [{"family": "SMITH"}, {"family": "Brown"}]
    assert doi_utils.extract_surnames(a) == ["doe", "smith"]
    assert doi_utils.surnames_overlap(a, b) == {"smith"}


def test_load_dataset_dois_filters_empty_rows(tmp_path: Path):
    csv_path = tmp_path / "datasets.csv"
    csv_path.write_text(
        "dataset,doi\nds1,10.1000/abc\nds2,\nds3, https://doi.org/10.1000/def \n",
        encoding="utf-8",
    )
    pairs = doi_utils.load_dataset_dois(csv_path)
    assert pairs == [("ds1", "10.1000/abc"), ("ds3", "https://doi.org/10.1000/def")]

"""Per-subject demographics on dataset pages cover fNIRS recordings.

The page asks ``/datasets/{id}/participants`` first and pages through
``/records`` when that endpoint is unavailable. The fallback filter listed the
suffixes eeg/ieeg/emg/meg only, the same allow-list the API endpoint had, so
fNIRS datasets (records stored with suffix ``fnirs``, or ``nirs`` when built
from the BIDS file name) never got an age-by-sex chart. These tests stub the
HTTP helper, so they stay offline.
"""

from __future__ import annotations

import json
import urllib.parse

import pytest
from tests_support.dataset_page import load_dataset_page_module


@pytest.fixture
def data_loaders(monkeypatch):
    module = load_dataset_page_module("data_loaders")
    monkeypatch.delenv("EEGDASH_NO_API", raising=False)
    return module


class _FakeApi:
    """Answers ``_get_json`` calls; the participants endpoint is "not deployed"."""

    def __init__(self, records):
        self.records = records
        self.urls: list[str] = []

    def __call__(self, url, **_kwargs):
        self.urls.append(url)
        if "/participants" in url:
            return None
        return {"success": True, "data": self.records}


def _records_filter(url: str) -> dict:
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    return json.loads(query["filter"][0])


def test_fallback_filter_includes_fnirs_suffixes(data_loaders, monkeypatch):
    api = _FakeApi(
        [
            {"subject": "01", "participant_tsv": {"age": 0.5, "sex": "F"}},
            {"subject": "02", "participant_tsv": {"age": 0.75, "sex": "M"}},
        ]
    )
    monkeypatch.setattr(data_loaders, "_get_json", api)

    rows = data_loaders._fetch_participants_from_records("NM000175")

    assert rows == [
        {"subject": "01", "age": 0.5, "sex": "F"},
        {"subject": "02", "age": 0.75, "sex": "M"},
    ]
    records_urls = [u for u in api.urls if "/records" in u]
    assert records_urls, "fallback /records query was not issued"
    suffixes = _records_filter(records_urls[0])["suffix"]["$in"]
    assert {"fnirs", "nirs"} <= set(suffixes)
    # The electrophysiology suffixes stay in the filter.
    assert {"eeg", "ieeg", "emg", "meg"} <= set(suffixes)


def test_fallback_filter_matches_dataset_and_skips_missing_files(
    data_loaders, monkeypatch
):
    api = _FakeApi([])
    monkeypatch.setattr(data_loaders, "_get_json", api)

    assert data_loaders._fetch_participants_from_records("NM000175") == []

    records_filter = _records_filter([u for u in api.urls if "/records" in u][0])
    assert records_filter["dataset"] == "nm000175"
    assert records_filter["_has_missing_files"] == {"$ne": True}


def test_primary_endpoint_rows_are_used_as_returned(data_loaders, monkeypatch):
    body = {
        "success": True,
        "data": [
            {"subject": "01", "participant_tsv": {"age": 34.0, "sex": "M"}},
            {"subject": "", "participant_tsv": {"age": 20.0}},
        ],
    }
    seen: list[str] = []

    def fake_get_json(url, **_kwargs):
        seen.append(url)
        return body

    monkeypatch.setattr(data_loaders, "_get_json", fake_get_json)

    rows = data_loaders._fetch_participants_from_records("nm000175")

    assert rows == [{"subject": "01", "age": 34.0, "sex": "M"}]
    assert len(seen) == 1 and seen[0].endswith("/datasets/nm000175/participants")

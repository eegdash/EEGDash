"""prune_stale_records deletes the records a dataset's new digest no longer holds."""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import httpx
import respx
from _helpers import load_inject

API = "https://api.example.com"
URL = f"{API}/admin/eegdash_dev/records"


def _filters(route) -> list[dict]:
    return [
        json.loads(parse_qs(urlparse(str(c.request.url)).query)["filter"][0])
        for c in route.calls
    ]


@respx.mock
def test_one_delete_per_dataset_older_than_its_digest():
    inject = load_inject()
    route = respx.delete(URL).mock(
        return_value=httpx.Response(200, json={"deleted_count": 20})
    )
    records = [
        {
            "dataset": "nm000189",
            "bids_relpath": "a",
            "digested_at": "2026-08-26T11:02:13+00:00",
        },
        {
            "dataset": "nm000189",
            "bids_relpath": "b",
            "digested_at": "2026-08-26T11:02:13+00:00",
        },
        {
            "dataset": "nm000172",
            "bids_relpath": "c",
            "digested_at": "2026-08-26T11:01:00+00:00",
        },
    ]
    out = inject.prune_stale_records(records, API, "eegdash_dev", "tok")

    assert sorted(_filters(route), key=lambda f: f["dataset"]) == [
        {"dataset": "nm000172", "digested_at": {"$lt": "2026-08-26T11:01:00+00:00"}},
        {"dataset": "nm000189", "digested_at": {"$lt": "2026-08-26T11:02:13+00:00"}},
    ]
    assert out == {"deleted": {"nm000172": 20, "nm000189": 20}, "errors": []}
    assert route.calls[0].request.headers["authorization"] == "Bearer tok"


@respx.mock
def test_uses_the_oldest_timestamp_of_a_dataset():
    """A record of the current digest must never match the delete filter."""
    inject = load_inject()
    route = respx.delete(URL).mock(
        return_value=httpx.Response(200, json={"deleted_count": 0})
    )
    records = [
        {"dataset": "d", "digested_at": "2026-10-02T00:00:00+00:00"},
        {"dataset": "d", "digested_at": "2026-10-01T00:00:00+00:00"},
    ]
    out = inject.prune_stale_records(records, API, "eegdash_dev", "tok")
    assert _filters(route) == [
        {"dataset": "d", "digested_at": {"$lt": "2026-10-01T00:00:00+00:00"}}
    ]
    assert out["deleted"] == {}


@respx.mock
def test_records_without_timestamp_do_not_trigger_a_delete():
    inject = load_inject()
    route = respx.delete(URL).mock(return_value=httpx.Response(200, json={}))
    out = inject.prune_stale_records(
        [{"dataset": "d"}, {"dataset": "e", "digested_at": None}],
        API,
        "eegdash_dev",
        "tok",
    )
    assert not route.called
    assert out == {"deleted": {}, "errors": []}


def _fail_for_bad(request):
    flt = json.loads(parse_qs(urlparse(str(request.url)).query)["filter"][0])
    if flt["dataset"] == "bad":
        return httpx.Response(400, json={"detail": "nope"})
    return httpx.Response(200, json={"deleted_count": 3})


@respx.mock
def test_a_failed_delete_is_reported_and_others_continue():
    inject = load_inject()

    respx.delete(URL).mock(side_effect=_fail_for_bad)
    out = inject.prune_stale_records(
        [
            {"dataset": "bad", "digested_at": "2026-10-01"},
            {"dataset": "good", "digested_at": "2026-10-01"},
        ],
        API,
        "eegdash_dev",
        "tok",
    )
    assert out["deleted"] == {"good": 3}
    assert len(out["errors"]) == 1
    assert out["errors"][0].startswith("bad:")

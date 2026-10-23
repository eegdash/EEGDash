"""Model gateway normalization/$set and execute Stage 5's real prune filter.

No Mongo or writer endpoint is contacted: only a small in-memory API contract
and respx GET/DELETE handlers. This specifically detects post-admission loss.
"""

from __future__ import annotations

import copy
import json
from functools import partial

import httpx
import pytest
import respx

from .test_pin_guard import API, DB, RAW, mock_live, pinned_record, setup_main


def apply_gateway_set(state, records):
    """Gateway api/main.py:5000–5027 effective identity and retained fields."""
    for record in copy.deepcopy(records):
        if not record.get("bids_relpath") and record.get("bidspath"):
            record["bids_relpath"] = record["bidspath"]
        identity = (record["dataset"], record["bids_relpath"])
        state.setdefault(identity, {}).update(record)


def delete_matching_strings(state, request):
    """Evaluate the actual pruner's dataset + Mongo string $lt predicate."""
    flt = json.loads(request.url.params["filter"])
    cutoff = flt["digested_at"]["$lt"]
    removed = [
        key
        for key, row in state.items()
        if row.get("dataset") == flt["dataset"]
        and isinstance(row.get("digested_at"), str)
        and row["digested_at"] < cutoff
    ]
    for key in removed:
        del state[key]
    return httpx.Response(200, json={"deleted_count": len(removed)})


def contract_writers(monkeypatch, mod, state, writes):
    """Only in-memory $set writes; retain the production prune implementation."""
    from _helpers import load_inject

    real_prune = load_inject().prune_stale_records

    monkeypatch.setattr(mod, "inject_records", partial(in_memory_upsert, state, writes))
    monkeypatch.setattr(mod, "prune_stale_records", real_prune)
    # Real prune needs an HTTP client, not setup_main's writer-spy sentinel.
    # respx intercepts every DELETE; no live writer is reachable here.
    monkeypatch.setattr(mod, "make_authed_client", http_client)


def http_client(_token):
    return httpx.Client()


def in_memory_upsert(state, writes, records, *args, **kwargs):
    writes.append("inject_records")
    apply_gateway_set(state, records)
    return {"errors": [], "updated_count": len(records)}


@pytest.mark.parametrize(
    "stamp", ["omitted", "retained_old", "fresh", "no_prune", "non_string"]
)
def test_post_set_pinned_record_survives_real_prune(monkeypatch, tmp_path, stamp):
    live = pinned_record()
    incoming = copy.deepcopy(live)
    other = {
        "dataset": live["dataset"],
        "bids_relpath": "other.edf",
        "digested_at": "2026-10-23",
    }
    if stamp in ("omitted", "no_prune"):
        incoming.pop("digested_at")
    elif stamp == "fresh":
        incoming["digested_at"] = "2026-10-23"
    elif stamp == "non_string":
        incoming["digested_at"] = None
    records = [incoming] if stamp == "no_prune" else [incoming, other]
    state = {(live["dataset"], RAW): copy.deepcopy(live)}
    mod, writes = setup_main(monkeypatch, tmp_path, records)
    contract_writers(monkeypatch, mod, state, writes)
    should_refuse = stamp == "omitted"
    with respx.mock(assert_all_called=False) as router:
        mock_live(router, [live])
        router.delete(f"{API}/admin/{DB}/records").mock(
            side_effect=lambda req: delete_matching_strings(state, req)
        )
        outcome = mod.main()
    assert (live["dataset"], RAW) in state
    assert outcome == (1 if should_refuse else 0)
    assert state[(live["dataset"], RAW)]["storage"] == live["storage"]
    if should_refuse:
        assert not writes


@pytest.mark.parametrize("falsy", ["omitted", None, "", False, 0, []])
def test_legacy_duplicate_cannot_overwrite_checked_pin(monkeypatch, tmp_path, falsy):
    live = pinned_record()
    canonical = copy.deepcopy(live)
    legacy = copy.deepcopy(live)
    legacy.pop("bids_relpath")
    legacy["bidspath"] = RAW
    legacy["storage"].pop("version_ids")
    if falsy != "omitted":
        legacy["bids_relpath"] = falsy
    state = {(live["dataset"], RAW): copy.deepcopy(live)}
    mod, writes = setup_main(monkeypatch, tmp_path, [canonical, legacy])
    contract_writers(monkeypatch, mod, state, writes)
    with respx.mock(assert_all_called=False) as router:
        mock_live(router, [live])
        router.delete(f"{API}/admin/{DB}/records").mock(
            side_effect=lambda req: delete_matching_strings(state, req)
        )
        outcome = mod.main()
    assert state[(live["dataset"], RAW)]["storage"] == live["storage"]
    assert outcome == 1
    assert not writes


def aliased_server(seen, live, request):
    query = json.loads(request.url.params["filter"])
    seen.append(query)
    # Equivalent to gateway _twin_query: $eq is deliberately not aliased.
    aliased = query["dataset"] == "ds003768"
    if request.url.path.endswith("/count"):
        return httpx.Response(200, json={"count": int(aliased)})
    return httpx.Response(200, json={"data": [live] if aliased else []})


def test_exact_ds_target_does_not_inherit_gateway_on_alias(monkeypatch, tmp_path):
    """Server aliases bare ds id on empty results, but not the $eq operator."""
    live = pinned_record()
    new = copy.deepcopy(live)
    new["dataset"] = "ds003768"
    new["storage"].pop("version_ids")
    mod, writes = setup_main(monkeypatch, tmp_path, [new])
    # Same plan helper defaults, now the actual dataset scheduled to write.
    plan_builder = mod.build_injection_plan
    plan = plan_builder()
    plan.changed_ids = ["ds003768"]
    plan.datasets = [{"dataset_id": "ds003768"}]
    seen = []

    with respx.mock(assert_all_called=False) as router:
        router.get(f"{API}/api/{DB}/count").mock(
            side_effect=partial(aliased_server, seen, live)
        )
        router.get(f"{API}/api/{DB}/records").mock(
            side_effect=partial(aliased_server, seen, live)
        )
        assert mod.main() == 0
    assert writes
    assert all(q["dataset"] == {"$eq": "ds003768"} for q in seen)


def test_real_contract_controls_demonstrate_inherited_prune_and_alias_overwrite():
    """Prove the in-memory model actually exposes the two destructive effects."""
    live = pinned_record()
    state = {(live["dataset"], RAW): copy.deepcopy(live)}
    incoming = copy.deepcopy(live)
    incoming.pop("digested_at")
    apply_gateway_set(state, [incoming])
    request = httpx.Request(
        "DELETE",
        f"{API}/admin/{DB}/records",
        params={
            "filter": json.dumps(
                {"dataset": live["dataset"], "digested_at": {"$lt": "2026-10-23"}}
            )
        },
    )
    assert delete_matching_strings(state, request).json()["deleted_count"] == 1
    assert not state
    state = {(live["dataset"], RAW): copy.deepcopy(live)}
    legacy = copy.deepcopy(live)
    legacy.pop("bids_relpath")
    legacy["bidspath"] = RAW
    legacy["storage"].pop("version_ids")
    apply_gateway_set(state, [copy.deepcopy(live), legacy])
    assert "version_ids" not in state[(live["dataset"], RAW)]["storage"]

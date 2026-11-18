"""Explicit live historical identities must survive Stage 5 unchanged."""

from __future__ import annotations

import ast
import contextlib
import copy
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import respx
from _helpers import load_inject

from _inject_plan import InjectionPlan

API = "https://pin-guard.example"
DB = "eegdash_staging"
RAW = "sub-01/eeg/sub-01_task-rest_eeg.vhdr"
DEP = RAW.replace(".vhdr", ".eeg")


def pinned_record():
    return {
        "dataset": "on003768",
        "bids_relpath": RAW,
        "digested_at": "2026-10-19",
        "ntimes": 4770300,
        "nchans": 32,
        "sampling_frequency": 5000.0,
        "duration_seconds": 954.06,
        "ch_names": ["Cz", "Pz"],
        "storage": {
            "backend": "s3",
            "base": "s3://openneuro.org/ds003768",
            "raw_key": RAW,
            "dep_keys": [DEP],
            "annex_keys": {
                RAW: "MD5E-s4--header.vhdr",
                DEP: "MD5E-s305299200--historical.eeg",
            },
            "version_ids": {DEP: "approved-old-version"},
        },
    }


def record_writer_call(writes, name, *args, **kwargs):
    writes.append(name)
    return {"errors": [], "deleted": {}}


def setup_main(monkeypatch, tmp_path, incoming, *, dry_run=False):
    mod = load_inject()
    args = SimpleNamespace(
        input=tmp_path,
        datasets=None,
        sources=None,
        skip_validation=True,
        only_records=False,
        only_datasets=False,
        only_montages=False,
        skip_montages=False,
        force=True,
        api_url=API,
        database=DB,
        validator_sweep=None,
        dry_run=dry_run,
        token="not-a-real-token",
        batch_size=100,
        compute_stats=False,
    )
    monkeypatch.setattr(mod, "load_inject_config_from_argv", lambda: args)
    monkeypatch.setattr(mod, "find_digested_datasets", lambda *a: [tmp_path])
    plan = InjectionPlan(
        datasets=[{"dataset_id": "on003768"}],
        records=incoming,
        montages=[{"hash": "layout"}],
        changed_ids=["on003768"],
    )
    monkeypatch.setattr(mod, "build_injection_plan", lambda *a, **kw: plan)
    writes = []
    for name in (
        "inject_datasets",
        "inject_records",
        "inject_montages",
        "prune_stale_records",
    ):
        monkeypatch.setattr(
            mod, name, Mock(side_effect=partial(record_writer_call, writes, name))
        )
    monkeypatch.setattr(
        mod, "make_authed_client", lambda _: contextlib.nullcontext(object())
    )
    return mod, writes


def mock_live(router, rows):
    router.get(f"{API}/api/{DB}/count").mock(
        return_value=httpx.Response(200, json={"count": len(rows)})
    )
    return router.get(f"{API}/api/{DB}/records").mock(
        side_effect=lambda request: httpx.Response(
            200,
            json={
                "data": rows[
                    int(request.url.params.get("skip", "0")) : int(
                        request.url.params.get("skip", "0")
                    )
                    + int(request.url.params["limit"])
                ]
            },
        )
    )


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize(
    "damage",
    [
        "drop_pin",
        "change_pin",
        "change_key",
        "change_base",
        "drop_dep",
        "change_raw",
        "scientific",
        "absent",
        "duplicate",
    ],
)
def test_conflict_refuses_before_every_writer_even_force(
    monkeypatch, tmp_path, damage, capsys, force
):
    live = pinned_record()
    new = copy.deepcopy(live)
    st = new["storage"]
    if damage == "drop_pin":
        st.pop("version_ids")
    elif damage == "change_pin":
        st["version_ids"][DEP] = "latest"
    elif damage == "change_key":
        st["annex_keys"][DEP] = "MD5E-s206647891--current-short.eeg"
    elif damage == "change_base":
        st["base"] = "s3://nemar/on003768"
    elif damage == "drop_dep":
        st["dep_keys"] = []
    elif damage == "change_raw":
        st["annex_keys"][RAW] = "MD5E-s3--other-header.vhdr"
    elif damage == "scientific":
        new["ntimes"] = 1
    incoming = (
        []
        if damage == "absent"
        else [new, copy.deepcopy(new)]
        if damage == "duplicate"
        else [new]
    )
    mod, writes = setup_main(monkeypatch, tmp_path, incoming)
    mod.load_inject_config_from_argv().force = force
    with respx.mock(assert_all_called=False) as router:
        mock_live(router, [live])
        assert mod.main() == 1
    assert writes == []
    output = capsys.readouterr()
    assert "pin" in (output.out + output.err).lower()
    assert "on003768" in output.out + output.err


def test_matched_identity_allows_additive_unpinned_deps_and_provenance(
    monkeypatch, tmp_path
):
    live = pinned_record()
    new = copy.deepcopy(live)
    new["storage"]["dep_keys"].append("participants.tsv")
    new["_metadata_provenance"] = {"ntimes": "source_header"}
    new["digested_at"] = "2026-10-23"
    mod, writes = setup_main(monkeypatch, tmp_path, [new])
    with respx.mock(assert_all_called=False) as router:
        records = mock_live(router, [live])
        assert mod.main() == 0
        assert records.called
        assert all("eegdash_staging" in str(c.request.url) for c in router.calls)
    assert writes[0] == "inject_datasets"


@pytest.mark.parametrize(
    "bad_read", ["http", "json", "short", "wrong_dataset", "duplicate"]
)
def test_ambiguous_target_reads_refuse_before_writes(monkeypatch, tmp_path, bad_read):
    live = pinned_record()
    mod, writes = setup_main(monkeypatch, tmp_path, [copy.deepcopy(live)])
    with respx.mock(assert_all_called=False) as router:
        mock_live(router, [live])
        route = router.get(f"{API}/api/{DB}/records")
        if bad_read == "http":
            route.mock(return_value=httpx.Response(503))
        elif bad_read == "json":
            route.mock(return_value=httpx.Response(200, content=b'{"data": ['))
        elif bad_read == "short":
            route.mock(return_value=httpx.Response(200, json={"data": []}))
        elif bad_read == "wrong_dataset":
            route.mock(
                return_value=httpx.Response(
                    200, json={"data": [{**live, "dataset": "other"}]}
                )
            )
        else:
            router.get(f"{API}/api/{DB}/count").mock(
                return_value=httpx.Response(200, json={"count": 2})
            )
            route.mock(return_value=httpx.Response(200, json={"data": [live, live]}))
        assert mod.main() == 1
    assert writes == []


def test_dry_run_does_not_read_target_or_write(monkeypatch, tmp_path):
    mod, writes = setup_main(monkeypatch, tmp_path, [pinned_record()], dry_run=True)
    with respx.mock(assert_all_called=False) as router:
        assert mod.main() == 0
        assert not router.calls
    assert not writes


@pytest.mark.parametrize("damage", [None, "drop", "different", "raw_key", "backend"])
def test_raw_pin_identity(monkeypatch, tmp_path, damage):
    live = pinned_record()
    live["storage"].pop("version_ids")
    live["storage"]["version_id"] = "old-raw-version"
    new = copy.deepcopy(live)
    if damage == "drop":
        new["storage"].pop("version_id")
    elif damage == "different":
        new["storage"]["version_id"] = "new-raw-version"
    elif damage == "raw_key":
        new["storage"]["raw_key"] = "another.vhdr"
    elif damage == "backend":
        new["storage"]["backend"] = "nemar"
    mod, writes = setup_main(monkeypatch, tmp_path, [new])
    with respx.mock(assert_all_called=False) as router:
        mock_live(router, [live])
        assert mod.main() == (1 if damage else 0)
    assert bool(writes) == (damage is None)


def test_no_pin_dataset_has_normal_writers(monkeypatch, tmp_path):
    new = pinned_record()
    new["storage"].pop("version_ids")
    mod, writes = setup_main(monkeypatch, tmp_path, [new])
    with respx.mock(assert_all_called=False) as router:
        mock_live(router, [])
        assert mod.main() == 0
    assert writes == [
        "inject_datasets",
        "inject_records",
        "prune_stale_records",
        "inject_montages",
    ]


def test_live_pagination_checks_exact_target_and_all_pages(monkeypatch, tmp_path):
    live = []
    for i in range(101):
        record = pinned_record()
        record["bids_relpath"] = f"record-{i}.vhdr"
        live.append(record)
    mod, writes = setup_main(monkeypatch, tmp_path, copy.deepcopy(live))
    with respx.mock(assert_all_called=False) as router:
        records = mock_live(router, live)
        assert mod.main() == 0
        assert [int(c.request.url.params["skip"]) for c in records.calls] == [
            0,
            100,
            101,
        ]
        for call in router.calls:
            assert call.request.method == "GET"
            assert call.request.headers["cache-control"] == "no-cache"
            query = json.loads(call.request.url.params["filter"])
            assert query["dataset"] == {"$eq": "on003768"}
            assert "storage.version_id" in query["$or"][0]
    assert writes


def test_truncated_json_read_is_retried_then_succeeds(monkeypatch, tmp_path):
    live = pinned_record()
    mod, writes = setup_main(monkeypatch, tmp_path, [copy.deepcopy(live)])
    with respx.mock(assert_all_called=False) as router:
        mock_live(router, [live])
        route = router.get(f"{API}/api/{DB}/records").mock(
            side_effect=[
                httpx.Response(200, content=b'{"data": ['),
                httpx.Response(200, json={"data": [live]}),
                httpx.Response(200, json={"data": []}),
            ]
        )
        assert mod.main() == 0
        assert route.call_count == 3
    assert writes


def test_changed_count_refuses_before_dataset_writer(monkeypatch, tmp_path):
    live = pinned_record()
    mod, writes = setup_main(monkeypatch, tmp_path, [copy.deepcopy(live)])
    with respx.mock(assert_all_called=False) as router:
        mock_live(router, [live])
        router.get(f"{API}/api/{DB}/count").mock(
            side_effect=[
                httpx.Response(200, json={"count": 1}),
                httpx.Response(200, json={"count": 2}),
            ]
        )
        assert mod.main() == 1
    assert not writes


def test_unpinned_optional_removal_and_dep_order_are_not_pin_changes(
    monkeypatch, tmp_path
):
    live = pinned_record()
    live["storage"]["dep_keys"] += ["missing.dat", "participants.tsv"]
    new = copy.deepcopy(live)
    new["storage"]["dep_keys"] = ["participants.tsv", DEP]
    mod, writes = setup_main(monkeypatch, tmp_path, [new])
    with respx.mock(assert_all_called=False) as router:
        mock_live(router, [live])
        assert mod.main() == 0
    assert writes


@pytest.mark.parametrize("bad_count", [None, True, -1, "1"])
def test_malformed_counts_fail_closed(monkeypatch, tmp_path, bad_count):
    mod, writes = setup_main(monkeypatch, tmp_path, [pinned_record()])
    with respx.mock(assert_all_called=False) as router:
        route = router.get(f"{API}/api/{DB}/count").mock(
            return_value=httpx.Response(200, json={"count": bad_count})
        )
        assert mod.main() == 1
        assert route.call_count == 3
    assert not writes


def test_transient_http_read_retries_without_writing_early(monkeypatch, tmp_path):
    live = pinned_record()
    mod, writes = setup_main(monkeypatch, tmp_path, [copy.deepcopy(live)])
    with respx.mock(assert_all_called=False) as router:
        mock_live(router, [live])
        route = router.get(f"{API}/api/{DB}/records").mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(200, json={"data": [live]}),
                httpx.Response(200, json={"data": []}),
            ]
        )
        assert mod.main() == 0
        assert route.call_count == 3
    assert writes


def test_invalid_storage_fails_closed(monkeypatch, tmp_path):
    live = pinned_record()
    new = copy.deepcopy(live)
    new["storage"] = ["invalid"]
    mod, writes = setup_main(monkeypatch, tmp_path, [new])
    with respx.mock(assert_all_called=False) as router:
        mock_live(router, [live])
        assert mod.main() == 1
    assert not writes


def test_guard_module_is_in_installed_distribution():
    """The numbered CLI imports a bare module; it must be shipped in wheels."""
    config = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    block = config.split("py-modules = ", 1)[1].split("\n]", 1)[0] + "\n]"
    assert "_inject_pin_guard" in ast.literal_eval(block)

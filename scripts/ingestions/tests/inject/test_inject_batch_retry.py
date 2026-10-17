"""A record batch the gateway drops is re-sent, and a batch still lost fails the run.

On 2026-10-17 the production inject (3-inject-all, 60 NEMAR datasets) lost 5
of its 42 record batches (5,000 records), skipped the stale-record prune, and
still reported ``Errors: 0`` with exit code 0.
"""

from __future__ import annotations

import contextlib
import json

import pytest
from _helpers import load_inject

import _inject_config

_CALLS: list[tuple[int, int]] = []


def _records(n: int) -> list[dict]:
    return [
        {
            "dataset": "nm000108",
            "bids_relpath": f"sub-{i:03d}/emg/sub-{i:03d}_emg.bdf",
            "storage": {"base": "s3://nemar/nm000108", "backend": "nemar"},
        }
        for i in range(n)
    ]


def _first_try_of_batch_1_fails(batch_idx, batch, url, session, *, timeout):
    """Stand-in for ``_bulk_upsert_batch``: batch 1 fails on its first send."""
    _CALLS.append((batch_idx, len(batch)))
    if batch_idx == 1 and _CALLS.count((1, 1000)) == 1 and len(batch) == 1000:
        return {"inserted": 0, "updated": 0, "error": "Batch 1: 504 Gateway Timeout"}
    return {"inserted": len(batch), "updated": 0, "error": None}


def _always_fails(batch_idx, batch, url, session, *, timeout):
    return {"inserted": 0, "updated": 0, "error": f"Batch {batch_idx}: timeout"}


@pytest.fixture
def inject_mod(monkeypatch):
    _CALLS.clear()
    mod = load_inject()
    monkeypatch.setattr(mod, "DEFAULT_WORKERS", 1)
    return mod


def test_failed_batch_is_resent_in_chunks(inject_mod, monkeypatch):
    monkeypatch.setattr(inject_mod, "_bulk_upsert_batch", _first_try_of_batch_1_fails)
    result = inject_mod.inject_records(
        _records(2500), "https://api.example.com", "eegdash", "t", client=object()
    )
    assert result["errors"] == []
    assert result["inserted_count"] == 2500
    assert _CALLS.count((1, 100)) == 10  # the lost batch, re-sent 100 at a time


def test_retry_chunks_cap_bytes(inject_mod, monkeypatch):
    """on006545: ~250 KB records must not go 100 to a body."""
    big = [{**r, "pad": "x" * 400} for r in _records(10)]
    one = len(json.dumps(big[0]))
    monkeypatch.setattr(inject_mod, "_RETRY_CHUNK_BYTES", 2 * one + 10)
    chunks = inject_mod._retry_chunks(big)
    assert [len(c) for c in chunks] == [2, 2, 2, 2, 2]
    assert [r for c in chunks for r in c] == big


def _digested(tmp_path):
    ds = tmp_path / "nm000108"
    ds.mkdir()
    (ds / "nm000108_dataset.json").write_text(
        json.dumps({"dataset_id": "nm000108", "source": "nemar"})
    )
    (ds / "nm000108_records.json").write_text(json.dumps({"records": _records(3)}))
    return tmp_path


def _no_client(_token):
    return contextlib.nullcontext(object())


def _no_database_list(_api_url, _token):
    return None


def _must_not_prune(*_args, **_kwargs):
    pytest.fail("pruned after a lost batch")


def test_lost_batch_fails_the_run(inject_mod, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(inject_mod, "_bulk_upsert_batch", _always_fails)
    monkeypatch.setattr(inject_mod, "make_authed_client", _no_client)
    monkeypatch.setattr(inject_mod, "prune_stale_records", _must_not_prune)
    monkeypatch.setattr(
        _inject_config, "fetch_valid_databases_from_api", _no_database_list
    )
    monkeypatch.setenv("EEGDASH_ADMIN_TOKEN", "t")
    argv = ["5_inject.py", "--input", str(_digested(tmp_path)), "--database"]
    argv += ["eegdash", "--force", "--only-records", "--skip-validation"]
    monkeypatch.setattr("sys.argv", argv)
    assert inject_mod.main() == 1
    assert "Errors:     1" in capsys.readouterr().out

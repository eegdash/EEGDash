"""Golden-master tests for the Stage-5 InjectionPlan decision logic.

``build_injection_plan`` / ``InjectionPlan`` (the create / update / skip decision,
fingerprint comparison, and montage de-duplication) previously had ZERO test
coverage — the richest behavior in the pipeline was unfrozen. These tests freeze
that decision against the committed digest-snapshot corpus with the EEGDash API
stubbed (no network), so any drift fails here before it can ship.
"""

from __future__ import annotations

import _inject_plan
from _inject_plan import build_injection_plan, find_digested_datasets
from eegdash.testing import data_file

# All three committed snapshot datasets (one carries a montage).
ALL_IDS = ["ds_snapshot_eeg_montage", "ds_snapshot_manifest", "ds_snapshot_vhdr"]


def _corpus_dirs():
    return find_digested_datasets(data_file("digest_snapshots/outputs"))


def _build(**overrides):
    kw = {
        "want_datasets": True,
        "want_records": True,
        "want_montages": True,
        "force": False,
        "only_montages": False,
        "api_url": "http://stub",
        "database": "test",
    }
    kw.update(overrides)
    return build_injection_plan(_corpus_dirs(), **kw)


def _no_api(*args, **kwargs):
    """Stub asserting the EEGDash API is never consulted (force / only_montages paths)."""
    raise AssertionError("the EEGDash API must not be called on this path")


def _summary(plan) -> dict:
    """Sanitized decision artifact — the behaviorally-rich part of the plan."""
    return {
        "changed_ids": sorted(plan.changed_ids),
        "skipped_ids": sorted(plan.skipped_ids),
        "n_datasets": len(plan.datasets),
        "n_records": len(plan.records),
        "n_montages": len(plan.montages),
        "duplicate_montage_sightings": plan.duplicate_montage_sightings,
        "errors": plan.errors,
    }


def test_injection_plan_all_new_golden(monkeypatch):
    """Nothing exists remotely -> every dataset is 'changed'. Frozen decision artifact."""
    monkeypatch.setattr(_inject_plan, "fetch_existing_dataset", lambda *a, **k: None)
    assert _summary(_build()) == {
        "changed_ids": ALL_IDS,
        "skipped_ids": [],
        "n_datasets": 3,
        "n_records": 5,
        "n_montages": 1,
        "duplicate_montage_sightings": 0,
        "errors": [],
    }


def test_injection_plan_skips_when_fingerprints_match(monkeypatch):
    """Remote fingerprint == local fingerprint -> the dataset is skipped (round-trip)."""
    # force=True bypasses the API and stamps the fingerprints the plan would compute.
    fp_table = {
        d["dataset_id"]: d["ingestion_fingerprint"] for d in _build(force=True).datasets
    }
    assert set(fp_table) == set(ALL_IDS)
    assert all(fp_table.values())

    monkeypatch.setattr(
        _inject_plan,
        "fetch_existing_dataset",
        lambda api, db, did: {"ingestion_fingerprint": fp_table.get(did)},
    )
    plan = _build()
    assert sorted(plan.skipped_ids) == ALL_IDS
    assert plan.changed_ids == []
    assert plan.datasets == []
    assert plan.records == []


def test_injection_plan_changes_when_fingerprint_differs(monkeypatch):
    """Remote fingerprint differs -> the dataset is re-upserted (changed)."""
    monkeypatch.setattr(
        _inject_plan,
        "fetch_existing_dataset",
        lambda api, db, did: {"ingestion_fingerprint": "STALE"},
    )
    plan = _build()
    assert sorted(plan.changed_ids) == ALL_IDS
    assert plan.skipped_ids == []


def test_injection_plan_force_bypasses_api(monkeypatch):
    """force=True must not consult the API and marks everything changed."""
    monkeypatch.setattr(_inject_plan, "fetch_existing_dataset", _no_api)
    plan = _build(force=True)
    assert sorted(plan.changed_ids) == ALL_IDS
    assert plan.skipped_ids == []
    assert len(plan.datasets) == 3
    assert len(plan.records) == 5


def test_injection_plan_only_montages_skips_api_and_collects_montages(monkeypatch):
    """only_montages=True must not consult the API; montages are still collected."""
    monkeypatch.setattr(_inject_plan, "fetch_existing_dataset", _no_api)
    plan = _build(only_montages=True)
    assert plan.skipped_ids == []
    assert len(plan.montages) == 1

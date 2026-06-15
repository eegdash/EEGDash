"""CI-gate tests for ``scripts/validation/check_snapshot_health.py``.

These exercise ``evaluate_snapshot`` — the B2 gate that refuses to
publish a docs build whose :class:`DatasetSnapshot` is degraded
(fallback source, low row count, or partial API degradation).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_check_snapshot_health():
    """Load ``scripts/validation/check_snapshot_health.py`` as a module.

    The script lives outside the importable package tree (``scripts/``
    has no ``__init__.py`` on purpose: it's a CLI directory, not a
    library). For tests we load the file directly via ``importlib``
    rather than mutating ``sys.path`` globally. The walk up from this
    file keeps the loader working regardless of where the test file is
    relocated within the tree.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "scripts" / "validation" / "check_snapshot_health.py"
        if candidate.exists():
            script_path = candidate
            break
    else:
        raise FileNotFoundError("check_snapshot_health.py not found above test file")
    spec = importlib.util.spec_from_file_location(
        "check_snapshot_health_under_test", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def health_module():
    return _load_check_snapshot_health()


def test_ci_gate_passes_clean_live_snapshot(health_module, gate_snapshot):
    """Source=live, count over threshold, api_errors empty → OK."""
    snap = gate_snapshot(source="live", dataset_count=800, api_errors=[])
    assert health_module.evaluate_snapshot(snap, min_count=700) == []


def test_ci_gate_fails_on_fallback_source(health_module, gate_snapshot):
    """A cached snapshot is degraded, even with a full row count."""
    snap = gate_snapshot(source="cached", dataset_count=900, api_errors=[])
    failures = health_module.evaluate_snapshot(snap, min_count=700)
    assert any("source" in msg for msg in failures), failures


def test_ci_gate_fails_on_low_count(health_module, gate_snapshot):
    """Row count at or below the floor must fail the gate."""
    snap = gate_snapshot(source="live", dataset_count=700, api_errors=[])
    failures = health_module.evaluate_snapshot(snap, min_count=700)
    assert any("dataset_count" in msg for msg in failures), failures


def test_ci_gate_fails_on_partial_degradation_api_errors(health_module, gate_snapshot):
    """The motivating regression: chart-data dead, summary alive.

    ``_build_uncached`` records the chart-data failure on
    ``api_errors`` and then tags the snapshot ``source="live"``
    because the summary endpoint saved the day. The pre-fix gate
    looked only at ``source`` and ``dataset_count`` and let this
    through — losing the aggregations block silently on the docs
    site. The fixed gate catches it.
    """
    snap = gate_snapshot(
        source="live",
        dataset_count=900,
        api_errors=[
            "chart-data 404 at https://api.example/db/datasets/chart-data; "
            "trying summary"
        ],
    )
    failures = health_module.evaluate_snapshot(snap, min_count=700)
    assert any("api_errors" in msg for msg in failures), (
        f"partial degradation did not trip the gate; failures={failures}"
    )


def test_ci_gate_api_errors_ignored_on_fallback_source(health_module, gate_snapshot):
    """When the snapshot has already fallen back, ``api_errors`` will
    obviously be populated (every fallback path appends to it). The
    source-tag check has already failed, so we should not double-count
    the same degradation. This locks in that ``api_errors`` only
    contributes a *new* failure when ``source == "live"``.
    """
    snap = gate_snapshot(
        source="package-csv",
        dataset_count=900,
        api_errors=["chart-data error at ...", "summary error at ..."],
    )
    failures = health_module.evaluate_snapshot(snap, min_count=700)
    # Exactly one failure: the source-tag failure. No api_errors line.
    assert len(failures) == 1, failures
    assert "source" in failures[0]

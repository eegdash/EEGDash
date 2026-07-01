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


@pytest.mark.parametrize(
    "source, dataset_count, api_errors, min_count, expect_pass, "
    "expect_substr, expect_len_one",
    [
        # Source=live, count over threshold, api_errors empty → OK.
        ("live", 800, [], 700, True, None, False),
        # A cached snapshot is degraded, even with a full row count.
        ("cached", 900, [], 700, False, "source", False),
        # Row count at or below the floor must fail the gate.
        ("live", 700, [], 700, False, "dataset_count", False),
        # Partial degradation: chart-data dead, summary alive →
        # source stays "live" but api_errors is populated. The fixed gate
        # catches it.
        (
            "live",
            900,
            [
                "chart-data 404 at https://api.example/db/datasets/chart-data; "
                "trying summary"
            ],
            700,
            False,
            "api_errors",
            False,
        ),
        # Already fallen back (package-csv): api_errors is populated but
        # the source-tag check already failed, so exactly one failure —
        # api_errors must NOT double-count when source != "live".
        (
            "package-csv",
            900,
            ["chart-data error at ...", "summary error at ..."],
            700,
            False,
            "source",
            True,
        ),
    ],
    ids=[
        "passes-clean-live",
        "fails-fallback-source",
        "fails-low-count",
        "fails-partial-degradation-api-errors",
        "api-errors-ignored-on-fallback-source",
    ],
)
def test_snapshot_health_gate(
    health_module,
    gate_snapshot,
    source,
    dataset_count,
    api_errors,
    min_count,
    expect_pass,
    expect_substr,
    expect_len_one,
):
    """``evaluate_snapshot`` returns ``[]`` for a clean live snapshot and
    a failure list (each carrying a specific substring) for degraded
    ones. Collapses the five former ``test_ci_gate_*`` cases.
    """
    snap = gate_snapshot(
        source=source, dataset_count=dataset_count, api_errors=api_errors
    )
    failures = health_module.evaluate_snapshot(snap, min_count=min_count)
    if expect_pass:
        assert failures == []
        return
    assert any(expect_substr in msg for msg in failures), failures
    if expect_len_one:
        # Exactly one failure: the source-tag failure. No api_errors line.
        assert len(failures) == 1, failures
        assert "source" in failures[0]

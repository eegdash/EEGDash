#!/usr/bin/env python
"""CI gate: refuse to publish docs unless the dataset snapshot is healthy.

Implements the §3 B2 invariant from
``docs_pipeline_architecture_review.md`` and the §3 step 6 ``Post`` check
in ``docs_pipeline_validation_plan.md``:

- ``snapshot.source == "live"`` — anything else means the build is
  rendering stale or fallback data.
- ``snapshot.dataset_count > MIN_DATASET_COUNT`` — guards against the
  failure mode where the API returned 200 with zero rows (e.g. an
  ingestion crash mid-build).

Exit codes
----------
0
    Snapshot is healthy; safe to publish docs.
1
    Snapshot is unhealthy (fallback fired, count too low, or empty
    aggregations); refuse to publish.

Environment variables
---------------------
EEGDASH_API_BASE
    Override the default API base URL (``https://data.eegdash.org/api``).
    Useful for staging-shard health checks.
EEGDASH_DATABASE
    Override the default database name (``eegdash``).
EEGDASH_MIN_DATASET_COUNT
    Minimum acceptable ``snapshot.dataset_count`` (default: 700).
    The review's published target is "700+ datasets" so the gate fires
    at 700 to catch any silent half-publish.
DRY_API
    When set, force-builds the snapshot against the value (used by the
    red half of the CI red-green test — see the validation plan §3
    step 6).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the repo importable when this script runs from a fresh checkout
# without ``pip install -e .``.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eegdash.dataset.snapshot import DatasetSnapshot  # noqa: E402


def _resolve_int(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        print(
            f"WARNING: {env_name}={raw!r} is not an integer; using default {default}",
            file=sys.stderr,
        )
        return default


def main() -> int:
    api_base = os.environ.get("DRY_API") or os.environ.get(
        "EEGDASH_API_BASE", "https://data.eegdash.org/api"
    )
    database = os.environ.get("EEGDASH_DATABASE", "eegdash")
    min_count = _resolve_int("EEGDASH_MIN_DATASET_COUNT", 700)

    print(
        f"check_snapshot_health: probing {api_base} (database={database}), "
        f"min_dataset_count={min_count}"
    )

    snapshot = DatasetSnapshot.build(
        api_base=api_base, database=database, force_refresh=True
    )

    print(
        f"  source={snapshot.source} "
        f"dataset_count={snapshot.dataset_count} "
        f"fetched_at={snapshot.fetched_at.isoformat()}"
    )
    if snapshot.api_errors:
        print(f"  api_errors ({len(snapshot.api_errors)}):")
        for err in snapshot.api_errors:
            print(f"    - {err}")

    failures: list[str] = []
    if snapshot.source != "live":
        failures.append(f"snapshot.source is {snapshot.source!r}, expected 'live'")
    if snapshot.dataset_count <= min_count:
        failures.append(
            f"snapshot.dataset_count={snapshot.dataset_count} <= min_count={min_count}"
        )

    if failures:
        print("\nFAIL: build cannot publish — snapshot is degraded:")
        for msg in failures:
            print(f"  * {msg}")
        return 1

    print("\nOK: snapshot is healthy.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

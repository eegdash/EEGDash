"""Snapshot test for ``digest_dataset`` — byte-level output stability.

Phase 8 Stage-3 safety net. Stage 3 will extract the bodies of
``digest_dataset`` and ``digest_from_manifest`` into private helpers
that return :class:`EnumerationResult` instead of writing JSON. The
LOC canary catches function-existence drift but **not** byte-level
output drift. This test catches that.

How it works
------------

1. The committed snapshot under ``tests/fixtures/digest_snapshots/``
   was produced by running the current (pre-Stage-3) ``digest_dataset``
   against a minimal BIDS fixture (``ds_snapshot_vhdr`` — the
   ds002336 sub-xp101 VHDR triple wrapped in a synthetic BIDS root).
2. This test re-runs ``digest_dataset`` against the same fixture in
   a temp output dir, then compares the produced JSON files to the
   committed snapshot field-by-field.
3. Non-deterministic fields (timestamps, absolute paths) are
   redacted via :func:`_sanitize_for_snapshot` before comparison.

If Stage 3's refactor changes the JSON shape — even by one missing
field or a re-ordered list — this test fails loudly. It is the only
gate that catches that class of drift; do not delete or skip it.

When the snapshot legitimately needs to change (e.g. a schema migration
intentionally adds a field), update the snapshot files in this directory
in the same commit that lands the new behaviour. Reviewers must see
both the code change AND the snapshot diff together.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))


SNAPSHOT_DIR = Path(__file__).parent / "fixtures" / "digest_snapshots"
INPUTS_DIR = SNAPSHOT_DIR / "inputs"
SNAPSHOT_OUTPUTS_DIR = SNAPSHOT_DIR / "outputs"


def _load_digest_module():
    """Lazy-load 3_digest.py despite its digit-prefixed filename."""
    spec = importlib.util.spec_from_file_location(
        "_digest_snapshot_target", _INGEST_DIR / "3_digest.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sanitize_for_snapshot(obj: Any, *, dataset_id: str) -> Any:
    """Recursively redact non-deterministic fields from a JSON value.

    The two classes of non-determinism are:

    1. **Timestamps**: ``digested_at`` (and any other ISO-shape string
       fields ending in 'Z' or containing '+00:00') gets replaced with
       the sentinel ``"<TIMESTAMP>"``.
    2. **Absolute paths**: ``dataset_file``, ``records_file``, and
       ``montages_file`` keys hold absolute paths to a temp directory.
       Strip the directory prefix; keep only the final filename.

    Everything else is preserved verbatim.
    """
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "digested_at":
                out[k] = "<TIMESTAMP>"
            elif k in {"dataset_file", "records_file", "montages_file"}:
                # Keep just the basename for path stability across runs.
                out[k] = Path(str(v)).name if v else v
            else:
                out[k] = _sanitize_for_snapshot(v, dataset_id=dataset_id)
        return out
    if isinstance(obj, list):
        return [_sanitize_for_snapshot(item, dataset_id=dataset_id) for item in obj]
    return obj


def _read_snapshot(dataset_id: str, suffix: str) -> dict:
    """Read a committed snapshot JSON file."""
    path = SNAPSHOT_OUTPUTS_DIR / dataset_id / f"{dataset_id}_{suffix}.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def fresh_digest_output(tmp_path_factory) -> Path:
    """Run ``digest_dataset`` against the snapshot fixture in a temp dir.

    Returns the output directory (which contains ``ds_snapshot_vhdr/``).
    Module-scoped so the (expensive) digest runs ONCE per test session.
    """
    digest_mod = _load_digest_module()

    tmp_output = tmp_path_factory.mktemp("digest_snapshot_run")
    summary = digest_mod.digest_dataset(
        "ds_snapshot_vhdr",
        INPUTS_DIR,
        tmp_output,
    )
    assert summary["status"] == "success", (
        f"snapshot fixture digestion failed: {summary}"
    )
    return tmp_output


# ─── Per-file snapshot comparisons ────────────────────────────────────────


@pytest.mark.parametrize(
    "suffix",
    ["dataset", "records", "montages", "summary"],
)
def test_digest_output_matches_snapshot(
    fresh_digest_output: Path,
    suffix: str,
) -> None:
    """Each of the 4 output JSON files must byte-match the committed snapshot.

    Run after a Stage-3-style refactor: this test surfaces ANY behaviour
    drift in the produced JSON, no matter how subtle (extra field,
    re-ordered list, changed default value, missing record).
    """
    dataset_id = "ds_snapshot_vhdr"
    fresh_path = fresh_digest_output / dataset_id / f"{dataset_id}_{suffix}.json"
    assert fresh_path.exists(), f"digest didn't produce {fresh_path.name}"

    fresh = json.loads(fresh_path.read_text())
    snapshot = _read_snapshot(dataset_id, suffix)

    fresh_sanitized = _sanitize_for_snapshot(fresh, dataset_id=dataset_id)
    snapshot_sanitized = _sanitize_for_snapshot(snapshot, dataset_id=dataset_id)

    assert fresh_sanitized == snapshot_sanitized, (
        f"\n{suffix}.json drifted from the committed snapshot. "
        f"If this is INTENTIONAL (e.g. a schema migration), update "
        f"the snapshot file in the same commit:\n"
        f"  cp {fresh_path} {SNAPSHOT_OUTPUTS_DIR / dataset_id / fresh_path.name}\n"
        f"\nReviewers MUST see both the code change AND the snapshot diff.\n"
    )


def test_snapshot_record_count_stable(fresh_digest_output: Path) -> None:
    """The number of Records produced must match the snapshot.

    Subtle bug class this catches: an extraction that skips a file
    silently (because of an early return / continue / wrong branch).
    """
    dataset_id = "ds_snapshot_vhdr"
    fresh = json.loads(
        (fresh_digest_output / dataset_id / f"{dataset_id}_records.json").read_text()
    )
    snapshot = _read_snapshot(dataset_id, "records")
    assert len(fresh["records"]) == len(snapshot["records"]), (
        f"record count drifted: {len(fresh['records'])} vs snapshot's "
        f"{len(snapshot['records'])}"
    )


def test_snapshot_fingerprint_stable(fresh_digest_output: Path) -> None:
    """``ingestion_fingerprint`` is content-addressed — same input fixtures
    must produce the same fingerprint across digest runs.

    If this test fails on a Stage 3 refactor, it means the order or
    content of files fed into fingerprint_from_files changed, which is
    a more subtle drift than a field rename.
    """
    dataset_id = "ds_snapshot_vhdr"
    fresh = json.loads(
        (fresh_digest_output / dataset_id / f"{dataset_id}_dataset.json").read_text()
    )
    snapshot = _read_snapshot(dataset_id, "dataset")
    assert fresh.get("ingestion_fingerprint") == snapshot.get(
        "ingestion_fingerprint"
    ), (
        "ingestion_fingerprint drifted — same input files produced a "
        "different content hash. This is a deeper drift than a field "
        "rename; investigate before updating the snapshot."
    )

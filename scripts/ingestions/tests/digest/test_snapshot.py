"""Snapshot test for ``digest_dataset`` — byte-level output stability.

Runs ``digest_dataset`` against committed BIDS fixtures in a temp dir, then
compares the produced JSON to committed snapshots field-by-field.
Non-deterministic fields (timestamps, absolute paths) are redacted before comparison.

To update snapshots after an intentional schema change, copy the fresh output
file over the committed snapshot in the same commit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _helpers import load_digest

from eegdash.testing import data_file

SNAPSHOT_DIR = data_file("digest_snapshots")
INPUTS_DIR = SNAPSHOT_DIR / "inputs"
SNAPSHOT_OUTPUTS_DIR = SNAPSHOT_DIR / "outputs"


def _sanitize_for_snapshot(obj: Any, *, dataset_id: str) -> Any:
    """Redact non-deterministic fields: timestamps → ``"<TIMESTAMP>"``, absolute paths → basename."""
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
    path = SNAPSHOT_OUTPUTS_DIR / dataset_id / f"{dataset_id}_{suffix}.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def fresh_digest_output(tmp_path_factory) -> Path:
    """Run ``digest_dataset`` against the BIDS-fs snapshot fixture; module-scoped."""
    digest_mod = load_digest()

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


@pytest.fixture(scope="module")
def fresh_manifest_digest_output(tmp_path_factory) -> Path:
    """Run ``digest_dataset`` against the manifest-only snapshot fixture; module-scoped."""
    digest_mod = load_digest()
    tmp_output = tmp_path_factory.mktemp("manifest_snapshot_run")
    summary = digest_mod.digest_dataset(
        "ds_snapshot_manifest",
        INPUTS_DIR,
        tmp_output,
    )
    assert summary["status"] == "success", (
        f"manifest snapshot fixture digestion failed: {summary}"
    )
    return tmp_output


# ─── BIDS-fs path: per-file snapshot comparisons ──────────────────────────


@pytest.mark.parametrize(
    "suffix",
    ["dataset", "records", "montages", "summary"],
)
def test_digest_output_matches_snapshot(
    fresh_digest_output: Path,
    suffix: str,
) -> None:
    """Each of the 4 output JSON files must match the committed snapshot."""
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
    """Record count must match the snapshot (catches silent skips in extraction)."""
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
    """Same input fixtures must produce the same content-addressed fingerprint."""
    dataset_id = "ds_snapshot_vhdr"
    fresh = json.loads(
        (fresh_digest_output / dataset_id / f"{dataset_id}_dataset.json").read_text()
    )
    snapshot = _read_snapshot(dataset_id, "dataset")
    assert fresh.get("ingestion_fingerprint") == snapshot.get(
        "ingestion_fingerprint"
    ), "ingestion_fingerprint drifted — same inputs produced a different content hash."


# ─── Manifest path: per-file snapshot comparisons ─────────────────────────


@pytest.mark.parametrize(
    "suffix",
    ["dataset", "records", "montages", "summary"],
)
def test_manifest_digest_output_matches_snapshot(
    fresh_manifest_digest_output: Path,
    suffix: str,
) -> None:
    """Manifest-only digest produces 4 byte-stable JSON files (mirrors BIDS-fs test)."""
    dataset_id = "ds_snapshot_manifest"
    fresh_path = (
        fresh_manifest_digest_output / dataset_id / f"{dataset_id}_{suffix}.json"
    )
    assert fresh_path.exists(), f"manifest digest didn't produce {fresh_path.name}"

    fresh = json.loads(fresh_path.read_text())
    snapshot = _read_snapshot(dataset_id, suffix)

    fresh_sanitized = _sanitize_for_snapshot(fresh, dataset_id=dataset_id)
    snapshot_sanitized = _sanitize_for_snapshot(snapshot, dataset_id=dataset_id)

    assert fresh_sanitized == snapshot_sanitized, (
        f"\nmanifest {suffix}.json drifted from the committed snapshot. "
        f"If this is INTENTIONAL, update the snapshot file:\n"
        f"  cp {fresh_path} {SNAPSHOT_OUTPUTS_DIR / dataset_id / fresh_path.name}\n"
    )


def test_manifest_snapshot_record_count_stable(
    fresh_manifest_digest_output: Path,
) -> None:
    dataset_id = "ds_snapshot_manifest"
    fresh = json.loads(
        (
            fresh_manifest_digest_output / dataset_id / f"{dataset_id}_records.json"
        ).read_text()
    )
    snapshot = _read_snapshot(dataset_id, "records")
    assert len(fresh["records"]) == len(snapshot["records"]), (
        f"manifest record count drifted: {len(fresh['records'])} vs "
        f"snapshot's {len(snapshot['records'])}"
    )


def test_manifest_snapshot_total_files_in_summary(
    fresh_manifest_digest_output: Path,
) -> None:
    """Manifest summary carries ``total_files``; BIDS-fs path does not."""
    dataset_id = "ds_snapshot_manifest"
    summary = json.loads(
        (
            fresh_manifest_digest_output / dataset_id / f"{dataset_id}_summary.json"
        ).read_text()
    )
    assert "total_files" in summary
    snapshot_summary = _read_snapshot(dataset_id, "summary")
    assert summary["total_files"] == snapshot_summary["total_files"]

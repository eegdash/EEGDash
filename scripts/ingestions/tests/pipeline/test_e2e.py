"""End-to-end pipeline smoke test .

Runs the 3 → 4 → 5 stages against the committed snapshot fixtures
and verifies the inter-stage contract documented in
``PIPELINE-CONTRACT.md`` actually holds.

Before this test: the unit + snapshot tests verified Stage 3's output
shape, but nothing verified that Stage 4 (validate) and Stage 5
(inject --dry-run) would ACCEPT Stage 3's output. Stage 3 could
silently drift into shapes the downstream stages reject.

This test runs the actual `4_validate_output.py` and `5_inject.py`
scripts against the snapshot's existing output directory and asserts:

- Validate exits 0 with no errors
- Inject (dry-run) exits 0 and reports the expected dataset/record
  counts

The test is offline — Stage 5's `--dry-run` mode doesn't call the API.
"""

from __future__ import annotations

import json
import re as _re
import subprocess
import sys

import pytest
from _helpers import INGEST_DIR as _INGEST_DIR

from eegdash.testing import data_file

_SNAPSHOT_OUTPUTS = data_file("digest_snapshots/outputs")


def _python_run(script: str, *args: str) -> subprocess.CompletedProcess:
    """Run an ingest script as a subprocess; return the completed result.

    Uses the same Python that's running the tests so the venv stays
    consistent. Captures stdout / stderr; doesn't raise on non-zero
    exit — the tests assert exit codes explicitly.
    """
    return subprocess.run(
        [sys.executable, str(_INGEST_DIR / script), *args],
        capture_output=True,
        text=True,
        cwd=str(_INGEST_DIR),
        check=False,
        timeout=120,
    )


# ─── Stage 4: validate ────────────────────────────────────────────────────


def test_stage4_accepts_bids_snapshot():
    """Stage 4 validates the BIDS-fs snapshot output cleanly."""
    result = _python_run(
        "4_validate_output.py",
        "--input",
        str(_SNAPSHOT_OUTPUTS),
        "--json",
    )
    assert result.returncode == 0, (
        f"validate exit {result.returncode}:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # Parse the JSON report from stdout (validate prints it as the
    # final block).
    # The script's --json mode emits a single JSON object at the end.
    lines = result.stdout.strip().split("\n")
    # Walk back from the end until we find a closing brace, then back
    # to a matching opener.
    json_lines = []
    for line in lines:
        json_lines.append(line)
        if line.strip() == "}":
            # Found a closing brace; try to parse from start
            try:
                report = json.loads("\n".join(json_lines))
                break
            except json.JSONDecodeError:
                continue
    else:
        pytest.fail(f"couldn't find JSON in validate output:\n{result.stdout}")

    assert report["valid"] is True, f"validate reported invalid: {report}"
    assert report["stats"]["datasets_checked"] >= 2  # bids + manifest
    assert report["stats"]["records_checked"] >= 4  # 1 from bids + 3 from manifest
    assert report["stats"]["storage_errors"] == 0
    assert report["stats"]["empty_datasets"] == 0


def test_stage4_strict_mode_surfaces_unknown_source_warning():
    """``--strict`` treats warnings as errors.

    The BIDS-fs snapshot (ds_snapshot_vhdr) gets ``source = "unknown"``
    because the dataset_id doesn't match any known source pattern.
    Stage 4 emits an "Unknown source" warning; ``--strict`` promotes
    it to an error and exits 1. That's the documented behaviour.

    This test pins that promotion — if a future refactor changes
    warning semantics, this fires.
    """
    result = _python_run(
        "4_validate_output.py",
        "--input",
        str(_SNAPSHOT_OUTPUTS),
        "--strict",
    )
    # Non-zero exit because --strict turns the "Unknown source" warning
    # into an error.
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "unknown" in combined.lower() or "source" in combined.lower()


# ─── Stage 5: inject --dry-run ────────────────────────────────────────────


def test_stage5_dry_run_accepts_snapshot():
    """Stage 5 dry-run accepts the snapshot and reports the right counts.

    Verifies the inter-stage contract:
    - Stage 3 produces 4 JSON files per dataset.
    - Stage 5 reads them and validates against the Dataset / Record /
      Montage Pydantic schemas WITHOUT making network calls.
    - The summary reports the expected dataset/record counts.
    """
    result = _python_run(
        "5_inject.py",
        "--input",
        str(_SNAPSHOT_OUTPUTS),
        "--database",
        "eegdash_dev",  # logical name; dry-run doesn't connect
        "--dry-run",
    )
    assert result.returncode == 0, (
        f"inject dry-run failed: stdout={result.stdout}\nstderr={result.stderr}"
    )

    # Parse the summary block — Stage 5 prints a fixed-format ASCII
    # summary at the end. We assert the dataset + record counts.
    out = result.stdout
    assert "Datasets:" in out
    assert "Records Ins:" in out
    assert "[DRY RUN - no data uploaded]" in out

    # Extract the counts from the summary block. The format is
    # "Records Ins:4" (no space after colon — see Stage 5's print
    # statement), so we split on ":" rather than whitespace.

    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Datasets:"):
            m = _re.search(r"Datasets:\s*(\d+)", stripped)
            assert m, f"couldn't parse Datasets count: {stripped!r}"
            assert int(m.group(1)) >= 2  # bids + manifest snapshot fixtures
        elif stripped.startswith("Records Ins:"):
            m = _re.search(r"Records Ins:\s*(\d+)", stripped)
            assert m, f"couldn't parse Records Ins count: {stripped!r}"
            assert int(m.group(1)) >= 4  # 1 from bids + 3 from manifest


def test_stage5_dry_run_no_errors():
    """No errors reported in the summary block."""
    result = _python_run(
        "5_inject.py",
        "--input",
        str(_SNAPSHOT_OUTPUTS),
        "--database",
        "eegdash_dev",
        "--dry-run",
    )
    assert result.returncode == 0
    # The "Errors: 0" line in the summary block. Format varies — be
    # tolerant: any whitespace after the colon.

    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Errors:"):
            m = _re.search(r"Errors:\s*(\d+)", stripped)
            assert m, f"couldn't parse Errors line: {stripped!r}"
            n_errors = int(m.group(1))
            assert n_errors == 0, (
                f"inject dry-run reported {n_errors} errors:\n{result.stdout}"
            )
            return
    pytest.fail("no 'Errors:' line in inject summary")


# ─── 3 → 4 → 5 contract ──────────────────────────────────────────────────


def test_pipeline_contract_3_to_4_to_5():
    """The 4 JSON files produced by Stage 3 ARE the input to Stages 4 and 5.

    Pins the contract documented in ``PIPELINE-CONTRACT.md`` —
    stage_3_output_dir / <dataset_id> / <dataset_id>_{dataset,records,
    montages,summary}.json.
    """
    for dataset_id in ("ds_snapshot_vhdr", "ds_snapshot_manifest"):
        ds_dir = _SNAPSHOT_OUTPUTS / dataset_id
        assert ds_dir.exists(), f"snapshot missing: {ds_dir}"
        for suffix in ("dataset", "records", "montages", "summary"):
            path = ds_dir / f"{dataset_id}_{suffix}.json"
            assert path.exists(), f"snapshot missing file: {path}"
            # And each file is valid JSON.
            json.loads(path.read_text())

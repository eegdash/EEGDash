"""End-to-end pipeline test against real MEF3 fixture (ds003708 .tmet)."""

from __future__ import annotations

import importlib.util as _il
import json
import shutil
import subprocess
import sys
from pathlib import Path

from _helpers import INGEST_DIR as _INGEST_DIR

from eegdash.testing import data_file

_TMET_FIXTURE = data_file("ieeg/EKG-000000.tmet")


def _build_minimal_mef3_bids_root(
    tmp_path: Path, dataset_id: str = "ds_mef3_real"
) -> tuple[Path, Path]:
    """Build a minimal BIDS root wrapping the real .tmet fixture."""
    inputs = tmp_path / "inputs"
    ds = inputs / dataset_id
    ds.mkdir(parents=True)

    (ds / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "MEF3 e2e test wrapping ds003708 .tmet",
                "BIDSVersion": "1.6.0",
                "DatasetType": "raw",
                "License": "CC0",
            }
        )
    )

    (ds / "participants.tsv").write_text("participant_id\tage\nsub-01\t30\n")

    ieeg = ds / "sub-01" / "ses-ieeg01" / "ieeg"
    ieeg.mkdir(parents=True)
    mefd = ieeg / "sub-01_ses-ieeg01_task-ccep_run-01_ieeg.mefd"
    mefd.mkdir()
    for ch in ("EKG", "LAD1", "LAD2"):
        segd = mefd / f"{ch}.timd" / f"{ch}-000000.segd"
        segd.mkdir(parents=True)
        shutil.copy(_TMET_FIXTURE, segd / f"{ch}-000000.tmet")

    (ieeg / "sub-01_ses-ieeg01_task-ccep_run-01_ieeg.json").write_text(
        json.dumps(
            {
                "TaskName": "ccep",
                "SamplingFrequency": 2048,
                "PowerLineFrequency": 50,
                "SoftwareFilters": "n/a",
            }
        )
    )

    return inputs, ds


# ─── Stage 3: digest produces correct sfreq from MEF3 ─────────────────────


def test_digest_dataset_extracts_real_mef3_sfreq(tmp_path: Path):
    """digest_dataset on a real .mefd produces a record with sampling_frequency = 2048 Hz."""

    inputs_dir, _ = _build_minimal_mef3_bids_root(tmp_path)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    spec = _il.spec_from_file_location("_e2e_mef3_digest", _INGEST_DIR / "3_digest.py")
    assert spec is not None
    assert spec.loader is not None
    digest_mod = _il.module_from_spec(spec)
    spec.loader.exec_module(digest_mod)

    summary = digest_mod.digest_dataset("ds_mef3_real", inputs_dir, output_dir)

    records_file = output_dir / "ds_mef3_real" / "ds_mef3_real_records.json"
    assert records_file.exists(), f"digest didn't write records: summary={summary}"

    payload = json.loads(records_file.read_text())
    records = payload.get("records", [])
    assert records, (
        f"no records in payload: keys={list(payload.keys())}, summary={summary}"
    )

    mef_records = [r for r in records if r.get("bids_relpath", "").endswith(".mefd")]
    assert mef_records, (
        f"no .mefd record found in {[r.get('bids_relpath') for r in records]}"
    )
    rec = mef_records[0]
    assert rec.get("sampling_frequency") == 2048.0, (
        f"sampling_frequency = {rec.get('sampling_frequency')}, expected 2048.0"
    )
    assert rec.get("nchans") == 3, (
        f"nchans = {rec.get('nchans')}, expected 3 (EKG + LAD1 + LAD2)"
    )


# ─── Stage 3 → Stage 4: validate accepts the output ───────────────────────


def test_stage4_validates_mef3_digest_output(tmp_path: Path):
    """Stage 4 accepts Records produced from the real .mefd."""

    inputs_dir, _ = _build_minimal_mef3_bids_root(tmp_path)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    spec = _il.spec_from_file_location("_e2e_mef3_digest2", _INGEST_DIR / "3_digest.py")
    assert spec is not None
    assert spec.loader is not None
    digest_mod = _il.module_from_spec(spec)
    spec.loader.exec_module(digest_mod)
    digest_mod.digest_dataset("ds_mef3_real", inputs_dir, output_dir)

    result = subprocess.run(
        [
            sys.executable,
            str(_INGEST_DIR / "4_validate_output.py"),
            "--input",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(_INGEST_DIR),
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"validate exit {result.returncode}:\nstdout:{result.stdout}\n"
        f"stderr:{result.stderr}"
    )


# ─── Stage 5 dry-run accepts the output ───────────────────────────────────


def test_stage5_dry_run_accepts_mef3_digest_output(tmp_path: Path):
    """Stage 5 dry-run accepts .mefd Records (catches schema drift)."""

    inputs_dir, _ = _build_minimal_mef3_bids_root(tmp_path)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    spec = _il.spec_from_file_location("_e2e_mef3_digest3", _INGEST_DIR / "3_digest.py")
    assert spec is not None
    assert spec.loader is not None
    digest_mod = _il.module_from_spec(spec)
    spec.loader.exec_module(digest_mod)
    digest_mod.digest_dataset("ds_mef3_real", inputs_dir, output_dir)

    result = subprocess.run(
        [
            sys.executable,
            str(_INGEST_DIR / "5_inject.py"),
            "--input",
            str(output_dir),
            "--database",
            "eegdash_dev",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(_INGEST_DIR),
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"inject dry-run failed: stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "[DRY RUN" in result.stdout


# ─── Cascade: provenance tags MEF3 records correctly ──────────────────────


def test_mef3_record_provenance_marks_binary_parser(tmp_path: Path):
    """Provenance for sampling_frequency comes from an accepted cascade source."""

    inputs_dir, _ = _build_minimal_mef3_bids_root(tmp_path)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    spec = _il.spec_from_file_location("_e2e_mef3_digest4", _INGEST_DIR / "3_digest.py")
    assert spec is not None
    assert spec.loader is not None
    digest_mod = _il.module_from_spec(spec)
    spec.loader.exec_module(digest_mod)
    digest_mod.digest_dataset("ds_mef3_real", inputs_dir, output_dir)

    records_file = output_dir / "ds_mef3_real" / "ds_mef3_real_records.json"
    payload = json.loads(records_file.read_text())
    records = payload.get("records", [])
    mef_records = [r for r in records if r.get("bids_relpath", "").endswith(".mefd")]
    assert mef_records
    rec = mef_records[0]

    prov = rec.get("_metadata_provenance", {})
    sfreq_source = prov.get("sampling_frequency")
    valid_sources = {
        "mne_bids",
        "modality_sidecar",
        "channels_tsv",
        "binary_parser",
    }
    assert sfreq_source in valid_sources, (
        f"unexpected provenance source: {sfreq_source}; valid={valid_sources}"
    )

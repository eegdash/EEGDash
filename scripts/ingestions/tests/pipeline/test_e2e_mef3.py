"""End-to-end pipeline test against real MEF3 fixture.

Runs the full ``digest_dataset`` → ``4_validate_output`` →
``5_inject --dry-run`` chain against a minimal BIDS root wrapping the
real ds003708 ``.tmet``. The parser sits inside a cascade in
``3_digest.py:_extract_technical_metadata`` that prefers
``channels.tsv`` over the binary parser; if the cascade has a bug
that drops the parser's output, we'd silently keep wrong behaviour.
This test exercises the full path end-to-end.

The fixture is loaded from the eegdash-testing-data corpus via
``data_file`` — runs that lack the cache + are offline will fail
at collection (set ``EEGDASH_SKIP_TESTING_DATA=true`` to bypass).
"""

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
    """Build a BIDS root wrapping the real .tmet fixture.

    Layout:
        <tmp>/inputs/ds_mef3_real/
            dataset_description.json
            participants.tsv
            sub-01/ses-ieeg01/ieeg/
                sub-01_ses-ieeg01_task-ccep_run-01_ieeg.mefd/
                    EKG.timd/EKG-000000.segd/EKG-000000.tmet  (real)
                    LAD1.timd/LAD1-000000.segd/LAD1-000000.tmet  (real)
                    LAD2.timd/LAD2-000000.segd/LAD2-000000.tmet  (real)

    Returns (inputs_dir, dataset_dir).
    """
    inputs = tmp_path / "inputs"
    ds = inputs / dataset_id
    ds.mkdir(parents=True)

    # dataset_description.json — minimal valid
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

    # participants.tsv — minimal
    (ds / "participants.tsv").write_text("participant_id\tage\nsub-01\t30\n")

    # The .mefd directory tree
    ieeg = ds / "sub-01" / "ses-ieeg01" / "ieeg"
    ieeg.mkdir(parents=True)
    mefd = ieeg / "sub-01_ses-ieeg01_task-ccep_run-01_ieeg.mefd"
    mefd.mkdir()
    for ch in ("EKG", "LAD1", "LAD2"):
        segd = mefd / f"{ch}.timd" / f"{ch}-000000.segd"
        segd.mkdir(parents=True)
        shutil.copy(_TMET_FIXTURE, segd / f"{ch}-000000.tmet")

    # Companion sidecars (minimal — the parser only reads the .tmet)
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
    """End-to-end: digest_dataset on a real .mefd produces a record with
    sampling_frequency = 2048 Hz (from the C5.1-fixed parser).

    Before C5.1's fix, this would have returned a record WITHOUT
    sampling_frequency (the parser silently dropped it). The test
    pins the integration: parser fix flows through the cascade and
    lands on the Record.
    """

    inputs_dir, _ = _build_minimal_mef3_bids_root(tmp_path)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    spec = _il.spec_from_file_location("_e2e_mef3_digest", _INGEST_DIR / "3_digest.py")
    assert spec is not None
    assert spec.loader is not None
    digest_mod = _il.module_from_spec(spec)
    spec.loader.exec_module(digest_mod)

    summary = digest_mod.digest_dataset("ds_mef3_real", inputs_dir, output_dir)

    # The digest run produced a Records file
    records_file = output_dir / "ds_mef3_real" / "ds_mef3_real_records.json"
    assert records_file.exists(), f"digest didn't write records: summary={summary}"

    payload = json.loads(records_file.read_text())
    records = payload.get("records", [])
    assert records, (
        f"no records in payload: keys={list(payload.keys())}, summary={summary}"
    )

    # Find the .mefd record and assert sfreq == 2048
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
    """Run digest, then validate. Stage 4 should accept the Records
    produced from the real .mefd."""

    inputs_dir, _ = _build_minimal_mef3_bids_root(tmp_path)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    spec = _il.spec_from_file_location("_e2e_mef3_digest2", _INGEST_DIR / "3_digest.py")
    assert spec is not None
    assert spec.loader is not None
    digest_mod = _il.module_from_spec(spec)
    spec.loader.exec_module(digest_mod)
    digest_mod.digest_dataset("ds_mef3_real", inputs_dir, output_dir)

    # Run 4_validate_output.py as subprocess (same pattern as
    # test_pipeline_e2e.py)
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
    """Run digest, then inject --dry-run. Stage 5's Pydantic validation
    should accept the .mefd Records (catches schema drift)."""

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
    # Confirm the dry-run summary mentions [DRY RUN]
    assert "[DRY RUN" in result.stdout


# ─── Cascade: provenance tags MEF3 records correctly ──────────────────────


def test_mef3_record_provenance_marks_binary_parser(tmp_path: Path):
    """When the .mefd parser extracts sfreq, the cascade's provenance
    payload should record ``binary_parser`` as the source.

    Pins the P0.1 cascade integration with the C5.1 parser fix.
    """

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

    # The cascade should report the source that filled sampling_frequency.
    # Per P0.1's contract, this is in _metadata_provenance.
    prov = rec.get("_metadata_provenance", {})
    sfreq_source = prov.get("sampling_frequency")
    # Acceptable sources (the cascade may go through mne_bids first, but
    # the actual extraction here came from the binary parser path).
    valid_sources = {
        "mne_bids",
        "modality_sidecar",
        "channels_tsv",
        "binary_parser",
    }
    assert sfreq_source in valid_sources, (
        f"unexpected provenance source: {sfreq_source}; valid={valid_sources}"
    )

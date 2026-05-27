"""Tests for BIDS sidecar enrichment in extract_record / extract_dataset.

Before sidecar enrichment landed, the only BIDS-spec fields surfaced as structured Record
columns were the technical metadata (sfreq / nchans / ntimes / ch_names).
PowerLineFrequency, EEGReference, SoftwareFilters, Manufacturer,
EEGPlacementScheme — all required or recommended by BIDS — were either
unread or buried as raw bytes in sidecar_inline. This file pins the
new enrichment.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from eegdash.testing import data_file

_INGEST_DIR = Path(__file__).resolve().parent.parent


def _load_digest():
    spec = importlib.util.spec_from_file_location(
        "_c6_digest_target", _INGEST_DIR / "3_digest.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── _extract_bids_sidecar_fields ─────────────────────────────────────────


class _FakeBIDSDataset:
    """Minimal stand-in. ``bidsdir`` + ``get_bids_file_attribute('modality',
    ...)`` are the only attributes the helper needs. Sidecar values are
    discovered on disk via BIDS inheritance — write them as real files
    in tmp_path."""

    def __init__(self, modality: str = "eeg", bids_root: Path | None = None):
        self._modality = modality
        self.bidsdir = bids_root

    def get_bids_file_attribute(self, name: str, _bids_file: str):
        if name == "modality":
            return self._modality
        return None


def _build_bids_root_with_sidecar(
    tmp_path: Path,
    sidecar_payload: dict,
    modality: str = "eeg",
) -> tuple[_FakeBIDSDataset, str]:
    """Build a minimal BIDS root that contains a single recording +
    a custom modality JSON sidecar holding ``sidecar_payload``.

    Returns (fake_bids_dataset, data_filepath_str).
    """
    sub_dir = tmp_path / "sub-01" / modality
    sub_dir.mkdir(parents=True)
    data = sub_dir / f"sub-01_task-rest_{modality}.edf"
    data.touch()
    sidecar = sub_dir / f"sub-01_task-rest_{modality}.json"
    sidecar.write_text(json.dumps(sidecar_payload))
    return _FakeBIDSDataset(modality=modality, bids_root=tmp_path), str(data)


def test_sidecar_extracts_power_line_frequency(tmp_path: Path):
    digest = _load_digest()
    fake, data = _build_bids_root_with_sidecar(tmp_path, {"PowerLineFrequency": 60})
    out = digest._extract_bids_sidecar_fields(fake, data)
    assert out["power_line_frequency"] == 60


def test_sidecar_extracts_eeg_reference(tmp_path: Path):
    digest = _load_digest()
    fake, data = _build_bids_root_with_sidecar(
        tmp_path, {"EEGReference": "linked mastoids"}
    )
    out = digest._extract_bids_sidecar_fields(fake, data)
    assert out["eeg_reference"] == "linked mastoids"


def test_sidecar_extracts_software_filters_dict(tmp_path: Path):
    """SoftwareFilters is a nested dict in the BIDS spec
    ({"HighPass": 0.1, "LowPass": 60}); preserved as-is."""
    digest = _load_digest()
    fake, data = _build_bids_root_with_sidecar(
        tmp_path, {"SoftwareFilters": {"HighPass": 0.1, "LowPass": 60}}
    )
    out = digest._extract_bids_sidecar_fields(fake, data)
    assert out["software_filters"] == {"HighPass": 0.1, "LowPass": 60}


def test_sidecar_extracts_manufacturer_and_model(tmp_path: Path):
    digest = _load_digest()
    fake, data = _build_bids_root_with_sidecar(
        tmp_path,
        {
            "Manufacturer": "Brain Products",
            "ManufacturersModelName": "BrainAmp DC",
        },
    )
    out = digest._extract_bids_sidecar_fields(fake, data)
    assert out["manufacturer"] == "Brain Products"
    assert out["manufacturers_model_name"] == "BrainAmp DC"


def test_sidecar_extracts_eeg_placement_scheme(tmp_path: Path):
    """Feeds the _montage.py template matcher — direct integration."""
    digest = _load_digest()
    fake, data = _build_bids_root_with_sidecar(
        tmp_path, {"EEGPlacementScheme": "10-20"}
    )
    out = digest._extract_bids_sidecar_fields(fake, data)
    assert out["eeg_placement_scheme"] == "10-20"


def test_sidecar_skips_missing_fields(tmp_path: Path):
    """Fields not in the sidecar are omitted from the output."""
    digest = _load_digest()
    fake, data = _build_bids_root_with_sidecar(
        tmp_path,
        {"PowerLineFrequency": 50},  # only one set
    )
    out = digest._extract_bids_sidecar_fields(fake, data)
    assert "power_line_frequency" in out
    assert "eeg_reference" not in out
    assert "software_filters" not in out


def test_sidecar_skips_empty_string_values(tmp_path: Path):
    """Empty strings count as "not specified" — same as None."""
    digest = _load_digest()
    fake, data = _build_bids_root_with_sidecar(
        tmp_path,
        {"Manufacturer": "", "ManufacturersModelName": "BrainAmp"},
    )
    out = digest._extract_bids_sidecar_fields(fake, data)
    assert "manufacturer" not in out
    assert out["manufacturers_model_name"] == "BrainAmp"


def test_sidecar_handles_missing_sidecar(tmp_path: Path):
    """If no modality sidecar exists at any level, return empty dict."""
    digest = _load_digest()
    sub_dir = tmp_path / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)
    data = sub_dir / "sub-01_task-rest_eeg.edf"
    data.touch()
    # No sidecar JSON
    fake = _FakeBIDSDataset(modality="eeg", bids_root=tmp_path)
    out = digest._extract_bids_sidecar_fields(fake, str(data))
    assert out == {}


def test_sidecar_handles_malformed_json(tmp_path: Path):
    """A malformed sidecar JSON → empty dict, no exception."""
    digest = _load_digest()
    sub_dir = tmp_path / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)
    data = sub_dir / "sub-01_task-rest_eeg.edf"
    data.touch()
    (sub_dir / "sub-01_task-rest_eeg.json").write_text("{not valid json")
    fake = _FakeBIDSDataset(modality="eeg", bids_root=tmp_path)
    out = digest._extract_bids_sidecar_fields(fake, str(data))
    assert out == {}


def test_sidecar_handles_dataset_without_required_attrs():
    """If the dataset object lacks bidsdir / get_bids_file_attribute,
    return empty dict (no crash)."""
    digest = _load_digest()
    out = digest._extract_bids_sidecar_fields(object(), "x.edf")
    assert out == {}


def test_sidecar_inheritance_walks_up_to_subject_level(tmp_path: Path):
    """BIDS inheritance: a sidecar at the subject level applies to all
    recordings within that subject. Pinned so a refactor that breaks
    the walk-up is caught."""
    digest = _load_digest()
    sub_dir = tmp_path / "sub-01"
    eeg_dir = sub_dir / "eeg"
    eeg_dir.mkdir(parents=True)
    data = eeg_dir / "sub-01_task-rest_eeg.edf"
    data.touch()
    # Sidecar at subject level, not next to data file
    (sub_dir / "sub-01_eeg.json").write_text(
        json.dumps({"PowerLineFrequency": 60, "EEGReference": "Cz"})
    )
    fake = _FakeBIDSDataset(modality="eeg", bids_root=tmp_path)
    out = digest._extract_bids_sidecar_fields(fake, str(data))
    assert out["power_line_frequency"] == 60
    assert out["eeg_reference"] == "Cz"


# ─── _extract_channel_status_counts ───────────────────────────────────────


def test_channel_status_counts_finds_bad_channels(tmp_path: Path):
    """A channels.tsv with status column → bad_channels list + count."""
    digest = _load_digest()
    sub_dir = tmp_path / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)
    data = sub_dir / "sub-01_task-rest_eeg.edf"
    data.touch()
    (sub_dir / "sub-01_task-rest_channels.tsv").write_text(
        "name\ttype\tstatus\nCz\tEEG\tgood\nFz\tEEG\tbad\nOz\tEEG\tbad\nPz\tEEG\tgood\n"
    )
    fake = _FakeBIDSDataset(bids_root=tmp_path)
    out = digest._extract_channel_status_counts(fake, str(data))
    assert sorted(out["bad_channels"]) == ["Fz", "Oz"]
    assert out["bad_channels_count"] == 2


def test_channel_status_counts_reports_zero_when_all_good(tmp_path: Path):
    """status column exists; all channels good → bad_channels_count = 0."""
    digest = _load_digest()
    sub_dir = tmp_path / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)
    data = sub_dir / "sub-01_task-rest_eeg.edf"
    data.touch()
    (sub_dir / "sub-01_task-rest_channels.tsv").write_text(
        "name\tstatus\nCz\tgood\nFz\tgood\n"
    )
    fake = _FakeBIDSDataset(bids_root=tmp_path)
    out = digest._extract_channel_status_counts(fake, str(data))
    assert out["bad_channels_count"] == 0
    assert out["bad_channels"] == []


def test_channel_status_counts_returns_empty_when_no_status_column(
    tmp_path: Path,
):
    """channels.tsv without a 'status' column → empty dict."""
    digest = _load_digest()
    sub_dir = tmp_path / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)
    data = sub_dir / "sub-01_task-rest_eeg.edf"
    data.touch()
    (sub_dir / "sub-01_task-rest_channels.tsv").write_text(
        "name\ttype\nCz\tEEG\nFz\tEEG\n"
    )
    fake = _FakeBIDSDataset(bids_root=tmp_path)
    out = digest._extract_channel_status_counts(fake, str(data))
    assert out == {}


def test_channel_status_counts_returns_empty_when_no_channels_tsv(
    tmp_path: Path,
):
    """No channels.tsv anywhere in the BIDS tree → empty dict."""
    digest = _load_digest()
    sub_dir = tmp_path / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)
    data = sub_dir / "sub-01_task-rest_eeg.edf"
    data.touch()
    fake = _FakeBIDSDataset(bids_root=tmp_path)
    assert digest._extract_channel_status_counts(fake, str(data)) == {}


def test_channel_status_counts_case_insensitive_bad():
    """'BAD' / 'Bad' / 'bad' all count."""
    digest = _load_digest()
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sub_dir = td / "sub-01" / "eeg"
        sub_dir.mkdir(parents=True)
        data = sub_dir / "sub-01_task-rest_eeg.edf"
        data.touch()
        (sub_dir / "sub-01_task-rest_channels.tsv").write_text(
            "name\tstatus\nCz\tBAD\nFz\tBad\nOz\tbad\nPz\tgood\n"
        )
        fake = _FakeBIDSDataset(bids_root=td)
        out = digest._extract_channel_status_counts(fake, str(data))
        assert out["bad_channels_count"] == 3


# ─── _extract_dataset_description_extras ──────────────────────────────────


def test_dataset_extras_extracts_acknowledgements():
    digest = _load_digest()
    desc = {"Acknowledgements": "Funded by grant XYZ"}
    out = digest._extract_dataset_description_extras(desc)
    assert out["acknowledgements"] == "Funded by grant XYZ"


def test_dataset_extras_extracts_how_to_acknowledge():
    digest = _load_digest()
    desc = {"HowToAcknowledge": "Please cite our paper"}
    out = digest._extract_dataset_description_extras(desc)
    assert out["how_to_acknowledge"] == "Please cite our paper"


def test_dataset_extras_extracts_ethics_approvals():
    digest = _load_digest()
    desc = {"EthicsApprovals": ["IRB-12345", "IRB-67890"]}
    out = digest._extract_dataset_description_extras(desc)
    assert out["ethics_approvals"] == ["IRB-12345", "IRB-67890"]


def test_dataset_extras_skips_empty_lists():
    """Empty list = "not specified" — omit."""
    digest = _load_digest()
    desc = {
        "Acknowledgements": "real value",
        "EthicsApprovals": [],
        "ReferencesAndLinks": "",
    }
    out = digest._extract_dataset_description_extras(desc)
    assert out["acknowledgements"] == "real value"
    assert "ethics_approvals" not in out
    assert "references_and_links" not in out


def test_dataset_extras_extracts_generated_by():
    """GeneratedBy is a list of {Name, Version, ...} dicts for processed
    datasets — preserved as-is."""
    digest = _load_digest()
    desc = {
        "GeneratedBy": [
            {"Name": "fMRIPrep", "Version": "20.2.0"},
            {"Name": "MNE-BIDS-Pipeline", "Version": "1.2.3"},
        ]
    }
    out = digest._extract_dataset_description_extras(desc)
    assert len(out["generated_by"]) == 2
    assert out["generated_by"][0]["Name"] == "fMRIPrep"


# ─── Integration: extract_record + extract_dataset_metadata e2e ──────────


def test_extract_record_includes_new_bids_fields_in_synthetic_bids(
    tmp_path: Path,
):
    """End-to-end: build a tiny BIDS root with rich sidecar JSON,
    run digest_dataset, assert the new fields land on the record.

    Uses a VHDR fixture (existing CC0) plus a custom *_eeg.json that
    declares the BIDS-spec fields.
    """
    digest = _load_digest()

    # Build a tiny BIDS root via the existing snapshot fixture
    inputs_root = data_file("digest_snapshots/inputs/ds_snapshot_vhdr")
    if not inputs_root.exists():
        # No fixture → skip (e2e is conditional on the snapshot tree)
        import pytest

        pytest.skip("snapshot fixture not available")

    # Copy the snapshot to a tmp_path so we can mutate
    import shutil

    work_inputs = tmp_path / "inputs"
    shutil.copytree(inputs_root.parent, work_inputs)

    # Write a richer modality sidecar with the C6.1 fields
    ds_dir = work_inputs / "ds_snapshot_vhdr"
    sub_files = list(ds_dir.rglob("*_eeg.vhdr"))
    if not sub_files:
        import pytest

        pytest.skip("VHDR file missing in snapshot")

    vhdr_path = sub_files[0]
    sidecar_path = vhdr_path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(
            {
                "TaskName": "motorloc",
                "SamplingFrequency": 5000,
                "PowerLineFrequency": 50,
                "EEGReference": "Cz",
                "SoftwareFilters": {"HighPass": 0.1, "LowPass": 1000},
                "Manufacturer": "Brain Products",
                "ManufacturersModelName": "BrainAmp DC",
                "EEGPlacementScheme": "10-20",
                "Acknowledgements": "Test data",
            }
        )
    )

    # dataset_description.json with extras
    desc_path = ds_dir / "dataset_description.json"
    desc = json.loads(desc_path.read_text()) if desc_path.exists() else {}
    desc.update(
        {
            "Name": "VHDR snapshot",
            "BIDSVersion": "1.6.0",
            "Acknowledgements": "Funded by Lab Q",
            "HowToAcknowledge": "Cite our paper",
            "EthicsApprovals": ["IRB-001"],
        }
    )
    desc_path.write_text(json.dumps(desc))

    output_dir = tmp_path / "outputs"
    summary = digest.digest_dataset("ds_snapshot_vhdr", work_inputs, output_dir)
    assert summary["status"] == "success", summary

    # Assert the dataset doc carries the extras
    dataset_doc = json.loads(
        (output_dir / "ds_snapshot_vhdr" / "ds_snapshot_vhdr_dataset.json").read_text()
    )
    assert dataset_doc.get("acknowledgements") == "Funded by Lab Q"
    assert dataset_doc.get("how_to_acknowledge") == "Cite our paper"
    assert dataset_doc.get("ethics_approvals") == ["IRB-001"]

    # Assert the record carries the new fields
    records_doc = json.loads(
        (output_dir / "ds_snapshot_vhdr" / "ds_snapshot_vhdr_records.json").read_text()
    )
    assert records_doc["records"]
    rec = records_doc["records"][0]
    assert rec.get("power_line_frequency") == 50
    assert rec.get("eeg_reference") == "Cz"
    assert rec.get("software_filters") == {"HighPass": 0.1, "LowPass": 1000}
    assert rec.get("manufacturer") == "Brain Products"
    assert rec.get("manufacturers_model_name") == "BrainAmp DC"
    assert rec.get("eeg_placement_scheme") == "10-20"

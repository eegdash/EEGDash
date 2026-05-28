"""Tests for BIDS sidecar enrichment in extract_record / extract_dataset.

Before sidecar enrichment landed, the only BIDS-spec fields surfaced as
structured Record columns were the technical metadata (sfreq / nchans /
ntimes / ch_names). PowerLineFrequency, EEGReference, SoftwareFilters,
Manufacturer, EEGPlacementScheme — all required or recommended by BIDS —
were either unread or buried as raw bytes in sidecar_inline.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
from _helpers import INGEST_DIR as _INGEST_DIR
from eegdash.testing import data_file


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


@pytest.mark.parametrize(
    ("sidecar_payload", "expected"),
    [
        pytest.param(
            {"PowerLineFrequency": 60},
            {"power_line_frequency": 60},
            id="power_line_frequency",
        ),
        pytest.param(
            {"EEGReference": "linked mastoids"},
            {"eeg_reference": "linked mastoids"},
            id="eeg_reference",
        ),
        pytest.param(
            {"SoftwareFilters": {"HighPass": 0.1, "LowPass": 60}},
            {"software_filters": {"HighPass": 0.1, "LowPass": 60}},
            id="software_filters_dict",
        ),
        pytest.param(
            {"Manufacturer": "Brain Products", "ManufacturersModelName": "BrainAmp DC"},
            {
                "manufacturer": "Brain Products",
                "manufacturers_model_name": "BrainAmp DC",
            },
            id="manufacturer_and_model",
        ),
        pytest.param(
            {"EEGPlacementScheme": "10-20"},
            {"eeg_placement_scheme": "10-20"},
            id="eeg_placement_scheme",
        ),
    ],
)
def test_sidecar_extracts_fields(tmp_path: Path, sidecar_payload: dict, expected: dict):
    """_extract_bids_sidecar_fields surfaces each BIDS-spec field correctly.

    SoftwareFilters is a nested dict in the BIDS spec — preserved as-is.
    EEGPlacementScheme feeds the _montage.py template matcher.
    """
    digest = _load_digest()
    fake, data = _build_bids_root_with_sidecar(tmp_path, sidecar_payload)
    out = digest._extract_bids_sidecar_fields(fake, data)
    for key, value in expected.items():
        assert out[key] == value


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


@pytest.mark.parametrize(
    ("tsv_content", "expected"),
    [
        pytest.param(
            "name\ttype\tstatus\nCz\tEEG\tgood\nFz\tEEG\tbad\nOz\tEEG\tbad\nPz\tEEG\tgood\n",
            {"bad_channels": ["Fz", "Oz"], "bad_channels_count": 2},
            id="counts_bad_channels",
        ),
        pytest.param(
            "name\tstatus\nCz\tgood\nFz\tgood\n",
            {"bad_channels_count": 0, "bad_channels": []},
            id="no_bad_channels",
        ),
        pytest.param(
            "name\ttype\nCz\tEEG\nFz\tEEG\n",
            {},
            id="no_status_column",
        ),
        pytest.param(
            "name\tstatus\nCz\tBAD\nFz\tBad\nOz\tbad\nPz\tgood\n",
            {"bad_channels_count": 3},
            id="case_insensitive_bad",
        ),
    ],
)
def test_channel_status_counts(tmp_path: Path, tsv_content: str, expected: dict):
    """_extract_channel_status_counts reads channels.tsv correctly.

    'BAD' / 'Bad' / 'bad' all count; missing status column → empty dict.
    """
    digest = _load_digest()
    sub_dir = tmp_path / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)
    data = sub_dir / "sub-01_task-rest_eeg.edf"
    data.touch()
    (sub_dir / "sub-01_task-rest_channels.tsv").write_text(tsv_content)
    fake = _FakeBIDSDataset(bids_root=tmp_path)
    out = digest._extract_channel_status_counts(fake, str(data))
    if expected == {}:
        assert out == {}
    else:
        for key, value in expected.items():
            if key == "bad_channels":
                assert sorted(out[key]) == sorted(value)
            else:
                assert out[key] == value


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


# ─── _extract_dataset_description_extras ──────────────────────────────────


@pytest.mark.parametrize(
    ("desc_payload", "expected_key", "expected_value"),
    [
        pytest.param(
            {"Acknowledgements": "Funded by grant XYZ"},
            "acknowledgements",
            "Funded by grant XYZ",
            id="acknowledgements",
        ),
        pytest.param(
            {"HowToAcknowledge": "Please cite our paper"},
            "how_to_acknowledge",
            "Please cite our paper",
            id="how_to_acknowledge",
        ),
        pytest.param(
            {"EthicsApprovals": ["IRB-12345", "IRB-67890"]},
            "ethics_approvals",
            ["IRB-12345", "IRB-67890"],
            id="ethics_approvals",
        ),
    ],
)
def test_dataset_extras_extracts_field(desc_payload, expected_key, expected_value):
    """_extract_dataset_description_extras surfaces each BIDS dataset field."""
    digest = _load_digest()
    out = digest._extract_dataset_description_extras(desc_payload)
    assert out[expected_key] == expected_value


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

        pytest.skip("snapshot fixture not available")

    # Copy the snapshot to a tmp_path so we can mutate

    work_inputs = tmp_path / "inputs"
    shutil.copytree(inputs_root.parent, work_inputs)

    # Write a richer modality sidecar with the C6.1 fields
    ds_dir = work_inputs / "ds_snapshot_vhdr"
    sub_files = list(ds_dir.rglob("*_eeg.vhdr"))
    if not sub_files:
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

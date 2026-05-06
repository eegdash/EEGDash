"""E2E workflow tests for user-visible BIDS catalog journeys."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eegdash.dataset.bids_dataset import EEGBIDSDataset


def _write_channels_tsv(path: Path, channels: list[tuple[str, str]]) -> None:
    lines = ["name\ttype"] + [f"{name}\t{kind}" for name, kind in channels]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_synthetic_bids_dataset(tmp_path: Path, scenario: dict) -> Path:
    dataset_root = tmp_path / scenario["dataset_name"]
    dataset_root.mkdir()

    (dataset_root / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "Synthetic workflow dataset",
                "BIDSVersion": "1.8.0",
                "DatasetType": "raw",
            }
        ),
        encoding="utf-8",
    )

    participants_lines = ["participant_id\tgroup"]
    participants_lines.extend(f"{pid}\tcontrol" for pid in scenario["participants"])
    (dataset_root / "participants.tsv").write_text(
        "\n".join(participants_lines) + "\n",
        encoding="utf-8",
    )

    for rec in scenario["recordings"]:
        eeg_dir = dataset_root / f"sub-{rec['subject']}"
        if rec.get("session") is not None:
            eeg_dir = eeg_dir / f"ses-{rec['session']}"
        eeg_dir = eeg_dir / "eeg"
        eeg_dir.mkdir(parents=True, exist_ok=True)

        stem_parts = [f"sub-{rec['subject']}"]
        if rec.get("session") is not None:
            stem_parts.append(f"ses-{rec['session']}")
        stem_parts.append(f"task-{rec['task']}")
        stem_parts.append(f"run-{rec['run']}")
        stem_parts.append("eeg")
        stem = "_".join(stem_parts)

        (eeg_dir / f"{stem}.edf").write_text("", encoding="utf-8")
        (eeg_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "SamplingFrequency": rec["sfreq"],
                    "RecordingDuration": rec["duration"],
                    "EEGChannelCount": sum(
                        1 for _, kind in rec["channels"] if kind == "EEG"
                    ),
                    "EOGChannelCount": sum(
                        1 for _, kind in rec["channels"] if kind == "EOG"
                    ),
                }
            ),
            encoding="utf-8",
        )

        if scenario["channels_layout"] == "shared-per-session":
            channels_path = eeg_dir / "channels.tsv"
        else:
            channels_path = eeg_dir / f"{stem.replace('_eeg', '')}_channels.tsv"
        _write_channels_tsv(channels_path, rec["channels"])

    return dataset_root


WORKFLOW_SCENARIOS = [
    pytest.param(
        {
            "dataset_name": "ds-workflow-shared",
            "channels_layout": "shared-per-session",
            "participants": ["sub-01", "sub-02"],
            "recordings": [
                {
                    "subject": "01",
                    "session": "01",
                    "task": "rest",
                    "run": "01",
                    "sfreq": 128.0,
                    "duration": 10.0,
                    "channels": [("C3", "EEG"), ("C4", "EEG"), ("EOG1", "EOG")],
                },
                {
                    "subject": "01",
                    "session": "01",
                    "task": "rest",
                    "run": "02",
                    "sfreq": 128.0,
                    "duration": 12.0,
                    "channels": [("C3", "EEG"), ("C4", "EEG"), ("EOG1", "EOG")],
                },
                {
                    "subject": "02",
                    "session": "01",
                    "task": "rest",
                    "run": "01",
                    "sfreq": 256.0,
                    "duration": 8.0,
                    "channels": [("F3", "EEG"), ("F4", "EEG"), ("EOG2", "EOG")],
                },
            ],
            "expected": {
                "recordings": 3,
                "subjects": {"01", "02"},
                "tasks": {"rest"},
                "orphans": set(),
            },
        },
        id="shared-channels-resting-workflow",
    ),
    pytest.param(
        {
            "dataset_name": "ds-workflow-perfile",
            "channels_layout": "per-recording",
            "participants": ["sub-01", "sub-02", "sub-99"],
            "recordings": [
                {
                    "subject": "01",
                    "session": None,
                    "task": "rest",
                    "run": "01",
                    "sfreq": 200.0,
                    "duration": 6.0,
                    "channels": [("Cz", "EEG"), ("Pz", "EEG")],
                },
                {
                    "subject": "02",
                    "session": None,
                    "task": "motor",
                    "run": "01",
                    "sfreq": 200.0,
                    "duration": 6.0,
                    "channels": [("C3", "EEG"), ("C4", "EEG"), ("EMG1", "EOG")],
                },
                {
                    "subject": "02",
                    "session": None,
                    "task": "motor",
                    "run": "02",
                    "sfreq": 200.0,
                    "duration": 4.0,
                    "channels": [("C3", "EEG"), ("C4", "EEG"), ("Oz", "EEG")],
                },
            ],
            "expected": {
                "recordings": 3,
                "subjects": {"01", "02"},
                "tasks": {"rest", "motor"},
                "orphans": {"sub-99"},
            },
        },
        id="per-recording-channels-with-orphan-participant",
    ),
]


@pytest.mark.parametrize("scenario", WORKFLOW_SCENARIOS)
@pytest.mark.parametrize(
    "modalities", [None, ["eeg"]], ids=["all-ephy-default", "eeg-only"]
)
def test_user_can_build_bids_catalog_and_participant_qc_report(
    tmp_path: Path, scenario: dict, modalities: list[str] | None
) -> None:
    """Validate an end-to-end catalog+QC workflow over realistic BIDS variants."""
    dataset_root = _create_synthetic_bids_dataset(tmp_path, scenario)
    dataset = EEGBIDSDataset(
        data_dir=str(dataset_root),
        dataset=dataset_root.name,
        allow_symlinks=False,
        modalities=modalities,
    )

    files = sorted(dataset.get_files())
    assert len(files) == scenario["expected"]["recordings"]
    assert dataset.check_eeg_dataset()

    catalog = []
    for file_path in files:
        labels = dataset.channel_labels(file_path)
        num_times = dataset.num_times(file_path)
        rel_path = dataset.get_relative_bidspath(file_path)
        catalog.append(
            {
                "subject": dataset.get_bids_file_attribute("subject", file_path),
                "task": dataset.get_bids_file_attribute("task", file_path),
                "run": dataset.get_bids_file_attribute("run", file_path),
                "n_channels": len(labels),
                "num_times": num_times,
                "relative_path": rel_path,
            }
        )

    summary = {
        "recordings": len(catalog),
        "subjects": {entry["subject"] for entry in catalog},
        "tasks": {entry["task"] for entry in catalog},
    }
    assert summary["recordings"] == scenario["expected"]["recordings"]
    assert summary["subjects"] == scenario["expected"]["subjects"]
    assert summary["tasks"] == scenario["expected"]["tasks"]
    assert all(entry["n_channels"] > 0 for entry in catalog)
    assert all(entry["num_times"] > 0 for entry in catalog)
    assert all(
        entry["relative_path"].startswith(f"{dataset_root.name}/") for entry in catalog
    )

    participants = dataset.get_all_participants_tsv()
    assert set(participants) == set(scenario["participants"])

    orphan_participants = dataset.get_orphan_participants()
    assert set(orphan_participants) == scenario["expected"]["orphans"]

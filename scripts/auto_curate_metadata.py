import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def match_metadata(
    dataset_id: str,
    target_info: dict[str, Any],
) -> dict[str, Any]:
    """Infer metadata from target info."""
    metadata = {
        "is_clinical": "False",
        "clinical_purpose": "",
        "paradigm_modality": "",
        "cognitive_domain": "",
        "is_10_20_system": "",  # Leave blank unless sure
    }

    # 1. Is Clinical?
    # Check text fields
    text_content = (
        (target_info.get("name") or "")
        + " "
        + (target_info.get("readme") or "")
        + " "
        + (target_info.get("study_domain") or "")
    ).lower()

    clinical_keywords = [
        "patient",
        "disorder",
        "syndrome",
        "epilepsy",
        "stroke",
        "dementia",
        "autism",
        "adhd",
        "depression",
        "schizophrenia",
        "alzheimer",
        "parkinson",
        "clinical",
        "pathology",
        "diagnosis",
    ]

    if any(k in text_content for k in clinical_keywords):
        metadata["is_clinical"] = "True"
        # Try to guess purpose
        for k in [
            "epilepsy",
            "dementia",
            "alzheimer",
            "autism",
            "depression",
            "schizophrenia",
            "parkinson",
            "stroke",
        ]:
            if k in text_content:
                metadata["clinical_purpose"] = k
                break
        if not metadata["clinical_purpose"]:
            metadata["clinical_purpose"] = "Unspecified Clinical"

    # 2. Modality
    # Check experimental_modalities first (but this is usually recording modality like eeg/meg)
    # We want paradigm modality (visual, auditory, etc.)
    if "visual" in text_content:
        metadata["paradigm_modality"] = "visual"
    elif "auditory" in text_content:
        metadata["paradigm_modality"] = "auditory"
    elif "resting" in text_content or "resting-state" in text_content:
        metadata["paradigm_modality"] = "resting_state"
    elif "motor" in text_content or "movement" in text_content:
        metadata["paradigm_modality"] = "motor"
    else:
        metadata["paradigm_modality"] = "other"

    if (
        "audio-visual" in text_content
        or "audiovisual" in text_content
        or ("visual" in text_content and "auditory" in text_content)
    ):
        metadata["paradigm_modality"] = "multisensory"

    # 3. Cognitive Domain
    # Use 'tasks' or 'study_domain'
    tasks = target_info.get("tasks", [])
    if tasks:
        # Clean up task name
        task_name = tasks[0]
        # Basic mapping
        if "memory" in task_name.lower():
            metadata["cognitive_domain"] = "memory"
        elif "attention" in task_name.lower():
            metadata["cognitive_domain"] = "attention"
        elif "language" in task_name.lower():
            metadata["cognitive_domain"] = "language"
        elif "emotion" in task_name.lower():
            metadata["cognitive_domain"] = "emotion"
        else:
            metadata["cognitive_domain"] = task_name  # Fallback
    elif target_info.get("study_domain"):
        metadata["cognitive_domain"] = target_info.get("study_domain")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Auto-curate metadata from openneuro_targets.json"
    )
    parser.add_argument(
        "--targets-file", type=Path, default=Path("consolidated/openneuro_targets.json")
    )
    parser.add_argument(
        "--curation-file", type=Path, default=Path("scripts/metadata_curation.csv")
    )
    args = parser.parse_args()

    if not args.targets_file.exists():
        logging.error(f"Targets file not found: {args.targets_file}")
        return

    if not args.curation_file.exists():
        logging.error(
            f"Curation file not found: {args.curation_file}. Run update_dataset_metadata.py --export first."
        )
        # Create dummy if needed? Better to fail.
        return

    # Load targets
    with open(args.targets_file, "r") as f:
        targets_list = json.load(f)

    target_map = {t["dataset_id"]: t for t in targets_list}
    logging.info(f"Loaded {len(target_map)} targets.")

    # Process CSV
    rows = []
    updated_count = 0

    with open(args.curation_file, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            ds_id = row["dataset_id"]
            if ds_id in target_map:
                inferred = match_metadata(ds_id, target_map[ds_id])

                # Update fields if empty or force update?
                # Let's overwrite since we are auto-curating "everything"
                row["is_clinical"] = inferred["is_clinical"]
                row["clinical_purpose"] = inferred["clinical_purpose"]
                row["paradigm_modality"] = inferred["paradigm_modality"]
                row["cognitive_domain"] = inferred["cognitive_domain"]
                # row["is_10_20_system"] = inferred["is_10_20_system"] # Keep existing/empty

                updated_count += 1
            rows.append(row)

    # Write back
    with open(args.curation_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f"Updated {updated_count} rows in {args.curation_file}")


if __name__ == "__main__":
    main()

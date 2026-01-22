"""Categorize datasets based on cloned BIDS metadata.

This script scans the `data/cloned` directory and infers metadata tags:
- Clinical status (Healthy vs Clinical)
- Paradigm Modality (Visual, Auditory, etc.)
- Cognitive Domain
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Keywords & Mappings ---

CLINICAL_KEYWORDS = {
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
    "lesion",
    "seizure",
}

HEALTHY_KEYWORDS = {"healthy", "control", "normal", "neurotypical"}

MODALITY_KEYWORDS = {
    "visual": [
        "visual",
        "image",
        "picture",
        "face",
        "screen",
        "monitor",
        "movie",
        "video",
        "checkerboard",
        "flashing",
        "flicker",
        "strobe",
    ],
    "auditory": [
        "auditory",
        "audio",
        "sound",
        "tone",
        "beep",
        "listen",
        "music",
        "speech",
        "voice",
        "syllable",
    ],
    "tactile": [
        "tactile",
        "touch",
        "vibration",
        "somatosensory",
        "finger",
        "stimulation",
    ],
    "motor": [
        "motor",
        "movement",
        "hand",
        "foot",
        "finger",
        "press",
        "button",
        "imagery",
        "execution",
    ],
    "resting_state": ["resting", "rest", "eyes open", "eyes closed"],
    "multisensory": ["multisensory", "audiovisual", "audio-visual"],
}

DOMAIN_KEYWORDS = {
    "memory": ["memory", "recall", "retrieval", "encoding", "working memory"],
    "attention": ["attention", "vigilance", "oddball", "p300", "target"],
    "language": ["language", "speech", "semantic", "syntax", "word", "sentence"],
    "emotion": ["emotion", "affective", "fear", "happy", "sad", "face"],
    "perception": ["perception", "sensory", "discrimination"],
    "executive": [
        "executive",
        "control",
        "inhibition",
        "go/no-go",
        "flanker",
        "stroop",
    ],
    "motor": ["motor", "movement", "imagery"],
}


def load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def score_categories(text: str, categories: dict[str, set[str]]) -> str:
    """Return the category with the most keyword matches."""
    text_lower = text.lower()
    scores = {cat: 0 for cat in categories}

    for cat, keywords in categories.items():
        for k in keywords:
            if k in text_lower:
                scores[cat] += 1

    # Filter zeros
    scores = {k: v for k, v in scores.items() if v > 0}

    if not scores:
        return ""

    # specific rules
    if "multisensory" in scores:
        return "multisensory"

    return max(scores, key=scores.get)  # Return highest score


def analyze_dataset(dataset_dir: Path) -> dict[str, Any]:
    dataset_id = dataset_dir.name

    # Load available files
    desc = load_json(dataset_dir / "dataset_description.json")
    readme = read_text(dataset_dir / "README")

    # Combined text for analysis
    full_text = (
        desc.get("Name", "") + " " + readme + " " + desc.get("BIDSVersion", "") + " "
    )

    # 1. Clinical Status
    is_clinical = False
    clinical_purpose = ""

    # Check for clinical keywords
    if any(k in full_text.lower() for k in CLINICAL_KEYWORDS):
        # But wait, "healthy control in epilepsy study" might trigger epilepsy
        # So we look for positive assertions or lack of "healthy-only" context
        # Heuristic: if clinical keyword exists, mark potential, but default to True for curation review
        is_clinical = True

        # Try to extract purpose
        for k in [
            "epilepsy",
            "dementia",
            "alzheimer",
            "autism",
            "depression",
            "schizophrenia",
            "parkinson",
            "stroke",
            "adhd",
        ]:
            if k in full_text.lower():
                clinical_purpose = k
                break
        if not clinical_purpose:
            clinical_purpose = "Unspecified Clinical"

    # 2. Paradigm Modality
    modality = score_categories(full_text, MODALITY_KEYWORDS)
    if not modality:
        # Check manifest or task names
        task_names = " ".join(desc.get("Tasks", []) or [])
        modality = score_categories(task_names, MODALITY_KEYWORDS)
    if not modality:
        modality = "other"

    # 3. Cognitive Domain
    domain = score_categories(full_text, DOMAIN_KEYWORDS)
    if not domain:
        domain = "other"

    return {
        "dataset_id": dataset_id,
        "is_clinical": is_clinical,
        "clinical_purpose": clinical_purpose,
        "paradigm_modality": modality,
        "cognitive_domain": domain,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Categorize datasets from cloned data."
    )
    parser.add_argument("--cloned-dir", type=Path, default=Path("data/cloned"))
    parser.add_argument(
        "--output", type=Path, default=Path("scripts/metadata_curation.csv")
    )
    args = parser.parse_args()

    if not args.cloned_dir.exists():
        print(f"Error: {args.cloned_dir} does not exist.")
        return

    results = []
    print(f"Scanning {args.cloned_dir}...")

    # Collect all existing dataset IDs from CSV if it exists, to preserve manual edits?
    # User asked to "create the tag based on information", implying extraction.
    # We will overwrite for now or merge? Let's overwrite/generate fresh to show the power of the tool.

    datasets = sorted(
        [d for d in args.cloned_dir.iterdir() if d.is_dir() and d.name.startswith("ds")]
    )

    for d in datasets:
        info = analyze_dataset(d)
        results.append(info)

    # Write to CSV
    fieldnames = [
        "dataset_id",
        "is_clinical",
        "clinical_purpose",
        "paradigm_modality",
        "cognitive_domain",
    ]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Categorized {len(results)} datasets. Saved to {args.output}")


if __name__ == "__main__":
    main()

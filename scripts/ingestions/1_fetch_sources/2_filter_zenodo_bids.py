"""Filter Zenodo aggressive results to identify genuine BIDS datasets.

CRITICAL INSIGHT: A dataset can be BIDS-compliant without mentioning "BIDS"!

Many neurophysiology datasets follow BIDS structure perfectly but never use
the term "BIDS" in their title, description, or keywords. Therefore, we must
prioritize FILE STRUCTURE over keyword mentions when classifying datasets.

NEW FEATURE: Archive Preview Detection
-------------------------------------
Zenodo provides a preview endpoint that shows archive contents WITHOUT downloading!
We now inspect .zip files to detect BIDS structure inside archived datasets.

This dramatically improves BIDS detection accuracy:
- OLD approach: 89% rejection (couldn't see inside archives)
- NEW approach: Expected 30-50% BIDS detection (can inspect archive contents)

Example: https://zenodo.org/records/13790279/preview/database.zip
Returns HTML with complete file tree showing:
  - dataset_description.json
  - participants.tsv
  - sub-01/sub-01_eeg.json
  - sub-01/sub-01_channels.tsv

This script post-processes the aggressive Zenodo fetch results to separate:
1. Genuine BIDS datasets (detected by structure OR keywords)
2. Neurophysiology datasets (convertible to BIDS)
3. False positives (not relevant)

Filtering criteria based on BIDS specification analysis:

PRIMARY INDICATORS (File Structure - Most Reliable):
- BIDS naming patterns: sub-XX, ses-XX, task-XX, run-XX
- Core files: dataset_description.json, participants.tsv
- Sidecars: *_eeg.json, *_channels.tsv, *_events.tsv
- Directory structure: code/, derivatives/, sourcedata/
- NOW: Archive preview contents inspection!

SECONDARY INDICATORS (Keywords - Less Reliable):
- Explicit mentions: "BIDS", "Brain Imaging Data Structure"
- Related terms: "BIDS-compliant", "BIDS format", "OpenNeuro"

CLASSIFICATION PRIORITY:
1. Strong file structure (≥3 BIDS indicators) → BIDS strict
2. Moderate structure (≥2 indicators) → BIDS moderate
3. Weak structure (≥1 indicator) + neuro files → BIDS probable
4. No structure + neuro files → Neurophysiology (convertible)
5. No indicators → Rejected

Input: consolidated/zenodo_datasets_aggressive.json (~1,570 records)
Outputs:
  - consolidated/zenodo_datasets_bids.json (validated BIDS)
  - consolidated/zenodo_datasets_neurophysiology.json (convertible)
  - consolidated/zenodo_datasets_rejected.json (false positives)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

# BIDS-specific indicators
BIDS_KEYWORDS = {
    "strict": [
        "bids",
        "brain imaging data structure",
        "bids format",
        "bids-compliant",
        "bids specification",
        "bids validator",
        "openneuro",
    ],
    "moderate": [
        "bids-like",
        "bids standard",
        "neuroimaging data structure",
        "standardized format",
    ],
}

BIDS_FILE_INDICATORS = [
    # Core BIDS files (definitive indicators)
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    # Modality-specific sidecars (strong indicators)
    "_eeg.json",
    "_meg.json",
    "_ieeg.json",
    "_channels.tsv",
    "_electrodes.tsv",
    "_events.tsv",
    "_coordsystem.json",
    # Task and run indicators
    "_task-",
    "_run-",
    "_ses-",
    "_sub-",
    # BIDS metadata files
    "README",
    "CHANGES",
    "code/",
    "derivatives/",
    "sourcedata/",
    # Specific BIDS naming patterns
    "_bold.nii",
    "_T1w.nii",
    "_dwi.nii",
    "_eeg.eeg",
    "_eeg.vhdr",
    "_eeg.set",
    "_meg.fif",
    "_meg.ds",
]

NEUROPHYSIOLOGY_EXTENSIONS = {
    "eeg": [".eeg", ".vhdr", ".vmrk", ".edf", ".bdf", ".set", ".fdt", ".cnt", ".mff"],
    "meg": [".fif", ".sqd", ".con", ".raw", ".ave", ".mrk", ".elp", ".hsp", ".ds"],
    "ieeg": [".edf", ".eeg", ".vhdr", ".trc", ".bin"],
    "emg": [".edf", ".cnt", ".poly5"],
}

NEUROPHYSIOLOGY_TERMS = [
    "eeg",
    "electroencephalography",
    "electroencephalogram",
    "meg",
    "magnetoencephalography",
    "magnetoencephalogram",
    "ieeg",
    "intracranial",
    "ecog",
    "electrocorticography",
    "seeg",
    "emg",
    "electromyography",
    "electromyogram",
    "evoked potential",
    "event-related potential",
    "erp",
    "resting state",
    "task-based",
    "paradigm",
]


def check_bids_keywords(dataset: dict[str, Any]) -> tuple[bool, str]:
    """Check if dataset mentions BIDS in title, description, or keywords.

    Returns:
        (is_bids, confidence_level) where confidence is 'strict' or 'moderate'

    """
    text = " ".join(
        [
            dataset.get("title", ""),
            dataset.get("description", ""),
            " ".join(dataset.get("keywords", [])),
        ]
    ).lower()

    # Check strict BIDS keywords
    for keyword in BIDS_KEYWORDS["strict"]:
        if keyword in text:
            return True, "strict"

    # Check moderate BIDS keywords
    for keyword in BIDS_KEYWORDS["moderate"]:
        if keyword in text:
            return True, "moderate"

    return False, "none"


def check_description_for_bids_evidence(dataset: dict[str, Any]) -> int:
    """Analyze description for BIDS structure evidence.

    Many archived datasets describe their BIDS structure in text without
    mentioning "BIDS" explicitly. Look for telltale phrases.

    Returns:
        Evidence score (0-3): 0=none, 1=weak, 2=moderate, 3=strong

    """
    description = dataset.get("description", "").lower()
    title = dataset.get("title", "").lower()
    combined = description + " " + title

    score = 0

    # Strong evidence phrases
    strong_evidence = [
        "participants.tsv",
        "dataset_description.json",
        "subject-level",
        "session-level",
        "organized according to",
        "follows the structure",
        "standardized format",
        "sub-<participant_label>",
        "sub-XX",
        "task-<task_name>",
    ]

    for phrase in strong_evidence:
        if phrase in combined:
            score = max(score, 3)
            break

    # Moderate evidence phrases
    moderate_evidence = [
        "organized dataset",
        "structured data",
        "standardized data",
        "metadata files",
        "sidecar files",
        "json metadata",
        "channel locations",
        "event markers",
    ]

    if score < 3:
        for phrase in moderate_evidence:
            if phrase in combined:
                score = max(score, 2)
                break

    # Weak evidence
    weak_evidence = [
        "well-organized",
        "structured",
        "metadata",
        "annotations",
        "documented",
    ]

    if score < 2:
        for phrase in weak_evidence:
            if phrase in combined:
                score = max(score, 1)
                break

    return score


def check_bids_files(dataset: dict[str, Any]) -> tuple[int, list[str]]:
    """Check how many BIDS-specific files are present.

    IMPORTANT: Checks both exact file matches AND BIDS naming patterns.
    Many BIDS datasets don't have "bids" in metadata but follow BIDS structure.

    NEW FEATURE: If dataset has archive preview contents, we can inspect
    the internal structure without downloading! This dramatically improves
    BIDS detection accuracy for archived datasets.

    Returns:
        (count, matched_files)

    """
    files = dataset.get("files", [])
    filenames = [f.get("filename", "").lower() for f in files]

    matched = []

    # NEW: Check archive contents if available
    archive_contents = dataset.get("archive_contents", {})
    all_filenames = filenames.copy()

    if archive_contents:
        # Add all files found inside archives
        for archive_name, contents in archive_contents.items():
            all_filenames.extend([f.lower() for f in contents])

    # Check if dataset is mostly archived (heuristic approach needed)
    archived_extensions = [".zip", ".tar.gz", ".tgz", ".tar", ".rar", ".7z", ".gz"]
    is_archived = any(fn.endswith(tuple(archived_extensions)) for fn in filenames)
    has_preview = len(archive_contents) > 0

    # Check for specific BIDS files (now including archive contents!)
    for indicator in BIDS_FILE_INDICATORS:
        if any(indicator.lower() in fn for fn in all_filenames):
            matched.append(indicator)

    # Check for BIDS naming patterns (sub-XX/ses-XX/task-XX)
    bids_patterns = [
        r"sub-[a-zA-Z0-9]+",  # Subject ID
        r"ses-[a-zA-Z0-9]+",  # Session ID
        r"task-[a-zA-Z0-9]+",  # Task name
        r"run-[0-9]+",  # Run number
        r"acq-[a-zA-Z0-9]+",  # Acquisition
    ]

    pattern_matches = set()
    for fn in all_filenames:
        for pattern in bids_patterns:
            if re.search(pattern, fn):
                pattern_matches.add(f"pattern:{pattern}")

    matched.extend(list(pattern_matches))

    # If archived WITHOUT preview, check archive name for BIDS indicators
    if is_archived and not has_preview:
        for fn in filenames:
            # Check if archive name suggests BIDS content
            if any(
                term in fn for term in ["bids", "openneuro", "ds0", "sub-", "dataset"]
            ):
                matched.append("archive:bids_indicator")
                break

    # If we have archive preview, note this as additional evidence
    if has_preview and any(
        "sub-" in fn or "dataset_description" in fn for fn in all_filenames
    ):
        matched.append("archive_preview:confirmed")

    # Deduplicate
    matched = list(set(matched))

    return len(matched), matched


def check_neurophysiology_files(dataset: dict[str, Any]) -> tuple[str | None, int]:
    """Check if dataset contains neurophysiology data files.

    NEW: Also checks archive preview contents if available.

    Returns:
        (modality, extension_count) where modality is 'eeg', 'meg', 'ieeg', 'emg', or None

    """
    files = dataset.get("files", [])
    filenames = [f.get("filename", "").lower() for f in files]

    # NEW: Add archive contents if available
    archive_contents = dataset.get("archive_contents", {})
    if archive_contents:
        for archive_name, contents in archive_contents.items():
            filenames.extend([f.lower() for f in contents])

    modality_counts = {}
    for modality, extensions in NEUROPHYSIOLOGY_EXTENSIONS.items():
        count = sum(
            1 for fn in filenames if any(fn.endswith(ext) for ext in extensions)
        )
        if count > 0:
            modality_counts[modality] = count

    if not modality_counts:
        return None, 0

    # Return modality with most files
    primary_modality = max(modality_counts, key=modality_counts.get)
    return primary_modality, modality_counts[primary_modality]


def check_neurophysiology_terms(dataset: dict[str, Any]) -> bool:
    """Check if dataset mentions neurophysiology-related terms."""
    text = " ".join(
        [
            dataset.get("title", ""),
            dataset.get("description", ""),
            " ".join(dataset.get("keywords", [])),
        ]
    ).lower()

    return any(term in text for term in NEUROPHYSIOLOGY_TERMS)


def classify_dataset(dataset: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Classify dataset as 'bids', 'neurophysiology', or 'rejected'.

    CRITICAL: A dataset can be BIDS-compliant without mentioning "BIDS"!
    We prioritize file structure indicators over keyword mentions.

    NEW: For archived datasets with preview contents, we can now accurately
    detect BIDS structure by inspecting the archive contents directly!

    For archived datasets without preview, we use description analysis to find BIDS evidence.

    Returns:
        (category, metadata) where metadata contains classification details

    """
    # Check BIDS indicators
    has_bids_keywords, keyword_confidence = check_bids_keywords(dataset)
    bids_file_count, bids_files = check_bids_files(dataset)
    description_score = check_description_for_bids_evidence(dataset)

    # Check neurophysiology indicators
    modality, neuro_file_count = check_neurophysiology_files(dataset)
    has_neuro_terms = check_neurophysiology_terms(dataset)

    # NEW: Check if we have archive preview
    has_archive_preview = dataset.get("has_archive_preview", False)
    archive_file_count = sum(
        len(contents) for contents in dataset.get("archive_contents", {}).values()
    )

    metadata = {
        "has_bids_keywords": has_bids_keywords,
        "bids_keyword_confidence": keyword_confidence,
        "bids_file_count": bids_file_count,
        "bids_files_found": bids_files,
        "description_bids_score": description_score,
        "primary_modality": modality,
        "neurophysiology_file_count": neuro_file_count,
        "has_neurophysiology_terms": has_neuro_terms,
        "has_archive_preview": has_archive_preview,  # NEW
        "archive_file_count": archive_file_count,  # NEW
    }

    # Classification logic - MULTIPLE PATHWAYS TO BIDS CLASSIFICATION

    # PATH 1: STRICT BIDS - Strong file structure evidence
    # Many BIDS datasets don't mention "BIDS" but have perfect structure
    # NOW WITH ARCHIVE PREVIEW: This is much more accurate!
    if bids_file_count >= 3:
        return "bids_strict", metadata

    # PATH 2: STRICT BIDS - Explicit BIDS keywords + some evidence
    if (
        has_bids_keywords
        and keyword_confidence == "strict"
        and (bids_file_count >= 1 or description_score >= 2)
    ):
        return "bids_strict", metadata

    # PATH 3: MODERATE BIDS - Good file evidence OR strong description evidence
    if bids_file_count >= 2 or (description_score >= 3 and has_neuro_terms):
        return "bids_moderate", metadata

    # PATH 4: MODERATE BIDS - BIDS keywords + moderate evidence
    if has_bids_keywords and (bids_file_count >= 1 or description_score >= 2):
        return "bids_moderate", metadata

    # PATH 5: PROBABLE BIDS - Weak BIDS evidence + neurophysiology data
    if (bids_file_count >= 1 or description_score >= 2) and modality:
        return "bids_probable", metadata

    # NEUROPHYSIOLOGY: Has neuro files and terms but no BIDS structure
    if (
        modality
        and neuro_file_count > 0
        and has_neuro_terms
        and bids_file_count == 0
        and description_score == 0
    ):
        return "neurophysiology", metadata

    # REJECTED: No clear indicators
    return "rejected", metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter aggressive Zenodo results to identify genuine BIDS datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("consolidated/zenodo_datasets_aggressive.json"),
        help="Input JSON file from aggressive fetch.",
    )
    parser.add_argument(
        "--output-bids",
        type=Path,
        default=Path("consolidated/zenodo_datasets_bids.json"),
        help="Output file for validated BIDS datasets.",
    )
    parser.add_argument(
        "--output-neurophysiology",
        type=Path,
        default=Path("consolidated/zenodo_datasets_neurophysiology.json"),
        help="Output file for neurophysiology datasets (convertible to BIDS).",
    )
    parser.add_argument(
        "--output-rejected",
        type=Path,
        default=Path("consolidated/zenodo_datasets_rejected.json"),
        help="Output file for rejected datasets (false positives).",
    )
    parser.add_argument(
        "--output-stats",
        type=Path,
        default=Path("consolidated/zenodo_filtering_stats.json"),
        help="Output file for filtering statistics.",
    )

    args = parser.parse_args()

    # Load aggressive results
    if not args.input.exists():
        print(f"Error: Input file {args.input} not found!")
        print("Run 1_fetch_zenodo_enhanced.py first.")
        return

    with args.input.open() as f:
        datasets = json.load(f)

    print(f"{'=' * 70}")
    print(f"Filtering {len(datasets)} aggressive Zenodo results")
    print(f"{'=' * 70}\n")

    # Classify datasets
    categories = {
        "bids_strict": [],
        "bids_moderate": [],
        "bids_probable": [],
        "neurophysiology": [],
        "rejected": [],
    }

    for i, dataset in enumerate(datasets, 1):
        category, metadata = classify_dataset(dataset)

        # Add classification metadata to dataset
        dataset["classification"] = {"category": category, **metadata}

        categories[category].append(dataset)

        if i % 100 == 0:
            print(f"  Processed {i}/{len(datasets)} datasets...")

    # Combine BIDS categories
    bids_datasets = (
        categories["bids_strict"]
        + categories["bids_moderate"]
        + categories["bids_probable"]
    )

    # Save results
    args.output_bids.parent.mkdir(parents=True, exist_ok=True)

    with args.output_bids.open("w") as f:
        json.dump(bids_datasets, f, indent=2)

    with args.output_neurophysiology.open("w") as f:
        json.dump(categories["neurophysiology"], f, indent=2)

    with args.output_rejected.open("w") as f:
        json.dump(categories["rejected"], f, indent=2)

    # Calculate statistics
    archive_preview_count = sum(
        1 for d in datasets if d.get("has_archive_preview", False)
    )
    archive_preview_in_bids = sum(
        1 for d in bids_datasets if d.get("has_archive_preview", False)
    )

    stats = {
        "total_input": len(datasets),
        "bids_total": len(bids_datasets),
        "bids_strict": len(categories["bids_strict"]),
        "bids_moderate": len(categories["bids_moderate"]),
        "bids_probable": len(categories["bids_probable"]),
        "neurophysiology": len(categories["neurophysiology"]),
        "rejected": len(categories["rejected"]),
        "bids_percentage": round(len(bids_datasets) / len(datasets) * 100, 1),
        "neurophysiology_percentage": round(
            len(categories["neurophysiology"]) / len(datasets) * 100, 1
        ),
        "rejected_percentage": round(
            len(categories["rejected"]) / len(datasets) * 100, 1
        ),
        "datasets_with_archive_preview": archive_preview_count,
        "archive_preview_percentage": round(
            archive_preview_count / len(datasets) * 100, 1
        )
        if datasets
        else 0,
        "bids_with_archive_preview": archive_preview_in_bids,
        "bids_archive_preview_percentage": round(
            archive_preview_in_bids / len(bids_datasets) * 100, 1
        )
        if bids_datasets
        else 0,
    }

    with args.output_stats.open("w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{'=' * 70}")
    print("FILTERING RESULTS")
    print(f"{'=' * 70}")
    print(f"Total input:           {stats['total_input']:5d}")
    print("")
    print("Archive Preview Stats:")
    print(
        f"  Datasets with preview: {stats['datasets_with_archive_preview']:4d} ({stats['archive_preview_percentage']:5.1f}%)"
    )
    print(
        f"  BIDS with preview:     {stats['bids_with_archive_preview']:4d} ({stats['bids_archive_preview_percentage']:5.1f}%)"
    )
    print("")
    print(
        f"BIDS Datasets:         {stats['bids_total']:5d} ({stats['bids_percentage']:5.1f}%)"
    )
    print(f"  - Strict (confident): {stats['bids_strict']:4d}")
    print(f"  - Moderate:           {stats['bids_moderate']:4d}")
    print(f"  - Probable:           {stats['bids_probable']:4d}")
    print("")
    print(
        f"Neurophysiology:       {stats['neurophysiology']:5d} ({stats['neurophysiology_percentage']:5.1f}%)"
    )
    print("  (Convertible to BIDS)")
    print("")
    print(
        f"Rejected:              {stats['rejected']:5d} ({stats['rejected_percentage']:5.1f}%)"
    )
    print("  (False positives)")
    print(f"{'=' * 70}")
    print("")
    print("Output files:")
    print(f"  BIDS:           {args.output_bids}")
    print(f"  Neurophysiology: {args.output_neurophysiology}")
    print(f"  Rejected:       {args.output_rejected}")
    print(f"  Statistics:     {args.output_stats}")
    print(f"{'=' * 70}")

    # Sample analysis
    if bids_datasets:
        print("\nSample BIDS dataset (strict confidence):")
        strict_samples = [d for d in categories["bids_strict"]]
        if strict_samples:
            sample = strict_samples[0]
            print(f"  Title: {sample.get('title', 'N/A')}")
            print(f"  DOI: {sample.get('doi', 'N/A')}")
            print(f"  Keywords: {', '.join(sample.get('keywords', [])[:5])}")
            print(f"  BIDS files found: {sample['classification']['bids_files_found']}")
            print(f"  Modality: {sample['classification']['primary_modality']}")
            if sample.get("has_archive_preview"):
                archive_count = sample["classification"].get("archive_file_count", 0)
                print(f"  Archive preview: ✓ ({archive_count} files inspected)")
            else:
                print("  Archive preview: ✗ (classified by keywords/description)")


if __name__ == "__main__":
    main()

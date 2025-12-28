"""Fetch dataset metadata from local BIDS directories.

A generic script for scanning local BIDS-formatted directories and creating
dataset metadata with configurable storage URLs (S3, local paths, etc.).

Usage:
    # HBN NeurIPS2025 Challenge data
    python local_bids.py --input data/NeurIPS2025 --source hbn --s3-base s3://nmdatasets/NeurIPS25

    # Generic local BIDS dataset
    python local_bids.py --input /path/to/bids --source myproject --output consolidated/myproject.json

    # Skip mini releases for HBN
    python local_bids.py --input data/NeurIPS2025 --source hbn --skip-pattern "_mini_"
"""

import argparse
import json
import sys
from pathlib import Path

# Add ingestion paths before importing local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _serialize import save_datasets_deterministically, setup_paths

setup_paths()
from eegdash.records import create_dataset


def parse_participants_tsv(participants_file: Path) -> dict:
    """Parse participants.tsv to extract demographics."""
    demographics = {
        "subjects_count": 0,
        "ages": [],
        "sex_distribution": {},
    }

    if not participants_file.exists():
        return demographics

    try:
        import pandas as pd

        df = pd.read_csv(participants_file, sep="\t")
        demographics["subjects_count"] = len(df)

        # Extract ages
        if "age" in df.columns:
            ages = df["age"].dropna().tolist()
            demographics["ages"] = [int(a) if pd.notna(a) else None for a in ages]
            if ages:
                demographics["age_min"] = min(ages)
                demographics["age_max"] = max(ages)
                demographics["age_mean"] = sum(ages) / len(ages)

        # Extract sex distribution
        if "sex" in df.columns:
            sex_counts = df["sex"].value_counts().to_dict()
            demographics["sex_distribution"] = {
                str(k).lower(): int(v) for k, v in sex_counts.items()
            }
    except Exception as e:
        print(f"  Warning: Could not parse participants.tsv: {e}")

    return demographics


def read_readme(dataset_dir: Path) -> str | None:
    """Read README file from dataset directory."""
    # Try different README filenames
    readme_names = ["README", "README.md", "README.txt", "readme", "readme.md"]

    for name in readme_names:
        readme_file = dataset_dir / name
        if readme_file.exists():
            try:
                return readme_file.read_text(encoding="utf-8")
            except Exception as e:
                print(f"  Warning: Could not read {name}: {e}")

    return None


def scan_local_bids_dataset(
    dataset_dir: Path,
    storage_url: str,
    source: str,
    modality: str = "eeg",
    file_extensions: list[str] | None = None,
) -> dict | None:
    """Scan a local BIDS dataset directory and extract metadata.

    Args:
        dataset_dir: Path to the BIDS dataset directory
        storage_url: URL where the data is stored (S3, HTTP, etc.)
        source: Source identifier (e.g., "hbn", "openneuro")
        modality: Primary recording modality (default: "eeg")
        file_extensions: File extensions to count (default: common EEG formats)

    Returns:
        Dataset dict or None if invalid

    """
    if file_extensions is None:
        file_extensions = [".bdf", ".edf", ".vhdr", ".set", ".fif", ".eeg"]

    dataset_description_file = dataset_dir / "dataset_description.json"
    if not dataset_description_file.exists():
        print(f"  Skipping {dataset_dir.name}: no dataset_description.json")
        return None

    # Load dataset_description.json
    with open(dataset_description_file) as f:
        desc = json.load(f)

    # Parse participants.tsv
    participants_file = dataset_dir / "participants.tsv"
    demographics = parse_participants_tsv(participants_file)

    # Read README
    readme = read_readme(dataset_dir)

    # Count subjects by looking at sub-* directories
    subject_dirs = list(dataset_dir.glob("sub-*"))
    if not demographics["subjects_count"]:
        demographics["subjects_count"] = len(subject_dirs)

    # Extract tasks, sessions, and count files
    tasks = set()
    sessions = set()
    total_files = 0

    for sub_dir in subject_dirs:
        # Check for session directories
        ses_dirs = list(sub_dir.glob("ses-*"))
        if ses_dirs:
            for ses_dir in ses_dirs:
                sessions.add(ses_dir.name.replace("ses-", ""))
                modality_dir = ses_dir / modality
                if modality_dir.exists():
                    for ext in file_extensions:
                        for f in modality_dir.glob(f"*{ext}"):
                            total_files += 1
                            # Extract task from filename
                            parts = f.stem.split("_")
                            for part in parts:
                                if part.startswith("task-"):
                                    tasks.add(part.replace("task-", ""))
        else:
            # No sessions, check directly for modality folder
            modality_dir = sub_dir / modality
            if modality_dir.exists():
                for ext in file_extensions:
                    for f in modality_dir.glob(f"*{ext}"):
                        total_files += 1
                        parts = f.stem.split("_")
                        for part in parts:
                            if part.startswith("task-"):
                                tasks.add(part.replace("task-", ""))

    # Build dataset entry
    dataset_id = dataset_dir.name

    # Get metadata from dataset_description.json
    dataset_doi = desc.get("DatasetDOI")
    authors = desc.get("Authors", [])
    funding = desc.get("Funding", [])
    license_str = desc.get("License", "")

    # Build the dataset dict
    dataset = create_dataset(
        dataset_id=dataset_id,
        name=desc.get("Name", dataset_id),
        source=source,
        recording_modality=modality,
        modalities=[modality],
        bids_version=desc.get("BIDSVersion"),
        license=license_str,
        authors=authors,
        funding=funding,
        dataset_doi=dataset_doi,
        tasks=sorted(tasks) if tasks else [],
        sessions=sorted(sessions) if sessions else [],
        total_files=total_files,
        # Demographics (unpacked)
        subjects_count=demographics.get("subjects_count", 0),
        ages=demographics.get("ages", []),
        age_mean=demographics.get("age_mean"),
        sex_distribution=demographics.get("sex_distribution", {}),
        source_url=storage_url,
    )

    # Add README content
    if readme:
        dataset["readme"] = readme

    # Add storage URL as external link
    if "external_links" not in dataset:
        dataset["external_links"] = {}
    dataset["external_links"]["s3_bucket"] = storage_url

    # Add local_path for clone step (transient field, not stored in DB)
    # Use absolute path to ensure it works from any working directory
    dataset["local_path"] = str(dataset_dir.resolve())

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Fetch dataset metadata from local BIDS directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # HBN NeurIPS2025 Challenge data
    python local_bids.py --input data/NeurIPS2025 --source hbn \\
        --s3-base s3://nmdatasets/NeurIPS25 --output consolidated/hbn_full.json

    # Custom local BIDS dataset
    python local_bids.py --input /data/mybids --source myproject \\
        --output consolidated/myproject.json

    # Skip certain patterns
    python local_bids.py --input data/NeurIPS2025 --source hbn --skip-pattern "_mini_"
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing BIDS datasets (can have multiple subdirectories)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for dataset metadata",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source identifier (e.g., 'hbn', 'openneuro', 'myproject')",
    )
    parser.add_argument(
        "--s3-base",
        type=str,
        default=None,
        help="S3 bucket base URL. If provided, storage URLs will be s3-base/dataset_id",
    )
    parser.add_argument(
        "--storage-url",
        type=str,
        default=None,
        help="Direct storage URL for single dataset. Use --s3-base for multiple datasets.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="eeg",
        help="Primary recording modality (default: eeg)",
    )
    parser.add_argument(
        "--file-extensions",
        nargs="+",
        default=None,
        help="File extensions to count (default: .bdf .edf .vhdr .set .fif .eeg)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific dataset directories to process. Default: all subdirectories",
    )
    parser.add_argument(
        "--skip-pattern",
        type=str,
        default=None,
        help="Skip directories matching this pattern (e.g., '_mini_')",
    )
    parser.add_argument(
        "--single-dataset",
        action="store_true",
        help="Treat input directory as a single BIDS dataset (not parent of multiple)",
    )

    args = parser.parse_args()

    input_dir = args.input
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    print(f"Scanning BIDS datasets from: {input_dir}")
    print(f"Source: {args.source}")
    if args.s3_base:
        print(f"S3 base: {args.s3_base}")

    datasets = []
    total_subjects = 0
    total_files = 0

    if args.single_dataset:
        # Treat input as a single BIDS dataset
        dataset_dirs = [input_dir]
    else:
        # Find all subdirectories as potential datasets
        dataset_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name

        # Skip patterns
        if args.skip_pattern and args.skip_pattern in dataset_name:
            print(f"Skipping: {dataset_name} (matches skip pattern)")
            continue

        # Filter by specific datasets if provided
        if args.datasets and dataset_name not in args.datasets:
            continue

        print(f"\nProcessing: {dataset_name}")

        # Build storage URL
        if args.s3_base:
            storage_url = f"{args.s3_base}/{dataset_name}"
        elif args.storage_url:
            storage_url = args.storage_url
        else:
            storage_url = str(dataset_dir)

        dataset = scan_local_bids_dataset(
            dataset_dir=dataset_dir,
            storage_url=storage_url,
            source=args.source,
            modality=args.modality,
            file_extensions=args.file_extensions,
        )

        if dataset:
            datasets.append(dataset)
            subj_count = dataset.get("demographics", {}).get("subjects_count", 0)
            file_count = dataset.get("total_files", 0)
            total_subjects += subj_count
            total_files += file_count
            has_readme = "✓" if dataset.get("readme") else "✗"
            print(
                f"  -> {subj_count} subjects, {file_count} files, README: {has_readme}"
            )

    if not datasets:
        print("\nNo valid BIDS datasets found!")
        return 1

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_datasets_deterministically(datasets, args.output)

    print(f"\nSaved {len(datasets)} dataset entries to {args.output}")
    print("\nSummary:")
    print(f"  Total datasets: {len(datasets)}")
    print(f"  Total subjects: {total_subjects}")
    print(f"  Total files: {total_files}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

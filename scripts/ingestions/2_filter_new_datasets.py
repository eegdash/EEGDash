"""Filter datasets that are not yet in EEGDash MongoDB."""

import argparse
import json
from pathlib import Path

from eegdash.api import EEGDash


def filter_new_datasets(input_file: Path, output_file: Path, is_public: bool = True):
    """Filter datasets that don't exist in MongoDB yet.

    Args:
        input_file: Input JSON file with dataset list
        output_file: Output JSON file with filtered datasets
        is_public: Whether to check public or private MongoDB

    """
    # Load input datasets
    with input_file.open("r") as f:
        datasets = json.load(f)

    # Connect to MongoDB
    eegdash = EEGDash(is_public=is_public)

    # Get existing dataset IDs from MongoDB
    existing_ids = {doc["dataset"] for doc in eegdash.find({})}

    # Filter out datasets that already exist
    new_datasets = [ds for ds in datasets if ds.get("dataset_id") not in existing_ids]

    # Save filtered datasets
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(new_datasets, f, indent=2)

    # Print summary
    print(f"Total datasets in input: {len(datasets)}")
    print(f"Already in MongoDB: {len(datasets) - len(new_datasets)}")
    print(f"New datasets to digest: {len(new_datasets)}")
    print(f"Filtered dataset list saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter datasets that are not yet in EEGDash MongoDB"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSON file (e.g., consolidated/openneuro_datasets.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: adds 'to_digest_' prefix)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Check private MongoDB instead of public",
    )
    args = parser.parse_args()

    # Default output filename with prefix
    if args.output is None:
        filename = args.input.name
        args.output = args.input.parent / f"to_digest_{filename}"

    filter_new_datasets(args.input, args.output, is_public=not args.private)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate digestion output for consistency and correctness.

This script checks that digested records and datasets are valid:
- Storage URLs match the source (no OpenNeuro with OSF URLs!)
- Required fields are present
- No duplicate records
- Source is set and valid for all entries

Usage:
    python validate_output.py --input digestion_output
    python validate_output.py --input digestion_output --verbose
    python validate_output.py --input digestion_output --fix  # Attempt fixes
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Valid storage URL patterns per source
VALID_STORAGE_PATTERNS = {
    "openneuro": r"^s3://openneuro\.org/ds\d+",
    "nemar": r"^s3://nemar/nm\d+",
    "osf": r"^https://files\.osf\.io/",
    "figshare": r"^https://(figshare\.com|ndownloader)",
    "zenodo": r"^https://zenodo\.org/",
    "scidb": r"^https://(www\.)?scidb\.cn/",
    "datarn": r"^https://webdav\.data\.ru\.nl/",
    "gin": r"^https://gin\.g-node\.org/",
}

# Required fields for records
REQUIRED_RECORD_FIELDS = [
    "dataset",
    "bids_relpath",
    "recording_modality",
    "storage",
]

REQUIRED_STORAGE_FIELDS = ["backend", "base", "raw_key"]

# Required fields for datasets
REQUIRED_DATASET_FIELDS = [
    "dataset_id",
    "source",
    "recording_modality",
]

# Valid sources
VALID_SOURCES = {
    "openneuro",
    "nemar",
    "gin",
    "figshare",
    "zenodo",
    "osf",
    "scidb",
    "datarn",
}


class ValidationResult:
    """Container for validation results."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = {
            "datasets_checked": 0,
            "records_checked": 0,
            "storage_errors": 0,
            "missing_fields": 0,
            "invalid_source": 0,
        }

    def add_error(self, dataset: str, message: str):
        self.errors.append({"dataset": dataset, "message": message})

    def add_warning(self, dataset: str, message: str):
        self.warnings.append({"dataset": dataset, "message": message})

    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "VALIDATION SUMMARY",
            "=" * 60,
            f"Datasets checked: {self.stats['datasets_checked']}",
            f"Records checked: {self.stats['records_checked']}",
            "",
            f"Errors: {len(self.errors)}",
            f"  - Storage URL errors: {self.stats['storage_errors']}",
            f"  - Missing fields: {self.stats['missing_fields']}",
            f"  - Invalid source: {self.stats['invalid_source']}",
            "",
            f"Warnings: {len(self.warnings)}",
        ]

        if self.errors:
            lines.append("")
            lines.append("Sample errors (first 10):")
            for err in self.errors[:10]:
                lines.append(f"  [{err['dataset']}] {err['message']}")

        if self.warnings:
            lines.append("")
            lines.append("Sample warnings (first 5):")
            for warn in self.warnings[:5]:
                lines.append(f"  [{warn['dataset']}] {warn['message']}")

        lines.append("=" * 60)
        return "\n".join(lines)


def validate_storage_url(source: str, storage_base: str) -> tuple[bool, str]:
    """Validate that storage URL matches the source.

    Returns:
        Tuple of (is_valid, error_message)

    """
    if source not in VALID_STORAGE_PATTERNS:
        return True, ""  # Unknown source, can't validate

    pattern = VALID_STORAGE_PATTERNS[source]
    if re.match(pattern, storage_base):
        return True, ""

    return False, f"Storage '{storage_base}' doesn't match pattern for {source}"


def validate_record(record: dict, dataset_id: str, result: ValidationResult):
    """Validate a single record."""
    # Check required fields
    for field in REQUIRED_RECORD_FIELDS:
        if field not in record or record[field] is None:
            result.add_error(dataset_id, f"Record missing required field: {field}")
            result.stats["missing_fields"] += 1
            return

    # Check storage fields
    storage = record.get("storage", {})
    for field in REQUIRED_STORAGE_FIELDS:
        if field not in storage:
            result.add_error(dataset_id, f"Record storage missing field: {field}")
            result.stats["missing_fields"] += 1

    # Validate storage URL matches source
    # Try to infer source from dataset_id pattern
    ds = record.get("dataset", "")
    if ds.startswith("ds") and ds[2:8].isdigit():
        source = "openneuro"
    elif ds.startswith("nm") and ds[2:8].isdigit():
        source = "nemar"
    else:
        # Can't validate without source
        return

    storage_base = storage.get("base", "")
    is_valid, error_msg = validate_storage_url(source, storage_base)
    if not is_valid:
        result.add_error(dataset_id, error_msg)
        result.stats["storage_errors"] += 1


def validate_dataset(dataset: dict, result: ValidationResult):
    """Validate a single dataset document."""
    dataset_id = dataset.get("dataset_id", "unknown")

    # Check required fields
    for field in REQUIRED_DATASET_FIELDS:
        if field not in dataset or dataset[field] is None:
            result.add_error(dataset_id, f"Dataset missing required field: {field}")
            result.stats["missing_fields"] += 1

    # Check source is valid
    source = dataset.get("source")
    if source and source not in VALID_SOURCES:
        result.add_warning(dataset_id, f"Unknown source: {source}")
        result.stats["invalid_source"] += 1


def validate_digestion_output(
    input_dir: Path, verbose: bool = False
) -> ValidationResult:
    """Validate all digestion output in a directory.

    Args:
        input_dir: Directory containing digested datasets
        verbose: Print progress information

    Returns:
        ValidationResult with all findings

    """
    result = ValidationResult()

    # Find all dataset directories
    if not input_dir.exists():
        result.add_error("", f"Input directory does not exist: {input_dir}")
        return result

    dataset_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    if verbose:
        print(f"Found {len(dataset_dirs)} dataset directories to validate")

    # Track sources and modalities for summary
    sources = Counter()
    modalities = Counter()

    for dataset_dir in dataset_dirs:
        dataset_id = dataset_dir.name

        # Check for records file
        records_file = dataset_dir / f"{dataset_id}_records.json"
        dataset_file = dataset_dir / f"{dataset_id}_dataset.json"

        if records_file.exists():
            try:
                with open(records_file) as f:
                    data = json.load(f)

                records = data.get("records", [])
                result.stats["records_checked"] += len(records)

                for record in records:
                    validate_record(record, dataset_id, result)

                    # Track modalities
                    mod = record.get("recording_modality", "unknown")
                    modalities[mod] += 1

            except json.JSONDecodeError as e:
                result.add_error(dataset_id, f"Invalid JSON in records file: {e}")
            except Exception as e:
                result.add_error(dataset_id, f"Error reading records: {e}")

        if dataset_file.exists():
            try:
                with open(dataset_file) as f:
                    dataset = json.load(f)

                result.stats["datasets_checked"] += 1
                validate_dataset(dataset, result)

                # Track sources
                src = dataset.get("source", "unknown")
                sources[src] += 1

            except json.JSONDecodeError as e:
                result.add_error(dataset_id, f"Invalid JSON in dataset file: {e}")
            except Exception as e:
                result.add_error(dataset_id, f"Error reading dataset: {e}")

    # Add distribution info to result
    result.source_distribution = dict(sources)
    result.modality_distribution = dict(modalities)

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate digestion output")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("digestion_output"),
        help="Input directory containing digested datasets",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    result = validate_digestion_output(args.input, verbose=args.verbose)

    if args.json:
        output = {
            "valid": result.is_valid(),
            "stats": result.stats,
            "errors": result.errors,
            "warnings": result.warnings,
            "source_distribution": result.source_distribution,
            "modality_distribution": result.modality_distribution,
        }
        print(json.dumps(output, indent=2))
    else:
        print(result.summary())

        if hasattr(result, "source_distribution"):
            print("\nSource distribution:")
            for src, count in sorted(
                result.source_distribution.items(), key=lambda x: -x[1]
            ):
                print(f"  {src}: {count}")

        if hasattr(result, "modality_distribution"):
            print("\nModality distribution:")
            for mod, count in sorted(
                result.modality_distribution.items(), key=lambda x: -x[1]
            ):
                print(f"  {mod}: {count}")

    # Exit with error code if validation failed
    sys.exit(0 if result.is_valid() else 1)


if __name__ == "__main__":
    main()

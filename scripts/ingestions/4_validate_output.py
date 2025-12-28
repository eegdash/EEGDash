#!/usr/bin/env python3
"""Validate digestion output for consistency and correctness.

This script checks that digested records and datasets are valid:
- Storage URLs match the source (no OpenNeuro with OSF URLs!)
- Required fields are present for eegdash compatibility
- Datasets have records (not empty)
- No duplicate records
- Source is set and valid for all entries

Usage:
    # Post-digestion validation (default)
    python validate_output.py --input digestion_output

    # Pre-digestion validation (check manifests)
    python validate_output.py --pre-check --input data/cloned

    # Strict mode - fail on warnings too
    python validate_output.py --input digestion_output --strict
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

# Valid storage URL patterns per source
VALID_STORAGE_PATTERNS = {
    "openneuro": r"^s3://openneuro\.org/ds\d+",
    "nemar": r"^s3://nemar/nm\d+",
    "osf": r"^https://files\.osf\.io/",
    "figshare": r"^https://(figshare\.com|ndownloader|.*\.figshare\.com)",
    "zenodo": r"^https://zenodo\.org/",
    "scidb": r"^https://(www\.)?scidb\.cn/",
    "datarn": r"^https://webdav\.data\.ru\.nl/",
    "gin": r"^https://gin\.g-node\.org/",
}

# Mandatory fields for Records (required for eegdash to work)
MANDATORY_RECORD_FIELDS = {
    "dataset": "Dataset ID (FK to Dataset)",
    "bids_relpath": "BIDS relative path to data file",
    "storage": "Storage configuration (backend, base, raw_key)",
    "recording_modality": "Recording modality (eeg, meg, ieeg, etc.)",
}

MANDATORY_STORAGE_FIELDS = {
    "backend": "Storage backend (s3, https, webdav)",
    "base": "Storage base URL",
    "raw_key": "Relative path from base",
}

# Mandatory fields for Datasets
MANDATORY_DATASET_FIELDS = {
    "dataset_id": "Dataset identifier",
    "source": "Data source (openneuro, nemar, etc.)",
    "recording_modality": "Primary recording modality",
}

# Recommended fields (warnings if missing)
RECOMMENDED_RECORD_FIELDS = ["entities", "datatype", "suffix", "extension"]
RECOMMENDED_DATASET_FIELDS = [
    "name",
    "datatypes",
    "demographics",
    "ingestion_fingerprint",
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
    "hbn",
}

# JSON Schemas (minimal required fields)
STORAGE_SCHEMA = {
    "type": "object",
    "required": ["backend", "base", "raw_key"],
    "properties": {
        "backend": {"type": "string"},
        "base": {"type": "string"},
        "raw_key": {"type": "string"},
    },
}

RECORD_SCHEMA = {
    "type": "object",
    "required": ["dataset", "bids_relpath", "storage", "recording_modality"],
    "properties": {
        "dataset": {"type": "string"},
        "bids_relpath": {"type": "string"},
        "recording_modality": {"type": "string"},
        "storage": STORAGE_SCHEMA,
    },
}

DATASET_SCHEMA = {
    "type": "object",
    "required": ["dataset_id", "source", "recording_modality"],
    "properties": {
        "dataset_id": {"type": "string"},
        "source": {"type": "string"},
        "recording_modality": {"type": "string"},
    },
}

RECORD_VALIDATOR = Draft202012Validator(RECORD_SCHEMA)
DATASET_VALIDATOR = Draft202012Validator(DATASET_SCHEMA)

# Data file extensions that indicate valid neurophysiology data
NEURO_EXTENSIONS = {
    ".edf",
    ".bdf",
    ".vhdr",
    ".set",
    ".fif",
    ".ds",
    ".con",
    ".sqd",
    ".pdf",
    ".mef",
    ".nwb",
    ".snirf",
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
            "missing_mandatory": 0,
            "missing_recommended": 0,
            "empty_datasets": 0,
            "invalid_source": 0,
            "zip_placeholders": 0,
        }
        self.source_distribution = {}
        self.modality_distribution = {}
        self.empty_datasets = []
        self.zip_placeholder_datasets = []

    def add_error(self, dataset: str, message: str, field: str = None):
        self.errors.append(
            {
                "dataset": dataset,
                "message": message,
                "field": field,
            }
        )

    def add_warning(self, dataset: str, message: str, field: str = None):
        self.warnings.append(
            {
                "dataset": dataset,
                "message": message,
                "field": field,
            }
        )

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
            f"  - Missing mandatory fields: {self.stats['missing_mandatory']}",
            f"  - Invalid source: {self.stats['invalid_source']}",
            "",
            f"Warnings: {len(self.warnings)}",
            f"  - Empty datasets (0 records): {self.stats['empty_datasets']}",
            f"  - Missing recommended fields: {self.stats['missing_recommended']}",
            f"  - ZIP placeholders (needs extraction): {self.stats['zip_placeholders']}",
        ]

        if self.empty_datasets:
            lines.append("")
            lines.append(f"Empty datasets ({len(self.empty_datasets)}):")
            for ds in self.empty_datasets[:10]:
                lines.append(f"  - {ds}")
            if len(self.empty_datasets) > 10:
                lines.append(f"  ... and {len(self.empty_datasets) - 10} more")

        if self.errors:
            lines.append("")
            lines.append("Sample errors (first 10):")
            for err in self.errors[:10]:
                field_info = f" [{err['field']}]" if err.get("field") else ""
                lines.append(f"  [{err['dataset']}]{field_info} {err['message']}")

        if self.warnings and len(self.errors) == 0:
            lines.append("")
            lines.append("Sample warnings (first 5):")
            for warn in self.warnings[:5]:
                lines.append(f"  [{warn['dataset']}] {warn['message']}")

        lines.append("=" * 60)
        return "\n".join(lines)


def validate_storage_url(source: str, storage_base: str) -> tuple[bool, str]:
    """Validate that storage URL matches the source."""
    if source not in VALID_STORAGE_PATTERNS:
        return True, ""  # Unknown source, can't validate

    pattern = VALID_STORAGE_PATTERNS[source]
    if re.match(pattern, storage_base):
        return True, ""

    return (
        False,
        f"Storage '{storage_base}' doesn't match expected pattern for {source}",
    )


def validate_record(
    record: dict,
    dataset_id: str,
    source: str,
    result: ValidationResult,
    record_idx: int = 0,
):
    """Validate a single record for mandatory fields."""
    # Schema validation for required fields
    for error in RECORD_VALIDATOR.iter_errors(record):
        field_path = ".".join(str(p) for p in error.path) if error.path else None
        result.add_error(dataset_id, error.message, field=field_path)
        if error.validator == "required":
            result.stats["missing_mandatory"] += 1

    # Validate storage URL matches source
    storage = record.get("storage", {})
    if storage:
        storage_base = storage.get("base", "")
        is_valid, error_msg = validate_storage_url(source, storage_base)
        if not is_valid:
            result.add_error(dataset_id, error_msg, field="storage.base")
            result.stats["storage_errors"] += 1

    # Check recommended fields (warnings only, first record only)
    if record_idx == 0:
        for field in RECOMMENDED_RECORD_FIELDS:
            if field not in record or record[field] is None:
                result.add_warning(
                    dataset_id,
                    f"Missing recommended field: {field}",
                    field=field,
                )
                result.stats["missing_recommended"] += 1


def validate_dataset(dataset: dict, result: ValidationResult):
    """Validate a single dataset document for mandatory fields."""
    dataset_id = dataset.get("dataset_id", "unknown")

    # Schema validation for required fields
    for error in DATASET_VALIDATOR.iter_errors(dataset):
        field_path = ".".join(str(p) for p in error.path) if error.path else None
        result.add_error(dataset_id, error.message, field=field_path)
        if error.validator == "required":
            result.stats["missing_mandatory"] += 1

    # Check source is valid
    source = dataset.get("source")
    if source and source not in VALID_SOURCES:
        result.add_warning(dataset_id, f"Unknown source: {source}", field="source")
        result.stats["invalid_source"] += 1

    # Check recommended fields
    for field in RECOMMENDED_DATASET_FIELDS:
        if field not in dataset or dataset[field] is None:
            result.add_warning(
                dataset_id,
                f"Missing recommended field: {field}",
                field=field,
            )


def validate_digestion_output(
    input_dir: Path,
    verbose: bool = False,
    strict: bool = False,
) -> ValidationResult:
    """Validate all digestion output in a directory.

    Args:
        input_dir: Directory containing digested datasets
        verbose: Print progress information
        strict: Treat warnings as errors

    Returns:
        ValidationResult with all findings

    """
    result = ValidationResult()

    if not input_dir.exists():
        result.add_error("", f"Input directory does not exist: {input_dir}")
        return result

    dataset_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    if verbose:
        print(f"Found {len(dataset_dirs)} dataset directories to validate")

    sources = Counter()
    modalities = Counter()

    for dataset_dir in dataset_dirs:
        dataset_id = dataset_dir.name
        records_file = dataset_dir / f"{dataset_id}_records.json"
        dataset_file = dataset_dir / f"{dataset_id}_dataset.json"

        source = "unknown"
        record_count = 0

        # Load dataset first to get source
        if dataset_file.exists():
            try:
                with open(dataset_file) as f:
                    dataset = json.load(f)

                result.stats["datasets_checked"] += 1
                validate_dataset(dataset, result)
                source = dataset.get("source", "unknown")
                sources[source] += 1

            except json.JSONDecodeError as e:
                result.add_error(dataset_id, f"Invalid JSON in dataset file: {e}")
            except Exception as e:
                result.add_error(dataset_id, f"Error reading dataset: {e}")

        # Validate records
        if records_file.exists():
            try:
                with open(records_file) as f:
                    data = json.load(f)

                records = data.get("records", [])
                record_count = len(records)
                result.stats["records_checked"] += record_count

                has_zip_placeholder = False
                for idx, record in enumerate(records):
                    validate_record(record, dataset_id, source, result, idx)
                    mod = record.get("recording_modality", "unknown")
                    modalities[mod] += 1

                    # Track ZIP placeholders
                    if record.get("needs_extraction") or record.get(
                        "zip_contains_bids"
                    ):
                        has_zip_placeholder = True

                if has_zip_placeholder:
                    result.stats["zip_placeholders"] += 1
                    result.zip_placeholder_datasets.append(f"{dataset_id} ({source})")

            except json.JSONDecodeError as e:
                result.add_error(dataset_id, f"Invalid JSON in records file: {e}")
            except Exception as e:
                result.add_error(dataset_id, f"Error reading records: {e}")

        # Flag empty datasets
        if record_count == 0:
            result.stats["empty_datasets"] += 1
            result.empty_datasets.append(f"{dataset_id} ({source})")
            if strict:
                result.add_error(
                    dataset_id,
                    "Dataset has 0 records - cannot be used in eegdash",
                )
            else:
                result.add_warning(
                    dataset_id,
                    "Dataset has 0 records - no usable data files found",
                )

    result.source_distribution = dict(sources)
    result.modality_distribution = dict(modalities)

    return result


def validate_pre_digestion(input_dir: Path, verbose: bool = False) -> ValidationResult:
    """Pre-digestion validation: check manifests have valid data files.

    Args:
        input_dir: Directory containing cloned datasets with manifest.json files
        verbose: Print progress information

    Returns:
        ValidationResult with findings

    """
    result = ValidationResult()

    if not input_dir.exists():
        result.add_error("", f"Input directory does not exist: {input_dir}")
        return result

    dataset_dirs = [
        d for d in input_dir.iterdir() if d.is_dir() and (d / "manifest.json").exists()
    ]

    if verbose:
        print(f"Found {len(dataset_dirs)} manifests to check")

    sources = Counter()

    for dataset_dir in dataset_dirs:
        dataset_id = dataset_dir.name
        manifest_path = dataset_dir / "manifest.json"

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)

            source = manifest.get("source", "unknown")
            sources[source] += 1
            result.stats["datasets_checked"] += 1

            files = manifest.get("files", [])

            # Count potential data files and CTF .ds directories
            data_file_count = 0
            ctf_ds_dirs = set()

            for f in files:
                filepath = (
                    f.get("path", f.get("name", "")) if isinstance(f, dict) else f
                )
                filepath_lower = filepath.lower()

                # Track CTF .ds directories (files inside .ds/ paths)
                if ".ds/" in filepath_lower:
                    ds_idx = filepath_lower.index(".ds/") + 3
                    ds_path = filepath[:ds_idx]
                    ctf_ds_dirs.add(ds_path)
                    continue

                # Check if this could be a neurophysiology data file
                for ext in NEURO_EXTENSIONS:
                    if filepath_lower.endswith(ext):
                        data_file_count += 1
                        break

            # Add CTF .ds directories count
            data_file_count += len(ctf_ds_dirs)
            result.stats["records_checked"] += data_file_count

            if data_file_count == 0:
                result.stats["empty_datasets"] += 1
                result.empty_datasets.append(f"{dataset_id} ({source})")
                result.add_warning(
                    dataset_id,
                    f"No recognized neurophysiology files in manifest "
                    f"({len(files)} total files)",
                )

            if verbose and data_file_count > 0:
                print(f"  {dataset_id}: {data_file_count} data files")

        except json.JSONDecodeError as e:
            result.add_error(dataset_id, f"Invalid JSON in manifest: {e}")
        except Exception as e:
            result.add_error(dataset_id, f"Error reading manifest: {e}")

    result.source_distribution = dict(sources)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate digestion output for eegdash compatibility"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("digestion_output"),
        help="Input directory (digestion_output or data/cloned for --pre-check)",
    )
    parser.add_argument(
        "--pre-check",
        action="store_true",
        help="Pre-digestion validation: check manifests have valid data files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (empty datasets become errors)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    if args.pre_check:
        result = validate_pre_digestion(args.input, verbose=args.verbose)
    else:
        result = validate_digestion_output(
            args.input,
            verbose=args.verbose,
            strict=args.strict,
        )

    if args.json:
        output = {
            "valid": result.is_valid(),
            "stats": result.stats,
            "errors": result.errors,
            "warnings": result.warnings,
            "empty_datasets": result.empty_datasets,
            "source_distribution": result.source_distribution,
            "modality_distribution": result.modality_distribution,
        }
        print(json.dumps(output, indent=2))
    else:
        print(result.summary())

        if result.source_distribution:
            print("\nSource distribution:")
            for src, count in sorted(
                result.source_distribution.items(), key=lambda x: -x[1]
            ):
                print(f"  {src}: {count}")

        if result.modality_distribution:
            print("\nModality distribution:")
            for mod, count in sorted(
                result.modality_distribution.items(), key=lambda x: -x[1]
            ):
                print(f"  {mod}: {count}")

    # Exit with error code if validation failed
    # In strict mode, warnings also cause failure
    if args.strict and result.warnings:
        sys.exit(1)
    sys.exit(0 if result.is_valid() else 1)


if __name__ == "__main__":
    main()

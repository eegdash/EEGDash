"""Validation utilities for injection scripts.

This module re-exports the validation functions from 4_validate_output.py
for use in other ingestion scripts.
"""

import json
import re
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from eegdash.schemas import DatasetModel, RecordModel

# Valid storage URL patterns per source
VALID_STORAGE_PATTERNS = {
    "openneuro": r"^s3://openneuro\.org/ds\d+",
    "nemar": r"^s3://(nemar|nmdatasets)/",
    "osf": r"^https://files\.osf\.io/",
    "figshare": r"^https://(figshare\.com|ndownloader|.*\.figshare\.com)",
    "zenodo": r"^https://zenodo\.org/",
    "scidb": r"^https://(www\.)?scidb\.cn/",
    "datarn": r"^https://webdav\.data\.ru\.nl/",
    "gin": r"^https://gin\.g-node\.org/",
}

# Recommended fields (warnings if missing)
RECOMMENDED_RECORD_FIELDS = ["entities", "datatype", "suffix", "extension"]
RECOMMENDED_DATASET_FIELDS = [
    "name",
    "datatypes",
    "demographics",
    "ingestion_fingerprint",
]

# Data quality fields - these should have values for usable records
DATA_QUALITY_FIELDS = ["nchans", "sampling_frequency"]

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
            "missing_nchans": 0,
            "missing_sampling_frequency": 0,
        }
        self.source_distribution = {}
        self.modality_distribution = {}
        self.empty_datasets = []
        self.zip_placeholder_datasets = []
        self.data_quality_issues = []  # Datasets with missing nchans/sampling_frequency

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
        total_records = self.stats["records_checked"]
        missing_nchans = self.stats["missing_nchans"]
        missing_sampling_frequency = self.stats["missing_sampling_frequency"]
        nchans_pct = (missing_nchans / total_records * 100) if total_records > 0 else 0
        sampling_frequency_pct = (
            (missing_sampling_frequency / total_records * 100)
            if total_records > 0
            else 0
        )

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
            "",
            "Data Quality:",
            f"  - Records missing nchans: {missing_nchans} ({nchans_pct:.1f}%)",
            f"  - Records missing sampling_frequency: {missing_sampling_frequency} ({sampling_frequency_pct:.1f}%)",
        ]

        if self.empty_datasets:
            lines.append("")
            lines.append(f"Empty datasets ({len(self.empty_datasets)}):")
            for ds in self.empty_datasets[:10]:
                lines.append(f"  - {ds}")
            if len(self.empty_datasets) > 10:
                lines.append(f"  ... and {len(self.empty_datasets) - 10} more")

        if self.data_quality_issues:
            lines.append("")
            lines.append(
                f"Data quality issues ({len(self.data_quality_issues)} datasets):"
            )
            for issue in self.data_quality_issues[:10]:
                lines.append(f"  - {issue}")
            if len(self.data_quality_issues) > 10:
                lines.append(f"  ... and {len(self.data_quality_issues) - 10} more")

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


def _add_pydantic_errors(
    result: "ValidationResult",
    *,
    dataset_id: str,
    exc: ValidationError,
    prefix: str | None = None,
):
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", [])]
        if prefix:
            loc = [prefix, *loc]
        field_path = ".".join(loc) if loc else None
        result.add_error(dataset_id, err.get("msg", "Invalid value"), field=field_path)
        if err.get("type") == "missing":
            result.stats["missing_mandatory"] += 1


def validate_record(
    record: dict,
    dataset_id: str,
    source: str,
    result: ValidationResult,
    record_idx: int = 0,
):
    """Validate a single record for mandatory fields."""
    try:
        RecordModel.model_validate(record)
    except ValidationError as exc:
        _add_pydantic_errors(result, dataset_id=dataset_id, exc=exc, prefix="record")

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

    # Check data quality fields (nchans, sampling_frequency) - critical for usability
    nchans = record.get("nchans")
    sampling_frequency = record.get("sampling_frequency")
    if nchans is None or nchans == 0:
        result.stats["missing_nchans"] += 1
    if sampling_frequency is None or sampling_frequency == 0:
        result.stats["missing_sampling_frequency"] += 1


def validate_dataset(dataset: dict, result: ValidationResult):
    """Validate a single dataset document for mandatory fields."""
    dataset_id = dataset.get("dataset_id", "unknown")

    try:
        DatasetModel.model_validate(dataset)
    except ValidationError as exc:
        _add_pydantic_errors(result, dataset_id=dataset_id, exc=exc, prefix="dataset")

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
                has_missing_nchans = False
                has_missing_sampling_frequency = False
                for idx, record in enumerate(records):
                    validate_record(record, dataset_id, source, result, idx)
                    mods = record.get("recording_modality", ["unknown"])
                    if isinstance(mods, str):
                        mods = [mods]
                    for mod in mods:
                        modalities[mod] += 1

                    # Track ZIP placeholders
                    if record.get("needs_extraction") or record.get(
                        "zip_contains_bids"
                    ):
                        has_zip_placeholder = True

                    # Track data quality at dataset level
                    if record.get("nchans") is None or record.get("nchans") == 0:
                        has_missing_nchans = True
                    if (
                        record.get("sampling_frequency") is None
                        or record.get("sampling_frequency") == 0
                    ):
                        has_missing_sampling_frequency = True

                if has_missing_nchans or has_missing_sampling_frequency:
                    issue_parts = []
                    if has_missing_nchans:
                        issue_parts.append("nchans")
                    if has_missing_sampling_frequency:
                        issue_parts.append("sampling_frequency")
                    result.data_quality_issues.append(
                        f"{dataset_id} ({source}): missing {', '.join(issue_parts)}"
                    )

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

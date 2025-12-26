#!/usr/bin/env python3
"""Digest BIDS datasets and generate JSON records for MongoDB.

This script produces two types of documents:
- **Dataset**: One per dataset (metadata for discovery/filtering)
- **Record**: One per file (metadata for loading data)

Usage:
    # Digest all cloned datasets
    python 3_digest.py --input data/cloned --output digestion_output

    # Digest specific datasets
    python 3_digest.py --input data/cloned --output digestion_output --datasets ds002718 ds005506

    # Digest with parallel processing
    python 3_digest.py --input data/cloned --output digestion_output --workers 4
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from eegdash.records import create_dataset, create_record

# Storage base URLs per source
STORAGE_BASES = {
    "openneuro": "s3://openneuro.org",
    "nemar": "https://nemar.org/dataexplorer/detail",
    "gin": "https://gin.g-node.org/EEGManyLabs",
    "figshare": "https://figshare.com/ndownloader/files",
    "zenodo": "https://zenodo.org/records",
    "osf": "https://files.osf.io",
    "scidb": "https://www.scidb.cn",
    "datarn": "https://webdav.data.ru.nl/dcc",
}


def get_storage_base(dataset_id: str, source: str) -> str:
    """Get storage base URL for a dataset."""
    base = STORAGE_BASES.get(source, STORAGE_BASES["openneuro"])
    return f"{base}/{dataset_id}"


def get_storage_backend(source: str) -> str:
    """Get storage backend type for a source."""
    if source == "openneuro":
        return "s3"
    return "https"


def detect_source(dataset_dir: Path) -> str:
    """Detect source from manifest.json or dataset structure."""
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
                return manifest.get("source", "openneuro")
        except Exception:
            pass

    # Fallback: check dataset_id pattern
    dataset_id = dataset_dir.name
    if dataset_id.startswith("ds"):
        return "openneuro"
    elif "EEGManyLabs" in dataset_id:
        return "gin"

    return "openneuro"


def extract_dataset_metadata(
    bids_dataset,
    dataset_id: str,
    source: str,
    digested_at: str,
) -> dict[str, Any]:
    """Extract Dataset-level metadata from a BIDS dataset.

    Parameters
    ----------
    bids_dataset : EEGBIDSDataset
        The BIDS dataset object
    dataset_id : str
        Dataset identifier
    source : str
        Source name (openneuro, nemar, etc.)
    digested_at : str
        ISO 8601 timestamp

    Returns
    -------
    dict
        Dataset schema compliant metadata

    """
    bids_root = Path(bids_dataset.bidsdir)

    # Read dataset_description.json
    description = {}
    desc_path = bids_root / "dataset_description.json"
    if desc_path.exists():
        try:
            with open(desc_path) as f:
                description = json.load(f)
        except Exception:
            pass

    # Extract basic metadata
    name = description.get("Name", dataset_id)
    bids_version = description.get("BIDSVersion")
    license_info = description.get("License")
    authors = description.get("Authors", [])
    funding = description.get("Funding", [])
    dataset_doi = description.get("DatasetDOI")

    # Get files and detect modalities
    files = bids_dataset.get_files()
    modalities = set()
    tasks = set()
    sessions = set()

    for bids_file in files:
        mod = bids_dataset.get_bids_file_attribute("modality", bids_file)
        if mod:
            modalities.add(mod)
        task = bids_dataset.get_bids_file_attribute("task", bids_file)
        if task:
            tasks.add(task)
        session = bids_dataset.get_bids_file_attribute("session", bids_file)
        if session:
            sessions.add(session)

    # Determine primary modality
    modality_priority = ["eeg", "meg", "ieeg"]
    recording_modality = "eeg"
    for mod in modality_priority:
        if mod in modalities:
            recording_modality = mod
            break

    # Read participants.tsv for demographics
    subjects_count = 0
    ages = []
    sex_distribution = {}
    handedness_distribution = {}

    participants_path = bids_root / "participants.tsv"
    if participants_path.exists():
        try:
            df = pd.read_csv(
                participants_path, sep="\t", dtype="string", keep_default_na=False
            )
            subjects_count = len(df)

            # Extract ages
            age_col = None
            for col in ["age", "Age", "AGE"]:
                if col in df.columns:
                    age_col = col
                    break
            if age_col:
                for val in df[age_col]:
                    try:
                        age = int(float(val))
                        if 0 < age < 120:
                            ages.append(age)
                    except (ValueError, TypeError):
                        pass

            # Extract sex distribution
            sex_col = None
            for col in ["sex", "Sex", "SEX", "gender", "Gender"]:
                if col in df.columns:
                    sex_col = col
                    break
            if sex_col:
                for val in df[sex_col]:
                    val_lower = str(val).lower().strip()
                    if val_lower in ("m", "male"):
                        sex_distribution["m"] = sex_distribution.get("m", 0) + 1
                    elif val_lower in ("f", "female"):
                        sex_distribution["f"] = sex_distribution.get("f", 0) + 1
                    elif val_lower and val_lower not in (
                        "n/a",
                        "na",
                        "nan",
                        "unknown",
                        "",
                    ):
                        sex_distribution["o"] = sex_distribution.get("o", 0) + 1

            # Extract handedness distribution
            hand_col = None
            for col in ["handedness", "Handedness", "hand", "Hand"]:
                if col in df.columns:
                    hand_col = col
                    break
            if hand_col:
                for val in df[hand_col]:
                    val_lower = str(val).lower().strip()
                    if val_lower in ("r", "right"):
                        handedness_distribution["r"] = (
                            handedness_distribution.get("r", 0) + 1
                        )
                    elif val_lower in ("l", "left"):
                        handedness_distribution["l"] = (
                            handedness_distribution.get("l", 0) + 1
                        )
                    elif val_lower in ("a", "ambidextrous"):
                        handedness_distribution["a"] = (
                            handedness_distribution.get("a", 0) + 1
                        )

        except Exception:
            pass

    # Count subjects from directories if participants.tsv not available
    if subjects_count == 0:
        subjects_count = len(
            [d for d in bids_root.iterdir() if d.is_dir() and d.name.startswith("sub-")]
        )

    # Check for derivatives (processed data)
    data_processed = (bids_root / "derivatives").exists()

    # Build source URL
    source_url = None
    if source == "openneuro":
        source_url = f"https://openneuro.org/datasets/{dataset_id}"
    elif source == "nemar":
        source_url = f"https://nemar.org/dataexplorer/detail/{dataset_id}"
    elif source == "gin":
        source_url = f"https://gin.g-node.org/EEGManyLabs/{dataset_id}"

    # Create Dataset document
    dataset = create_dataset(
        dataset_id=dataset_id,
        name=name,
        source=source,
        recording_modality=recording_modality,
        modalities=sorted(modalities) if modalities else [recording_modality],
        bids_version=bids_version,
        license=license_info,
        authors=authors if isinstance(authors, list) else [authors] if authors else [],
        funding=funding if isinstance(funding, list) else [funding] if funding else [],
        dataset_doi=dataset_doi,
        tasks=sorted(tasks),
        sessions=sorted(sessions),
        total_files=len(files),
        data_processed=data_processed,
        subjects_count=subjects_count,
        ages=ages,
        age_mean=sum(ages) / len(ages) if ages else None,
        sex_distribution=sex_distribution if sex_distribution else None,
        handedness_distribution=handedness_distribution
        if handedness_distribution
        else None,
        source_url=source_url,
        digested_at=digested_at,
    )

    return dict(dataset)


def extract_record(
    bids_dataset,
    bids_file: str,
    dataset_id: str,
    source: str,
    digested_at: str,
) -> dict[str, Any]:
    """Extract Record metadata for a single BIDS file.

    Parameters
    ----------
    bids_dataset : EEGBIDSDataset
        The BIDS dataset object
    bids_file : str
        Path to the BIDS file
    dataset_id : str
        Dataset identifier
    source : str
        Source name (openneuro, nemar, etc.)
    digested_at : str
        ISO 8601 timestamp

    Returns
    -------
    dict
        Record schema compliant metadata

    """
    # Get BIDS entities
    subject = bids_dataset.get_bids_file_attribute("subject", bids_file)
    session = bids_dataset.get_bids_file_attribute("session", bids_file)
    task = bids_dataset.get_bids_file_attribute("task", bids_file)
    run = bids_dataset.get_bids_file_attribute("run", bids_file)
    modality = bids_dataset.get_bids_file_attribute("modality", bids_file) or "eeg"

    # Get BIDS relative path (without dataset prefix)
    bids_relpath = str(bids_dataset.get_relative_bidspath(bids_file))
    if bids_relpath.startswith(f"{dataset_id}/"):
        bids_relpath = bids_relpath[len(dataset_id) + 1 :]

    # Determine datatype and suffix
    datatype = modality
    suffix = modality

    # Get storage info
    storage_base = get_storage_base(dataset_id, source)
    storage_backend = get_storage_backend(source)

    # Find dependency files (channels.tsv, events.tsv, etc.)
    dep_keys = []
    bids_file_path = Path(bids_file)
    parent_dir = bids_file_path.parent
    base_name = bids_file_path.stem.rsplit("_", 1)[0]

    for dep_suffix in [
        "_channels.tsv",
        "_events.tsv",
        "_electrodes.tsv",
        "_coordsystem.json",
    ]:
        dep_file = parent_dir / f"{base_name}{dep_suffix}"
        if dep_file.exists() or dep_file.is_symlink():
            try:
                dep_relpath = dep_file.relative_to(bids_dataset.bidsdir)
                dep_keys.append(str(dep_relpath))
            except ValueError:
                pass

    # Create record using the schema
    record = create_record(
        dataset=dataset_id,
        storage_base=storage_base,
        bids_relpath=bids_relpath,
        subject=subject,
        session=session,
        task=task,
        run=str(run) if run is not None else None,
        dep_keys=dep_keys,
        datatype=datatype,
        suffix=suffix,
        storage_backend=storage_backend,
        recording_modality=modality,
        digested_at=digested_at,
    )

    return dict(record)


def digest_dataset(
    dataset_id: str,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Digest a single dataset and generate JSON output.

    Produces:
    - {dataset_id}_dataset.json: Dataset-level metadata
    - {dataset_id}_records.json: Per-file Record metadata

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    input_dir : Path
        Directory containing cloned datasets
    output_dir : Path
        Directory for output JSON files

    Returns
    -------
    dict
        Summary of digestion results

    """
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    dataset_dir = input_dir / dataset_id
    dataset_output_dir = output_dir / dataset_id

    if not dataset_dir.exists():
        return {
            "status": "skipped",
            "dataset_id": dataset_id,
            "reason": "directory not found",
        }

    # Detect source
    source = detect_source(dataset_dir)

    # Generate timestamp
    digested_at = datetime.now(timezone.utc).isoformat()

    # Load BIDS dataset
    try:
        bids_dataset = EEGBIDSDataset(
            data_dir=str(dataset_dir),
            dataset=dataset_id,
            allow_symlinks=True,
        )
    except Exception as e:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "error": f"Failed to load BIDS dataset: {e}",
        }

    files = bids_dataset.get_files()
    if not files:
        return {
            "status": "empty",
            "dataset_id": dataset_id,
            "reason": "no EEG files found",
        }

    # Extract Dataset metadata
    try:
        dataset_meta = extract_dataset_metadata(
            bids_dataset, dataset_id, source, digested_at
        )
    except Exception as e:
        dataset_meta = {
            "dataset_id": dataset_id,
            "source": source,
            "error": str(e),
        }

    # Extract Record metadata for each file
    records = []
    errors = []

    for bids_file in files:
        try:
            record = extract_record(
                bids_dataset, bids_file, dataset_id, source, digested_at
            )
            records.append(record)
        except Exception as e:
            errors.append({"file": str(bids_file), "error": str(e)})

    if not records:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "error": "No records extracted",
            "errors": errors,
        }

    # Create output directory
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Save Dataset document
    dataset_path = dataset_output_dir / f"{dataset_id}_dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(dataset_meta, f, indent=2, default=_json_serializer)

    # Save Records document
    records_path = dataset_output_dir / f"{dataset_id}_records.json"
    records_data = {
        "dataset": dataset_id,
        "source": source,
        "digested_at": digested_at,
        "record_count": len(records),
        "records": records,
    }
    with open(records_path, "w") as f:
        json.dump(records_data, f, indent=2, default=_json_serializer)

    # Save summary
    summary = {
        "status": "success",
        "dataset_id": dataset_id,
        "source": source,
        "record_count": len(records),
        "error_count": len(errors),
        "dataset_file": str(dataset_path),
        "records_file": str(records_path),
    }

    summary_path = dataset_output_dir / f"{dataset_id}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _json_serializer(obj):
    """Handle non-serializable objects."""
    import numpy as np

    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, set):
        return sorted(list(obj))
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def find_datasets(input_dir: Path, datasets: list[str] | None = None) -> list[str]:
    """Find dataset IDs in input directory."""
    if datasets:
        return datasets

    found = []
    for d in input_dir.iterdir():
        if d.is_dir() and (
            d.name.startswith("ds")
            or d.name.startswith("nm")
            or "EEGManyLabs" in d.name
        ):
            found.append(d.name)

    return sorted(found)


def main():
    parser = argparse.ArgumentParser(
        description="Digest BIDS datasets and generate Dataset + Record JSON for MongoDB."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/cloned"),
        help="Directory containing cloned datasets (default: data/cloned/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("digestion_output"),
        help="Output directory for JSON files (default: digestion_output/)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific dataset IDs to digest (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of datasets to process (for testing)",
    )

    args = parser.parse_args()

    # Find datasets
    dataset_ids = find_datasets(args.input, args.datasets)
    if args.limit:
        dataset_ids = dataset_ids[: args.limit]

    print(f"Found {len(dataset_ids)} datasets to digest")
    print(f"Workers: {args.workers}")
    print("=" * 60)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Process datasets
    results = []
    stats = {"success": 0, "error": 0, "skipped": 0, "empty": 0}

    if args.workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(digest_dataset, ds_id, args.input, args.output): ds_id
                for ds_id in dataset_ids
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Digesting"
            ):
                result = future.result()
                results.append(result)
                status = result.get("status", "error")
                stats[status] = stats.get(status, 0) + 1
    else:
        # Sequential processing
        for ds_id in tqdm(dataset_ids, desc="Digesting"):
            result = digest_dataset(ds_id, args.input, args.output)
            results.append(result)
            status = result.get("status", "error")
            stats[status] = stats.get(status, 0) + 1

    # Save batch summary
    batch_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_datasets": len(dataset_ids),
        "stats": stats,
        "total_records": sum(
            r.get("record_count", 0) for r in results if r.get("status") == "success"
        ),
    }

    batch_summary_path = args.output / "BATCH_SUMMARY.json"
    with open(batch_summary_path, "w") as f:
        json.dump(batch_summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("DIGESTION SUMMARY")
    print("=" * 60)
    print(f"  Success:  {stats['success']}")
    print(f"  Skipped:  {stats['skipped']}")
    print(f"  Empty:    {stats['empty']}")
    print(f"  Error:    {stats['error']}")
    print(f"\nTotal records: {batch_summary['total_records']}")
    print(f"Batch summary: {batch_summary_path}")
    print("=" * 60)

    return 0 if stats["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

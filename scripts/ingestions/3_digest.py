#!/usr/bin/env python3
"""Digest BIDS datasets and generate JSON records for MongoDB.

Unified digestion script supporting both minimal and full modes.

Modes:
    - minimal: Core Record schema fields only (fast, for production)
    - full: Core + enriched metadata (complete, for debugging)

Usage:
    # Digest all cloned datasets (minimal mode - recommended)
    python 3_digest.py --input data/cloned --output digestion_output

    # Digest with full metadata
    python 3_digest.py --input data/cloned --output digestion_output --mode full

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

from tqdm import tqdm

from eegdash.records import create_record

# Storage base URLs per source
STORAGE_BASES = {
    "openneuro": "s3://openneuro.org",
    "nemar": "https://nemar.org/dataexplorer/detail",  # NEMAR uses HTTPS
    "gin": "https://gin.g-node.org/EEGManyLabs",  # GIN WebDAV
    "figshare": "https://figshare.com/ndownloader/files",  # Figshare CDN
    "zenodo": "https://zenodo.org/records",  # Zenodo
    "osf": "https://files.osf.io",  # OSF storage
    "scidb": "https://www.scidb.cn",  # SciDB
    "datarn": "https://webdav.data.ru.nl/dcc",  # data.ru.nl WebDAV
}

# Enriched fields (not part of core Record schema)
ENRICHED_FIELDS = {
    "participant_tsv",
    "eeg_json",
    "channels_tsv",
    "events_tsv",
    "bidsdependencies",
    "sampling_frequency",
    "nchans",
    "ntimes",
}


def get_storage_base(dataset_id: str, source: str) -> str:
    """Get storage base URL for a dataset.
    
    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    source : str
        Source name (openneuro, nemar, etc.)
    
    Returns
    -------
    str
        Full storage base URL
    """
    base = STORAGE_BASES.get(source, STORAGE_BASES["openneuro"])
    
    if source == "openneuro":
        return f"{base}/{dataset_id}"
    elif source == "nemar":
        return f"{base}/{dataset_id}"
    elif source == "gin":
        return f"{base}/{dataset_id}"
    else:
        return f"{base}/{dataset_id}"


def get_storage_backend(source: str) -> str:
    """Get storage backend type for a source.
    
    Parameters
    ----------
    source : str
        Source name
    
    Returns
    -------
    str
        Backend type: 's3', 'https', or 'local'
    """
    if source == "openneuro":
        return "s3"
    return "https"


def extract_record(
    bids_dataset,
    bids_file: str,
    dataset_id: str,
    source: str,
    mode: str,
    digested_at: str,
) -> dict[str, Any]:
    """Extract metadata for a single BIDS file using Record schema.

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
    mode : str
        'minimal' for core fields only, 'full' for all metadata
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
    # Remove dataset_id prefix if present
    if bids_relpath.startswith(f"{dataset_id}/"):
        bids_relpath = bids_relpath[len(dataset_id) + 1:]
    
    # Determine datatype and suffix from path
    path_parts = Path(bids_relpath).parts
    datatype = modality  # eeg, meg, ieeg
    suffix = modality
    
    # Get storage info
    storage_base = get_storage_base(dataset_id, source)
    storage_backend = get_storage_backend(source)
    
    # Find dependency files (channels.tsv, events.tsv, etc.)
    dep_keys = []
    bids_file_path = Path(bids_file)
    parent_dir = bids_file_path.parent
    base_name = bids_file_path.stem.rsplit("_", 1)[0]  # Remove suffix like _eeg
    
    for dep_suffix in ["_channels.tsv", "_events.tsv", "_electrodes.tsv", "_coordsystem.json"]:
        dep_file = parent_dir / f"{base_name}{dep_suffix}"
        if dep_file.exists() or (dep_file.is_symlink()):
            # Get relative path
            try:
                dep_relpath = dep_file.relative_to(bids_dataset.bids_dir)
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

    if mode == "full":
        # Add enriched metadata
        enriched = {}
        
        # Get sampling info from JSON sidecars
        sfreq = bids_dataset.get_bids_file_attribute("sfreq", bids_file)
        nchans = bids_dataset.get_bids_file_attribute("nchans", bids_file)
        ntimes = bids_dataset.get_bids_file_attribute("ntimes", bids_file)
        
        if sfreq is not None:
            enriched["sampling_frequency"] = sfreq
        if nchans is not None:
            enriched["nchans"] = nchans
        if ntimes is not None:
            enriched["ntimes"] = ntimes
        
        # Load full metadata including participant info, EEG JSON, etc.
        try:
            from eegdash.bids_eeg_metadata import load_eeg_attrs_from_bids_file
            full_attrs = load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
            enriched.update({k: v for k, v in full_attrs.items() if k in ENRICHED_FIELDS})
        except Exception:
            pass
        
        record["_enriched"] = enriched

    return dict(record)  # Convert TypedDict to regular dict


def detect_source(dataset_dir: Path) -> str:
    """Detect source from manifest.json or dataset structure.
    
    Parameters
    ----------
    dataset_dir : Path
        Path to dataset directory
    
    Returns
    -------
    str
        Source name (openneuro, nemar, gin, figshare, zenodo, osf, scidb, datarn)
    """
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


def digest_dataset(
    dataset_id: str,
    input_dir: Path,
    output_dir: Path,
    mode: str = "minimal",
) -> dict[str, Any]:
    """Digest a single dataset and generate JSON output.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    input_dir : Path
        Directory containing cloned datasets
    output_dir : Path
        Directory for output JSON files
    mode : str
        'minimal' or 'full'

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
            allow_symlinks=True,  # Enable metadata extraction from git-annex
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

    # Extract records
    records = []
    errors = []

    for bids_file in files:
        try:
            record = extract_record(bids_dataset, bids_file, dataset_id, source, mode, digested_at)
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

    # Save records
    if mode == "minimal":
        # Single output file with all records
        output_path = dataset_output_dir / f"{dataset_id}_records.json"
        output_data = {
            "dataset": dataset_id,
            "source": source,
            "digested_at": digested_at,
            "record_count": len(records),
            "records": records,
        }
    else:
        # Split into core and enriched for full mode
        core_records = []
        enriched_records = []

        for record in records:
            # Core = all Record schema fields (without _enriched)
            core = {k: v for k, v in record.items() if k != "_enriched"}
            core_records.append(core)
            
            # Enriched = the _enriched dict with data_name link
            enriched = record.get("_enriched", {})
            enriched["data_name"] = record.get("data_name")  # Link key
            enriched_records.append(enriched)

        # Save core
        core_path = dataset_output_dir / f"{dataset_id}_core.json"
        with open(core_path, "w") as f:
            json.dump({
                "dataset": dataset_id,
                "source": source,
                "digested_at": digested_at,
                "record_count": len(core_records),
                "records": core_records,
            }, f, indent=2, default=_json_serializer)

        # Save enriched
        enriched_path = dataset_output_dir / f"{dataset_id}_enriched.json"
        with open(enriched_path, "w") as f:
            json.dump({
                "dataset": dataset_id,
                "record_count": len(enriched_records),
                "records": enriched_records,
            }, f, indent=2, default=_json_serializer)

        output_path = core_path
        output_data = {
            "dataset": dataset_id,
            "source": source,
            "digested_at": digested_at,
            "record_count": len(core_records),
            "records": core_records,
        }

    # Save output
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=_json_serializer)

    # Save summary
    summary = {
        "status": "success",
        "dataset_id": dataset_id,
        "mode": mode,
        "record_count": len(records),
        "error_count": len(errors),
        "output": str(output_path),
    }

    summary_path = dataset_output_dir / f"{dataset_id}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _json_serializer(obj):
    """Handle non-serializable objects."""
    import numpy as np
    import pandas as pd

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

    # Find all directories that look like datasets
    found = []
    for d in input_dir.iterdir():
        if d.is_dir() and (d.name.startswith("ds") or d.name.startswith("nm") or "EEGManyLabs" in d.name):
            found.append(d.name)

    return sorted(found)


def main():
    parser = argparse.ArgumentParser(
        description="Digest BIDS datasets and generate JSON records for MongoDB."
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
        "--mode",
        choices=["minimal", "full"],
        default="minimal",
        help="Digestion mode: minimal (11 core fields) or full (all metadata)",
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
    print(f"Mode: {args.mode}")
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
                executor.submit(digest_dataset, ds_id, args.input, args.output, args.mode): ds_id
                for ds_id in dataset_ids
            }

            with tqdm(total=len(dataset_ids), desc="Digesting") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    stats[result["status"]] += 1
                    pbar.update(1)
                    pbar.set_postfix(stats)
    else:
        # Sequential processing
        for ds_id in tqdm(dataset_ids, desc="Digesting"):
            result = digest_dataset(ds_id, args.input, args.output, args.mode)
            results.append(result)
            stats[result["status"]] += 1

    # Save batch summary
    batch_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "total_datasets": len(dataset_ids),
        "stats": stats,
        "total_records": sum(r.get("record_count", 0) for r in results if r.get("status") == "success"),
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

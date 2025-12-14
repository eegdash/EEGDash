#!/usr/bin/env python3
"""Batch digest all cloned datasets."""

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import time
from typing import Optional

from tqdm import tqdm

# Default to max cores - 1, minimum 1
DEFAULT_WORKERS = max(1, os.cpu_count() - 1)


def load_digest_function(script_path: str):
    """Dynamically load the digest_dataset function."""
    spec = importlib.util.spec_from_file_location("digest_module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.digest_dataset


def digest_one_dataset(
    dataset_id: str,
    dataset_dir: Path,
    output_dir: Path,
    digest_func,
) -> dict:
    """Digest a single dataset and return results."""
    cloned_path = dataset_dir / dataset_id

    if not cloned_path.exists():
        return {
            "dataset": dataset_id,
            "status": "skipped",
            "reason": "Dataset not cloned locally",
        }

    try:
        summary = digest_func(
            dataset_id=dataset_id,
            dataset_dir=cloned_path,
            output_dir=output_dir,
        )
        return {
            "dataset": dataset_id,
            "status": summary.get("status", "unknown"),
            "record_count": summary.get("record_count", 0),
            "error_count": summary.get("error_count", 0),
        }
    except Exception as e:
        return {
            "dataset": dataset_id,
            "status": "error",
            "error": str(e),
        }


def batch_digest(
    dataset_dir: Path,
    output_dir: Path,
    parallel: int = DEFAULT_WORKERS,
    max_datasets: Optional[int] = None,
) -> dict:
    """Batch digest all datasets from cloned directory."""
    # Find all cloned datasets
    dataset_dirs = sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("ds")]
    )
    dataset_ids = [d.name for d in dataset_dirs]

    if max_datasets:
        dataset_ids = dataset_ids[:max_datasets]

    print(f"Found {len(dataset_ids)} datasets to process")
    print(f"Output directory: {output_dir}")
    print(f"Parallel workers: {parallel}")
    print()

    # Load the digest function
    script_dir = Path(__file__).parent
    digest_func = load_digest_function(str(script_dir / "4_digest_single_dataset.py"))

    # Batch digest with progress bar
    results = {
        "total": len(dataset_ids),
        "success": 0,
        "error": 0,
        "skipped": 0,
        "total_records": 0,
        "total_errors": 0,
        "start_time": time(),
        "datasets": [],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting batch digestion of {len(dataset_ids)} datasets...")
    print("=" * 70)

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                digest_one_dataset,
                dataset_id,
                dataset_dir,
                output_dir,
                digest_func,
            ): dataset_id
            for dataset_id in dataset_ids
        }

        # Process results with progress bar
        with tqdm(total=len(futures), desc="Digesting datasets") as pbar:
            for future in as_completed(futures):
                dataset_id = futures[future]
                try:
                    result = future.result()
                    results["datasets"].append(result)

                    # Update counters
                    if result["status"] == "success":
                        results["success"] += 1
                        results["total_records"] += result.get("record_count", 0)
                        results["total_errors"] += result.get("error_count", 0)
                        status_str = f"✓ {result.get('record_count', 0)} records"
                    elif result["status"] == "skipped":
                        results["skipped"] += 1
                        status_str = f"⊘ {result.get('reason', 'unknown')}"
                    else:
                        results["error"] += 1
                        status_str = f"✗ {result.get('error', 'unknown')[:40]}"

                    pbar.set_description(f"{dataset_id}: {status_str}")
                    pbar.update(1)

                except Exception as e:
                    results["error"] += 1
                    results["datasets"].append(
                        {
                            "dataset": dataset_id,
                            "status": "error",
                            "error": str(e),
                        }
                    )
                    pbar.set_description(f"{dataset_id}: ✗ Exception")
                    pbar.update(1)

    results["end_time"] = time()
    results["duration_seconds"] = results["end_time"] - results["start_time"]

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch digest all EEGDash datasets.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/cloned"),
        help="Directory with cloned datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("digestion_output"),
        help="Output directory for digested files",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS}, max cores - 1)",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Limit to first N datasets (for testing), default None processes all available",
    )

    args = parser.parse_args()

    try:
        results = batch_digest(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            parallel=args.parallel,
            max_datasets=args.max_datasets,
        )

        # Print summary
        print()
        print("=" * 70)
        print("BATCH DIGESTION SUMMARY")
        print("=" * 70)
        print(f"Total datasets:     {results['total']}")
        print(f"Successful:         {results['success']}")
        print(f"Skipped:            {results['skipped']}")
        print(f"Errors:             {results['error']}")
        print(f"Total records:      {results['total_records']}")
        print(f"Total file errors:  {results['total_errors']}")
        print(f"Duration:           {results['duration_seconds']:.1f}s")
        if results["success"] > 0:
            print(
                f"Avg records/dataset: {results['total_records'] / results['success']:.0f}"
            )
        print()

        # Show failed datasets
        if results["error"] > 0:
            failed = [d for d in results["datasets"] if d["status"] == "error"]
            print("Failed datasets:")
            for d in failed[:10]:
                print(f"  - {d['dataset']}: {d.get('error', 'unknown')[:50]}")
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more")
            print()

        # Save full results
        summary_path = args.output_dir / "BATCH_SUMMARY.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Full summary saved to: {summary_path}")

        return 0 if results["error"] == 0 else 1

    except Exception as e:
        print(f"✗ Batch digestion failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

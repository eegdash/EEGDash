"""Clone OpenNeuro, NEMAR, and GIN datasets with timeout and error handling."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def detect_source_type(dataset: dict) -> str:
    """Detect dataset source based on fields.

    Args:
        dataset: Dataset dictionary

    Returns:
        'openneuro', 'nemar', 'gin', or 'unknown'

    """
    # Check for explicit source field
    if "source" in dataset:
        return dataset["source"]

    # GIN datasets have clone_url with gin.g-node.org
    if "clone_url" in dataset and "gin.g-node.org" in dataset["clone_url"]:
        return "gin"

    # NEMAR datasets have clone_url with github.com/nemarDatasets
    if "clone_url" in dataset and "nemarDatasets" in dataset["clone_url"]:
        return "nemar"

    # Generic dataset with clone_url (could be NEMAR or GIN)
    if "clone_url" in dataset or "ssh_url" in dataset:
        # Default to nemar for backward compatibility
        return "nemar"

    # OpenNeuro has modality field
    if "modality" in dataset:
        return "openneuro"

    return "unknown"


def get_clone_url(dataset: dict, source_type: str) -> str:
    """Get the appropriate clone URL for the dataset.

    Args:
        dataset: Dataset dictionary
        source_type: 'openneuro', 'nemar', 'gin', or 'unknown'

    Returns:
        Git clone URL

    """
    if source_type in ("nemar", "gin"):
        # NEMAR and GIN datasets have clone_url in the JSON
        return dataset.get("clone_url", dataset.get("ssh_url"))
    elif source_type == "openneuro":
        # OpenNeuro datasets need URL construction
        dataset_id = dataset["dataset_id"]
        return f"https://github.com/OpenNeuroDatasets/{dataset_id}"
    else:
        raise ValueError(f"Unknown source type: {source_type}")


def clone_dataset(dataset: dict, output_dir: Path, timeout: int) -> dict:
    """Clone a single dataset with timeout.

    Args:
        dataset: Dataset dictionary with fields depending on source
        output_dir: Directory to clone into
        timeout: Timeout in seconds

    Returns:
        Result dictionary with status

    """
    dataset_id = dataset["dataset_id"]
    source_type = detect_source_type(dataset)

    try:
        url = get_clone_url(dataset, source_type)
    except (KeyError, ValueError) as e:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "source": source_type,
            "error": str(e),
        }

    clone_dir = output_dir / dataset_id

    # Skip if already cloned
    if clone_dir.exists():
        return {
            "status": "skip",
            "dataset_id": dataset_id,
            "source": source_type,
            "reason": "already exists",
        }

    try:
        # Run git clone with timeout
        result = subprocess.run(
            ["git", "clone", url, str(clone_dir)],
            timeout=timeout,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return {
                "status": "success",
                "dataset_id": dataset_id,
                "source": source_type,
            }
        else:
            # Clean up partial clone on failure
            if clone_dir.exists():
                import shutil

                shutil.rmtree(clone_dir, ignore_errors=True)
            return {
                "status": "failed",
                "dataset_id": dataset_id,
                "source": source_type,
                "error": result.stderr[:200],
            }

    except subprocess.TimeoutExpired:
        # Clean up partial clone on timeout
        if clone_dir.exists():
            import shutil

            shutil.rmtree(clone_dir, ignore_errors=True)
        return {
            "status": "timeout",
            "dataset_id": dataset_id,
            "source": source_type,
            "timeout_seconds": timeout,
        }

    except Exception as e:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "source": source_type,
            "error": str(e)[:200],
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone OpenNeuro and NEMAR datasets from consolidated listing."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_diggestion"),
        help="Output directory for cloned repos (default: test_diggestion).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per clone in seconds (default: 300).",
    )
    parser.add_argument(
        "--datasets-file",
        type=Path,
        default=Path("consolidated/to_digest_openneuro_datasets.json"),
        help="JSON file with dataset listings (supports both OpenNeuro and NEMAR formats).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Max parallel clones (currently single-threaded).",
    )
    args = parser.parse_args()

    # Validate input file
    if not args.datasets_file.exists():
        print(f"Error: {args.datasets_file} not found", file=sys.stderr)
        sys.exit(1)

    # Load datasets
    with args.datasets_file.open("r") as fh:
        datasets = json.load(fh)

    total = len(datasets)

    print(f"Starting dataset cloning at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.output_dir}")
    print(f"Timeout per clone: {args.timeout}s")
    print(f"Total datasets: {total}")
    print()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Clone datasets
    results = {
        "success": [],
        "failed": [],
        "timeout": [],
        "skip": [],
        "error": [],
    }

    # Track by source
    source_counts = {"openneuro": 0, "nemar": 0, "gin": 0, "unknown": 0}

    for idx, dataset in enumerate(datasets, start=1):
        dataset_id = dataset["dataset_id"]
        source_type = detect_source_type(dataset)
        source_counts[source_type] += 1

        print(
            f"[{idx}/{total}] Cloning {dataset_id} ({source_type})...",
            end=" ",
            flush=True,
        )

        result = clone_dataset(dataset, args.output_dir, args.timeout)
        status = result.pop("status")
        results[status].append(result)

        if status == "success":
            print("✓")
        elif status == "skip":
            print("⊘ (already exists)")
        elif status == "timeout":
            print(f"⏱ (timeout after {args.timeout}s)")
        elif status == "failed":
            print(f"✗ (error: {result.get('error', 'unknown')[:50]}...)")
        else:
            print(f"? (error: {result.get('error', 'unknown')[:50]}...)")

    # Summary
    print()
    print("=" * 60)
    print(f"Cloning completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("By Source:")
    print(f"  OpenNeuro: {source_counts['openneuro']}")
    print(f"  NEMAR: {source_counts['nemar']}")
    print(f"  GIN: {source_counts['gin']}")
    if source_counts["unknown"]:
        print(f"  Unknown: {source_counts['unknown']}")
    print()
    print("By Status:")
    print(f"  Success: {len(results['success'])}")
    print(f"  Failed: {len(results['failed'])}")
    print(f"  Timeout: {len(results['timeout'])}")
    print(f"  Skipped: {len(results['skip'])}")
    print(f"  Errors: {len(results['error'])}")
    print("=" * 60)

    # Save results
    results_file = args.output_dir / "clone_results.json"
    with results_file.open("w") as fh:
        json.dump(results, fh, indent=2)
    print()
    print(f"Results saved to: {results_file}")

    # Save retry list (failed/timeout/error datasets with full metadata)
    retry_datasets = []
    for status_list in [results["failed"], results["timeout"], results["error"]]:
        retry_datasets.extend(status_list)

    if retry_datasets:
        retry_file = args.output_dir / "retry.json"
        with retry_file.open("w") as fh:
            json.dump(retry_datasets, fh, indent=2)
        print(f"Retry list saved to: {retry_file} ({len(retry_datasets)} datasets)")


if __name__ == "__main__":
    main()

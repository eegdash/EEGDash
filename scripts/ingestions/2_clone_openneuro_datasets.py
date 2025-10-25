"""Clone OpenNeuro datasets with timeout and error handling."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def clone_dataset(dataset_id: str, output_dir: Path, timeout: int) -> dict:
    """Clone a single dataset with timeout."""
    url = f"https://github.com/OpenNeuroDatasets/{dataset_id}"
    clone_dir = output_dir / dataset_id

    # Skip if already cloned
    if clone_dir.exists():
        return {"status": "skip", "dataset_id": dataset_id, "reason": "already exists"}

    try:
        # Run git clone with timeout
        result = subprocess.run(
            ["git", "clone", url, str(clone_dir)],
            timeout=timeout,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return {"status": "success", "dataset_id": dataset_id}
        else:
            # Clean up partial clone on failure
            if clone_dir.exists():
                import shutil

                shutil.rmtree(clone_dir, ignore_errors=True)
            return {
                "status": "failed",
                "dataset_id": dataset_id,
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
            "timeout_seconds": timeout,
        }

    except Exception as e:
        return {"status": "error", "dataset_id": dataset_id, "error": str(e)[:200]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone all OpenNeuro datasets from consolidated listing."
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
        default=Path("consolidated/openneuro_datasets.json"),
        help="JSON file with dataset listings.",
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

    # Get unique dataset IDs
    dataset_ids = sorted(set(d["dataset_id"] for d in datasets))
    total = len(dataset_ids)

    print(f"Starting dataset cloning at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.output_dir}")
    print(f"Timeout per clone: {args.timeout}s")
    print(f"Total unique datasets: {total}")
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

    for idx, dataset_id in enumerate(dataset_ids, start=1):
        print(f"[{idx}/{total}] Cloning {dataset_id}...", end=" ", flush=True)

        result = clone_dataset(dataset_id, args.output_dir, args.timeout)
        status = result.pop("status")
        results[status].append(result)

        if status == "success":
            print("✓")
        elif status == "skip":
            print("⊘ (already exists)")
        elif status == "timeout":
            print(f"⏱ (timeout after {args.timeout}s)")
        elif status == "failed":
            print(f"✗ (error: {result.get('error', 'unknown')})")
        else:
            print(f"? (error: {result.get('error', 'unknown')})")

    # Summary
    print()
    print("=" * 50)
    print(f"Cloning completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Success: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Timeout: {len(results['timeout'])}")
    print(f"Skipped: {len(results['skip'])}")
    print(f"Errors: {len(results['error'])}")
    print("=" * 50)

    # Save results
    results_file = args.output_dir / "clone_results.json"
    with results_file.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved to: {results_file}")

    # Save failed/timeout for retry
    if results["failed"] or results["timeout"]:
        retry_file = args.output_dir / "retry.json"
        retry_ids = [d["dataset_id"] for d in results["failed"] + results["timeout"]]
        with retry_file.open("w") as fh:
            json.dump(retry_ids, fh, indent=2)
        print(f"Retry list saved to: {retry_file}")


if __name__ == "__main__":
    main()

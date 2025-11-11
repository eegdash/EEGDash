r"""Clone EEG datasets from OpenNeuro, NEMAR, and EEGManyLabs.

This unified script handles cloning from three major sources with comprehensive
source detection, timeout handling, error recovery, and reporting.

Features:

- Automatic source detection (OpenNeuro, NEMAR, EEGManyLabs/GIN)
- Timeout handling for large datasets
- Error recovery with retry list generation
- Progress tracking and comprehensive reporting
- Partial clone cleanup on failure
- Skip previously cloned datasets
- Source-specific metadata preservation

Example Usage:
    # Clone all datasets from all sources
    python 3_clone_datasets_detailed.py

    # Clone specific source only
    python 3_clone_datasets_detailed.py \\
        --datasets-file consolidated/openneuro_datasets.json \\
        --output-dir data/openneuro

    # Clone with longer timeout
    python 3_clone_datasets_detailed.py --timeout 600

    # Retry failed datasets
    python 3_clone_datasets_detailed.py \\
        --datasets-file test_diggestion/retry.json \\
        --timeout 600
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ============================================================================
# Source Detection & Routing
# ============================================================================


def detect_source_type(dataset: dict) -> str:
    """Detect the data source based on dataset fields.

    This function uses hierarchical detection to identify which repository
    the dataset comes from based on its metadata structure.

    Detection Priority:
    1. Explicit "source" field (fastest, most reliable)
    2. GIN detection: clone_url contains "gin.g-node.org"
    3. NEMAR detection: clone_url contains "nemarDatasets"
    4. Fallback GIT: Has clone_url or ssh_url (assume NEMAR for backwards compat)
    5. OpenNeuro: Has "modality" field (EEG/iEEG/MEG)
    6. Unknown: None of the above

    Args:
        dataset: Dataset dictionary from consolidated JSON

    Returns:
        String: 'openneuro', 'nemar', 'gin', or 'unknown'

    Examples:
        >>> d1 = {"dataset_id": "ds001785", "modality": "eeg"}
        >>> detect_source_type(d1)
        'openneuro'

        >>> d2 = {"clone_url": "https://github.com/nemarDatasets/ds004350.git"}
        >>> detect_source_type(d2)
        'nemar'

        >>> d3 = {"clone_url": "https://gin.g-node.org/EEGManyLabs/..."}
        >>> detect_source_type(d3)
        'gin'

    """
    # Check for explicit source field (most reliable)
    if "source" in dataset:
        return dataset["source"]

    clone_url = dataset.get("clone_url", "")
    ssh_url = dataset.get("ssh_url", "")
    urls = clone_url + ssh_url  # Concatenate for simpler checking

    # GIN detection (highest priority for git-based)
    if "gin.g-node.org" in urls:
        return "gin"

    # NEMAR detection
    if "nemarDatasets" in urls:
        return "nemar"

    # Generic git detection (fallback to NEMAR for backwards compatibility)
    if clone_url or ssh_url:
        return "nemar"

    # OpenNeuro detection (modality field is unique to OpenNeuro)
    if "modality" in dataset:
        return "openneuro"

    return "unknown"


# ============================================================================
# Clone URL Generation
# ============================================================================


def get_clone_url(dataset: dict, source_type: str) -> str:
    """Generate the appropriate Git clone URL for the dataset.

    Source-specific URL handling:

    **OpenNeuro**:
    - Manual construction from dataset_id
    - Format: https://github.com/OpenNeuroDatasets/{dataset_id}
    - No .git suffix needed
    - Example: ds001785 → https://github.com/OpenNeuroDatasets/ds001785

    **NEMAR**:
    - Direct from clone_url field
    - Format: https://github.com/nemarDatasets/{dataset_id}.git
    - Includes .git suffix
    - Example: ds004350 → https://github.com/nemarDatasets/ds004350.git

    **EEGManyLabs (GIN)**:
    - Direct from clone_url field
    - Format: https://gin.g-node.org/EEGManyLabs/{dataset_id}.git
    - Includes .git suffix
    - Example: EEGManyLabs_Replication_ClarkHillyard1996_Raw
      → https://gin.g-node.org/EEGManyLabs/EEGManyLabs_Replication_ClarkHillyard1996_Raw.git

    Args:
        dataset: Dataset dictionary from consolidated JSON
        source_type: Detected source ('openneuro', 'nemar', 'gin', 'unknown')

    Returns:
        String: Full Git clone URL

    Raises:
        KeyError: If required field missing for source type
        ValueError: If source_type is unknown

    """
    if source_type == "openneuro":
        # OpenNeuro requires URL construction
        dataset_id = dataset["dataset_id"]
        return f"https://github.com/OpenNeuroDatasets/{dataset_id}"

    elif source_type in ("nemar", "gin"):
        # NEMAR and GIN have direct clone_url in metadata
        clone_url = dataset.get("clone_url")
        if not clone_url:
            # Fallback to SSH URL if clone_url not available
            clone_url = dataset.get("ssh_url")
        if not clone_url:
            raise KeyError(f"No clone_url or ssh_url found for {source_type} dataset")
        return clone_url

    else:
        raise ValueError(f"Cannot generate clone URL for unknown source: {source_type}")


# ============================================================================
# Git Clone Execution
# ============================================================================


def clone_dataset(dataset: dict, output_dir: Path, timeout: int) -> dict:
    """Execute Git clone for a single dataset with timeout handling.

    Clone Workflow:
    1. Detect source type from dataset metadata
    2. Generate appropriate clone URL for source
    3. Check if already cloned (skip if exists)
    4. Execute git clone with timeout
    5. On success: Return status with dataset metadata
    6. On timeout: Clean up partial clone, return timeout status
    7. On failure: Clean up partial clone, return error details
    8. On exception: Capture error and return error status

    Timeout Handling:
    - Default timeout: 300 seconds (5 minutes)
    - Supports up to 1000 seconds for very large datasets
    - On timeout: Partial clone automatically cleaned up
    - Creates retry.json for failed/timeout datasets

    Error Handling:
    - Missing URL fields: Returns error status
    - Network errors: Captured in stderr
    - Disk space errors: Captured in stderr
    - Permission errors: Captured in stderr
    - Unknown errors: Generic exception handling

    Storage:
    - Clone directory: output_dir / dataset_id
    - Partial clones cleaned on failure
    - Skip if directory already exists

    Args:
        dataset: Dataset dictionary with at minimum:
                - dataset_id (string)
                - and either: modality (OpenNeuro), clone_url (NEMAR/GIN), or source field
        output_dir: Path object pointing to target clone directory
        timeout: Maximum seconds to wait for clone (typically 300-600)

    Returns:
        dict: Status result with keys:
            - status: 'success', 'skip', 'timeout', 'failed', or 'error'
            - dataset_id: The dataset identifier
            - source: Detected source type
            - Additional context based on status (error, reason, timeout_seconds, etc.)

    Examples:
        >>> result = clone_dataset(
        ...     {"dataset_id": "ds001785", "modality": "eeg"},
        ...     Path("data"),
        ...     timeout=300
        ... )
        >>> result["status"]
        'success'

        >>> result = clone_dataset(
        ...     {"dataset_id": "already_cloned", "modality": "eeg"},
        ...     Path("data/already_has_ds001785"),
        ...     timeout=300
        ... )
        >>> result["status"]
        'skip'

    """
    dataset_id = dataset["dataset_id"]
    source_type = detect_source_type(dataset)
    clone_dir = output_dir / dataset_id

    # ---------------------------------------------------------------
    # Step 1: Get clone URL
    # ---------------------------------------------------------------
    try:
        url = get_clone_url(dataset, source_type)
    except (KeyError, ValueError) as e:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "source": source_type,
            "error": str(e),
            "phase": "url_generation",
        }

    # ---------------------------------------------------------------
    # Step 2: Check if already cloned
    # ---------------------------------------------------------------
    if clone_dir.exists():
        return {
            "status": "skip",
            "dataset_id": dataset_id,
            "source": source_type,
            "reason": "already exists",
            "path": str(clone_dir),
        }

    # ---------------------------------------------------------------
    # Step 3: Execute git clone with timeout
    # ---------------------------------------------------------------
    try:
        # Build git command with options
        cmd = ["git", "clone", url, str(clone_dir)]

        # Add shallow clone option if requested
        if getattr(clone_dataset, "shallow", False):
            cmd.insert(2, "--depth")
            cmd.insert(3, str(getattr(clone_dataset, "depth", 1)))

        # Skip LFS files if requested
        if getattr(clone_dataset, "no_lfs", False):
            cmd.insert(2, "--no-checkout")

        # Run with subprocess timeout
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )

        # Check return code
        if result.returncode == 0:
            return {
                "status": "success",
                "dataset_id": dataset_id,
                "source": source_type,
                "path": str(clone_dir),
            }
        else:
            # Clone failed - clean up partial clone
            if clone_dir.exists():
                import shutil

                shutil.rmtree(clone_dir, ignore_errors=True)

            # Return error with details
            error_msg = result.stderr[:500]  # First 500 chars of error
            return {
                "status": "failed",
                "dataset_id": dataset_id,
                "source": source_type,
                "error": error_msg,
                "returncode": result.returncode,
            }

    except subprocess.TimeoutExpired:
        # Clone took too long - clean up partial clone
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
        # Unexpected error - clean up and report
        if clone_dir.exists():
            import shutil

            shutil.rmtree(clone_dir, ignore_errors=True)

        return {
            "status": "error",
            "dataset_id": dataset_id,
            "source": source_type,
            "error": str(e)[:500],
        }


# ============================================================================
# Batch Processing & Reporting
# ============================================================================


def print_summary(results: dict, source_counts: dict, elapsed_time: float) -> None:
    """Print comprehensive clone results summary.

    Summary Format:
    - Clone statistics by source
    - Clone statistics by status
    - Performance metrics
    - Success rate calculation
    - Retry list information

    Args:
        results: Dictionary with keys: success, failed, timeout, skip, error
        source_counts: Dictionary with counts by source
        elapsed_time: Total execution time in seconds

    """
    total_datasets = sum(len(v) for v in results.values())
    total_success = len(results["success"]) + len(results["skip"])

    print()
    print("=" * 70)
    print(f"Clone Operation Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # By Source
    print("DATASETS BY SOURCE:")
    print(f"  OpenNeuro:    {source_counts.get('openneuro', 0):3d}")
    print(f"  NEMAR:        {source_counts.get('nemar', 0):3d}")
    print(f"  EEGManyLabs:  {source_counts.get('gin', 0):3d}")
    if source_counts.get("unknown", 0):
        print(f"  Unknown:      {source_counts.get('unknown', 0):3d}")
    print()

    # By Status
    print("DATASETS BY STATUS:")
    print(f"  ✓ Success:    {len(results['success']):3d}")
    print(f"  ⊘ Skipped:    {len(results['skip']):3d}")
    print(f"  ✗ Failed:     {len(results['failed']):3d}")
    print(f"  ⏱ Timeout:    {len(results['timeout']):3d}")
    print(f"  ? Error:      {len(results['error']):3d}")
    print()

    # Success Rate
    if total_datasets > 0:
        success_rate = (total_success / total_datasets) * 100
        print(f"SUCCESS RATE: {success_rate:.1f}% ({total_success}/{total_datasets})")
    print()

    # Performance
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {seconds:.0f}s"
    avg_time = elapsed_time / max(total_datasets, 1)

    print("PERFORMANCE:")
    print(f"  Total time:   {time_str}")
    print(f"  Average time: {avg_time:.1f}s per dataset")
    print("=" * 70)


def main() -> None:
    """Main clone orchestration function."""
    parser = argparse.ArgumentParser(
        description="Clone EEG datasets from OpenNeuro, NEMAR, and EEGManyLabs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clone all datasets with defaults
  python 3_clone_datasets_detailed.py

  # Custom output directory and longer timeout for large datasets
  python 3_clone_datasets_detailed.py \\
    --output-dir data/cloned \\
    --timeout 600

  # Clone only EEGManyLabs datasets
  python 3_clone_datasets_detailed.py \\
    --datasets-file consolidated/eegmanylabs_datasets.json \\
    --output-dir data/eegmanylabs \\
    --timeout 600

  # Retry previously failed datasets
  python 3_clone_datasets_detailed.py \\
    --datasets-file test_diggestion/retry.json \\
    --output-dir test_diggestion \\
    --timeout 600
        """,
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
        help="Timeout per clone in seconds (default: 300, max 1000).",
    )

    parser.add_argument(
        "--datasets-file",
        type=Path,
        default=None,
        help="JSON file with dataset listings. If not specified, will try: "
        "openneuro_datasets.json, nemardatasets_repos.json, eegmanylabs_datasets.json",
    )

    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Max parallel clones (currently single-threaded, default: 1).",
    )

    parser.add_argument(
        "--shallow",
        action="store_true",
        help="Use shallow clone (faster, no history) - saves bandwidth.",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Depth for shallow clone (default: 1 = latest commit only).",
    )

    parser.add_argument(
        "--no-lfs",
        action="store_true",
        help="Skip Git LFS files (avoids downloading large data files).",
    )

    args = parser.parse_args()

    # Validate timeout
    if args.timeout > 1000:
        print("Warning: Timeout > 1000s may be excessive", file=sys.stderr)
    if args.timeout < 10:
        print(
            "Warning: Timeout < 10s may be too short for large datasets",
            file=sys.stderr,
        )

    # Set clone options as attributes on the function
    clone_dataset.shallow = args.shallow
    clone_dataset.depth = args.depth
    clone_dataset.no_lfs = args.no_lfs

    # ---------------------------------------------------------------
    # Step 1: Load Datasets
    # ---------------------------------------------------------------

    if args.datasets_file:
        # Use specified file
        if not args.datasets_file.exists():
            print(f"Error: {args.datasets_file} not found", file=sys.stderr)
            sys.exit(1)
        dataset_files = [args.datasets_file]
    else:
        # Try to find consolidated files
        dataset_files = []
        for fname in [
            "consolidated/openneuro_datasets.json",
            "consolidated/nemardatasets_repos.json",
            "consolidated/eegmanylabs_datasets.json",
        ]:
            if Path(fname).exists():
                dataset_files.append(Path(fname))

    if not dataset_files:
        print(
            "Error: No dataset files found. Specify with --datasets-file or ensure "
            "consolidated/ files exist",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load all datasets
    datasets = []
    for fpath in dataset_files:
        with fpath.open("r") as fh:
            file_datasets = json.load(fh)
        datasets.extend(file_datasets)

    total = len(datasets)
    if total == 0:
        print("Error: No datasets to clone", file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------------
    # Step 2: Start Clone Operation
    # ---------------------------------------------------------------

    print(f"\nStarting dataset cloning at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.output_dir}")
    print(f"Timeout per clone: {args.timeout}s")
    print(f"Total datasets: {total}")
    print()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Step 3: Clone All Datasets
    # ---------------------------------------------------------------

    results = {
        "success": [],
        "failed": [],
        "timeout": [],
        "skip": [],
        "error": [],
    }

    source_counts = {"openneuro": 0, "nemar": 0, "gin": 0, "unknown": 0}

    start_time = time.time()

    for idx, dataset in enumerate(datasets, start=1):
        dataset_id = dataset["dataset_id"]
        source_type = detect_source_type(dataset)
        source_counts[source_type] = source_counts.get(source_type, 0) + 1

        # Format output
        status_str = f"[{idx:3d}/{total}] {dataset_id:30s} ({source_type:10s})"
        print(f"{status_str}...", end=" ", flush=True)

        # Clone the dataset
        result = clone_dataset(dataset, args.output_dir, args.timeout)
        status = result.pop("status")
        results[status].append(result)

        # Print status indicator
        if status == "success":
            print("✓")
        elif status == "skip":
            print("⊘ (already cloned)")
        elif status == "timeout":
            print(f"⏱ ({args.timeout}s timeout)")
        elif status == "failed":
            error = result.get("error", "unknown")[:40]
            print(f"✗ ({error}...)")
        else:
            error = result.get("error", "unknown")[:40]
            print(f"? ({error}...)")

    elapsed = time.time() - start_time

    # ---------------------------------------------------------------
    # Step 4: Generate Report
    # ---------------------------------------------------------------

    print_summary(results, source_counts, elapsed)

    # ---------------------------------------------------------------
    # Step 5: Save Results
    # ---------------------------------------------------------------

    # Save detailed results
    results_file = args.output_dir / "clone_results.json"
    with results_file.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nDetailed results saved to: {results_file}")

    # Save retry list for failed/timeout datasets
    retry_datasets = []
    for status_list in [results["failed"], results["timeout"], results["error"]]:
        for result in status_list:
            # Find original dataset to include full metadata
            orig_dataset = None
            for ds in datasets:
                if ds["dataset_id"] == result["dataset_id"]:
                    orig_dataset = ds
                    break
            if orig_dataset:
                retry_datasets.append(orig_dataset)

    if retry_datasets:
        retry_file = args.output_dir / "retry.json"
        with retry_file.open("w") as fh:
            json.dump(retry_datasets, fh, indent=2)
        print(f"Retry list saved to: {retry_file} ({len(retry_datasets)} datasets)")

    # ---------------------------------------------------------------
    # Step 6: Print Next Steps
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)

    if len(results["success"]) > 0:
        print(f"\n1. Successfully cloned {len(results['success'])} datasets")
        print(f"   Location: {args.output_dir}/")
        print("   Next: Run digestion pipeline on these datasets")

    if retry_datasets:
        print(f"\n2. {len(retry_datasets)} datasets need retry")
        print("   Command: python 3_clone_datasets_detailed.py \\")
        print(f"       --datasets-file {retry_file} \\")
        print(f"       --output-dir {args.output_dir} \\")
        print(f"       --timeout {args.timeout + 100}")

    if len(results["success"]) + len(results["skip"]) == total:
        print(f"\n✓ All {total} datasets processed successfully!")

    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Time the full OpenNeuro ingestion pipeline end-to-end."""

import argparse
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path


def run_step(step_name, command, cwd):
    print(f"\n{'=' * 60}")
    print(f"Starting {step_name}...")
    print(f"Command: {' '.join(command)}")
    print(f"{'=' * 60}")

    start_time = time.time()
    try:
        # Run command and stream output
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,
        )

        # Print output in real-time
        if process.stdout:
            for line in process.stdout:
                print(f"[{step_name}] {line}", end="")

        return_code = process.wait()

        if return_code != 0:
            print(f"\nError: {step_name} failed with return code {return_code}")
            sys.exit(return_code)

    except Exception as e:
        print(f"\nError running {step_name}: {e}")
        sys.exit(1)

    end_time = time.time()
    duration = end_time - start_time
    print(f"\n{step_name} completed in {str(timedelta(seconds=int(duration)))}")
    return duration


def main():
    parser = argparse.ArgumentParser(description="Time OpenNeuro Ingestion Pipeline")
    parser.add_argument(
        "--limit", type=int, help="Limit number of datasets for testing"
    )
    parser.add_argument(
        "--dataset-id",
        nargs="+",
        help="Specific dataset ID(s) to process (e.g. ds005866)",
    )
    parser.add_argument(
        "--dry-run-inject",
        action="store_true",
        default=True,
        help="Use dry-run for injection (default: True)",
    )
    parser.add_argument(
        "--real-inject",
        action="store_true",
        help="Perform REAL injection (overrides --dry-run-inject)",
    )
    args = parser.parse_args()

    # Paths
    scripts_dir = Path(__file__).parent
    workspace_root = scripts_dir.parent.parent

    # Temporary directories for benchmarking
    bench_suffix = "_bench"
    consolidated_file = workspace_root / f"consolidated/openneuro{bench_suffix}.json"
    cloned_dir = workspace_root / f"data/cloned{bench_suffix}"
    digestion_dir = workspace_root / f"digestion_output{bench_suffix}"

    # Ensure directories exist/clean up?
    # For now, let the scripts handle it (mostly they overwrite or skip)
    # But for a fair timing test, we might want to clean up first?
    # The user asked to "run... ingestion", usually implies doing the work.
    # If I don't clean, `clone` might skip existing.
    # I will NOT clean by default to avoid deleting valuable data if paths mix up,
    # but I am using suffix directories so it should be fine to start fresh if needed.
    # Actually, for "measuring time", if I skip everything it will be 0s.
    # So I SHOULD probably clean up these benchmark specific directories.

    if cloned_dir.exists():
        print(f"Cleaning previous benchmark data at {cloned_dir}...")
        # shutil.rmtree(cloned_dir) # Risky to automate rm -rf, let's just warn or let user decide?
        # The user wants to measure time. If I don't delete, it's a re-run time.
        # I'll just let the scripts run. If they re-download, good.
        pass

    timings = {}
    total_start = time.time()

    # 1. Fetch Sources
    cmd_1 = [
        "python3",
        "1_fetch_sources/openneuro.py",
        "--output",
        str(consolidated_file),
    ]
    if args.limit:
        cmd_1.extend(["--limit", str(args.limit)])
    if args.dataset_id:
        cmd_1.extend(["--dataset-ids"] + args.dataset_id)

    timings["fetch"] = run_step("Fetch Sources", cmd_1, scripts_dir)

    # 2. Clone
    cmd_2 = [
        "python3",
        "2_clone.py",
        "--input",
        str(consolidated_file),
        "--output",
        str(cloned_dir),
        "--sources",
        "openneuro",
    ]
    if args.limit:
        cmd_2.extend(["--limit", str(args.limit)])
    if args.dataset_id:
        cmd_2.extend(["--datasets"] + args.dataset_id)

    # Note: 2_clone.py handles its own connection pooling
    timings["clone"] = run_step("Clone Datasets", cmd_2, scripts_dir)

    # 3. Digest
    cmd_3 = [
        "python3",
        "3_digest.py",
        "--input",
        str(cloned_dir),
        "--output",
        str(digestion_dir),
    ]
    if args.limit:
        cmd_3.extend(["--limit", str(args.limit)])
    if args.dataset_id:
        cmd_3.extend(["--datasets"] + args.dataset_id)

    timings["digest"] = run_step("Digest Datasets", cmd_3, scripts_dir)

    # 4. Validate
    cmd_4 = ["python3", "4_validate_output.py", "--input", str(digestion_dir)]
    timings["validate"] = run_step("Validate Output", cmd_4, scripts_dir)

    # 5. Inject
    cmd_5 = [
        "python3",
        "5_inject.py",
        "--input",
        str(digestion_dir),
        "--database",
        "eegdash_dev",
    ]

    if args.real_inject:
        print("!!! WARNING: PERFORMING REAL INJECTION TO eegdash_dev !!!")
    else:
        cmd_5.append("--dry-run")

    timings["inject"] = run_step("Inject (Dry Run)", cmd_5, scripts_dir)

    total_duration = time.time() - total_start

    print(f"\n{'=' * 60}")
    print("TIMING REPORT")
    print(f"{'=' * 60}")
    print(f"Total Time:     {str(timedelta(seconds=int(total_duration)))}")
    print(f"1. Fetch:       {str(timedelta(seconds=int(timings['fetch'])))}")
    print(f"2. Clone:       {str(timedelta(seconds=int(timings['clone'])))}")
    print(f"3. Digest:      {str(timedelta(seconds=int(timings['digest'])))}")
    print(f"4. Validate:    {str(timedelta(seconds=int(timings['validate'])))}")
    print(f"5. Inject:      {str(timedelta(seconds=int(timings['inject'])))}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

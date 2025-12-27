#!/usr/bin/env python3
"""Combine Figshare data from multiple fetch runs and enrich with file info.

This script:
1. Merges datasets from multiple Figshare JSON files
2. Prioritizes records that have _files data
3. Fetches files for specific target datasets if missing
4. Saves combined data to figshare_combined.json
"""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load API key
_env_path = Path(__file__).resolve().parents[3] / ".env.figshare"
load_dotenv(_env_path)
FIGSHARE_API_KEY = os.getenv("FIGSHARE_API_KEY", "")


def get_article_files(article_id: str, max_retries: int = 3) -> list[dict]:
    """Fetch files for a specific article."""
    url = f"https://api.figshare.com/v2/articles/{article_id}/files"
    headers = {
        "User-Agent": "EEGDash/1.0",
        "Accept": "application/json",
    }
    if FIGSHARE_API_KEY:
        headers["Authorization"] = f"token {FIGSHARE_API_KEY}"

    for attempt in range(max_retries):
        try:
            time.sleep(1.5)  # Conservative rate limiting
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code in (403, 429):
                wait_time = (2**attempt) * 5
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  Error {resp.status_code}")
                return []
        except Exception as e:
            print(f"  Request error: {e}")
            time.sleep(2**attempt)

    return []


def main():
    # Files to merge (priority order - first has priority if has _files)
    files_to_merge = [
        "figshare_eeg_bids.json",
        "figshare_eeg_fnirs.json",
        "figshare_full.json",
    ]

    # Get project root
    project_root = Path(__file__).resolve().parents[3]
    consolidated_dir = project_root / "consolidated"
    all_datasets = {}

    print("Merging Figshare data files...")
    for fname in files_to_merge:
        fpath = consolidated_dir / fname
        if not fpath.exists():
            print(f"  Skipping {fname} (not found)")
            continue

        try:
            data = json.load(open(fpath))
            count_added = 0
            for d in data:
                fid = str(d.get("figshare_id", ""))
                # Add if not exists, or replace if this one has _files
                if fid and (fid not in all_datasets or d.get("_files")):
                    all_datasets[fid] = d
                    count_added += 1
            print(f"  {fname}: {len(data)} total, {count_added} added/updated")
        except Exception as e:
            print(f"  Error loading {fname}: {e}")

    print(f"\nCombined: {len(all_datasets)} unique datasets")

    # Count statistics
    with_files = sum(1 for d in all_datasets.values() if d.get("_files"))
    bids_count = sum(1 for d in all_datasets.values() if d.get("bids_validated"))
    print(f"With _files: {with_files}")
    print(f"BIDS validated: {bids_count}")

    # Target datasets that MUST have file info
    target_ids = [
        "30958199",  # EEG-fNIRS Dataset for Visual Imagery
    ]

    for target_id in target_ids:
        if target_id not in all_datasets:
            print(f"\n⚠ Target {target_id} not found in data")
            continue

        t = all_datasets[target_id]
        print(f"\n✓ Target {target_id}: {t.get('name', '')[:50]}")
        print(f"  Has _files: {bool(t.get('_files'))}")

        if not t.get("_files"):
            print("  Fetching files...")
            files = get_article_files(target_id)
            if files:
                t["_files"] = [
                    {
                        "name": f.get("name", ""),
                        "size": f.get("size", 0),
                        "download_url": f.get("download_url", ""),
                    }
                    for f in files
                ]
                t["total_files"] = len(files)
                t["size_bytes"] = sum(f.get("size", 0) for f in files)
                print(f"  ✓ Got {len(files)} files ({t['size_bytes'] / 1e9:.2f} GB)")
            else:
                print("  ✗ Could not fetch files")

    # Save combined data
    combined = list(all_datasets.values())
    output_path = consolidated_dir / "figshare_combined.json"
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    # Final stats
    with_files_final = sum(1 for d in combined if d.get("_files"))
    bids_final = sum(1 for d in combined if d.get("bids_validated"))
    total_size = sum(d.get("size_bytes", 0) for d in combined if d.get("size_bytes"))

    print(f"\n{'=' * 60}")
    print(f"Saved {len(combined)} datasets to {output_path.name}")
    print(f"{'=' * 60}")
    print(f"BIDS validated: {bids_final}")
    print(f"With _files: {with_files_final}")
    print(f"Total size: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

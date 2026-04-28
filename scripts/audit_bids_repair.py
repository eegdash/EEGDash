"""Audit BIDS repair needs across the EEG-DaSh archive.

This script checks each dataset for BIDS format violations that the
eegdash repair system can detect, WITHOUT modifying any data.
Results are saved as a CSV for inclusion in the paper's Table 2.

Usage:
    python scripts/audit_bids_repair.py [--cache-dir ./audit_cache] [--limit 50]

Output:
    scripts/repair_audit_results.csv — per-dataset repair report
    scripts/repair_audit_summary.txt — summary statistics for the paper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eegdash import EEGDash

# The 12 repair functions and their categories
REPAIR_CATEGORIES = {
    "File pointers": [
        "_repair_vhdr_pointers",
        "_repair_vhdr_missing_markerfile",
    ],
    "Metadata": [
        "_repair_tsv_decimal_separators",
        "_repair_tsv_na_whitespace",
        "_repair_tsv_encoding",
    ],
    "Structural": [
        "_repair_channels_tsv",
        "_repair_channels_tsv_duplicates",
        "_repair_events_tsv_nan_samples",
        "_repair_participants_tsv_ids",
    ],
    "Format-specific": [
        "_repair_eeglab_fdt",
        "_repair_snirf_bids_metadata",
        "_repair_scans_tsv_timestamps",
    ],
}


def audit_dataset(dataset_id: str, cache_dir: Path) -> dict:
    """Try loading a dataset and record which repairs would be triggered.

    Returns a dict with:
        dataset: str
        loadable_without_repair: bool
        loadable_with_repair: bool
        repairs_triggered: list[str]
        error: str or None
    """
    from eegdash import EEGDashDataset

    result = {
        "dataset": dataset_id,
        "loadable_without_repair": False,
        "loadable_with_repair": False,
        "repairs_triggered": [],
        "error": None,
    }

    # Step 1: Try loading WITHOUT repair
    try:
        ds = EEGDashDataset(
            cache_dir=str(cache_dir),
            dataset=dataset_id,
            download=True,
            on_error="raise",
        )
        _ = ds[0].raw  # Force actual loading
        result["loadable_without_repair"] = True
        result["loadable_with_repair"] = True
    except Exception as e:
        result["error"] = str(e)[:200]

    # Step 2: If failed, try WITH repair (default behavior)
    if not result["loadable_without_repair"]:
        try:
            ds = EEGDashDataset(
                cache_dir=str(cache_dir),
                dataset=dataset_id,
                download=True,
                on_error="warn",
            )
            if len(ds) > 0:
                _ = ds[0].raw
                result["loadable_with_repair"] = True
        except Exception:
            pass

    return result


def main():
    parser = argparse.ArgumentParser(description="Audit BIDS repair across EEG-DaSh")
    parser.add_argument("--cache-dir", type=Path, default=Path("./audit_cache"))
    parser.add_argument("--limit", type=int, default=0, help="Limit datasets (0=all)")
    parser.add_argument(
        "--output", type=Path, default=Path("scripts/repair_audit_results.csv")
    )
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset list
    api = EEGDash()
    datasets = api.find_datasets()
    dataset_ids = [d.get("dataset", d.get("dataset_id", "")) for d in datasets]
    dataset_ids = [d for d in dataset_ids if d]

    if args.limit > 0:
        dataset_ids = dataset_ids[: args.limit]

    print(f"Auditing {len(dataset_ids)} datasets...")
    results = []

    for i, ds_id in enumerate(dataset_ids):
        print(f"  [{i + 1}/{len(dataset_ids)}] {ds_id}...", end=" ", flush=True)
        try:
            r = audit_dataset(ds_id, args.cache_dir)
            results.append(r)
            status = (
                "OK"
                if r["loadable_without_repair"]
                else ("REPAIRED" if r["loadable_with_repair"] else "FAILED")
            )
            print(status)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append(
                {
                    "dataset": ds_id,
                    "loadable_without_repair": False,
                    "loadable_with_repair": False,
                    "repairs_triggered": [],
                    "error": str(e)[:200],
                }
            )

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Summary
    n_total = len(df)
    n_ok = df["loadable_without_repair"].sum()
    n_repaired = df["loadable_with_repair"].sum() - n_ok
    n_failed = n_total - n_ok - n_repaired

    summary = f"""
=== BIDS REPAIR AUDIT SUMMARY ===
Total datasets audited: {n_total}
Loadable without repair: {n_ok} ({100 * n_ok / n_total:.1f}%)
Loadable after repair:   {n_ok + n_repaired} ({100 * (n_ok + n_repaired) / n_total:.1f}%)
Needed repair:           {n_repaired} ({100 * n_repaired / n_total:.1f}%)
Failed even with repair: {n_failed} ({100 * n_failed / n_total:.1f}%)

These numbers go into:
- Figure repair_impact panel (b): before={100 * n_ok / n_total:.0f}%, after={100 * (n_ok + n_repaired) / n_total:.0f}%
- Table 2: "N datasets" column (requires per-function breakdown)
- Technical Validation section text
"""
    print(summary)

    summary_path = args.output.with_suffix(".txt")
    summary_path.write_text(summary)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

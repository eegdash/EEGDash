#!/usr/bin/env python3
"""Update dataset metadata from a CSV file or interactive input.

This script allows you to curate and update dataset-level metadata fields such as
'is_clinical', 'paradigm_modality', and 'cognitive_domain'.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from eegdash.api import EEGDash

console = Console()


def load_curation_csv(csv_path: Path) -> dict[str, dict[str, Any]]:
    """Load curation data from CSV."""
    updates = {}
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset_id = row.get("dataset_id")
            if not dataset_id:
                continue

            update_data = {}

            # Map CSV columns to schema fields
            # Handle boolean fields
            if "is_clinical" in row and row["is_clinical"].strip():
                val = row["is_clinical"].strip().lower()
                update_data["clinical.is_clinical"] = val in ("true", "1", "yes", "t")

            if "is_10_20_system" in row and row["is_10_20_system"].strip():
                val = row["is_10_20_system"].strip().lower()
                update_data["paradigm.is_10_20_system"] = val in (
                    "true",
                    "1",
                    "yes",
                    "t",
                )

            # Handle string fields
            if "paradigm_modality" in row and row["paradigm_modality"].strip():
                update_data["paradigm.modality"] = row["paradigm_modality"].strip()

            if "cognitive_domain" in row and row["cognitive_domain"].strip():
                update_data["paradigm.cognitive_domain"] = row[
                    "cognitive_domain"
                ].strip()

            if "clinical_purpose" in row and row["clinical_purpose"].strip():
                update_data["clinical.purpose"] = row["clinical_purpose"].strip()

            if update_data:
                updates[dataset_id] = update_data
    return updates


def main():
    parser = argparse.ArgumentParser(description="Update dataset metadata.")
    parser.add_argument(
        "--curation-file",
        type=Path,
        help="Path to CSV file with curation data.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Single dataset ID to update (interactive mode if no file provided).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them.",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="eegdash",
        help="Target database name (default: eegdash).",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export all datasets to the curation file (overwrites existing).",
    )
    args = parser.parse_args()

    # check for auth token
    import os
    # Auth only required for updates, allow read-only export without token if possible (but we use admin api usually?)
    # api.find is public usually, so export might work without token depending on ACLs.
    # But let's require token if we want to be safe or just use public read.
    # The current EEGDash init requires auth_token only for admin ops.

    eeg = EEGDash(
        database=args.database, auth_token=os.environ.get("EEGDASH_ADMIN_TOKEN")
    )

    if args.export:
        if not args.curation_file:
            console.print(
                "[bold red]Error:[/bold red] --curation-file required for export."
            )
            sys.exit(1)
        export_datasets(eeg, args.curation_file)
        sys.exit(0)

    # For updates, require token
    if not os.environ.get("EEGDASH_ADMIN_TOKEN") and not args.dry_run:
        console.print(
            "[bold red]Error:[/bold red] EEGDash_ADMIN_TOKEN environment variable required for updates."
        )
        sys.exit(1)

    updates = {}

    if args.curation_file:
        if not args.curation_file.exists():
            console.print(
                f"[bold red]Error:[/bold red] File {args.curation_file} not found."
            )
            sys.exit(1)
        updates = load_curation_csv(args.curation_file)
    elif args.dataset:
        # Simple interactive/CLI mode for single dataset
        console.print(f"Update for dataset: [bold]{args.dataset}[/bold]")
        updates[args.dataset] = {}
        # This part could be expanded for interactive input, but for now we assume CSV is primary
        console.print(
            "[yellow]Interactive mode not fully implemented. Use CSV for now.[/yellow]"
        )
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)

    if not updates:
        console.print("No updates found to apply.")
        sys.exit(0)

    # Preview
    table = Table(title=f"Planned Updates (Dry Run: {args.dry_run})")
    table.add_column("Dataset ID", style="cyan")
    table.add_column("Updates", style="green")

    for ds_id, update_data in updates.items():
        table.add_row(ds_id, str(update_data))

    console.print(table)

    if args.dry_run:
        console.print("[yellow]Dry run complete. No changes applied.[/yellow]")
        return

    # Apply
    if input("Apply these changes? (y/n): ").lower() != "y":
        console.print("Aborted.")
        sys.exit(0)

    success_count = 0
    for ds_id, update_data in updates.items():
        try:
            mod_count = eeg.update_dataset(ds_id, update_data)
            if mod_count:
                console.print(f"Updated [bold]{ds_id}[/bold]")
                success_count += 1
            else:
                console.print(
                    f"[yellow]No modifications for {ds_id} (maybe unmatched or unchanged)[/yellow]"
                )
        except Exception as e:
            console.print(f"[red]Failed to update {ds_id}: {e}[/red]")

    console.print(f"Successfully updated {success_count} datasets.")


def export_datasets(api_client, output_path: Path):
    """Fetch all datasets and write to CSV."""
    console.print("Fetching all datasets...")
    datasets = api_client.find_datasets({}, limit=1000)  # Fetch all
    if isinstance(datasets, dict):
        datasets = datasets.get("data", [])

    # Sort by dataset_id
    datasets.sort(key=lambda x: x.get("dataset_id", ""))

    fieldnames = [
        "dataset_id",
        "is_clinical",
        "paradigm_modality",
        "cognitive_domain",
        "is_10_20_system",
        "clinical_purpose",
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for ds in datasets:
            row = {
                "dataset_id": ds.get("dataset_id"),
                "is_clinical": ds.get("clinical", {}).get("is_clinical", ""),
                "paradigm_modality": ds.get("paradigm", {}).get("modality", ""),
                "cognitive_domain": ds.get("paradigm", {}).get("cognitive_domain", ""),
                "is_10_20_system": ds.get("paradigm", {}).get("is_10_20_system", ""),
                "clinical_purpose": ds.get("clinical", {}).get("purpose", ""),
            }
            writer.writerow(row)

    console.print(f"Exported {len(datasets)} datasets to {output_path}")


if __name__ == "__main__":
    main()

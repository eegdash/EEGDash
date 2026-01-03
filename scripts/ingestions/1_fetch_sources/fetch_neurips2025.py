import re
import sys
from pathlib import Path

# Add ingestion paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _serialize import save_datasets_deterministically
from local_bids import scan_local_bids_dataset


def main():
    input_dir = Path("data/NeurIPS2025")
    output_file = Path("ingestions/consolidated/neurips2025.json")
    source = "nemar"
    s3_root = "s3://nmdatasets/NeurIPS25"

    datasets = []

    # Pattern to extract i and extra from EEG2025r{i}{extra}
    pattern = re.compile(r"EEG2025r(\d+)(.*)")

    for d in sorted(input_dir.iterdir()):
        if not d.is_dir():
            continue

        match = pattern.match(d.name)
        if not match:
            print(f"Skipping {d.name}: does not match pattern")
            continue

        i, extra = match.groups()
        # Mapping: R{i}_{extra}_L100_bdf
        s3_mapped_name = f"R{i}_{extra}_L100_bdf"
        storage_url = f"{s3_root}/{s3_mapped_name}"

        print(f"Processing {d.name} -> {storage_url}")

        dataset = scan_local_bids_dataset(
            dataset_dir=d, storage_url=storage_url, source=source, modality="eeg"
        )

        if dataset:
            datasets.append(dataset)

    if datasets:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_datasets_deterministically(datasets, output_file)
        print(f"\nSaved {len(datasets)} datasets to {output_file}")
    else:
        print("No datasets found!")


if __name__ == "__main__":
    main()

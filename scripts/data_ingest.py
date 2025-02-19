import argparse
from eegdash import EEGDash

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple command line argument parser")

    # Add arguments
    parser.add_argument('--data', type=str, default="/mnt/nemar/openneuro/ds004186", help="Path to data directory (Default: /mnt/nemar/openneuro/ds004186)")
    parser.add_argument('--dataset', type=str, default="ds004186", help="Dataset name (Default: ds004186)")

    # Parse the arguments
    args = parser.parse_args()
    print('Arguments:', args)

    obj = EEGDash(
        is_public=False,
    )
    # hbn_datasets = ['ds005507','ds005506', 'ds005510', 'ds005512','ds005505','ds005508','ds005509','ds005514','ds005511']
    datasets = ['ds002718']
    for ds in datasets:
        obj.add_bids_dataset(dataset=ds, data_dir=f'/mnt/nemar/openneuro/{ds}', raw_format='eeglab', overwrite=True)

if __name__ == "__main__":
    main()

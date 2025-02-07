import argparse
from eegdash.signalstore_data_utils import SignalstoreOpenneuro

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple command line argument parser")

    # Add arguments
    parser.add_argument('--data', type=str, default="/mnt/nemar/openneuro/ds004186", help="Path to data directory (Default: /mnt/nemar/openneuro/ds004186)")
    parser.add_argument('--dataset', type=str, default="ds004186", help="Dataset name (Default: ds004186)")

    # Parse the arguments
    args = parser.parse_args()
    print('Arguments:', args)

    signalstore = SignalstoreOpenneuro(
        is_public=False,
        local_filesystem=False,
    )
    hbn_datasets = ['ds005514','ds005511','ds005509','ds005508','ds005507','ds005506', 'ds005510', 'ds005512','ds005505']
    # for ds in hbn_datasets:
    signalstore.add_bids_dataset(dataset='ds004186', data_dir=f'/Users/arno/Downloads/old_mar2024/ds004186subset', raw_format='eeglab')

if __name__ == "__main__":
    main()

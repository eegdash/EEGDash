import sys 
sys.path.append('..')
import argparse
from signalstore_data_utils import SignalstoreBIDS

def add_bids_dataset(args):
    signalstore_aws = SignalstoreBIDS(
        dbconnectionstring='mongodb://23.21.113.214:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.1',
        local_filesystem=False,
        project_name='eegdash',
    )
    signalstore_aws.add_bids_dataset(dataset=args.dataset, data_dir=args.data, raw_format='eeglab')

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple command line argument parser")

    # Add arguments
    parser.add_argument('--data', type=str, default="/mnt/nemar/openneuro/ds004186", help="Path to data directory (Default: /mnt/nemar/openneuro/ds004186)")
    parser.add_argument('--dataset', type=str, default="ds004186", help="Dataset name (Default: ds004186)")

    # Parse the arguments
    args = parser.parse_args()
    print('Arguments:', args)

    add_bids_dataset(args)

if __name__ == "__main__":
    main()
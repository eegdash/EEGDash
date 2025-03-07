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
    # datasets = ['ds005507','ds005506', 'ds005510', 'ds005512','ds005505','ds005508','ds005509','ds005514','ds005511']
    hbn_datasets = ['ds005873','ds004855','ds004854','ds004853','ds004852','ds004851','ds004850','ds004849','ds004844','ds004843','ds004842','ds004841','ds004661','ds004660','ds004657','ds004362','ds004123','ds004122','ds004121','ds004120','ds004119','ds004118','ds004117','ds004106','ds004105','ds003645','ds003061','ds002893','ds002691','ds002680','ds002578']
    failed_ds = ['ds005873', "ds004148"]
    hed_datasets = ['ds004853','ds004852','ds004851','ds004850','ds004849','ds004844','ds004843','ds004842','ds004841','ds004661','ds004660','ds004657','ds004362','ds004123','ds004122','ds004121','ds004120','ds004119','ds004118','ds004117','ds004106','ds004105','ds003645','ds003061','ds002893','ds002691','ds002680','ds002578']
    eeglab_datasets = ["ds004362", "ds005514", "ds002181","ds004554", "ds005697", "ds004151", "ds003800", "ds004350","ds004105", "ds004785", "ds004504", "ds004122", "ds004118","ds004121", "ds004635", "ds005787", "ds005512", "ds005079","ds004120", "ds004119", "ds005178", "ds004019", "ds005342","ds004745", "ds004502", "ds005505", "ds005034", "ds004563","ds002680", "ds003774", "ds004123", "ds003805", "ds005506","ds003838", "ds005507", "ds004040", "ds005511", "ds002718","ds002691", "ds003690", "ds003061", "ds005672", "ds003775","ds004106", "ds005410", "ds005508", "ds005510", "ds005509","ds002578", "ds003620"]
    datasets = ["ds002578", "ds003620"]
    for ds in datasets:
        obj.add_bids_dataset(dataset=ds, data_dir=f'/mnt/nemar/openneuro/{ds}', raw_format='eeglab', overwrite=True)

if __name__ == "__main__":
    main()

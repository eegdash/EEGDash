from pathlib import Path
import re
import scipy.io as sio
import numpy as np
import xarray as xr
import os
from os import scandir, walk
from signalstore.store import UnitOfWorkProvider
# from mongomock import MongoClient
from pymongo.mongo_client import MongoClient
from fsspec.implementations.local import LocalFileSystem
from fsspec import get_mapper
from fsspec.implementations.dirfs import DirFileSystem
import fsspec
import mne
import pandas as pd
import json
import s3fs
from data_utils import BIDSDataset
# from dask.distributed import LocalCluster


class SignalstoreBIDS():
    AWS_BUCKET = 'eegdash'
    def __init__(self, 
                 project_name=AWS_BUCKET,
                 dbconnectionstring="mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.3.1",
                 local_filesystem=True,
                 ):
        # tmp_dir = TemporaryDirectory()
        # print(tmp_dir.name)
        # Create data storage location

        # uri = "mongodb+srv://dtyoung112:XbiUEbzmCacjafGu@cluster0.6jtigmc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0" # mongodb free atlas server
        # Create a new client and connect to the server
        client = MongoClient(dbconnectionstring)
        memory_store = {}
        filesystem = self.set_up_filesystem(is_local=local_filesystem)
        self.uow_provider = UnitOfWorkProvider(
            mongo_client=client,
            filesystem=filesystem,
            memory_store=memory_store,
            default_filetype='zarr'
        )

        self.project_name=project_name
        self.uow = self.uow_provider(self.project_name)
        # self.load_domain_models()

    def set_up_filesystem(self, is_local=True):
        if is_local:
            cache_path='/mnt/nemar/dtyoung/eeg-ssl-data/signalstore/hbn'                  # path where signalstore netCDF files are stored
            # Create a directory for the dataset
            store_path = Path(cache_path)
            if not os.path.exists(store_path):
                os.makedirs(store_path)

            filesystem = LocalFileSystem()
            tmp_dir_fs = DirFileSystem(
                store_path,
                filesystem=filesystem
            )
            return tmp_dir_fs
        else:
            s3 = s3fs.S3FileSystem(client_kwargs={'region_name': 'us-east-2'})
            return s3

    def load_domain_models(self):
        cwd = Path.cwd()
        domain_models_path = cwd / f"DomainModels/{self.project_name}/data_models.json"
        metamodel_path = cwd / f"DomainModels/{self.project_name}/metamodels.json"
        property_path = cwd / f"DomainModels/{self.project_name}/property_models.json"
        with open(metamodel_path) as f:
            metamodels = json.load(f)

        with open(property_path) as f:
            property_models = json.load(f)
            
        # load domain models json file
        with open(domain_models_path) as f:
            domain_models = json.load(f)
            
        with self.uow as uow:
            for property_model in property_models:
                uow.domain_models.add(property_model)
                model = uow.domain_models.get(property_model['schema_name'])
                print('property model: ', model['schema_name'])
            for metamodel in metamodels:
                uow.domain_models.add(metamodel)
                model = uow.domain_models.get(metamodel['schema_name'])
                print('meta model: ', model['schema_name'])
            for domain_model in domain_models:
                uow.domain_models.add(domain_model)
                model = uow.domain_models.get(domain_model['schema_name'])
                print('domain model: ', model['schema_name'])
                uow.commit()

    def extract_attribute(self, pattern, filename):
        match = re.search(pattern, filename)
        return match.group(1) if match else None

    def load_eeg_data_from_bids_file(self, bids_dataset: BIDSDataset, bids_file):
        '''
        bids_file must be a file of the bids_dataset
        '''
        if bids_file not in bids_dataset.files:
            raise ValueError(f'{bids_file} not in {bids_dataset.dataset}')

        f = os.path.basename(bids_file)

        attrs = {
            'schema_ref': 'eeg_signal',
            'data_name': f'{bids_dataset.dataset}_{f}',
            # 'dataset': bids_dataset.dataset,
            'subject': bids_dataset.subject(bids_file),
            'task': bids_dataset.task(bids_file),
            'session': bids_dataset.session(bids_file),
            'run': bids_dataset.run(bids_file),
            'modality': 'EEG',
        }

        eeg_data = bids_dataset.load_and_preprocess_raw(bids_file)
        print('data shape:', eeg_data.shape)
    
        fs = bids_dataset.sfreq(bids_file)
        attrs['sampling_frequency'] = fs
        max_time = eeg_data.shape[1] / fs
        time_steps = np.linspace(0, max_time, eeg_data.shape[1]).squeeze() # in seconds
        # print('time steps', len(time_steps))

        # replace eeg.set with channels.tsv
        # todo this is still a hacky way
        channels_tsv = bids_dataset.get_bids_metadata_files(bids_file, 'channels.tsv')
        channels_tsv = Path(channels_tsv[0]) 
        if channels_tsv.exists():
            channels = pd.read_csv(channels_tsv, sep='\t') 
            # get channel names from channel_coords
            channel_names = channels['name'].values

        eeg_xarray = xr.DataArray(
            data=eeg_data,
            dims=['channel','time'],
            coords={
                'time': time_steps,
                'channel': channel_names
            },
            attrs=attrs
        )
        return eeg_xarray

    def exist(self, schema_ref='eeg_signal', data_name=''):
        with self.uow as uow:
            query = {
                "schema_ref": schema_ref,
                "data_name": data_name
            }
            sessions = uow.data.find(query)
            if len(sessions) > 0:
                return True
            else:
                return False

    def add_bids_dataset(self, dataset, data_dir, raw_format='eeglab'):
        bids_dataset = BIDSDataset(
            data_dir=data_dir,
            dataset=dataset,
            raw_format=raw_format,
        )
        for bids_file in bids_dataset.get_files():
            print('bids raw file', bids_file)

            signalstore_data_id = f"{dataset}_{os.path.basename(bids_file)}"
            if self.exist(data_name=signalstore_data_id):
                print('data already exist. skipped')
                continue
            else:
                eeg_xarray = self.load_eeg_data_from_bids_file(bids_dataset, bids_file)
                with self.uow as uow:
                    print('adding data', eeg_xarray.attrs['data_name'])
                    uow.data.add(eeg_xarray)
                    uow.commit()

    def remove_all(self):
        with self.uow as uow:
            sessions = uow.data.find({})
            print(len(sessions))
            for session in range(len(sessions)):
                uow.data.remove(session['schema_ref'], session['data_name'])
                uow.commit()

            uow.purge()
            
            print('Verifying deletion job. Dataset length: ', len(uow.data.find({})))

    def find(self, query:dict, validate=False, get_data=False):
        '''
        query: {
            'dataset': 'dsxxxx',

        }'''
        with self.uow as uow:
            sessions = uow.data.find(query, validate=validate, get_data=get_data)
            if sessions:
                print(f'Found {len(sessions)} records')
                return sessions
            else:
                return []

if __name__ == "__main__":
    # sstore_hbn = SignalstoreHBN()
    # sstore_hbn.add_data()
    # sstore_ds004584 = SignalstoreHBN(
    #     data_path='/mnt/nemar/openneuro/ds004584',
    #     dataset_name='eegdash',
    #     local_filesystem=False,
    #     dbconnectionstring='mongodb://23.21.113.214:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.1'
    # )
    # sstore_ds004584.load_domain_models()
    # sstore_ds004584.add_data()
    pass

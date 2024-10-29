from pathlib import Path
import re
import scipy.io as sio
import numpy as np
import xarray as xr
import os
from os import scandir, walk
from signalstore import UnitOfWorkProvider
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
# from dask.distributed import LocalCluster


class SignalstoreHBN():
    def __init__(self, 
                 data_path='/mnt/nemar/openneuro/ds004186',                                     # path to raw data
                 dataset_name="healthy-brain-network",                                          # TODO right now this is resting state data --> rename it to differentiate between tasks later
                 dbconnectionstring="mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.15",
                 local_filesystem=True,
                 ):
        # tmp_dir = TemporaryDirectory()
        # print(tmp_dir.name)

        # Create data storage location
        self.dataset_name = dataset_name
        self.data_path = Path(data_path)

        
        
        # uri = "mongodb+srv://dtyoung112:XbiUEbzmCacjafGu@cluster0.6jtigmc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0" # mongodb free atlas server
        # Create a new client and connect to the server
        if not dbconnectionstring:
            dbconnectionstring = 'mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.15'
        client = MongoClient(dbconnectionstring)
        memory_store = {}
        filesystem = self.set_up_filesystem(is_local=local_filesystem)
        self.uow_provider = UnitOfWorkProvider(
            mongo_client=client,
            filesystem=filesystem,
            memory_store=memory_store,
            default_filetype='zarr'
        )

        self.uow = self.uow_provider(self.dataset_name)
        # self.load_domain_models()
        # self.add_data()

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
            s3 = s3fs.S3FileSystem()
            return s3

    def load_domain_models(self):
        cwd = Path.cwd()
        domain_models_path = cwd / f"DomainModels/{self.dataset_name}/data_models.json"
        metamodel_path = cwd / f"DomainModels/{self.dataset_name}/metamodels.json"
        property_path = cwd / f"DomainModels/{self.dataset_name}/property_models.json"
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

    def load_eeg_data_from_bids(self, bids_data_path):
        allowed_extentions = ['.set', '.bdf', '.eeg']
        for entry in scandir(bids_data_path):
            if entry.is_dir() and entry.name.startswith('sub-'):
                subject_dir = entry.name
                subject = subject_dir.split('-')[1]
                subject_dir_path = bids_data_path / subject_dir

                for root, dirs, files in os.walk(subject_dir_path):
                    for f in files:
                        # if f ends with any of the allowed extentions
                        # get task from file name from '*task_<taskname>*'
                        if any(f.endswith(ext) for ext in allowed_extentions):
                            attrs = {
                                'schema_ref': 'eeg_signal',
                                'data_name': f'{self.dataset_name}_{f}',
                                'subject': subject,
                                'modality': 'EEG',
                            }

                            task = re.search(r'task-(.*)_', f)
                            session = re.search(r'ses-(.*)_', f)
                            run = re.search(r'run-(.*)_', f)
                            print('task', type(task.group(1)))
                            attrs['task'] = task.group(1) if task else ""
                            # attrs['session'] = session.group(1) if session else ""
                            # attrs['run'] = run.group(1) if run else ""

                            raw_file = Path(root) / f
                            print('raw file', raw_file)

                            if raw_file.name.endswith('.set'):
                                print('reading set file')
                                EEG = mne.io.read_raw_eeglab(os.path.join(raw_file), preload=True)
                                eeg_data = EEG.get_data()
                                print('data shape:', eeg_data.shape)
                        
                            
                            json_file = raw_file.with_suffix('.json')
                            print('json file', json_file.name)
                            if json_file.exists():
                                with open(json_file) as f:
                                    eeg_json = json.load(f)
                                    fs = int(eeg_json['SamplingFrequency'])
                                    attrs['sampling_frequency'] = fs
                                    max_time = eeg_data.shape[1] / fs
                                    time_steps = np.linspace(0, max_time, eeg_data.shape[1]).squeeze() # in seconds
                                    # print('time steps', len(time_steps))

                            # replace eeg.set with channels.tsv
                            channels_tsv = raw_file.with_suffix('.tsv')
                            channels_tsv = channels_tsv.with_name(channels_tsv.name.replace('_eeg', '_channels'))
                            print('channels file', channels_tsv.name)
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
                            yield eeg_xarray

    def add_data(self):
        for eeg_xarray in self.load_eeg_data_from_bids(self.data_path):
            with self.uow_provider(self.dataset_name) as uow:
                query = {
                    "schema_ref": eeg_xarray.attrs['schema_ref'],
                    "data_name": eeg_xarray.attrs['data_name']
                }
                sessions = uow.data.find(query)
                if len(sessions) == 0:
                    print('adding data', eeg_xarray.attrs['data_name'])
                    # if self.__cache_exist(eeg_xarray.attrs['schema_ref'] + '__' + eeg_xarray.attrs['data_name']):
                    #     attrs = eeg_xarray.attrs
                    #     attrs['has_file'] = True
                    #     uow.data.add(attrs)
                    # else:
                    uow.data.add(eeg_xarray)
                    uow.commit()

    def remove_all(self):
        with self.uow_provider(self.dataset_name) as uow:
            sessions = uow.data.find({})
            print(len(sessions))
            for session in range(len(sessions)):
                uow.data.remove(session['schema_ref'], session['data_name'])
                uow.commit()

            uow.purge()
            
            print('Verifying deletion job. Dataset length: ', len(uow.data.find({})))

    def query_data(self, query={}, validate=False, get_data=False):
        with self.uow_provider(self.dataset_name) as uow:
            sessions = uow.data.find(query, validate=validate, get_data=get_data)
            if sessions:
                print(f'Found {len(sessions)} records')
                return sessions
            else:
                return []

    def __cache_exist(self, id):
        print(self.cache_path / (id+".nc"))
        return os.path.exists(self.cache_path / (id+".nc"))

if __name__ == "__main__":
    # sstore_hbn = SignalstoreHBN()
    # sstore_hbn.add_data()
    sstore_ds004584 = SignalstoreHBN(
        data_path='/mnt/nemar/openneuro/ds004584',
        dataset_name='eegdash',
        local_filesystem=False,
        dbconnectionstring='mongodb://23.21.113.214:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.1'
    )
    sstore_ds004584.load_domain_models()
    sstore_ds004584.add_data()

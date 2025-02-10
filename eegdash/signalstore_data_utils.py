from pathlib import Path
from dotenv import load_dotenv
import re
import numpy as np
import xarray as xr
import os
from signalstore.store import UnitOfWorkProvider
# from mongomock import MongoClient
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.dirfs import DirFileSystem
import pandas as pd
import json
import s3fs
from signalstore.store.data_access_objects import FileSystemDAO
from .data_utils import BIDSDataset
import tempfile
import mne
from joblib import Parallel, delayed

class SignalstoreOpenneuro():
    AWS_BUCKET = 's3://openneuro.org'
    PROJECT_NAME = 'eegdash'
    def __init__(self, 
                 dbconnectionstring="mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.3.1",
                 is_public=False,
                 local_filesystem=True,
                 ):
        self.is_public = is_public
        self.project_name = self.PROJECT_NAME
        if is_public:
            dbconnectionstring='mongodb+srv://eegdash-user:mdzoMjQcHWTVnKDq@cluster0.vz35p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
        else:
            load_dotenv()
            dbconnectionstring = os.getenv('DB_CONNECTION_STRING')

        # Create a new client and connect to the server
        client = MongoClient(dbconnectionstring, server_api=ServerApi('1'))
        # Send a ping to confirm a successful connection
        try:
            client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)

        memory_store = {}
        self.filesystem = self.set_up_filesystem(is_local=local_filesystem)
        self.uow_provider = UnitOfWorkProvider(
            mongo_client=client,
            filesystem=self.filesystem,
            memory_store=memory_store,
            default_filetype='zarr'
        )

        self.uow = self.uow_provider(self.PROJECT_NAME)
        self.load_domain_models()

    def set_up_filesystem(self, is_local=True):
        if is_local:
            cache_path='/mnt/nemar/dtyoung/eeg-dash-data'                  # path where signalstore netCDF files are stored
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
            s3 = s3fs.S3FileSystem(anon=True, client_kwargs={'region_name': 'us-east-2'})
            return s3

    def load_domain_models(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cwd = Path(dir_path)
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
                if not uow.domain_models.exists(property_model['schema_name']):
                    uow.domain_models.add(property_model)
                    model = uow.domain_models.get(property_model['schema_name'])
                    print('property model: ', model['schema_name'])
            for metamodel in metamodels:
                if not uow.domain_models.exists(metamodel['schema_name']):
                    uow.domain_models.add(metamodel)
                    model = uow.domain_models.get(metamodel['schema_name'])
                    print('meta model: ', model['schema_name'])
            for domain_model in domain_models:
                if not uow.domain_models.exists(domain_model['schema_name']):
                    uow.domain_models.add(domain_model)
                    model = uow.domain_models.get(domain_model['schema_name'])
                    print('domain model: ', model['schema_name'])
                uow.commit()

    def extract_attribute(self, pattern, filename):
        match = re.search(pattern, filename)
        return match.group(1) if match else None

    def load_eeg_attrs_from_bids_file(self, bids_dataset: BIDSDataset, bids_file):
        '''
        bids_file must be a file of the bids_dataset
        '''
        if bids_file not in bids_dataset.files:
            raise ValueError(f'{bids_file} not in {bids_dataset.dataset}')
        f = os.path.basename(bids_file)
        dsnumber = bids_dataset.dataset
        # extract openneuro path by finding the first occurrence of the dataset name in the filename and remove the path before that
        openneuro_path = dsnumber + bids_file.split(dsnumber)[1]

        attrs = {
            'schema_ref': 'eeg_signal',
            'data_name': f'{bids_dataset.dataset}_{f}',
            'dataset': bids_dataset.dataset,
            'bidspath': openneuro_path,
            'subject': bids_dataset.subject(bids_file),
            'nchans': bids_dataset.num_channels(bids_file),
            'ntimes': bids_dataset.num_times(bids_file),
            'channel_types': bids_dataset.channel_types(bids_file),
            'channel_names': bids_dataset.channel_labels(bids_file),
            'task': bids_dataset.task(bids_file),
            'session': bids_dataset.session(bids_file),
            'run': bids_dataset.run(bids_file),
            'sampling_frequency': bids_dataset.sfreq(bids_file), 
            'modality': 'EEG',
        }

        return attrs

    def load_eeg_data_from_s3(self, s3path):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.set') as tmp:
            with self.filesystem.open(s3path) as s3_file:
                tmp.write(s3_file.read())
            tmp_path = tmp.name
            eeg_data = self.load_eeg_data_from_bids_file(tmp_path)
            os.unlink(tmp_path)
            return eeg_data

    def load_eeg_data_from_bids_file(self,  bids_file, eeg_attrs=None):
        '''
        bids_file must be a file of the bids_dataset
        '''
        EEG = mne.io.read_raw_eeglab(bids_file)
        eeg_data = EEG.get_data()
    
        fs = EEG.info['sfreq']
        max_time = eeg_data.shape[1] / fs
        time_steps = np.linspace(0, max_time, eeg_data.shape[1]).squeeze() # in seconds

        channel_names = EEG.ch_names

        eeg_xarray = xr.DataArray(
            data=eeg_data,
            dims=['channel','time'],
            coords={
                'time': time_steps,
                'channel': channel_names
            },
            # attrs=attrs
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

    def add_bids_dataset(self, dataset, data_dir, raw_format='eeglab', overwrite=True):
        '''
        Create new records for the dataset in the MongoDB database if not found
        '''
        if self.is_public:
            raise ValueError('This operation is not allowed for public users')

        bids_dataset = BIDSDataset(
            data_dir=data_dir,
            dataset=dataset,
            raw_format=raw_format,
        )
        for bids_file in bids_dataset.get_files():
            print('bids raw file', bids_file)

            signalstore_data_id = f"{dataset}_{os.path.basename(bids_file)}"

            if self.exist(data_name=signalstore_data_id):
                if overwrite:
                    eeg_attrs = self.load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
                    with self.uow as uow:
                        # Assume raw data already exists on Openneuro, recreating record only
                        print('updating record', eeg_attrs['data_name'])
                        uow.data.update_record(eeg_attrs)
                        uow.commit()
                else:
                    print('data already exist and not overwriting. skipped')
                    continue
            else:
                eeg_attrs = self.load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
                with self.uow as uow:
                    # Assume raw data already exists on Openneuro, recreating record only
                    eeg_attrs['has_file'] = True
                    print('adding record', eeg_attrs['data_name'])
                    uow.data.add(eeg_attrs)
                    uow.commit()

    def update_bids_dataset(self, dataset, data_dir, raw_format='eeglab'):
        '''
        Create new records for the dataset in the MongoDB database if not found
        '''
        if self.is_public:
            raise ValueError('This operation is not allowed for public users')

        bids_dataset = BIDSDataset(
            data_dir=data_dir,
            dataset=dataset,
            raw_format=raw_format,
        )
        for bids_file in bids_dataset.get_files():
            print('bids raw file', bids_file)

            signalstore_data_id = f"{dataset}_{os.path.basename(bids_file)}"

            if not self.exist(data_name=signalstore_data_id):
                raise ValueError('data not found')
            else:
                self.remove(data_name=signalstore_data_id)

                eeg_attrs = self.load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
                with self.uow as uow:
                    # Assume raw data already exists on Openneuro, recreating record only
                    eeg_attrs['has_file'] = True
                    print('adding record', eeg_attrs['data_name'])
                    uow.data.add(eeg_attrs)
                    uow.commit()

    def remove(self, schema_ref='eeg_signal', data_name=''):
        if self.is_public:
            raise ValueError('This operation is not allowed for public users')

        print('Removing record', data_name)
        with self.uow as uow:
            sessions = uow.data.find({'schema_ref': schema_ref, 'data_name': data_name})
            if len(sessions) > 0:
                for session in sessions:
                    uow.data.remove(session['schema_ref'], session['data_name'])
                    uow.commit()
            uow.purge()
            assert len(uow.data.find({'schema_ref': schema_ref, 'data_name': data_name})) == 0, 'Data still exists'

    def remove_all(self):
        if self.is_public:
            raise ValueError('This operation is not allowed for public users')

        with self.uow as uow:
            sessions = uow.data.find({})
            print(len(sessions))
            for session in range(len(sessions)):
                uow.data.remove(session['schema_ref'], session['data_name'])
                uow.commit()

            uow.purge()
            
            print('Verifying deletion job. Dataset length: ', len(uow.data.find({})))

    def find(self, query:dict, validate=False):
        '''
        query: {
            'dataset': 'dsxxxx',

        }'''
        with self.uow as uow:
            sessions = uow.data.find(query, validate=validate)
            if sessions:
                print(f'Found {len(sessions)} records')
                return sessions
            else:
                return []

    def get_s3path(self, record):
        return f"{self.AWS_BUCKET}/{record['bidspath']}"

    def get(self, query:dict, validate=False):
        '''
        query: {
            'dataset': 'dsxxxx',

        }'''
        with self.uow as uow:
            sessions = uow.data.find(query, validate=validate)
            results = []
            if sessions:
                print(f'Found {len(sessions)} records')
                results = Parallel(n_jobs=-1, prefer="threads", verbose=1)(
                    delayed(self.load_eeg_data_from_s3)(self.get_s3path(session)) for session in sessions
                )
            return results

class SignalstoreBIDS():
    AWS_BUCKET = 'eegdash'
    def __init__(self, 
                 project_name=AWS_BUCKET,
                 dbconnectionstring="mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.3.1",
                 is_public=False,
                 local_filesystem=True,
                 ):
        self.is_public = is_public
        if is_public:
            dbconnectionstring='mongodb+srv://eegdash-user:mdzoMjQcHWTVnKDq@cluster0.vz35p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
        else:
            load_dotenv()
            dbconnectionstring = os.getenv('DB_CONNECTION_STRING')

        # Create a new client and connect to the server
        client = MongoClient(dbconnectionstring, server_api=ServerApi('1'))
        # Send a ping to confirm a successful connection
        try:
            client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)

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
            cache_path='/mnt/nemar/dtyoung/eeg-ssl-data'                  # path where signalstore netCDF files are stored
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
            if self.is_public:
                s3 = s3fs.S3FileSystem(anon=True, client_kwargs={'region_name': 'us-east-2'})
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

    def load_eeg_attrs_from_bids_file(self, bids_dataset: BIDSDataset, bids_file):
        '''
        bids_file must be a file of the bids_dataset
        '''
        if bids_file not in bids_dataset.files:
            raise ValueError(f'{bids_file} not in {bids_dataset.dataset}')
        f = os.path.basename(bids_file)
        attrs = {
            'schema_ref': 'eeg_signal',
            'data_name': f'{bids_dataset.dataset}_{f}',
            'dataset': bids_dataset.dataset,
            'subject': bids_dataset.subject(bids_file),
            'task': bids_dataset.task(bids_file),
            'session': bids_dataset.session(bids_file),
            'run': bids_dataset.run(bids_file),
            'sampling_frequency': bids_dataset.sfreq(bids_file), 
            'modality': 'EEG',
        }

        return attrs

    def load_eeg_data_from_bids_file(self, bids_dataset: BIDSDataset, bids_file, eeg_attrs=None):
        '''
        bids_file must be a file of the bids_dataset
        '''
        if bids_file not in bids_dataset.files:
            raise ValueError(f'{bids_file} not in {bids_dataset.dataset}')

        attrs = self.load_eeg_attrs_from_bids_file(bids_dataset, bids_file) if eeg_attrs is None else eeg_attrs

        eeg_data = bids_dataset.load_and_preprocess_raw(bids_file)
        print('data shape:', eeg_data.shape)
    
        fs = attrs['sampling_frequency']
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

    def add_bids_dataset(self, dataset, data_dir, raw_format='eeglab', overwrite=False, record_only=False):
        if self.is_public:
            raise ValueError('This operation is not allowed for public users')

        bids_dataset = BIDSDataset(
            data_dir=data_dir,
            dataset=dataset,
            raw_format=raw_format,
        )
        for bids_file in bids_dataset.get_files():
            print('bids raw file', bids_file)

            signalstore_data_id = f"{dataset}_{os.path.basename(bids_file)}"
            if overwrite:
                self.remove(signalstore_data_id)

            if self.exist(data_name=signalstore_data_id):
                print('data already exist. skipped')
                continue
            else:
                eeg_attrs = self.load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
                with self.uow as uow:
                    # Assume raw data already exists, recreating record only
                    eeg_attrs['has_file'] = True
                    print('adding record', eeg_attrs['data_name'])
                    uow.data.add(eeg_attrs)
                    uow.commit()
                if  not record_only:
                    eeg_xarray = self.load_eeg_data_from_bids_file(bids_dataset, bids_file, eeg_attrs)
                    with self.uow as uow:
                        print('adding data', eeg_xarray.attrs['data_name'])
                        uow.data.add(eeg_xarray)
                        uow.commit()

    def remove(self, schema_ref='eeg_signal', data_name=''):
        if self.is_public:
            raise ValueError('This operation is not allowed for public users')

        with self.uow as uow:
            sessions = uow.data.find({'schema_ref': schema_ref, 'data_name': data_name})
            if len(session) > 0:
                for session in range(len(sessions)):
                    uow.data.remove(session['schema_ref'], session['data_name'])
                    uow.commit()

    def remove_all(self):
        if self.is_public:
            raise ValueError('This operation is not allowed for public users')

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

    def get(self, query:dict, validate=False):
        '''
        query: {
            'dataset': 'dsxxxx',

        }'''
        with self.uow as uow:
            sessions = uow.data.find(query, validate=validate, get_data=True)
            if sessions:
                print(f'Found {len(sessions)} records')
                return sessions
            else:
                return []

class OpenneuroFileSystemDAO(FileSystemDAO):
    def __init__(self):
        filesystem = s3fs.S3FileSystem(anon=True, client_kwargs={'region_name': 'us-east-2'})
        super().__init__(filesystem, project_dir='openneuro.org')
    
    def get(self, schema_ref, data_name, version_timestamp=0, nth_most_recent=1, data_adapter=None):
        """Gets an object from the Openneuro S3 bucket.
        Arguments:
            schema_ref {str} -- The type of object to get.
            data_name {str} -- The name of the object to get.
            version_timestamp {str} -- The version_timestamp of the object to get.
        Raises:
            FileSystemDAOFileNotFoundError -- If the object is not found.
        Returns:
            dict -- The object.
        """
        self._check_args(
            schema_ref=schema_ref,
            data_name=data_name,
            nth_most_recent=nth_most_recent,
            version_timestamp=version_timestamp,
            data_adapter=data_adapter
            )
        if data_adapter is None:
            data_adapter = self._default_data_adapter
        else:
            data_adapter.set_filesystem(self._fs)
        path = self._get_file_path(schema_ref, data_name, version_timestamp, nth_most_recent, data_adapter)
        if path is None:
            return None
        data_object = data_adapter.read_file(path)
        data_object = self._deserialize(data_object)
        return data_object
    
        
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

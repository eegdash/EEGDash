import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import mne
import numpy as np
import pymongo
import s3fs
import xarray as xr
from braindecode.datasets import BaseConcatDataset, BaseDataset
from dotenv import load_dotenv
from joblib import Parallel, delayed
from pymongo import DeleteOne, InsertOne, MongoClient, UpdateOne

from .data_config import config as data_config
from .data_utils import EEGBIDSDataset, EEGDashBaseDataset, EEGDashBaseRaw


class EEGDash:
    AWS_BUCKET = 's3://openneuro.org'
    def __init__(self, 
                 is_public=True):
        # Load config file
        # config_path = Path(__file__).parent / 'config.json'
        # with open(config_path, 'r') as f:
        #     self.config = json.load(f)

        self.config = data_config
        if is_public:
            DB_CONNECTION_STRING="mongodb+srv://eegdash-user:mdzoMjQcHWTVnKDq@cluster0.vz35p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        else:
            load_dotenv()
            DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING')

        self.__client = pymongo.MongoClient(DB_CONNECTION_STRING)
        self.__db = self.__client['eegdash']
        self.__collection = self.__db['records']   

        self.is_public = is_public
        self.filesystem = s3fs.S3FileSystem(anon=True, client_kwargs={'region_name': 'us-east-2'})
    
    def find(self, *args):
        results = self.__collection.find(*args)
        
        # convert to list using get_item on each element
        return [result for result in results]

    def exist(self, query:dict):
        accepted_query_fields = ['data_name', 'dataset']
        assert all(field in accepted_query_fields for field in query.keys())
        sessions = self.find(query)
        return len(sessions) > 0

    def _validate_input(self, record:dict):
        input_types = {
            'data_name': str,
            'dataset': str,
            'bidspath': str,
            'subject': str,
            'task': str,
            'session': str,
            'run': str,
            'sampling_frequency': float,
            'modality': str,
            'nchans': int,
            'ntimes': int,
            'channel_types': list,
            'channel_names': list,
        }
        if 'data_name' not in record:
            raise ValueError("Missing key: data_name")
        # check if args are in the keys and has correct type
        for key,value in record.items():
            if key not in input_types:
                raise ValueError(f"Invalid input: {key}")
            if not isinstance(value, input_types[key]):
                raise ValueError(f"Invalid input: {key}")

        return record

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

    def get_raw_extensions(self, bids_file, bids_dataset: EEGBIDSDataset):
        bids_file = Path(bids_file)
        extensions = {
            '.set': ['.set', '.fdt'], # eeglab
            '.edf': ['.edf'], # european
            '.vhdr': ['.eeg', '.vhdr', '.vmrk', '.dat', '.raw'], # brainvision
            '.bdf': ['.bdf'], # biosemi
        }
        return [str(bids_dataset.get_relative_bidspath(bids_file.with_suffix(suffix))) for suffix in extensions[bids_file.suffix] if bids_file.with_suffix(suffix).exists()]

    def load_eeg_attrs_from_bids_file(self, bids_dataset: EEGBIDSDataset, bids_file):
        '''
        bids_file must be a file of the bids_dataset
        '''
        if bids_file not in bids_dataset.files:
            raise ValueError(f'{bids_file} not in {bids_dataset.dataset}')

        # Initialize attrs with None values for all expected fields
        attrs = {field: None for field in self.config['attributes'].keys()}

        f = os.path.basename(bids_file)
        dsnumber = bids_dataset.dataset
        # extract openneuro path by finding the first occurrence of the dataset name in the filename and remove the path before that
        openneuro_path = dsnumber + bids_file.split(dsnumber)[1]

        # Update with actual values where available
        try:
            participants_tsv = bids_dataset.subject_participant_tsv(bids_file)
        except Exception as e:
            print(f"Error getting participants_tsv: {str(e)}")
            participants_tsv = None
            
        try:
            eeg_json = bids_dataset.eeg_json(bids_file)
        except Exception as e:
            print(f"Error getting eeg_json: {str(e)}")
            eeg_json = None
            
        bids_dependencies_files = self.config['bids_dependencies_files']
        bidsdependencies = []
        for extension in bids_dependencies_files:
            try:
                dep_path = bids_dataset.get_bids_metadata_files(bids_file, extension)
                dep_path = [str(bids_dataset.get_relative_bidspath(dep)) for dep in dep_path]
                bidsdependencies.extend(dep_path)
            except Exception as e:
                pass
                
        bidsdependencies.extend(self.get_raw_extensions(bids_file, bids_dataset))

        # Define field extraction functions with error handling
        field_extractors = {
            'data_name': lambda: f'{bids_dataset.dataset}_{f}',
            'dataset': lambda: bids_dataset.dataset,
            'bidspath': lambda: openneuro_path,
            'subject': lambda: bids_dataset.get_bids_file_attribute('subject', bids_file),
            'task': lambda: bids_dataset.get_bids_file_attribute('task', bids_file),
            'session': lambda: bids_dataset.get_bids_file_attribute('session', bids_file),
            'run': lambda: bids_dataset.get_bids_file_attribute('run', bids_file),
            'modality': lambda: bids_dataset.get_bids_file_attribute('modality', bids_file),
            'sampling_frequency': lambda: bids_dataset.get_bids_file_attribute('sfreq', bids_file),
            'nchans': lambda: bids_dataset.get_bids_file_attribute('nchans', bids_file),
            'ntimes': lambda: bids_dataset.get_bids_file_attribute('ntimes', bids_file),
            'participant_tsv': lambda: participants_tsv,
            'eeg_json': lambda: eeg_json,
            'bidsdependencies': lambda: bidsdependencies,
        }
        
        # Dynamically populate attrs with error handling
        for field, extractor in field_extractors.items():
            try:
                attrs[field] = extractor()
            except Exception as e:
                print(f"Error extracting {field}: {str(e)}")
                attrs[field] = None

        return attrs

    def add_bids_dataset(self, dataset, data_dir, overwrite=True):
        '''
        Create new records for the dataset in the MongoDB database if not found
        '''
        if self.is_public:
            raise ValueError('This operation is not allowed for public users')

        if not overwrite and self.exist({'dataset': dataset}):
            print(f'Dataset {dataset} already exists in the database')
            return
        try:
            bids_dataset = EEGBIDSDataset(
                data_dir=data_dir,
                dataset=dataset,
            )
        except Exception as e:
            print(f'Error creating bids dataset {dataset}: {str(e)}')
            raise e
        requests = []
        for bids_file in bids_dataset.get_files():
            try:
                data_id = f"{dataset}_{os.path.basename(bids_file)}"

                if self.exist({'data_name':data_id}):
                    if overwrite:
                        eeg_attrs = self.load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
                        requests.append(self.update_request(eeg_attrs))
                else:
                    eeg_attrs = self.load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
                    requests.append(self.add_request(eeg_attrs))
            except:
                print('error adding record', bids_file)

        print('Number of database requests', len(requests))

        if requests:
            result = self.__collection.bulk_write(requests, ordered=False)
            print(f"Inserted: {result.inserted_count}")
            print(f"Modified: {result.modified_count}")
            print(f"Deleted: {result.deleted_count}")
            print(f"Upserted: {result.upserted_count}")
            print(f"Errors: {result.bulk_api_result.get('writeErrors', [])}")

    def get(self, query:dict):
        '''
        query: {
            'dataset': 'dsxxxx',

        }'''
        sessions = self.find(query)
        results = []
        if sessions:
            print(f'Found {len(sessions)} records')
            results = Parallel(n_jobs=-1 if len(sessions) > 1 else 1, prefer="threads", verbose=1)(
                delayed(self.load_eeg_data_from_s3)(self.get_s3path(session)) for session in sessions
            )
        return results

    def add_request(self, record:dict):
        return InsertOne(record)

    def add(self, record:dict):
        try:
            # input_record = self._validate_input(record)
            self.__collection.insert_one(record)
        # silent failing
        except ValueError as e:
            print(f"Failed to validate record: {record['data_name']}")
            print(e)
        except: 
            print(f"Error adding record: {record['data_name']}")

    def update_request(self, record:dict):
        return UpdateOne({'data_name': record['data_name']}, {'$set': record})

    def update(self, record:dict):
        try:
            self.__collection.update_one({'data_name': record['data_name']}, {'$set': record})
        except: # silent failure
            print(f'Error updating record {record["data_name"]}')

    def remove_field(self, record, field):
        self.__collection.update_one({'data_name': record['data_name']}, {'$unset': {field: 1}})
    
    def remove_field_from_db(self, field):
        self.__collection.update_many({}, {'$unset': {field: 1}})
    
    @property
    def collection(self):
        return self.__collection

class EEGDashDataset(BaseConcatDataset):
    # CACHE_DIR = '.eegdash_cache'
    def __init__(
        self,
        query:dict=None,
        data_dir:str | list =None,
        dataset:str | list =None,
        description_fields: list[str]=['subject', 'session', 'run', 'task', 'age', 'gender', 'sex'],
        cache_dir:str='.eegdash_cache',
        **kwargs
    ):
        self.cache_dir = cache_dir
        if query:
            datasets = self.find_datasets(query, description_fields, **kwargs)
        elif data_dir:
            if type(data_dir) == str:
                datasets = self.load_bids_dataset(dataset, data_dir, description_fields)
            else:
                assert len(data_dir) == len(dataset), 'Number of datasets and their directories must match' 
                datasets = []
                for i in range(len(data_dir)):
                    datasets.extend(self.load_bids_dataset(dataset[i], data_dir[i], description_fields))
        # convert to list using get_item on each element
        super().__init__(datasets)

    
    def find_key_in_nested_dict(self, data, target_key):
        if isinstance(data, dict):
            if target_key in data:
                return data[target_key]
            for value in data.values():
                result = self.find_key_in_nested_dict(value, target_key)
                if result is not None:
                    return result
        return None

    def find_datasets(self, query:dict, description_fields:list[str], **kwargs):
        eegdashObj = EEGDash()
        datasets = []
        for record in eegdashObj.find(query):
            description = {}
            for field in description_fields:
                value = self.find_key_in_nested_dict(record, field)
                if value:
                    description[field] = value
            datasets.append(EEGDashBaseDataset(record, self.cache_dir, description=description, **kwargs))
        return datasets

    def load_bids_dataset(self, dataset, data_dir, description_fields: list[str],raw_format='eeglab', **kwargs):
        '''
        '''
        def get_base_dataset_from_bids_file(bids_dataset, bids_file):
            record = eegdashObj.load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
            description = {}
            for field in description_fields:
                value = self.find_key_in_nested_dict(record, field)
                if value:
                    description[field] = value
            return EEGDashBaseDataset(record, self.cache_dir, description=description, **kwargs)

        bids_dataset = EEGBIDSDataset(
            data_dir=data_dir,
            dataset=dataset,
            raw_format=raw_format,
        )
        eegdashObj = EEGDash()
        datasets = Parallel(n_jobs=-1, prefer="threads", verbose=1)(
                delayed(get_base_dataset_from_bids_file)(bids_dataset, bids_file) for bids_file in bids_dataset.get_files()
            )
        return datasets

def main():
    eegdash = EEGDash()
    record = eegdash.find({'dataset': 'ds005511', 'subject': 'NDARUF236HM7'})
    print(record)

if __name__ == '__main__':
    main()
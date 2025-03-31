from typing import List
import pymongo
from dotenv import load_dotenv
import os
from pathlib import Path
import s3fs
from joblib import Parallel, delayed
import tempfile
import mne
import numpy as np
import xarray as xr
from .data_utils import BIDSDataset, EEGDashBaseRaw, EEGDashBaseDataset
from braindecode.datasets import BaseDataset, BaseConcatDataset
from collections import defaultdict
from pymongo import MongoClient, InsertOne, UpdateOne, DeleteOne

class EEGDash:
    AWS_BUCKET = 's3://openneuro.org'
    def __init__(self, 
                 is_public=True):
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

    def exist(self, data_name=''):
        query = {
            "data_name": data_name
        }
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

        bids_dependencies_files = ['dataset_description.json', 'participants.tsv', 'events.tsv', 'events.json', 'eeg.json', 'electrodes.tsv', 'channels.tsv', 'coordsystem.json']
        bidsdependencies = []
        for extension in bids_dependencies_files:
            dep_path = bids_dataset.get_bids_metadata_files(bids_file, extension)
            dep_path = [str(bids_dataset.get_relative_bidspath(dep)) for dep in dep_path]

            bidsdependencies.extend(dep_path)

        participants_tsv = bids_dataset.subject_participant_tsv(bids_file)
        eeg_json = bids_dataset.eeg_json(bids_file)
        attrs = {
            'data_name': f'{bids_dataset.dataset}_{f}',
            'dataset': bids_dataset.dataset,
            'bidspath': openneuro_path,
            'subject': bids_dataset.subject(bids_file),
            'task': bids_dataset.task(bids_file),
            'session': bids_dataset.session(bids_file),
            'run': bids_dataset.run(bids_file),
            'modality': 'EEG',
            'nchans': bids_dataset.num_channels(bids_file),
            'ntimes': bids_dataset.num_times(bids_file),
            'participant_tsv': participants_tsv,
            'eeg_json': eeg_json,
            'bidsdependencies': bidsdependencies,
        }

        return attrs

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
        requests = []
        for bids_file in bids_dataset.get_files():
            try:
                data_id = f"{dataset}_{os.path.basename(bids_file)}"

                if self.exist(data_name=data_id):
                    if overwrite:
                        eeg_attrs = self.load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
                        requests.append(UpdateOne(self.update_request(eeg_attrs)))
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
    

class EEGDashDataset(BaseConcatDataset):
    CACHE_DIR = '.eegdash_cache'
    def __init__(
        self,
        query:dict=None,
        data_dir:str | list =None,
        dataset:str | list =None,
        description_fields: list[str]=['subject', 'session', 'run', 'task', 'age', 'gender', 'sex'],
        **kwargs
    ):
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
            datasets.append(EEGDashBaseDataset(record, self.CACHE_DIR, description=description, **kwargs))
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
            return EEGDashBaseDataset(record, self.CACHE_DIR, description=description, **kwargs)

        bids_dataset = BIDSDataset(
            data_dir=data_dir,
            dataset=dataset,
            raw_format=raw_format,
        )
        eegdashObj = EEGDash()
        datasets = Parallel(n_jobs=-1, prefer="threads", verbose=1)(
                delayed(get_base_dataset_from_bids_file)(bids_dataset, bids_file) for bids_file in bids_dataset.get_files()
            )
        # datasets = []
        # for bids_file in bids_dataset.get_files():
        #     record = eegdashObj.load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
        #     description = {}
        #     for field in description_fields:
        #         value = self.find_key_in_nested_dict(record, field)
        #         if value:
        #             description[field] = value
        #     datasets.append(EEGDashBaseDataset(record, self.CACHE_DIR, description=description, **kwargs))
        return datasets

def main():
    eegdash = EEGDash()
    record = eegdash.find({'dataset': 'ds005511', 'subject': 'NDARUF236HM7'})
    print(record)

if __name__ == '__main__':
    main()
from typing import List
import pymongo
from dotenv import load_dotenv
import os
import s3fs
from joblib import Parallel, delayed
import tempfile
import mne
import numpy as np
import xarray as xr
from .data_utils import BIDSDataset, EEGDashBaseRaw
from braindecode.datasets import BaseDataset, BaseConcatDataset

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
    
    def findrecord(self, *args):
        results = self.__collection.find(*args)
        
        # convert to list using get_item on each element
        return [result for result in results]

    def find(self, query:dict, description_fields:List[str]=['sampling_frequency', 'nchans', 'ntimes']):
        datasets = []
        for record in self.findrecord(query):
            sfreq = record['sampling_frequency']
            nchans = record['nchans']
            ntimes = record['ntimes']
            ch_names = record['channel_names']
            ch_types = record['channel_types']
            s3_path = self.get_s3path(record)
            description = {}
            for field in description_fields:
                description[field] = record[field]
            datasets.append(BaseDataset(EEGDashBaseRaw(s3_path, {'sfreq': sfreq, 'nchans': nchans, 'n_times': ntimes, 'ch_types': ch_types, 'ch_names': ch_names}, preload=False),
                                        description=description)) 
        # convert to list using get_item on each element
        if len(datasets) == 0:
            return []
        else:
            return BaseConcatDataset(datasets)
    def exist(self, data_name=''):
        query = {
            "data_name": data_name
        }
        sessions = self.find(query)
        return len(sessions) > 0

    def add(self, record:dict):
        try:
            input_record = self._validate_input(record)
            self.__collection.insert_one(input_record)
        # silent failing
        except ValueError as e:
            print(f"Failed to validate record: {record['data_name']}")
            print(e)
        except: 
            print(f"Error adding record: {record['data_name']}")

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

        participants_tsv = bids_dataset.subject_participant_tsv(bids_file)
        eeg_json = bids_dataset.eeg_json(bids_file)
        channel_tsv = bids_dataset.channel_tsv(bids_file)
        attrs = {
            'data_name': f'{bids_dataset.dataset}_{f}',
            'dataset': bids_dataset.dataset,
            'bidspath': openneuro_path,
            'subject': bids_dataset.subject(bids_file),
            'task': bids_dataset.task(bids_file),
            'session': bids_dataset.session(bids_file),
            'run': bids_dataset.run(bids_file),
            'modality': 'EEG',
            'participant_tsv': participants_tsv,
            'eeg_json': eeg_json,
            'channel_tsv': channel_tsv,
            'rawdatainfo': {
                'sampling_frequency': bids_dataset.sfreq(bids_file), 
                'nchans': bids_dataset.num_channels(bids_file),
                'ntimes': bids_dataset.num_times(bids_file),
                'channel_types': bids_dataset.channel_types(bids_file),
                'channel_names': bids_dataset.channel_labels(bids_file),
            },
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
        for bids_file in bids_dataset.get_files():
            print('bids raw file', bids_file)

            signalstore_data_id = f"{dataset}_{os.path.basename(bids_file)}"

            if self.exist(data_name=signalstore_data_id):
                if overwrite:
                    eeg_attrs = self.load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
                    print('updating record', eeg_attrs['data_name'])
                    self.update(eeg_attrs)
                else:
                    print('data already exist and not overwriting. skipped')
                    continue
            else:
                eeg_attrs = self.load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
                # Assume raw data already exists on Openneuro, recreating record only
                print('adding record', eeg_attrs['data_name'])
                self.add(eeg_attrs)

    def get_s3path(self, record):
        return f"{self.AWS_BUCKET}/{record['bidspath']}"

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

    def update(self, record:dict):
        try:
            self.__collection.update_one({'schema_ref': 'eeg_signal', 'data_name': record['data_name']}, {'$set': record})
        except: # silent failure
            print(f'Error updating record {record["data_name"]}')

def main():
    eegdash = EEGDash()
    record = eegdash.find({'dataset': 'ds005511', 'subject': 'NDARUF236HM7'})
    print(record)

if __name__ == '__main__':
    main()
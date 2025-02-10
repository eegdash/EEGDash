from eegdash.signalstore_data_utils import SignalstoreBIDS

class EEGDash:
    def __init__(self):
        
        DB_CONNECTION_STRING="mongodb+srv://eegdash-user:mdzoMjQcHWTVnKDq@cluster0.vz35p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        import pymongo
        self.client = pymongo.MongoClient(DB_CONNECTION_STRING)
        self.db = self.client['eegdash']
        self.collection = self.db['records']   
    
    def find(self, *args):
        results = self.collection.find(*args)
        
        # convert to list using get_item on each element
        return [result for result in results]
    
    def get(self, *args):
        
        # repackage to BrainDecodeDataset
        
        # sessions = self.sstore.get(*args) # return xarray.Dataset
        
        # Download from S3
        
        # return BrainDecodeDataset(sessions)
        
        return self.sstore.get(*args)

def main():
    eegdash = EEGDash()
    record = eegdash.find({'dataset': 'ds005511', 'subject': 'NDARUF236HM7'})
    print(record)

if __name__ == '__main__':
    main()
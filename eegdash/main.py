from eegdash.signalstore_data_utils import SignalstoreBIDS

class EEGDash:
    def __init__(self):
        self.sstore = SignalstoreBIDS(
            # dbconnectionstring='mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.3.1',
            dbconnectionstring='mongodb+srv://eegdash-user:mdzoMjQcHWTVnKDq@cluster0.vz35p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
            is_public=True,
            local_filesystem=False,
            project_name='eegdash'
        )
    
    def find(self, *args):
        return self.sstore.find(*args)
    
    def get(self, *args):
        return self.sstore.get(*args)

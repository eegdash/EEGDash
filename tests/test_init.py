from torch.utils.data import Dataset

from eegdash import EEGDash, EEGDashDataset
import unittest

class TestEEGDash(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.eeg_dash = EEGDash()

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        if hasattr(self, 'eeg_dash'):
            self.eeg_dash.close()

    def test_set_import_instanciate_eegdash(self):
        assert isinstance(self.eeg_dash, EEGDash)

        eeg_pytorch_dataset_instance = EEGDashDataset(
            {"dataset": "ds005514", "task": "RestingState", "subject": "NDARDB033FW5"}
        )
        assert isinstance(eeg_pytorch_dataset_instance, Dataset)

    def test_dataset_api(self):
        record = self.eeg_dash.find({"dataset": "ds005511", "subject": "NDARUF236HM7"})
        assert isinstance(record, list)

    def test_number_recordings(self):
        records = self.eeg_dash.find({})
        assert isinstance(records, list)
        assert len(records) >= 55088
        # As of the last known count in 9 of jun of 2025, there are 55088 recordings in the dataset

if __name__ == "__main__":
    unittest.main()
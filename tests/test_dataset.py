import pytest

from eegdash.dataset import EEGChallengeDataset


@pytest.mark.skip("Skipping test for EEGChallengeDataset initialization")
def test_eeg_challenge_dataset_initialization():
    """Test the initialization of EEGChallengeDataset."""
    dataset = EEGChallengeDataset(release="R5")
    assert dataset.s3_bucket == "s3://nmdatasets/NeurIPS25/R5_L100"
    assert (
        dataset.datasets[0].s3file
        == "s3://nmdatasets/NeurIPS25/R5_L100/ds005509/sub-NDARFB322DRA/eeg/sub-NDARFB322DRA_task-RestingState_eeg.set"
    )

from eegdash.dataset import EEGChallengeDataset

dataset_dir = "/Users/baristim/mne_data/eeg_challenge_completed"
dataset = EEGChallengeDataset(cache_dir=dataset_dir, release="R3", mini=False)

print(dataset)

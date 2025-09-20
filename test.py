from eegdash.dataset import EEGChallengeDataset
from eegdash.paths import get_default_cache_dir

cache_dir = get_default_cache_dir()
dataset = EEGChallengeDataset(release="R5", cache_dir=cache_dir)

print(dataset)
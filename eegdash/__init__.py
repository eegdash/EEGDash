from .api import EEGDash, EEGDashDataset
from .dataset import EEGChallengeDataset
from .hbn import preprocessing
from .utils import __init__mongo_client

__init__mongo_client()


__all__ = ["EEGDash", "EEGDashDataset", "EEGChallengeDataset", "preprocessing"]

__version__ = "0.3.7"

from .api import EEGDash, EEGDashDataset
from .dataset import EEGChallengeDataset
from .hbn import preprocessing
from .utils import _init_mongo_client

_init_mongo_client()

__all__ = ["EEGDash", "EEGDashDataset", "EEGChallengeDataset", "preprocessing"]

__version__ = "0.3.9"
